from __future__ import annotations

import hashlib
import http.server
import json
import socket
import struct
import threading
import time
import uuid
from pathlib import Path

import pytest

import freetoken.checkpoint.step9b_executor as executor_module
from freetoken.checkpoint.step9b_executor import (
    ACCEPTED_SOURCE_INVENTORY,
    AcquisitionManifest,
    BodyTransferDisabled,
    Downloader,
    ExecutorError,
    MIN_DISK_RESERVE_BYTES,
    ResumeRejected,
    SourceEntry,
    Step9BExecutor,
    TransferBudget,
    UrllibTransport,
    _atomic_json,
    _publish_component_receipt,
    _receipt_matches,
)


PIN = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
WORKSPACE = Path("Z:/Qwen38-FlashNext-Cluster")


def zroot(label: str) -> Path:
    path = WORKSPACE / "artifacts" / "stage7f-test-fixtures" / f"{label}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def source_entry(name: str, body: bytes, *, etag: str = "fixture-etag", safetensors: bool = False) -> SourceEntry:
    header_length = header_sha = None
    if safetensors:
        (header_length,) = struct.unpack("<Q", body[:8])
        header_sha = hashlib.sha256(body[8 : 8 + header_length]).hexdigest()
    return SourceEntry(
        filename=name,
        byte_length=len(body),
        source_class="PLE",
        acquisition_order=1,
        repository=REPO,
        revision=PIN,
        accepted_etag=f'"{etag}"',
        lfs_oid_sha256=hashlib.sha256(body).hexdigest(),
        accepted_header_length=header_length,
        accepted_header_sha256=header_sha,
    )


def small_manifest(row: SourceEntry, *, cap: int = 1_000_000) -> AcquisitionManifest:
    return AcquisitionManifest(REPO, PIN, (row,), (), ACCEPTED_SOURCE_INVENTORY, row.byte_length, cap)


class FixtureServer:
    def __init__(self, body: bytes, *, etag: str = "fixture-etag", mode: str = "normal", interrupt_at: int = 0):
        self.body = body
        self.etag = etag
        self.mode = mode
        self.interrupt_at = interrupt_at
        self.requests: list[tuple[str, str | None, str | None]] = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_HEAD(self):
                owner.requests.append(("HEAD", self.headers.get("Range"), self.headers.get("If-Range")))
                self.send_response(200)
                self.send_header("Content-Length", str(len(owner.body)))
                if owner.mode != "missing_head_etag":
                    self.send_header("ETag", f'"{owner.etag}"')
                self.end_headers()

            def do_GET(self):
                range_header = self.headers.get("Range")
                owner.requests.append(("GET", range_header, self.headers.get("If-Range")))
                start = int(range_header.removeprefix("bytes=").split("-")[0]) if range_header else 0
                if range_header and owner.mode == "ignore_range":
                    start = 0
                    status = 200
                else:
                    status = 206 if range_header else 200
                payload = owner.body[start:]
                if owner.mode == "oversized":
                    payload += b"!"
                if owner.mode == "undersized":
                    payload = payload[:-1]
                self.send_response(status)
                if owner.mode != "missing_etag":
                    self.send_header("ETag", f'"{owner.etag}"')
                self.send_header("Content-Length", str(len(payload)))
                if status == 206:
                    end = len(owner.body) - 1
                    if owner.mode == "malformed_range":
                        end -= 1
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(owner.body)}")
                self.end_headers()
                if owner.mode == "interrupt":
                    self.wfile.write(payload[: owner.interrupt_at])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                self.wfile.write(payload)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/fixture"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class LocalDownloader(Downloader):
    def __init__(self, *args, server: FixtureServer, **kwargs):
        self.server = server
        super().__init__(*args, transport=UrllibTransport(), **kwargs)

    def _url(self, row: SourceEntry) -> str:
        return self.server.url

    def resolve_hf_metadata(self, row: SourceEntry) -> dict:
        return {"commit": PIN, "size": row.byte_length, "etag": row.lfs_oid_sha256, "xet_file_hash": row.accepted_etag.strip('"'), "body_bytes": 0}


def local_downloader(server: FixtureServer, row: SourceEntry, root: Path, *, cap: int = 1_000_000) -> LocalDownloader:
    return LocalDownloader(
        root,
        small_manifest(row, cap=cap),
        server=server,
        execute=True,
        allow_network_body=True,
        budget=TransferBudget(cap, state_path=root / "budget.json"),
    )


def test_local_http_clean_download_promotes_and_receipts():
    body = b"contract-complete-body"
    row = source_entry("fixture.bin", body)
    root = zroot("http-clean")
    with FixtureServer(body) as server:
        result = local_downloader(server, row, root).acquire(row)
    assert result["state"] == "SOURCE_COMPLETE"
    assert (root / row.filename).read_bytes() == body
    assert not (root / f"{row.filename}.partial").exists()
    receipts = list((root / ".step9b-receipts").glob("*.receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["completion"] == "SOURCE_COMPLETE"
    assert receipt["resolved_commit"] == PIN
    assert receipt["source_inventory_fingerprint"] == ACCEPTED_SOURCE_INVENTORY


def test_body_without_required_etag_fails_before_promotion():
    body = b"missing-etag"
    row = source_entry("missing-etag.bin", body)
    root = zroot("missing-etag")
    with FixtureServer(body, mode="missing_etag") as server:
        with pytest.raises(ExecutorError, match="omitted required ETag"):
            local_downloader(server, row, root).acquire(row)
    assert not (root / row.filename).exists()


def test_head_without_required_etag_fails_before_body():
    body = b"missing-head-etag"
    row = source_entry("missing-head-etag.bin", body)
    root = zroot("missing-head-etag")
    with FixtureServer(body, mode="missing_head_etag") as server:
        downloader = local_downloader(server, row, root)
        downloader.resolve_hf_metadata = lambda _row: downloader.validate_metadata(_row, downloader.transport.head(server.url))
        with pytest.raises(ExecutorError, match="omitted required ETag"):
            downloader.acquire(row)
    assert not any(request[0] == "GET" for request in server.requests)


def test_transfer_budget_exact_cap_passes_and_next_received_byte_is_counted():
    body = b"exact-cap"
    row = source_entry("exact-cap.bin", body)
    root = zroot("exact-cap")
    with FixtureServer(body) as server:
        result = local_downloader(server, row, root, cap=len(body)).acquire(row)
    assert result["state"] == "SOURCE_COMPLETE"
    persisted = TransferBudget(len(body), state_path=root / "budget.json")
    assert persisted.transferred == len(body)
    with pytest.raises(ExecutorError, match="transfer cap exceeded"):
        persisted.reserve(1)
    assert TransferBudget(len(body), state_path=root / "budget.json").transferred == len(body) + 1


def test_interrupted_download_persists_partial_then_exact_resume():
    body = b"0123456789abcdef"
    row = source_entry("resume.bin", body)
    root = zroot("http-resume")
    with FixtureServer(body, mode="interrupt", interrupt_at=5) as server:
        downloader = local_downloader(server, row, root)
        with pytest.raises(ExecutorError):
            downloader.acquire(row)
    partial = root / "resume.bin.partial"
    meta = root / "resume.bin.partial.meta.json"
    assert partial.read_bytes() == body[:5]
    assert json.loads(meta.read_text())["partial_length"] == 5
    with FixtureServer(body) as server:
        result = local_downloader(server, row, root).acquire(row)
        get = [request for request in server.requests if request[0] == "GET"][-1]
    assert get[1] == "bytes=5-"
    assert get[2] == row.lfs_oid_sha256
    assert result["resumed_from"] == 5
    assert (root / "resume.bin").read_bytes() == body


def test_complete_partial_is_revalidated_and_promoted_without_another_body():
    body = b"complete-partial"
    row = source_entry("complete-partial.bin", body)
    root = zroot("complete-partial")
    partial = root / f"{row.filename}.partial"
    partial.write_bytes(body)
    with FixtureServer(body) as server:
        downloader = local_downloader(server, row, root)
        remote = downloader.resolve_hf_metadata(row)
        _atomic_json(partial.with_name(partial.name + ".meta.json"), downloader._identity(row, len(body), remote))
        result = downloader.acquire(row)
    assert result["recovered_complete_partial"] is True
    assert result["body_bytes_this_run"] == 0
    assert (root / row.filename).read_bytes() == body
    assert not any(request[0] == "GET" for request in server.requests)


def test_crash_after_source_promotion_before_receipt_recovers_without_body(monkeypatch):
    body = b"promotion-crash"
    row = source_entry("promotion-crash.bin", body)
    root = zroot("promotion-crash")
    partial = root / f"{row.filename}.partial"
    partial.write_bytes(body)
    with FixtureServer(body) as server:
        downloader = local_downloader(server, row, root)
        remote = downloader.resolve_hf_metadata(row)
        identity = partial.with_name(partial.name + ".meta.json")
        _atomic_json(identity, downloader._identity(row, len(body), remote))
        real_atomic = executor_module._atomic_json

        def fail_receipt(path, value):
            if path.name.endswith(".receipt.json"):
                raise OSError("injected receipt publication crash")
            return real_atomic(path, value)

        monkeypatch.setattr(executor_module, "_atomic_json", fail_receipt)
        with pytest.raises(OSError, match="publication crash"):
            downloader.acquire(row)
        assert (root / row.filename).read_bytes() == body
        assert not list((root / ".step9b-receipts").glob("*.receipt.json"))
        monkeypatch.setattr(executor_module, "_atomic_json", real_atomic)
        recovered = local_downloader(server, row, root).acquire(row)
    assert recovered["state"] == "SKIP_VALID_FINAL"
    assert not any(request[0] == "GET" for request in server.requests)


@pytest.mark.parametrize("mode,error", [("ignore_range", ResumeRejected), ("malformed_range", ResumeRejected)])
def test_resume_rejects_invalid_range_semantics(mode, error):
    body = b"abcdefghijk"
    row = source_entry("range.bin", body)
    root = zroot(mode)
    partial = root / "range.bin.partial"
    partial.write_bytes(body[:3])
    with FixtureServer(body, mode=mode) as server:
        downloader = local_downloader(server, row, root)
        _atomic_json(partial.with_name(partial.name + ".meta.json"), downloader._identity(row, 3, downloader.resolve_hf_metadata(row)))
        with pytest.raises(error):
            downloader.acquire(row)
    assert partial.read_bytes() == body[:3]


def test_etag_drift_rejects_partial_before_body():
    body = b"etag-drift"
    row = source_entry("etag.bin", body, etag="old")
    root = zroot("etag")
    partial = root / "etag.bin.partial"
    partial.write_bytes(body[:2])
    with FixtureServer(body, etag="new") as server:
        downloader = local_downloader(server, row, root)
        old = {"commit": PIN, "etag": '"old"', "xet_file_hash": "old"}
        _atomic_json(partial.with_name(partial.name + ".meta.json"), downloader._identity(row, 2, old))
        downloader.resolve_hf_metadata = lambda _row: {"commit": PIN, "etag": '"new"', "xet_file_hash": "new"}
        with pytest.raises(ResumeRejected):
            downloader.acquire(row)
    assert not any(request[0] == "GET" for request in server.requests)


@pytest.mark.parametrize("mode", ["oversized", "undersized"])
def test_wrong_body_length_never_promotes(mode):
    body = b"length-contract"
    row = source_entry("length.bin", body)
    root = zroot(mode)
    with FixtureServer(body, mode=mode) as server:
        with pytest.raises(ExecutorError):
            local_downloader(server, row, root).acquire(row)
    assert not (root / row.filename).exists()


def test_sha_and_safetensors_header_mismatch_never_promote():
    header = b'{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    body = struct.pack("<Q", len(header)) + header + b"x"
    row = source_entry("header.safetensors", body, safetensors=True)
    bad_row = SourceEntry(**{**row.__dict__, "accepted_header_sha256": "0" * 64})
    root = zroot("header")
    with FixtureServer(body) as server:
        with pytest.raises(ExecutorError, match="header hash"):
            local_downloader(server, bad_row, root).acquire(bad_row)
    assert not (root / bad_row.filename).exists()
    sha_row = SourceEntry(**{**row.__dict__, "accepted_header_sha256": row.accepted_header_sha256, "lfs_oid_sha256": "0" * 64})
    with FixtureServer(body) as server:
        with pytest.raises(ExecutorError, match="SHA/LFS"):
            local_downloader(server, sha_row, zroot("sha")).acquire(sha_row)


def test_existing_valid_final_recovers_receipt_without_body():
    body = b"already-complete"
    row = source_entry("complete.bin", body)
    root = zroot("recover-source")
    (root / row.filename).write_bytes(body)
    with FixtureServer(body) as server:
        result = local_downloader(server, row, root).acquire(row)
    assert result["state"] == "SKIP_VALID_FINAL"
    assert not server.requests
    receipt = next((root / ".step9b-receipts").glob("*.receipt.json"))
    assert json.loads(receipt.read_text())["recovered_after_promotion"] is True


def test_invalid_existing_final_is_preserved_and_rejected():
    body = b"expected"
    row = source_entry("invalid.bin", body)
    root = zroot("invalid-final")
    final = root / row.filename
    final.write_bytes(b"corrupt!")
    with FixtureServer(body) as server:
        with pytest.raises(ExecutorError):
            local_downloader(server, row, root).acquire(row)
    assert final.read_bytes() == b"corrupt!"
    assert not server.requests


def test_transfer_budget_persists_and_counts_rejected_byte():
    root = zroot("budget")
    state = root / "budget.json"
    first = TransferBudget(3, state_path=state)
    first.reserve(3)
    second = TransferBudget(3, state_path=state)
    assert second.transferred == 3
    with pytest.raises(ExecutorError):
        second.reserve(1)
    assert json.loads(state.read_text())["transferred"] == 4


def test_component_precommit_and_receipt_recovery_contract():
    root = zroot("component-receipt")
    receipt = root / "B2.json"
    value = {"stage": "B2", "target_sha256": "a" * 64, "builder_commit": "b" * 40}
    _publish_component_receipt(receipt, value)
    assert _receipt_matches(receipt, value)
    assert json.loads(receipt.with_suffix(".json.precommit").read_text())["completion"] == "VALIDATED_PRECOMMIT"
    receipt.write_text("{}")
    assert not _receipt_matches(receipt, value)


def test_component_source_binding_rehash_rejects_post_acquisition_mutation():
    body = b"verified-source"
    row = source_entry("bound-source.bin", body)
    root = zroot("source-binding")
    source_root = root / "source"
    source_root.mkdir()
    (source_root / row.filename).write_bytes(body)
    executor = object.__new__(Step9BExecutor)
    executor.source_root = source_root
    executor.scratch_root = root / "scratch"
    executor.source_inventory_fingerprint = ACCEPTED_SOURCE_INVENTORY
    executor.manifest = type("Manifest", (), {"revision": PIN})()
    executor.downloader = Downloader(source_root, small_manifest(row), execute=False)
    receipt = executor.scratch_root / "receipts" / "sources" / f"{row.acquisition_order:03d}-{row.filename}.receipt.json"
    _atomic_json(receipt, {
        "completion": "SOURCE_COMPLETE",
        "entry": row.__dict__,
        "source_inventory_fingerprint": ACCEPTED_SOURCE_INVENTORY,
        "resolved_commit": PIN,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    })
    assert Step9BExecutor._source_bindings(executor, [row])[0]["sha256"] == hashlib.sha256(body).hexdigest()
    (source_root / row.filename).write_bytes(b"mutated-source!")
    with pytest.raises(ExecutorError, match="SHA/LFS"):
        Step9BExecutor._source_bindings(executor, [row])


def test_cancel_before_body_leaves_no_final():
    body = b"cancel-me"
    row = source_entry("cancel.bin", body)
    root = zroot("cancel")
    with FixtureServer(body) as server:
        downloader = local_downloader(server, row, root)
        downloader.cancel()
        with pytest.raises(ExecutorError, match="cancelled"):
            downloader.acquire(row)
    assert not (root / row.filename).exists()
    assert not any(request[0] == "GET" for request in server.requests)


class MemoryResponse:
    def __init__(self, body: bytes, *, fail: bool = False, delay: float = 0.05):
        self.status = 200
        self.headers = {"Content-Length": str(len(body)), "ETag": '"fixture-etag"'}
        self.body = body
        self.fail = fail
        self.delay = delay
        self.closed = False

    def iter_bytes(self, chunk_bytes=8 << 20):
        midpoint = max(1, len(self.body) // 2)
        yield self.body[:midpoint]
        time.sleep(self.delay)
        if self.fail:
            raise OSError("injected stream failure")
        yield self.body[midpoint:]

    def close(self):
        self.closed = True


class ConcurrentTransport:
    def __init__(self, body: bytes, *, fail_name: str | None = None):
        self.body = body
        self.fail_name = fail_name
        self.responses: list[MemoryResponse] = []

    def head(self, url, *, headers=None):
        response = MemoryResponse(b"", delay=0)
        response.headers = {"Content-Length": str(len(self.body)), "ETag": '"fixture-etag"'}
        return response

    def get(self, url, *, headers=None, allow_body=False):
        name = url.rsplit("/", 1)[-1]
        response = MemoryResponse(self.body, fail=name == self.fail_name)
        self.responses.append(response)
        return response


def test_max_two_concurrent_bodies_and_failure_isolation():
    body = b"same-body-for-each-file"
    rows = [source_entry(f"f{index}.bin", body) for index in range(3)]
    rows = [SourceEntry(**{**row.__dict__, "acquisition_order": index + 1}) for index, row in enumerate(rows)]
    manifest = AcquisitionManifest(REPO, PIN, tuple(rows), (), ACCEPTED_SOURCE_INVENTORY, len(body) * 3, 1_000_000)
    root = zroot("concurrency")
    transport = ConcurrentTransport(body, fail_name="f1.bin")
    downloader = Downloader(root, manifest, transport=transport, execute=True, allow_network_body=True, max_concurrent=2, budget=TransferBudget(1_000_000))
    results: dict[str, object] = {}

    def run(row):
        try:
            results[row.filename] = downloader.acquire(row)
        except Exception as exc:
            results[row.filename] = exc

    threads = [threading.Thread(target=run, args=(row,)) for row in rows]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert downloader.max_active == 2
    assert isinstance(results["f1.bin"], OSError)
    assert (root / "f0.bin").read_bytes() == body
    assert (root / "f2.bin").read_bytes() == body
    assert not (root / "f1.bin").exists()
    assert all(response.closed for response in transport.responses)


def test_concurrent_admission_cannot_overcommit_transfer_cap():
    body = b"123456"
    rows = [source_entry(f"cap-{index}.bin", body) for index in range(2)]
    rows = [SourceEntry(**{**row.__dict__, "acquisition_order": index + 1}) for index, row in enumerate(rows)]
    manifest = AcquisitionManifest(REPO, PIN, tuple(rows), (), ACCEPTED_SOURCE_INVENTORY, 12, 10)
    root = zroot("concurrent-cap")
    transport = ConcurrentTransport(body)
    budget = TransferBudget(10, state_path=root / "budget.json")
    downloader = Downloader(root, manifest, transport=transport, execute=True, allow_network_body=True, max_concurrent=2, budget=budget)
    results: list[object] = []

    def run(row):
        try:
            results.append(downloader.acquire(row))
        except Exception as exc:
            results.append(exc)

    threads = [threading.Thread(target=run, args=(row,)) for row in rows]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, ExecutorError) for result in results) == 1
    assert budget.transferred == len(body)
    assert len(transport.responses) == 1


def test_wrong_resolved_commit_rejected_before_body(monkeypatch):
    body = b"wrong-commit"
    row = source_entry("commit.bin", body)
    root = zroot("wrong-commit")
    downloader = Downloader(root, small_manifest(row), transport=UrllibTransport(), execute=True, allow_network_body=True)

    class Meta:
        commit_hash = "0" * 40
        size = len(body)
        etag = row.lfs_oid_sha256
        xet_file_data = type("Xet", (), {"file_hash": row.accepted_etag.strip('"')})()

    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", lambda **kwargs: Meta())
    with pytest.raises(ExecutorError, match="resolved commit"):
        downloader.acquire(row)
    assert not (root / row.filename).exists()


def test_real_host_body_kill_switch_in_dry_run():
    body = b"never-transfer"
    row = source_entry("kill.bin", body)

    class KillTransport:
        body_calls = 0

        def head(self, *args, **kwargs):
            raise AssertionError("dry run must not perform metadata traffic per file")

        def get(self, *args, **kwargs):
            self.body_calls += 1
            raise AssertionError("REAL_HF_BODY_KILL_SWITCH")

    transport = KillTransport()
    downloader = Downloader(zroot("kill-switch"), small_manifest(row), transport=transport, execute=False, allow_network_body=False)
    assert downloader.acquire(row)["state"] == "PLANNED"
    assert transport.body_calls == 0


def test_q3_final_target_without_receipt_is_adopted(monkeypatch):
    root = zroot("q3-adopt")
    target = root / "target"
    scratch = root / "scratch"
    target.mkdir()
    data = target / "ple-q3-000.bin"
    manifest_path = target / "ple-q3.json"
    data.write_bytes(b"q3!!")
    manifest_path.write_text("{}")

    class Reader:
        def __init__(self, path):
            self.manifest = {"segment_count": 128}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def gather(self, rows):
            assert rows == [0, 2_500_011, 2_500_012, 160_000_768, 317_501_524, 320_001_535]

    monkeypatch.setattr(executor_module, "Q3_BYTES", 4)
    monkeypatch.setattr("freetoken.checkpoint.q3_ple.plan_q3_ple_production", lambda: {"segment_count": 128, "total_bytes": 4})
    monkeypatch.setattr("freetoken.checkpoint.q3_ple.Q3PLEReader", Reader)
    monkeypatch.setattr("freetoken.checkpoint.q3_ple.write_q3_ple_from_safetensors", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must adopt, not rebuild")))
    executor = object.__new__(Step9BExecutor)
    executor.execute = True
    executor.target_root = target
    executor.scratch_root = scratch
    executor.source_root = root / "source"
    executor.source_inventory_fingerprint = ACCEPTED_SOURCE_INVENTORY
    executor.builder_commit = "b" * 40
    executor.manifest = type("Manifest", (), {"revision": PIN, "rows_for_stage": lambda self, stage: ()})()
    executor._source_bindings = lambda rows: []
    executor._disk_gate = lambda: 0
    executor._host_gate = lambda: {}
    result = Step9BExecutor.convert_and_validate_q3(executor)
    assert result["recovered_after_promotion"] is True
    receipt = scratch / "receipts" / "B2-q3.json"
    assert json.loads(receipt.read_text())["completion"] == "COMPONENT_COMPLETE"


def test_c6_controller_uses_isolated_runtime_subprocess(monkeypatch):
    root = zroot("c6-controller")
    executor = object.__new__(Step9BExecutor)
    executor.execute = True
    executor.runtime_worktree = WORKSPACE / "workstreams" / "freetoken-qwen4-pr257-fit"
    executor.runtime_commit = "0307a6114c57b0efc61bc17688f3288fe0bf1dc7"
    executor.target_root = root / "target"
    executor.target_root.mkdir()
    executor.scratch_root = root / "scratch"
    executor.toolchain_root = WORKSPACE / "artifacts" / "toolchain" / "step9b-pr257"
    executor._disk_gate = lambda: 0
    executor._host_gate = lambda: {}
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return type("Completed", (), {"returncode": 0, "stdout": '{"status":"C6_STATIC_OK"}', "stderr": ""})()

    monkeypatch.setattr(executor_module.subprocess, "run", fake_run)
    result = Step9BExecutor.run_c6_static_reopen(executor)
    assert result["inference"] is False
    assert Path(observed["cwd"]) == executor.runtime_worktree
    assert observed["env"]["PYTHONPATH"].split(executor_module.os.pathsep)[0] == str(executor.runtime_worktree / "python")
    script = observed["command"][2]
    for required in ("Q3PLEFileTable", "FileExpertSource", "HostBank", "_apply_pr257_hardware_fit_policy", "graph_requested=True"):
        assert required in script


def test_real_manifest_dry_run_reconciles_all_206_rows():
    path = WORKSPACE / "results" / "FREETOKEN-QWEN4-001" / "step9_acquisition_manifest.json"
    manifest = AcquisitionManifest.load(path)
    assert len(manifest.entries) == 206
    assert len(manifest.metadata) == 9
    assert manifest.expected_weight_bytes == 135_195_303_851
    assert sum(row.byte_length for row in manifest.metadata) == 57_176_714
    assert len([row for row in manifest.entries if row.source_class == "EXPERT"]) == 192
    assert {row.layer_id for row in manifest.entries if row.source_class == "EXPERT"} == set(range(48))


def test_projected_initial_disk_requirement_matches_frozen_strategy_a(monkeypatch):
    manifest_path = WORKSPACE / "results" / "FREETOKEN-QWEN4-001" / "step9_acquisition_manifest.json"
    source = zroot("projection-source")
    target = zroot("projection-target")
    scratch = zroot("projection-scratch")
    logs = zroot("projection-logs")
    runtime = WORKSPACE / "workstreams" / "freetoken-qwen4-pr257-fit"
    executor = Step9BExecutor(
        manifest_path, source, target, scratch, logs,
        builder_commit="b64a342ea8e5ccac39a7619747b4a7b3e37466f3",
        runtime_worktree=runtime,
        runtime_commit="0307a6114c57b0efc61bc17688f3288fe0bf1dc7",
        source_inventory_fingerprint=ACCEPTED_SOURCE_INVENTORY,
        min_disk_free=MIN_DISK_RESERVE_BYTES,
    )
    assert executor._projected_required_free() == 309_257_827_893
    plan = executor.dry_run_plan()
    assert len(plan["planned_files"]) == 215
    assert all(name.startswith("model-plefp8-") for name in plan["planned_files"][9:19])
    assert plan["planned_files"][-4:] == [row.filename for row in executor.manifest.rows_for_stage("B4")]
    assert len(plan["source_receipts"]) == 215
    assert len(plan["component_receipts"]) == 53


def test_retention_true_fails_before_any_body(monkeypatch):
    manifest_path = WORKSPACE / "results" / "FREETOKEN-QWEN4-001" / "step9_acquisition_manifest.json"
    root = zroot("retention")
    executor = Step9BExecutor(
        manifest_path, root / "source", root / "target", root / "scratch", root / "logs",
        builder_commit="b64a342ea8e5ccac39a7619747b4a7b3e37466f3",
        runtime_worktree=WORKSPACE / "workstreams" / "freetoken-qwen4-pr257-fit",
        runtime_commit="0307a6114c57b0efc61bc17688f3288fe0bf1dc7",
        source_inventory_fingerprint=ACCEPTED_SOURCE_INVENTORY,
        source_retirement_authorized=True,
    )
    with pytest.raises(ExecutorError, match="retirement"):
        executor.preflight()


def test_preflight_rejects_payload_without_transfer_budget_provenance():
    manifest_path = WORKSPACE / "results" / "FREETOKEN-QWEN4-001" / "step9_acquisition_manifest.json"
    root = zroot("budget-provenance")
    executor = Step9BExecutor(
        manifest_path, root / "source", root / "target", root / "scratch", root / "logs",
        builder_commit="b64a342ea8e5ccac39a7619747b4a7b3e37466f3",
        runtime_worktree=WORKSPACE / "workstreams" / "freetoken-qwen4-pr257-fit",
        runtime_commit="0307a6114c57b0efc61bc17688f3288fe0bf1dc7",
        source_inventory_fingerprint=ACCEPTED_SOURCE_INVENTORY,
    )
    executor.source_root.mkdir(parents=True, exist_ok=True)
    first = executor.manifest.all_entries[0]
    (executor.source_root / first.filename).write_bytes(b"not-trusted")
    with pytest.raises(ExecutorError, match="transfer-budget provenance"):
        executor.preflight()


def test_synthetic_controller_run_uses_complete_stage_order(monkeypatch):
    events: list[str] = []
    executor = object.__new__(Step9BExecutor)
    executor.execute = True
    executor.preflight = lambda: events.append("preflight")
    executor.acquire_metadata = lambda: events.append("B1")
    executor.acquire_ple = lambda: events.append("B2-source")
    executor.convert_and_validate_q3 = lambda: events.append("B2-q3")
    executor.acquire_expert_layer = lambda layer: events.append(f"B3-source-{layer}")
    executor.convert_and_validate_expert = lambda layer: events.append(f"B3-target-{layer}")
    executor.acquire_active = lambda: events.append("B4-source")
    executor.convert_and_validate_active = lambda: events.append("B4-active")
    executor.finalize_artifact = lambda: (events.append("B5") or {"state": "COMPLETE"})
    executor.run_c6_static_reopen = lambda: (events.append("C6") or {"state": "COMPLETE"})
    executor.closeout = lambda: (events.append("closeout") or {"state": "CLOSED"})
    result = Step9BExecutor.run(executor)
    assert events[:5] == ["preflight", "B1", "B2-source", "B2-q3", "B3-source-0"]
    assert events[-5:] == ["B4-source", "B4-active", "B5", "C6", "closeout"]
    assert sum(item.startswith("B3-target-") for item in events) == 48
    assert result["state"] == "COMPLETE"


@pytest.mark.parametrize("fail_at", ["after_B2", "between_L17_L18", "after_all_experts", "after_B5", "during_C6"])
def test_controller_restart_boundaries_stop_then_resume_without_duplicate_completion(fail_at):
    completed: set[str] = set()
    attempts: dict[str, int] = {}

    def boundary(name: str):
        attempts[name] = attempts.get(name, 0) + 1
        if name in completed:
            return
        if attempts[name] == 1 and name == fail_at:
            raise ExecutorError(f"injected {name}")
        completed.add(name)

    def make_executor():
        executor = object.__new__(Step9BExecutor)
        executor.execute = True
        executor.preflight = lambda: boundary("preflight")
        executor.acquire_metadata = lambda: boundary("B1")
        executor.acquire_ple = lambda: boundary("B2-source")
        executor.convert_and_validate_q3 = lambda: (boundary("B2-target"), boundary("after_B2"))
        executor.acquire_expert_layer = lambda layer: boundary("between_L17_L18") if layer == 18 else None
        executor.convert_and_validate_expert = lambda layer: (boundary(f"B3-{layer}"), boundary("after_all_experts") if layer == 47 else None)
        executor.acquire_active = lambda: boundary("B4-source")
        executor.convert_and_validate_active = lambda: boundary("B4-target")
        executor.finalize_artifact = lambda: (boundary("B5"), boundary("after_B5"), {"state": "COMPLETE"})[-1]
        executor.run_c6_static_reopen = lambda: (boundary("during_C6"), boundary("C6"), {"state": "COMPLETE"})[-1]
        executor.closeout = lambda: (boundary("closeout"), {"state": "CLOSED"})[-1]
        return executor

    with pytest.raises(ExecutorError, match="injected"):
        Step9BExecutor.run(make_executor())
    result = Step9BExecutor.run(make_executor())
    assert result["state"] == "COMPLETE"
    assert "closeout" in completed
    # Receipt-backed stages may be revisited, but their synthetic completion is
    # published once and subsequent calls take the restart-skip path.
    assert all(name in completed for name in ("B1", "B2-target", "B3-17", "B3-47", "B4-target", "B5", "C6"))
