from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
import ctypes
from ctypes import wintypes
from pathlib import Path

import pytest

import freetoken.checkpoint.step9b_executor as module
from freetoken.checkpoint.step9b_executor import (
    ACCEPTED_SOURCE_INVENTORY,
    AcquisitionManifest,
    AtomicJsonReplaceError,
    Downloader,
    ResumeRejected,
    SourceEntry,
    TransferBudget,
    _atomic_json,
)


ROOT = Path("Z:/Qwen38-FlashNext-Cluster/artifacts/stage7h-test-fixtures")
PIN = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"


def zroot(name: str) -> Path:
    path = ROOT / f"{name}-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def v2_row(name: str = "partial.safetensors", length: int = 32) -> SourceEntry:
    lfs = hashlib.sha256(b"lfs-" + name.encode()).hexdigest()
    xet = hashlib.sha256(b"xet-" + name.encode()).hexdigest()
    return SourceEntry(
        filename=name,
        byte_length=length,
        source_class="PLE",
        acquisition_order=1,
        repository=REPO,
        revision=PIN,
        git_blob_id="a" * 40,
        lfs_oid_sha256=lfs,
        xet_file_hash=xet,
        allowed_body_etags=(lfs, xet),
        identity_version=2,
    )


def downloader_for(root: Path, row: SourceEntry, transferred: int) -> Downloader:
    manifest = AcquisitionManifest(REPO, PIN, (row,), (), ACCEPTED_SOURCE_INVENTORY, row.byte_length, 10_000, schema_version=2)
    budget = TransferBudget(10_000, transferred=transferred, state_path=root / "budget.json")
    return Downloader(root, manifest, budget=budget, execute=True, allow_network_body=False)


def _deny_delete_handle(path: Path):
    """Open *path* without FILE_SHARE_DELETE so Windows replace is denied."""
    if os.name != "nt":
        pytest.skip("Windows share-lock integration is Windows-only")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000, 0x00000001, None, 3, 0x80, None)  # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel32, handle


def _close_handle(kernel32, handle):
    kernel32.CloseHandle(handle)


def test_windows_deny_delete_lock_releases_within_retry_window():
    if os.name != "nt":
        pytest.skip("Windows share-lock integration is Windows-only")
    root = zroot("atomic-lock-release")
    destination = root / "state.json"
    _atomic_json(destination, {"sequence": 1})
    kernel32, handle = _deny_delete_handle(destination)

    def release():
        time.sleep(0.20)
        _close_handle(kernel32, handle)

    thread = threading.Thread(target=release)
    thread.start()
    try:
        _atomic_json(destination, {"sequence": 2}, replace_deadline_seconds=1.5, replace_backoff_seconds=0.03)
    finally:
        thread.join(timeout=2)
    assert json.loads(destination.read_text()) == {"sequence": 2}
    assert not list(root.glob(".state.json.partial-*"))


def test_windows_deny_delete_lock_beyond_deadline_preserves_old_and_orphan():
    if os.name != "nt":
        pytest.skip("Windows share-lock integration is Windows-only")
    root = zroot("atomic-lock-timeout")
    destination = root / "state.json"
    _atomic_json(destination, {"sequence": 1})
    kernel32, handle = _deny_delete_handle(destination)
    try:
        with pytest.raises(AtomicJsonReplaceError, match="preserved_orphan=.*state.json"):
            _atomic_json(destination, {"sequence": 2}, replace_attempts=4, replace_deadline_seconds=0.10, replace_backoff_seconds=0.02)
        assert json.loads(destination.read_text()) == {"sequence": 1}
        assert list(root.glob(".state.json.partial-*"))
    finally:
        _close_handle(kernel32, handle)


def test_repeated_atomic_publication_has_no_malformed_reads():
    root = zroot("atomic-stress")
    destination = root / "state.json"
    _atomic_json(destination, {"sequence": 0})
    stop = threading.Event()
    malformed: list[str] = []
    observed: list[int] = []

    def reader():
        while not stop.is_set():
            try:
                value = json.loads(destination.read_text())
                sequence = int(value["sequence"])
                if observed and sequence < observed[-1]:
                    malformed.append("non-monotonic")
                observed.append(sequence)
            except OSError:
                # Windows may briefly deny/open-race the pathname while the
                # directory entry is replaced.  That is not a malformed read.
                pass
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                malformed.append("malformed")
            # Yield a bounded replacement window instead of continuously
            # reacquiring a deny-delete CRT handle and starving the writer.
            time.sleep(0.002)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for sequence in range(1, 101):
            _atomic_json(destination, {"sequence": sequence}, replace_backoff_seconds=0.001)
    finally:
        stop.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert malformed == []
    assert json.loads(destination.read_text())["sequence"] == 100


def _write_orphan_case(root: Path, row: SourceEntry, *, mutate=None, canonical=None, extra=None):
    partial = root / f"{row.filename}.partial"
    partial.write_bytes(bytes(range(row.byte_length)))
    downloader = downloader_for(root, row, transferred=row.byte_length)
    remote = {"commit": PIN, "metadata_etag": row.lfs_oid_sha256, "xet_file_hash": row.xet_file_hash, "observed_body_etag": row.xet_file_hash}
    identity = partial.with_name(partial.name + ".meta.json")
    orphan = identity.with_name(f".{identity.name}.partial-invalid")
    orphan_value = downloader._identity(row, row.byte_length, remote)
    if mutate:
        mutate(orphan_value)
    orphan.write_text(json.dumps(orphan_value))
    if extra is not None:
        extra_path = identity.with_name(f".{identity.name}.partial-second")
        extra_path.write_text(json.dumps(extra))
    if canonical is not None:
        identity.write_text(json.dumps(canonical))
    return downloader, row, partial, identity, remote


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("revision", "deadbeef", "revision"),
        ("expected_git_blob_id", "b" * 40, "expected_git_blob_id"),
        ("expected_lfs_oid_sha256", "c" * 64, "expected_lfs_oid_sha256"),
        ("expected_xet_file_hash", "d" * 64, "expected_xet_file_hash"),
        ("observed_body_etag", "e" * 64, "observed body ETag"),
        ("source_inventory_fingerprint", "f" * 64, "source_inventory_fingerprint"),
        ("acquisition_order", 9, "acquisition_order"),
    ],
)
def test_orphan_identity_negative_matrix(field, value, match):
    root = zroot(f"orphan-negative-{field}")
    row = v2_row(length=64)
    mutate = lambda identity: identity.__setitem__(field, value)
    downloader, row, partial, identity, remote = _write_orphan_case(root, row, mutate=mutate)
    with pytest.raises(ResumeRejected, match=match):
        downloader.recover_partial_identity_checkpoint(row, partial=partial, identity=identity, remote=remote)
    assert partial.stat().st_size == 64
    assert downloader.budget.transferred == 64


def test_orphan_canonical_ahead_rejected():
    root = zroot("orphan-canonical-ahead")
    row = v2_row(length=64)
    downloader = downloader_for(root, row, transferred=64)
    remote = {"commit": PIN, "metadata_etag": row.lfs_oid_sha256, "xet_file_hash": row.xet_file_hash, "observed_body_etag": row.xet_file_hash}
    partial = root / f"{row.filename}.partial"
    partial.write_bytes(bytes(range(64)))
    identity = partial.with_name(partial.name + ".meta.json")
    identity.write_text(json.dumps(downloader._identity(row, 65, remote)))
    orphan = identity.with_name(f".{identity.name}.partial-ahead")
    orphan.write_text(json.dumps(downloader._identity(row, 64, remote)))
    with pytest.raises(ResumeRejected, match="ahead"):
        downloader.recover_partial_identity_checkpoint(row, partial=partial, identity=identity, remote=remote)


def test_conflicting_multiple_orphans_fail_closed():
    root = zroot("orphan-conflict")
    row = v2_row(length=64)
    downloader = downloader_for(root, row, transferred=64)
    remote = {"commit": PIN, "metadata_etag": row.lfs_oid_sha256, "xet_file_hash": row.xet_file_hash, "observed_body_etag": row.xet_file_hash}
    partial = root / f"{row.filename}.partial"
    partial.write_bytes(bytes(range(64)))
    identity = partial.with_name(partial.name + ".meta.json")
    first = downloader._identity(row, 64, remote)
    second = dict(first)
    second["observed_body_etag"] = row.lfs_oid_sha256
    identity.with_name(f".{identity.name}.partial-a").write_text(json.dumps(first))
    identity.with_name(f".{identity.name}.partial-b").write_text(json.dumps(second))
    with pytest.raises(ResumeRejected, match="AMBIGUOUS"):
        downloader.recover_partial_identity_checkpoint(row, partial=partial, identity=identity, remote=remote)


def test_malformed_orphan_rejected():
    root = zroot("orphan-malformed")
    row = v2_row(length=64)
    downloader = downloader_for(root, row, transferred=64)
    partial = root / f"{row.filename}.partial"
    partial.write_bytes(bytes(range(64)))
    identity = partial.with_name(partial.name + ".meta.json")
    identity.with_name(f".{identity.name}.partial-malformed").write_text("not-json")
    with pytest.raises(ResumeRejected, match="invalid orphan"):
        downloader.recover_partial_identity_checkpoint(row, partial=partial, identity=identity, remote={"commit": PIN})


def test_canonical_missing_valid_orphan_is_adopted():
    root = zroot("orphan-canonical-missing")
    row = v2_row(length=64)
    downloader = downloader_for(root, row, transferred=64)
    remote = {"commit": PIN, "metadata_etag": row.lfs_oid_sha256, "xet_file_hash": row.xet_file_hash, "observed_body_etag": row.xet_file_hash}
    partial = root / f"{row.filename}.partial"
    partial.write_bytes(bytes(range(64)))
    identity = partial.with_name(partial.name + ".meta.json")
    identity.with_name(f".{identity.name}.partial-orphan").write_text(json.dumps(downloader._identity(row, 64, remote)))
    result = downloader.recover_partial_identity_checkpoint(row, partial=partial, identity=identity, remote=remote)
    assert result and result["state"] == "ADOPTED"
    assert json.loads(identity.read_text())["partial_length"] == 64


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_atomic_json_retries_known_windows_transient(monkeypatch, tmp_path: Path, winerror: int):
    path = tmp_path / "state.json"
    _atomic_json(path, {"sequence": 1})
    original = os.replace
    attempts = {"count": 0}

    def flaky(source, destination):
        attempts["count"] += 1
        if attempts["count"] <= 2:
            error = PermissionError(winerror, "transient lock")
            error.winerror = winerror
            raise error
        return original(source, destination)

    monkeypatch.setattr(module.os, "replace", flaky)
    _atomic_json(path, {"sequence": 2}, replace_backoff_seconds=0, replace_deadline_seconds=1)
    assert json.loads(path.read_text()) == {"sequence": 2}
    assert attempts["count"] == 3


def test_atomic_json_nontransient_is_immediate_and_preserves_orphan(monkeypatch, tmp_path: Path):
    path = tmp_path / "state.json"
    _atomic_json(path, {"sequence": 1})
    original = os.replace
    calls = {"count": 0}

    def fail(source, destination):
        calls["count"] += 1
        error = OSError("invalid path")
        error.winerror = 87
        raise error

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="invalid path"):
        _atomic_json(path, {"sequence": 2})
    assert calls["count"] == 1
    assert json.loads(path.read_text()) == {"sequence": 1}
    assert list(tmp_path.glob(".state.json.partial-*"))
    monkeypatch.setattr(module.os, "replace", original)


def test_atomic_json_permanent_transient_preserves_canonical_and_orphan(monkeypatch, tmp_path: Path):
    path = tmp_path / "state.json"
    _atomic_json(path, {"sequence": 1})

    def fail(source, destination):
        error = PermissionError(5, "locked")
        error.winerror = 5
        raise error

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(AtomicJsonReplaceError, match="preserved_orphan=.*state.json"):
        _atomic_json(path, {"sequence": 2}, replace_attempts=2, replace_backoff_seconds=0)
    assert json.loads(path.read_text()) == {"sequence": 1}
    assert list(tmp_path.glob(".state.json.partial-*"))


def test_orphan_adoption_is_exact_and_body_ledger_unchanged(tmp_path: Path):
    root = zroot("orphan-adopt")
    row = v2_row(length=64)
    partial = root / f"{row.filename}.partial"
    body = bytes(range(64))
    partial.write_bytes(body)
    downloader = downloader_for(root, row, transferred=len(body))
    remote = {"commit": PIN, "metadata_etag": row.lfs_oid_sha256, "xet_file_hash": row.xet_file_hash, "observed_body_etag": row.xet_file_hash}
    identity = partial.with_name(partial.name + ".meta.json")
    _atomic_json(identity, downloader._identity(row, 16, remote))
    orphan = identity.with_name(f".{identity.name}.partial-test")
    orphan.write_text(json.dumps(downloader._identity(row, len(body), remote)))
    before = hashlib.sha256(body).hexdigest()
    result = downloader.recover_partial_identity_checkpoint(row, partial=partial, identity=identity, remote=remote)
    assert result and result["state"] == "ADOPTED"
    assert json.loads(identity.read_text())["partial_length"] == len(body)
    assert partial.stat().st_size == len(body)
    assert hashlib.sha256(partial.read_bytes()).hexdigest() == before
    assert downloader.budget.transferred == len(body)


def test_orphan_wrong_length_rejected_without_body_or_ledger_mutation(tmp_path: Path):
    root = zroot("orphan-invalid")
    row = v2_row(length=64)
    partial = root / f"{row.filename}.partial"
    partial.write_bytes(bytes(range(64)))
    downloader = downloader_for(root, row, transferred=64)
    remote = {"commit": PIN, "metadata_etag": row.lfs_oid_sha256, "xet_file_hash": row.xet_file_hash, "observed_body_etag": row.xet_file_hash}
    identity = partial.with_name(partial.name + ".meta.json")
    orphan = identity.with_name(f".{identity.name}.partial-invalid")
    orphan.write_text(json.dumps({**downloader._identity(row, 63, remote), "partial_length": 63}))
    with pytest.raises(ResumeRejected, match="physical partial"):
        downloader.recover_partial_identity_checkpoint(row, partial=partial, identity=identity, remote=remote)
    assert downloader.budget.transferred == 64
    assert partial.stat().st_size == 64


def test_resume_plan_is_recorded_before_body_disabled(tmp_path: Path):
    root = zroot("resume-plan")
    row = v2_row(length=64)
    partial = root / f"{row.filename}.partial"
    partial.write_bytes(bytes(range(64)))
    downloader = downloader_for(root, row, transferred=64)
    remote = {"commit": PIN, "metadata_etag": row.lfs_oid_sha256, "xet_file_hash": row.xet_file_hash, "observed_body_etag": row.xet_file_hash}
    identity = partial.with_name(partial.name + ".meta.json")
    _atomic_json(identity, downloader._identity(row, 16, remote, observed_body_etag=row.xet_file_hash))
    # Partial length is intentionally made consistent with the body so the
    # body-disabled path reaches the exact resume-plan seam.
    _atomic_json(identity, downloader._identity(row, 64, remote, observed_body_etag=row.xet_file_hash))
    # A complete partial would be promoted; use a shorter body instead.
    partial.write_bytes(bytes(range(32)))
    _atomic_json(identity, downloader._identity(row, 32, remote, observed_body_etag=row.xet_file_hash))
    downloader.resolve_hf_metadata = lambda _row: remote
    with pytest.raises(module.BodyTransferDisabled):
        downloader.acquire(row)
    assert downloader.resume_plans[row.filename] == {
        "filename": row.filename,
        "range_start": 32,
        "remaining_bytes": 32,
        "if_range": row.xet_file_hash,
        "body_request_authorized": False,
    }
