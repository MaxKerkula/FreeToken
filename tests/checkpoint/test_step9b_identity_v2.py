from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from freetoken.checkpoint.step9b_executor import (
    ACQUISITION_MANIFEST_V1,
    ACQUISITION_MANIFEST_V2,
    ACCEPTED_SOURCE_INVENTORY,
    ExecutorError,
    BodyTransferDisabled,
    Downloader,
    AcquisitionManifest,
    ResumeRejected,
    _atomic_json,
    SourceEntry,
    generate_acquisition_manifest_v2,
    migrate_source_receipt_v1_to_v2,
)


PIN = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"


def _hex(char: str, length: int) -> str:
    return (char * length)[:length]


def _v2_row(*, name: str = "model.safetensors", git: str | None = None) -> SourceEntry:
    lfs = _hex("a", 64)
    xet = _hex("b", 64)
    if git is None:
        git = _hex("c", 40)
    return SourceEntry(
        filename=name,
        byte_length=4,
        source_class="BF16",
        acquisition_order=1,
        repository=REPO,
        revision=PIN,
        git_blob_id=git,
        lfs_oid_sha256=lfs,
        xet_file_hash=xet,
        allowed_body_etags=(lfs, xet),
        identity_version=2,
    )


def test_v2_rows_keep_git_and_lfs_xet_body_identities_distinct():
    row = _v2_row()
    assert row.metadata_etag == row.lfs_oid_sha256
    assert set(row.allowed_body_etags) == {row.lfs_oid_sha256, row.xet_file_hash}
    assert row.git_blob_id not in row.allowed_body_etags
    with pytest.raises(ExecutorError, match="allowed_body_etags"):
        SourceEntry(**{**row.__dict__, "allowed_body_etags": (row.git_blob_id, row.lfs_oid_sha256)})


def test_v2_weight_requires_git_lfs_and_xet_and_v2_input_cannot_infer_xet():
    row = _v2_row()
    with pytest.raises(ExecutorError, match="git_blob_id provenance"):
        SourceEntry(**{**row.__dict__, "git_blob_id": None})
    with pytest.raises(ExecutorError, match="weight row requires"):
        SourceEntry(**{**row.__dict__, "lfs_oid_sha256": None, "xet_file_hash": None, "allowed_body_etags": (row.git_blob_id,)})
    raw = {**row.__dict__, "xet_file_hash": None, "allowed_body_etags": None, "accepted_etag": row.xet_file_hash}
    with pytest.raises(ExecutorError, match="Xet hash"):
        SourceEntry.from_mapping(raw, schema_version=2)


def test_git_only_v2_row_allows_only_git_body_etag():
    row = SourceEntry(
        filename="config.json",
        byte_length=4,
        source_class="METADATA",
        acquisition_order=1,
        repository=REPO,
        revision=PIN,
        git_blob_id=_hex("d", 40),
        allowed_body_etags=(_hex("d", 40),),
        identity_version=2,
    )
    assert row.semantic_kind == "GIT"
    with pytest.raises(ExecutorError, match="allowed_body_etags"):
        SourceEntry(**{**row.__dict__, "allowed_body_etags": (_hex("e", 40),)})


def test_manifest_generator_requires_frozen_xet_for_lfs_metadata():
    lfs = _hex("a", 64)
    raw = {
        "schema": ACQUISITION_MANIFEST_V1,
        "repository": REPO,
        "revision": PIN,
        "required_small_metadata": [{
            "filename": "tokenizer.json",
            "byte_length": 4,
            "git_blob_id": _hex("d", 40),
            "lfs_oid_sha256": lfs,
        }],
        "source_weight_shards": [],
    }
    with pytest.raises(ExecutorError, match="xet_file_hash"):
        generate_acquisition_manifest_v2(raw)
    generated = generate_acquisition_manifest_v2(raw, identity_overrides={"tokenizer.json": {"xet_file_hash": _hex("b", 64)}})
    assert generated["schema"] == ACQUISITION_MANIFEST_V2
    row = generated["required_small_metadata"][0]
    assert row["allowed_body_etags"] == [lfs, _hex("b", 64)]
    assert _hex("d", 40) not in row["allowed_body_etags"]


def test_receipt_migration_preserves_v1_bytes_and_records_predecessor_sha(tmp_path: Path):
    body = {"completion": "SOURCE_COMPLETE", "body_bytes": 0, "observed_etag": _hex("b", 64)}
    receipt = tmp_path / "001-model.receipt.json"
    original = json.dumps(body, sort_keys=True).encode("utf-8")
    receipt.write_bytes(original)
    row = _v2_row()
    migrated = migrate_source_receipt_v1_to_v2(
        receipt,
        row=row,
        source_inventory_fingerprint=ACCEPTED_SOURCE_INVENTORY,
        observed_metadata_etag=row.lfs_oid_sha256,
        observed_xet_file_hash=row.xet_file_hash,
        observed_body_etag=row.xet_file_hash,
    )
    predecessor = Path(str(receipt) + ".v1")
    assert predecessor.read_bytes() == original
    assert migrated["predecessor_receipt_sha256"] == hashlib.sha256(original).hexdigest()
    assert json.loads(receipt.read_text())["receipt_version"] == 2
    assert json.loads(receipt.read_text())["body_bytes"] == 0
    assert migrated["new_body_bytes"] == 0
    assert migrated["lifetime_transfer_accounting"] == "unchanged"
    assert migrated["migration_reason"] == "storage_identity_v2"
    assert migrated["original_body_bytes"] == 0
    assert migrated["original_completion"] == "SOURCE_COMPLETE"


class _Response:
    def __init__(self, status: int, headers: dict[str, str], body: bytes = b""):
        self.status = status
        self.headers = headers
        self._body = body

    def iter_bytes(self, chunk_bytes: int = 8 << 20):
        if self._body:
            yield self._body

    def close(self):
        return None


class _SyntheticTransport:
    def __init__(self, body: bytes, metadata_etag: str, xet: str | None, body_etag: str | None):
        self.body = body
        self.metadata_etag = metadata_etag
        self.xet = xet
        self.body_etag = body_etag
        self.get_headers: list[dict[str, str]] = []
        self.head_calls = 0

    def head(self, url: str, *, headers=None):
        self.head_calls += 1
        values = {"Content-Length": str(len(self.body)), "ETag": self.metadata_etag, "X-Repo-Commit": PIN}
        if self.xet:
            values["X-Xet-Hash"] = self.xet
        return _Response(200, values)

    def get(self, url: str, *, headers=None, allow_body=False):
        self.get_headers.append(dict(headers or {}))
        headers = dict(headers or {})
        start = int(headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        payload = self.body[start:]
        response_headers = {"Content-Length": str(len(payload)), "ETag": self.body_etag or ""}
        if start:
            response_headers["Content-Range"] = f"bytes {start}-{len(self.body)-1}/{len(self.body)}"
            return _Response(206, response_headers, payload)
        return _Response(200, response_headers, payload)


def _manifest_for_v2(row: SourceEntry) -> AcquisitionManifest:
    return AcquisitionManifest(REPO, PIN, (row,), (), ACCEPTED_SOURCE_INVENTORY, row.byte_length, 1_000_000, schema_version=2)


def test_strict_metadata_checks_git_lfs_xet_mismatch_and_absence():
    body = b"metadata"
    lfs = hashlib.sha256(body).hexdigest()
    xet = _hex("b", 64)
    git = _hex("c", 40)
    row = SourceEntry("meta.bin", len(body), "BF16", 1, REPO, PIN, git_blob_id=git, lfs_oid_sha256=lfs, xet_file_hash=xet, allowed_body_etags=(lfs, xet), identity_version=2)
    downloader = Downloader(Path("Z:/Qwen38-FlashNext-Cluster/artifacts/stage7f-test-fixtures/meta-check"), _manifest_for_v2(row), execute=True, allow_network_body=True)
    good = _Response(200, {"Content-Length": str(len(body)), "ETag": lfs, "X-Xet-Hash": xet, "X-Repo-Commit": PIN})
    assert downloader.validate_metadata(row, good)["xet_file_hash"] == xet
    for headers, message in [
        ({"Content-Length": str(len(body)), "ETag": git, "X-Xet-Hash": xet, "X-Repo-Commit": PIN}, "metadata ETag"),
        ({"Content-Length": str(len(body)), "ETag": lfs, "X-Repo-Commit": PIN}, "Xet"),
        ({"Content-Length": str(len(body)), "ETag": lfs, "X-Xet-Hash": _hex("d", 64), "X-Repo-Commit": PIN}, "Xet"),
    ]:
        with pytest.raises(ExecutorError, match=message):
            downloader.validate_metadata(row, _Response(200, headers))
    git_row = SourceEntry("config.json", len(body), "METADATA", 1, REPO, PIN, git_blob_id=git, allowed_body_etags=(git,), identity_version=2)
    with pytest.raises(ExecutorError, match="unexpectedly exposed Xet"):
        downloader.validate_metadata(git_row, _Response(200, {"Content-Length": str(len(body)), "ETag": git, "X-Xet-Hash": xet, "X-Repo-Commit": PIN}))


@pytest.mark.parametrize("body_etag", ["lfs", "xet", "git", "arbitrary", None])
def test_body_etag_allowlist_accepts_only_lfs_or_xet(body_etag: str | None, tmp_path: Path):
    body = b"body-etag"
    lfs = hashlib.sha256(body).hexdigest()
    xet = _hex("b", 64)
    git = _hex("c", 40)
    values = {"lfs": lfs, "xet": xet, "git": git, "arbitrary": "transport"}
    etag = values.get(body_etag) if body_etag else None
    row = SourceEntry("body.bin", len(body), "BF16", 1, REPO, PIN, git_blob_id=git, lfs_oid_sha256=lfs, xet_file_hash=xet, allowed_body_etags=(lfs, xet), identity_version=2)
    transport = _SyntheticTransport(body, lfs, xet, etag)
    root = Path("Z:/Qwen38-FlashNext-Cluster/artifacts/stage7f-test-fixtures") / f"body-etag-{body_etag or 'missing'}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    downloader = Downloader(root, _manifest_for_v2(row), transport=transport, execute=True, allow_network_body=True)
    if body_etag in {"lfs", "xet"}:
        assert downloader.acquire(row)["state"] == "SOURCE_COMPLETE"
    else:
        with pytest.raises(ExecutorError, match="ETag"):
            downloader.acquire(row)
        assert not (root / row.filename).exists()


def test_v2_partial_identity_contains_distinct_metadata_and_body_fields(tmp_path: Path):
    body = b"partial"
    lfs = hashlib.sha256(body).hexdigest()
    xet = _hex("b", 64)
    row = SourceEntry("partial.bin", len(body), "BF16", 1, REPO, PIN, git_blob_id=_hex("c", 40), lfs_oid_sha256=lfs, xet_file_hash=xet, allowed_body_etags=(lfs, xet), identity_version=2)
    manifest = _manifest_for_v2(row)
    downloader = Downloader(Path("Z:/Qwen38-FlashNext-Cluster/artifacts/stage7f-test-fixtures/partial-fields"), manifest, execute=True, allow_network_body=True)
    identity = downloader._identity(row, 2, {"commit": PIN, "metadata_etag": lfs, "xet_file_hash": xet}, observed_body_etag='"' + xet + '"')
    assert identity["observed_metadata_etag"] == lfs
    assert identity["observed_body_etag"] == '"' + xet + '"'
    assert identity["expected_xet_file_hash"] == xet


@pytest.mark.parametrize("body_etag", ['"' + "a" * 64 + '"', "b" * 64])
def test_resume_if_range_uses_exact_persisted_body_etag(body_etag: str, tmp_path: Path):
    body = b"resume-body"
    lfs = hashlib.sha256(body).hexdigest()
    xet = "a" * 64 if body_etag.startswith('"') else "b" * 64
    row = SourceEntry("resume-v2.bin", len(body), "BF16", 1, REPO, PIN, git_blob_id=_hex("c", 40), lfs_oid_sha256=lfs, xet_file_hash=xet, allowed_body_etags=(lfs, xet), identity_version=2)
    root = Path("Z:/Qwen38-FlashNext-Cluster/artifacts/stage7f-test-fixtures") / f"resume-v2-{xet[0]}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    transport = _SyntheticTransport(body, lfs, xet, body_etag)
    downloader = Downloader(root, _manifest_for_v2(row), transport=transport, execute=True, allow_network_body=True)
    partial = root / f"{row.filename}.partial"
    partial.write_bytes(body[:3])
    _atomic_json(partial.with_name(partial.name + ".meta.json"), downloader._identity(row, 3, {"commit": PIN, "metadata_etag": lfs, "xet_file_hash": xet}, observed_body_etag=body_etag))
    result = downloader.acquire(row)
    assert result["state"] == "SOURCE_COMPLETE"
    assert transport.get_headers[-1]["If-Range"] == body_etag


def test_receipt_migration_rejects_invalid_or_ambiguous_predecessor(tmp_path: Path):
    row = SourceEntry("receipt.bin", 1, "METADATA", 1, REPO, PIN, git_blob_id=_hex("c", 40), allowed_body_etags=(_hex("c", 40),), identity_version=2)
    receipt = tmp_path / "bad.receipt.json"
    receipt.write_text("not-json", encoding="utf-8")
    with pytest.raises(ExecutorError, match="invalid v1"):
        migrate_source_receipt_v1_to_v2(receipt, row=row, source_inventory_fingerprint=ACCEPTED_SOURCE_INVENTORY)
    receipt.write_text(json.dumps({"completion": "SOURCE_COMPLETE"}), encoding="utf-8")
    predecessor = Path(str(receipt) + ".v1")
    predecessor.write_text("different", encoding="utf-8")
    with pytest.raises(ExecutorError, match="predecessor"):
        migrate_source_receipt_v1_to_v2(receipt, row=row, source_inventory_fingerprint=ACCEPTED_SOURCE_INVENTORY)


def test_body_disabled_restart_migrates_final_then_stops_before_next_get(tmp_path: Path):
    body = b"final-body"
    lfs = hashlib.sha256(body).hexdigest()
    xet = _hex("b", 64)
    row = SourceEntry("final.bin", len(body), "BF16", 1, REPO, PIN, git_blob_id=_hex("c", 40), lfs_oid_sha256=lfs, xet_file_hash=xet, allowed_body_etags=(lfs, xet), identity_version=2)
    root = Path("Z:/Qwen38-FlashNext-Cluster/artifacts/stage7f-test-fixtures/body-disabled-restart")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    (root / row.filename).write_bytes(body)
    receipt_root = root / "receipts"
    receipt_root.mkdir()
    receipt = receipt_root / "001-final.bin.receipt.json"
    receipt.write_text(json.dumps({
        "completion": "SOURCE_COMPLETE",
        "observed_etag": xet,
        "body_bytes": 0,
        "final_path": str(root / row.filename),
        "expected_bytes": len(body),
        "bytes": len(body),
        "expected_lfs_oid": lfs,
        "observed_lfs_oid": lfs,
        "sha256": lfs,
    }), encoding="utf-8")
    transport = _SyntheticTransport(body, lfs, xet, '"' + xet + '"')
    downloader = Downloader(root, _manifest_for_v2(row), receipt_root=receipt_root, transport=transport, execute=True, allow_network_body=False)
    assert downloader.acquire(row)["state"] == "SKIP_VALID_FINAL"
    assert (Path(str(receipt) + ".v1")).is_file()
    # A second restart must accept the durable JSON list representation in the
    # migrated v2 receipt without rewriting or redownloading anything.
    migrated_bytes = receipt.read_bytes()
    assert downloader.acquire(row)["state"] == "SKIP_VALID_FINAL"
    assert receipt.read_bytes() == migrated_bytes
    missing = SourceEntry("next.bin", len(body), "BF16", 2, REPO, PIN, git_blob_id=_hex("d", 40), lfs_oid_sha256=lfs, xet_file_hash=xet, allowed_body_etags=(lfs, xet), identity_version=2)
    with pytest.raises(BodyTransferDisabled):
        downloader.acquire(missing)
    assert transport.head_calls >= 2
    assert len(transport.get_headers) == 0
