"""Bounded, payload-free tests for the Step 9B controller."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

import pytest

from freetoken.checkpoint.step9b_executor import (
    AcquisitionManifest,
    BodyTransferDisabled,
    Downloader,
    SourceEntry,
    TransferBudget,
)


class Response:
    def __init__(self, status, body=b"", *, headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    def iter_bytes(self, chunk_bytes=8 << 20):
        for offset in range(0, len(self._body), 3):
            yield self._body[offset : offset + 3]

    def close(self):
        pass


class Transport:
    def __init__(self, body, etag='"etag"'):
        self.body = body
        self.etag = etag
        self.get_headers = []

    def head(self, url, *, headers=None):
        return Response(200, headers={"Content-Length": str(len(self.body)), "ETag": self.etag})

    def get(self, url, *, headers=None, allow_body=False):
        assert allow_body
        self.get_headers.append(dict(headers or {}))
        start = int((headers or {}).get("Range", "bytes=0-").split("=")[1].split("-")[0])
        if start:
            return Response(206, self.body[start:], headers={"Content-Length": str(len(self.body) - start), "Content-Range": f"bytes {start}-{len(self.body)-1}/{len(self.body)}", "ETag": self.etag})
        return Response(200, self.body, headers={"Content-Length": str(len(self.body)), "ETag": self.etag})


def entry(body: bytes) -> SourceEntry:
    return SourceEntry("fixture.bin", len(body), "METADATA", 1, "RadixArk/Qwen3.8-Flash-Next-NVFP4", "7b719225242aacd3dbd3f9407468c2ee9a9d2594", accepted_etag='"etag"', lfs_oid_sha256=hashlib.sha256(body).hexdigest())


def manifest_for(row: SourceEntry) -> AcquisitionManifest:
    return AcquisitionManifest("RadixArk/Qwen3.8-Flash-Next-NVFP4", "7b719225242aacd3dbd3f9407468c2ee9a9d2594", (row,) * 206, (row,) * 9, "8572d200e31b344faff0fda f0dc72aa4726c1f062443d4109531b62ca63f66eb".replace(" ", ""), row.byte_length * 206, 10_000)


def z_test_root(label: str) -> Path:
    root = Path("Z:/Qwen38-FlashNext-Cluster/artifacts/stage7f-test-fixtures") / f"{label}-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    return root


def test_dry_run_cannot_request_body(tmp_path):
    body = b"hello world"
    row = entry(body)
    root = z_test_root("dry-run")
    downloader = Downloader(root, manifest_for(row), transport=Transport(body), execute=False)
    assert downloader.acquire(row)["state"] == "PLANNED"
    with pytest.raises(BodyTransferDisabled):
        Downloader(root, manifest_for(row), transport=Transport(body), execute=True, allow_network_body=False).acquire(row)


def test_download_resume_and_cap(tmp_path):
    root = z_test_root("resume")
    body = b"0123456789abcdef"
    row = entry(body)
    manifest = manifest_for(row)
    transport = Transport(body)
    downloader = Downloader(root, manifest, transport=transport, execute=True, allow_network_body=True, budget=TransferBudget(10_000))
    partial = root / "fixture.bin.partial"
    partial.write_bytes(body[:5])
    from freetoken.checkpoint.step9b_executor import _atomic_json
    _atomic_json(partial.with_name(partial.name + ".meta.json"), downloader._identity(row, 5, {"resolved_commit": row.revision, "etag": '"etag"'}))
    result = downloader.acquire(row)
    assert result["state"] == "SOURCE_COMPLETE"
    assert (root / "fixture.bin").read_bytes() == body
    assert transport.get_headers[-1]["Range"] == "bytes=5-"
    assert downloader.budget.transferred == len(body) - 5


def test_transfer_budget_rejects_extra_byte(tmp_path):
    budget = TransferBudget(3)
    budget.reserve(3)
    with pytest.raises(Exception):
        budget.reserve(1)
