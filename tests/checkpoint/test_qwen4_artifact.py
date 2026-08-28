from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from freetoken.checkpoint.qwen4_artifact import (
    FORMAT,
    TEXT_ONLY_MARKER,
    configure_mixed_expert_sources,
    build_mixed_expert_sources,
    load_qwen4_artifact_manifest,
    qwen4_text_only_marker,
)
from freetoken.moe.expert_source import FileExpertSource, RAW_RECORD_BYTES


@pytest.fixture
def z_fixture_dir():
    root = Path.cwd() / ".stage7-test-fixtures" / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    try:
        assert (root.drive or "").upper() == "Z:", root
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class _FakeCache:
    bank_schema = FileExpertSource.bank_schema
    decode_target = "gpu"
    prefill_overlap = False
    cache_size = 512
    num_layers = 2
    num_experts = 1

    def set_bank_sources(self, sources):
        self.bank_sources = sources

    def set_file_sources(self, sources):
        self.file_sources = sources


def _manifest(root: Path, sidecar: Path, digest: str) -> Path:
    data = {
        "format": FORMAT,
        "version": 1,
        "text_only": True,
        "source": {"repository": "synthetic", "revision": "test", "inventory_sha256": "0" * 64},
        "minimum_freetoken_commit": "0" * 40,
        "tvm_ffi_patch_sha256": "0" * 64,
        "active": {"format": "nvfp4_w4a16_v1", "path": ".", "bytes": 0, "sha256": "0" * 64},
        "ple": {
            "format": "q3_ple_32",
            "manifest": "ple-q3.json",
            "data_bytes": 0,
            "sha256": "0" * 64,
            "required_volume": "Z:",
        },
        "experts": {
            "format": "ftexpert1_nvfp4_v1",
            "files": [{"layer": 0, "path": sidecar.name, "bytes": sidecar.stat().st_size, "sha256": digest}],
            "file_tier_layers": [0],
            "resident_layers": [1],
            "required_volume": "Z:",
        },
        "metadata": {"config_sha256": "0" * 64},
        "complete_artifact_fingerprint": "0" * 64,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_qwen4_marker_is_explicit_and_unknown_fails_closed():
    class Config:
        freetoken_text_only = TEXT_ONLY_MARKER

    assert qwen4_text_only_marker(Config()) is True
    Config.freetoken_text_only = "future_policy"
    with pytest.raises(ValueError, match="unsupported freetoken_text_only"):
        qwen4_text_only_marker(Config())


def test_manifest_reopens_and_wires_mixed_sources(z_fixture_dir):
    sidecar = z_fixture_dir / "experts-L00.nvfp4"
    digest = FileExpertSource.create_synthetic(
        sidecar, num_experts=1, records=[bytes([7]) * RAW_RECORD_BYTES], layer_id=0
    )
    manifest_path = _manifest(z_fixture_dir, sidecar, digest)
    manifest = load_qwen4_artifact_manifest(manifest_path, require=True)
    assert manifest is not None
    assert manifest.file_tier_layers == (0,)
    assert manifest.resident_layers == (1,)

    resident = {name: [None, object()] for name in FileExpertSource.bank_schema}
    cache = _FakeCache()
    sources = configure_mixed_expert_sources(cache, manifest, resident)
    assert sorted(sources) == [0]
    assert all(cache.bank_sources[name][0] is None for name in cache.bank_schema)
    assert all(cache.bank_sources[name][1] is not None for name in cache.bank_schema)
    assert cache.file_sources[0].layer_id == 0
    cache.file_sources[0].close()


def test_resident_builder_streams_one_record_without_full_layer(z_fixture_dir):
    first_path = z_fixture_dir / "experts-L00.nvfp4"
    first_digest = FileExpertSource.create_synthetic(
        first_path, num_experts=1, records=[bytes([8]) * RAW_RECORD_BYTES], layer_id=0
    )
    resident_path = z_fixture_dir / "experts-L01.nvfp4"
    digest = FileExpertSource.create_synthetic(
        resident_path, num_experts=1, records=[bytes([9]) * RAW_RECORD_BYTES], layer_id=1
    )
    manifest_data = {
        "format": FORMAT,
        "version": 1,
        "text_only": True,
        "active": {"format": "nvfp4_w4a16_v1", "path": "."},
        "ple": {"format": "q3_ple_32", "manifest": "ple-q3.json", "sha256": "0" * 64},
        "experts": {
            "format": "ftexpert1_nvfp4_v1",
            "files": [{"layer": 0, "path": first_path.name, "bytes": first_path.stat().st_size, "sha256": first_digest},
                      {"layer": 1, "path": resident_path.name, "bytes": resident_path.stat().st_size, "sha256": digest}],
            "file_tier_layers": [],
            "resident_layers": [0, 1],
            "required_volume": "Z:",
        },
    }
    path = z_fixture_dir / "manifest.json"
    path.write_text(json.dumps(manifest_data), encoding="utf-8")
    manifest = load_qwen4_artifact_manifest(path, require=True)
    resident_sources, file_sources = build_mixed_expert_sources(
        manifest,
        num_experts=1,
        resident_residency=["pageable", "pageable"],
        allocator=lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    assert file_sources == {}
    assert resident_sources["gate_up_packed"][0].shape == (1, 1280, 1280)
    assert int(resident_sources["gate_up_packed"][1][0, 0, 0]) == 9
