from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
import torch
from types import SimpleNamespace

from freetoken.checkpoint.qwen4_artifact import (
    FORMAT,
    TEXT_ONLY_MARKER,
    build_qwen4_modular_artifact,
    configure_mixed_expert_sources,
    build_mixed_expert_sources,
    finalize_qwen4_modular_manifest,
    load_qwen4_artifact_manifest,
    qwen4_text_only_marker,
)
from freetoken.checkpoint.ftw import FTWWriter
from freetoken.checkpoint.q3_ple import ROW_VALUES, write_q3_ple_sidecar
from freetoken.models.qwen4_exp.weight import iter_active_nvfp4_runtime_entries
from freetoken.models.weight import load_weight
from freetoken.moe.expert_source import (
    FileExpertSource,
    RAW_RECORD_BYTES,
    write_expert_sidecar,
)


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
    cpu_layer_ids = frozenset()

    def set_bank_sources(self, sources, layer_residency=None):
        self.bank_sources = sources
        self.layer_residency = layer_residency

    def set_file_sources(self, sources):
        self.file_sources = sources


def _manifest(root: Path, sidecar: Path, digest: str) -> Path:
    source_fingerprint = hashlib.sha256(b"synthetic").hexdigest()
    active_root = root / "qwen4-active-v1.ftw"
    active_root.mkdir(exist_ok=True)
    active_index = active_root / "freetoken_weight.json"
    active_index.write_text(
        json.dumps({"source_inventory_sha256": source_fingerprint}), encoding="utf-8"
    )
    active_digest = __import__("hashlib").sha256(active_index.read_bytes()).hexdigest()
    ple_manifest = root / "ple-q3.json"
    ple_manifest.write_text(
        json.dumps({"source_fingerprint": source_fingerprint}), encoding="utf-8"
    )
    resident_sidecar = root / "experts-L01.nvfp4"
    resident_digest = FileExpertSource.create_synthetic(
        resident_sidecar, num_experts=1, records=[bytes([8]) * RAW_RECORD_BYTES], layer_id=1
    )
    config_path = root / "config.json"
    config_path.write_bytes(b"")
    config_digest = hashlib.sha256(b"").hexdigest()
    data = {
        "format": FORMAT,
        "version": 1,
        "artifact_schema": FORMAT,
        "text_only": True,
        "source": {"repository": "synthetic", "revision": "test", "inventory_sha256": source_fingerprint},
        "minimum_freetoken_commit": "0" * 40,
        "tvm_ffi_patch_sha256": "0" * 64,
        "active": {
            "format": "nvfp4_w4a16_v1",
            "path": active_root.name,
            "bytes": active_index.stat().st_size,
            "files": [{"path": str(active_index.relative_to(root)), "bytes": active_index.stat().st_size, "sha256": active_digest}],
        },
        "ple": {
            "format": "q3_ple_32",
            "manifest": "ple-q3.json",
            "data_bytes": 0,
            "sha256": "0" * 64,
            "required_volume": "Z:",
        },
        "experts": {
            "format": "ftexpert1_nvfp4_v1",
            "files": [
                {"layer": 0, "path": sidecar.name, "bytes": sidecar.stat().st_size, "sha256": digest, "source_fingerprint": source_fingerprint},
                {"layer": 1, "path": resident_sidecar.name, "bytes": resident_sidecar.stat().st_size, "sha256": resident_digest, "source_fingerprint": source_fingerprint},
            ],
            "file_tier_layers": [0],
            "resident_layers": [1],
            "required_volume": "Z:",
        },
        "metadata": {"files": [{"path": "config.json", "bytes": 0, "sha256": config_digest}]},
    }
    data["complete_artifact_fingerprint"] = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
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


def test_active_component_hash_fails_closed(z_fixture_dir):
    sidecar = z_fixture_dir / "experts-L00.nvfp4"
    digest = FileExpertSource.create_synthetic(
        sidecar, num_experts=1, records=[bytes([7]) * RAW_RECORD_BYTES], layer_id=0
    )
    manifest = load_qwen4_artifact_manifest(_manifest(z_fixture_dir, sidecar, digest), require=True, allow_synthetic_geometry=True)
    assert manifest is not None
    manifest.verify_active()
    manifest.active_files[0].path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="length mismatch|SHA-256 mismatch"):
        manifest.verify_active()


def test_manifest_unknown_schema_and_fingerprint_fail_closed(z_fixture_dir):
    sidecar = z_fixture_dir / "experts-L00.nvfp4"
    digest = FileExpertSource.create_synthetic(
        sidecar, num_experts=1, records=[bytes([7]) * RAW_RECORD_BYTES], layer_id=0
    )
    path = _manifest(z_fixture_dir, sidecar, digest)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["artifact_schema"] = "future-v99"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_schema"):
        load_qwen4_artifact_manifest(path, require=True, allow_synthetic_geometry=True)

    data["artifact_schema"] = FORMAT
    data["complete_artifact_fingerprint"] = "f" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_qwen4_artifact_manifest(path, require=True, allow_synthetic_geometry=True)


def test_manifest_rejects_component_source_fingerprint_drift(z_fixture_dir):
    sidecar = z_fixture_dir / "experts-L00.nvfp4"
    digest = FileExpertSource.create_synthetic(
        sidecar, num_experts=1, records=[bytes([7]) * RAW_RECORD_BYTES], layer_id=0
    )
    path = _manifest(z_fixture_dir, sidecar, digest)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["experts"]["files"][0]["source_fingerprint"] = "f" * 64
    data.pop("complete_artifact_fingerprint")
    data["complete_artifact_fingerprint"] = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="expert sidecar source fingerprint mismatch"):
        load_qwen4_artifact_manifest(path, require=True, allow_synthetic_geometry=True)


def test_production_loader_rejects_reduced_geometry(z_fixture_dir):
    sidecar = z_fixture_dir / "experts-L00.nvfp4"
    digest = FileExpertSource.create_synthetic(
        sidecar, num_experts=1, records=[bytes([7]) * RAW_RECORD_BYTES], layer_id=0
    )
    path = _manifest(z_fixture_dir, sidecar, digest)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["source"]["repository"] = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
    data["source"]["revision"] = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
    data["minimum_freetoken_commit"] = "846504bf9d81119cb72400e6c5a3cc860f2b1dd8"
    data["tvm_ffi_patch_sha256"] = "889310b8152a147a6552a3e451b3251a7df70cdc8e6e4c1c87c7adf3854182ec"
    data.pop("complete_artifact_fingerprint")
    data["complete_artifact_fingerprint"] = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="active payload does not match"):
        load_qwen4_artifact_manifest(path, require=True)


def test_manifest_rejects_metadata_tamper_and_out_of_root_component(z_fixture_dir):
    sidecar = z_fixture_dir / "experts-L00.nvfp4"
    digest = FileExpertSource.create_synthetic(
        sidecar, num_experts=1, records=[bytes([7]) * RAW_RECORD_BYTES], layer_id=0
    )
    path = _manifest(z_fixture_dir, sidecar, digest)
    (z_fixture_dir / "config.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="length mismatch|SHA-256 mismatch"):
        load_qwen4_artifact_manifest(path, require=True, allow_synthetic_geometry=True)

    path = _manifest(z_fixture_dir, sidecar, digest)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["experts"]["files"][0]["path"] = str(sidecar.parent.parent / sidecar.name)
    data.pop("complete_artifact_fingerprint")
    data["complete_artifact_fingerprint"] = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="outside artifact root"):
        load_qwen4_artifact_manifest(path, require=True, allow_synthetic_geometry=True)


def test_manifest_reopens_and_wires_mixed_sources(z_fixture_dir):
    sidecar = z_fixture_dir / "experts-L00.nvfp4"
    digest = FileExpertSource.create_synthetic(
        sidecar, num_experts=1, records=[bytes([7]) * RAW_RECORD_BYTES], layer_id=0
    )
    manifest_path = _manifest(z_fixture_dir, sidecar, digest)
    manifest = load_qwen4_artifact_manifest(manifest_path, require=True, allow_synthetic_geometry=True)
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
    source_fingerprint = hashlib.sha256(b"synthetic").hexdigest()
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
        "artifact_schema": FORMAT,
        "text_only": True,
        "source": {"repository": "synthetic", "revision": "test", "inventory_sha256": source_fingerprint},
        "minimum_freetoken_commit": "0" * 40,
        "tvm_ffi_patch_sha256": "0" * 64,
        "active": {"format": "nvfp4_w4a16_v1", "path": ".", "files": [{"path": "manifest.json", "bytes": 0, "sha256": "0" * 64}]},
        "ple": {"format": "q3_ple_32", "manifest": "ple-q3.json", "sha256": "0" * 64},
        "experts": {
            "format": "ftexpert1_nvfp4_v1",
            "files": [{"layer": 0, "path": first_path.name, "bytes": first_path.stat().st_size, "sha256": first_digest, "source_fingerprint": source_fingerprint},
                      {"layer": 1, "path": resident_path.name, "bytes": resident_path.stat().st_size, "sha256": digest, "source_fingerprint": source_fingerprint}],
            "file_tier_layers": [],
            "resident_layers": [0, 1],
            "required_volume": "Z:",
        },
        "metadata": {"files": [{"path": "config.json", "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}]},
    }
    manifest_data["complete_artifact_fingerprint"] = hashlib.sha256(
        json.dumps(manifest_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = z_fixture_dir / "manifest.json"
    (z_fixture_dir / "config.json").write_bytes(b"")
    (z_fixture_dir / "ple-q3.json").write_text(
        json.dumps({"source_fingerprint": source_fingerprint}), encoding="utf-8"
    )
    path.write_text(json.dumps(manifest_data), encoding="utf-8")
    manifest = load_qwen4_artifact_manifest(path, require=True, allow_synthetic_geometry=True)
    resident_sources, file_sources = build_mixed_expert_sources(
        manifest,
        num_experts=1,
        resident_residency=["pageable", "pageable"],
        allocator=lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    assert file_sources == {}
    assert resident_sources["gate_up_packed"][0].shape == (1, 1280, 1280)
    assert int(resident_sources["gate_up_packed"][1][0, 0, 0]) == 9


def _tiny_geometry():
    return {
        "gate_up_packed": ((4, 4), torch.uint8),
        "gate_up_scale": ((4, 1), torch.float8_e4m3fn),
        "gate_up_global": ((4,), torch.float16),
        "down_packed": ((2, 16), torch.uint8),
        "down_scale": ((2, 1), torch.float8_e4m3fn),
        "down_global": ((2,), torch.float16),
    }


def _tiny_planes(value: int):
    return {
        name: torch.full(shape, 1 if dtype == torch.float8_e4m3fn else value, dtype=dtype)
        for name, (shape, dtype) in _tiny_geometry().items()
    }


def test_end_to_end_synthetic_modular_artifact_reopens_normal_paths(z_fixture_dir, monkeypatch):
    source_fingerprint = "4" * 64
    config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "freetoken_text_only": TEXT_ONLY_MARKER,
        "freetoken_active_quant": "nvfp4_w4a16_v1",
    }
    (z_fixture_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (z_fixture_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    active = z_fixture_dir / "qwen4-active-v1.ftw"
    writer = FTWWriter(str(active), shard_limit=4096 * 16)
    fused = torch.arange(64, dtype=torch.float32).reshape(4, 16).to(torch.bfloat16)
    protected = torch.arange(16, dtype=torch.bfloat16)
    entries = iter_active_nvfp4_runtime_entries(
        [
            ("model.layers.0.self_attn.qkv_proj.weight", fused),
            ("model.layers.0.self_attn.index_qk_proj.weight", protected),
        ]
    )
    emitted_names = []
    for name, tensor in entries:
        emitted_names.append(name)
        writer.add_tensor(name, tensor)
    writer.finalize({
        "artifact_format": "qwen4_modular_v1",
        "source_inventory_sha256": source_fingerprint,
    })
    assert emitted_names == [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.qkv_proj.weight_scale",
        "model.layers.0.self_attn.qkv_proj.weight_global",
        "model.layers.0.self_attn.index_qk_proj.weight",
    ]

    write_q3_ple_sidecar(
        ([float((row + column) % 9 - 4) for column in range(ROW_VALUES)] for row in range(4)),
        z_fixture_dir / "ple-q3-000.bin",
        z_fixture_dir / "ple-q3.json",
        source_fingerprint=source_fingerprint,
        weight_scale=1.0,
        segment_rows=2,
    )
    experts = {}
    for layer in range(2):
        path = z_fixture_dir / f"experts-L{layer:02d}.nvfp4"
        write_expert_sidecar(
            path,
            [_tiny_planes(layer + expert + 1) for expert in range(2)],
            layer_id=layer,
            source_fingerprint=source_fingerprint,
            num_experts=2,
            geometry=_tiny_geometry(),
        )
        experts[layer] = path.name

    finalized = finalize_qwen4_modular_manifest(
        z_fixture_dir,
        source_repository="synthetic/qwen4",
        source_revision="3" * 40,
        source_inventory_sha256=source_fingerprint,
        minimum_freetoken_commit="846504bf9d81119cb72400e6c5a3cc860f2b1dd8",
        tvm_ffi_patch_sha256="889310b8152a147a6552a3e451b3251a7df70cdc8e6e4c1c87c7adf3854182ec",
        expert_paths=experts,
        file_tier_layers=[0],
        metadata_paths=["config.json", "tokenizer_config.json"],
        expert_num_experts=2,
        allow_synthetic_geometry=True,
    )
    assert finalized["experts"]["resident_layers"] == [1]
    manifest = load_qwen4_artifact_manifest(z_fixture_dir, require=True, allow_synthetic_geometry=True)
    assert manifest is not None
    manifest.verify_active()
    import freetoken.checkpoint.qwen4_artifact as artifact_module

    original_loader = artifact_module.load_qwen4_artifact_manifest
    monkeypatch.setattr(
        artifact_module,
        "load_qwen4_artifact_manifest",
        lambda _path, **_kwargs: manifest,
    )
    loaded = dict(load_weight(str(z_fixture_dir), torch.device("cpu"), include_moe_experts=False))
    assert set(loaded) == set(emitted_names)
    resident, file_sources = build_mixed_expert_sources(
        manifest,
        num_experts=2,
        resident_residency=["pageable", "pageable"],
        allocator=lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    assert all(resident[name][0] is None for name in FileExpertSource.bank_schema)
    assert all(resident[name][1] is not None for name in FileExpertSource.bank_schema)
    assert set(file_sources) == {0}
    file_sources[0].close()

    # Normal Qwen4 host-load dispatch selects Q3 from the manifest; no manual
    # load_q3_ple_weights injection is involved.
    calls = []
    fake = SimpleNamespace(load_q3_ple_weights=lambda path: calls.append(path))
    from freetoken.models.qwen4_exp.model import Qwen4ExpModel

    Qwen4ExpModel.load_host_weights(fake, str(z_fixture_dir), dummy=False)
    assert calls == [str(z_fixture_dir / "ple-q3.json")]
    monkeypatch.setattr(artifact_module, "load_qwen4_artifact_manifest", original_loader)


def test_production_orchestrator_sequences_all_modular_components(z_fixture_dir, monkeypatch):
    source = z_fixture_dir / "source"
    target = z_fixture_dir / "target"
    source.mkdir()
    inventory = "5" * 64

    import freetoken.checkpoint.convert as convert_module
    import freetoken.checkpoint.q3_ple as q3_module
    import freetoken.moe.expert_source as expert_module

    def fake_convert(_source, out_dir, **kwargs):
        assert kwargs["artifact_format"] == "qwen4_modular_v1"
        assert kwargs["source_inventory_sha256"] == inventory
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(
            json.dumps({
                "freetoken_text_only": TEXT_ONLY_MARKER,
                "freetoken_active_quant": "nvfp4_w4a16_v1",
            }),
            encoding="utf-8",
        )
        active = out / "qwen4-active-v1.ftw"
        writer = FTWWriter(str(active), shard_limit=4096 * 4)
        writer.add_tensor("protected.weight", torch.ones(1, dtype=torch.bfloat16))
        return writer.finalize({
            "source_inventory_sha256": inventory,
            "copied_metadata": ["config.json"],
        })

    def fake_q3(_source, data_path, manifest_path, **kwargs):
        assert kwargs["source_fingerprint"] == inventory
        return write_q3_ple_sidecar(
            ([0.0] * ROW_VALUES for _ in range(2)),
            data_path,
            manifest_path,
            source_fingerprint=inventory,
            weight_scale=1.0,
            segment_rows=1,
        )

    def fake_expert(_source, path, **kwargs):
        return write_expert_sidecar(
            path,
            [_tiny_planes(kwargs["layer_id"] + 1) for _ in range(kwargs["num_experts"])],
            layer_id=kwargs["layer_id"],
            source_fingerprint=kwargs["source_fingerprint"],
            num_experts=kwargs["num_experts"],
            geometry=kwargs["geometry"],
        )

    monkeypatch.setattr(convert_module, "convert_checkpoint", fake_convert)
    monkeypatch.setattr(q3_module, "write_q3_ple_from_safetensors", fake_q3)
    monkeypatch.setattr(expert_module, "write_expert_sidecar_from_safetensors", fake_expert)
    manifest = build_qwen4_modular_artifact(
        source,
        target,
        source_repository="synthetic/qwen4",
        source_revision="6" * 40,
        source_inventory_sha256=inventory,
        minimum_freetoken_commit="7" * 40,
        ple_split_parts=1,
        expert_layers=(0, 1),
        file_tier_layers=(0,),
        expert_num_experts=2,
        expert_geometry=_tiny_geometry(),
        allow_synthetic_geometry=True,
    )
    assert manifest["active"]["format"] == "nvfp4_w4a16_v1"
    assert manifest["ple"]["source_fingerprint"] == inventory
    assert [item["layer"] for item in manifest["experts"]["files"]] == [0, 1]
    assert (target / "manifest.json").is_file()
