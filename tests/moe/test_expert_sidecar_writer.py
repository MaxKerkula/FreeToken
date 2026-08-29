from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from freetoken.moe.expert_source import (
    ExpertSourceError,
    FileExpertSource,
    MAGIC,
    adapt_expert_tensor_record,
    write_expert_sidecar,
    write_expert_sidecar_from_safetensors,
)


@pytest.fixture
def z_dir():
    root = Path.cwd() / ".stage7-test-fixtures" / uuid4().hex
    root.mkdir(parents=True)
    assert (root.drive or "").upper() == "Z:"
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _geometry():
    return {
        "gate_up_packed": ((4, 4), torch.uint8),
        "gate_up_scale": ((4, 1), torch.float8_e4m3fn),
        "gate_up_global": ((4,), torch.float16),
        "down_packed": ((2, 16), torch.uint8),
        "down_scale": ((2, 1), torch.float8_e4m3fn),
        "down_global": ((2,), torch.float16),
    }


def _planes(value: int):
    out = {}
    for name, (shape, dtype) in _geometry().items():
        if dtype == torch.float8_e4m3fn:
            out[name] = torch.full(shape, 1, dtype=dtype)
        else:
            out[name] = torch.full(shape, value, dtype=dtype)
    return out


def _source_record(value: int):
    out = {}
    for projection, rows, width in (("gate_proj", 2, 4), ("up_proj", 2, 4), ("down_proj", 2, 16)):
        out[f"{projection}.weight"] = torch.full((rows, width), value, dtype=torch.uint8)
        out[f"{projection}.weight_scale"] = torch.full((rows, max(1, width // 16)), 1, dtype=torch.float8_e4m3fn)
        out[f"{projection}.weight_scale_2"] = torch.tensor(1.5, dtype=torch.float32)
        out[f"{projection}.input_scale"] = torch.tensor(1.0, dtype=torch.float32)
    return out


def test_writer_reduced_geometry_reopens_and_is_deterministic(z_dir):
    first = z_dir / "layer-00.ftex"
    second = z_dir / "layer-00-copy.ftex"
    kwargs = dict(layer_id=7, source_fingerprint=b"source", num_experts=3, geometry=_geometry())
    result = write_expert_sidecar(first, [_planes(i) for i in range(3)], **kwargs)
    result_copy = write_expert_sidecar(second, [_planes(i) for i in range(3)], **kwargs)
    assert result["format"] == "FTEXPERT1"
    assert result["raw_record_bytes"] == 66
    assert result["record_bytes"] == 4096
    assert result["sample_ids"] == (0, 1, 2)
    assert first.read_bytes() == second.read_bytes()
    assert result["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert result["sha256"] == result_copy["sha256"]
    with FileExpertSource(first, num_experts=3, expected_sha256=result["sha256"], expected_layer_id=7) as source:
        assert source.read_record(0)["gate_up_packed"].flatten()[0].item() == 0
        assert source.read_record(2)["gate_up_packed"].flatten()[0].item() == 2
        assert source.record_bytes == 4096


def test_source_tensor_adapter_validates_twelve_names_and_expands_globals(z_dir):
    source_record = _source_record(9)
    adapted = adapt_expert_tensor_record(source_record)
    assert adapted["gate_up_packed"].shape == (4, 4)
    assert adapted["gate_up_scale"].shape == (4, 1)
    assert adapted["gate_up_global"].dtype == torch.float16
    assert adapted["gate_up_global"].tolist() == [1.5] * 4
    assert adapted["down_global"].tolist() == [1.5] * 2
    result = write_expert_sidecar(
        z_dir / "source.ftex", [source_record], layer_id=0, source_fingerprint=b"source", num_experts=1, geometry=_geometry()
    )
    with FileExpertSource(result["path"], num_experts=1) as source:
        assert source.read_record(0)["gate_up_packed"].flatten()[0].item() == 9


def test_production_writer_streams_indexed_safetensor_experts(z_dir):
    import json
    from safetensors.torch import save_file

    prefix = "model.language_model.layers.3.mlp.experts"
    tensors = {}
    weight_map = {}
    for expert_id in range(2):
        for suffix, tensor in _source_record(5 + expert_id).items():
            name = f"{prefix}.{expert_id}.{suffix}"
            tensors[name] = tensor
            weight_map[name] = "layer-3.safetensors"
    save_file(tensors, z_dir / "layer-3.safetensors")
    (z_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    output = z_dir / "experts-L03.nvfp4"
    result = write_expert_sidecar_from_safetensors(
        z_dir,
        output,
        layer_id=3,
        source_fingerprint="a" * 64,
        num_experts=2,
        geometry=_geometry(),
    )
    assert result["sample_ids"] == (0, 1)
    with FileExpertSource(output, num_experts=2, expected_layer_id=3) as source:
        assert int(source.read_record(0)["gate_up_packed"][0, 0]) == 5
        assert int(source.read_record(1)["down_packed"][0, 0]) == 6


def test_writer_accepts_explicit_id_and_named_pairs(z_dir):
    named = [(name, value) for name, value in _planes(4).items()]
    path = z_dir / "explicit.ftex"
    write_expert_sidecar(
        path,
        [(0, named)],
        layer_id=0,
        source_fingerprint="fixture",
        num_experts=1,
        geometry=_geometry(),
    )
    with FileExpertSource(path, num_experts=1) as source:
        assert source.read_record(0)["down_global"].flatten()[0].item() == 4


@pytest.mark.parametrize("records", [
    [(0, _planes(1)), (0, _planes(2))],
    [(0, _planes(1))],
    [(0, _planes(1)), (2, _planes(2))],
])
def test_writer_rejects_duplicate_or_missing_ids_without_publishing(z_dir, records):
    path = z_dir / "bad.ftex"
    with pytest.raises(ValueError, match="(duplicate|missing|outside)"):
        write_expert_sidecar(path, records, layer_id=0, source_fingerprint=b"x", num_experts=2, geometry=_geometry())
    assert not path.exists()
    assert not Path(str(path) + ".partial").exists()


def test_writer_partial_and_payload_corruption_fail_closed(z_dir):
    path = z_dir / "corrupt.ftex"
    result = write_expert_sidecar(path, [_planes(1)], layer_id=0, source_fingerprint=b"x", num_experts=1, geometry=_geometry())
    with path.open("r+b") as handle:
        handle.seek(4096 + 1)
        handle.write(b"x")
    with pytest.raises(ExpertSourceError, match="payload hash mismatch"):
        FileExpertSource(path, num_experts=1)
    path.unlink()
    partial = Path(str(path) + ".partial")
    partial.write_bytes(b"partial")
    with pytest.raises(ExpertSourceError):
        FileExpertSource(partial, num_experts=1)


def test_writer_emits_ft_expert_magic(z_dir):
    path = z_dir / "magic.ftex"
    write_expert_sidecar(path, [_planes(1)], layer_id=0, source_fingerprint=b"x", num_experts=1, geometry=_geometry())
    assert path.read_bytes()[: len(MAGIC)] == MAGIC
