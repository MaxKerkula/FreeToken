from __future__ import annotations

import json
from pathlib import Path

import pytest

from freetoken.checkpoint.step9b_executor import (
    ExecutorError,
    PRODUCTION_PLE_SEGMENT_COUNT,
    PRODUCTION_PLE_SOURCE_LAYER_ID,
    _resolve_production_ple_source_layer,
)


def _key(layer: int, index: int) -> str:
    return (
        f"model.language_model.layers.{layer}.ple.ple_embedding."
        f"ngram_embedding.shard_{index}.weight"
    )


def _scale_key(layer: int) -> str:
    return (
        f"model.language_model.layers.{layer}.ple.ple_embedding."
        "ngram_embedding.weight_scale"
    )


def _write_index(path: Path, weight_map: dict[str, str]) -> Path:
    path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return path


def _production_map() -> dict[str, str]:
    result = {
        _key(PRODUCTION_PLE_SOURCE_LAYER_ID, index): f"model-plefp8-{index // 13:05d}.safetensors"
        for index in range(PRODUCTION_PLE_SEGMENT_COUNT)
    }
    result[_scale_key(PRODUCTION_PLE_SOURCE_LAYER_ID)] = "model-plefp8-00009.safetensors"
    return result


def test_resolves_exact_layer_one_128_segment_contract(tmp_path: Path) -> None:
    index = _write_index(tmp_path / "model.safetensors.index.json", _production_map())
    assert _resolve_production_ple_source_layer(index) == 1


@pytest.mark.parametrize("missing", [0, 63, 127])
def test_rejects_missing_logical_segment(tmp_path: Path, missing: int) -> None:
    weight_map = _production_map()
    del weight_map[_key(1, missing)]
    index = _write_index(tmp_path / "model.safetensors.index.json", weight_map)
    with pytest.raises(ExecutorError, match="logical segment mismatch"):
        _resolve_production_ple_source_layer(index)


def test_rejects_missing_global_scale(tmp_path: Path) -> None:
    weight_map = _production_map()
    del weight_map[_scale_key(1)]
    index = _write_index(tmp_path / "model.safetensors.index.json", weight_map)
    with pytest.raises(ExecutorError, match="global scale is missing"):
        _resolve_production_ple_source_layer(index)


def test_rejects_stale_layer_two_namespace(tmp_path: Path) -> None:
    weight_map = {
        _key(2, index): "ple.safetensors"
        for index in range(PRODUCTION_PLE_SEGMENT_COUNT)
    }
    weight_map[_scale_key(2)] = "ple.safetensors"
    index = _write_index(tmp_path / "model.safetensors.index.json", weight_map)
    with pytest.raises(ExecutorError, match=r"expected only layer 1, found \[2\]"):
        _resolve_production_ple_source_layer(index)


def test_rejects_competing_layer_namespace(tmp_path: Path) -> None:
    weight_map = _production_map()
    weight_map[_key(2, 0)] = "other.safetensors"
    index = _write_index(tmp_path / "model.safetensors.index.json", weight_map)
    with pytest.raises(ExecutorError, match=r"expected only layer 1, found \[1, 2\]"):
        _resolve_production_ple_source_layer(index)


def test_rejects_malformed_ple_shard_suffix(tmp_path: Path) -> None:
    weight_map = _production_map()
    weight_map[
        "model.language_model.layers.1.ple.ple_embedding."
        "ngram_embedding.shard_seventeen.weight"
    ] = "ple.safetensors"
    index = _write_index(tmp_path / "model.safetensors.index.json", weight_map)
    with pytest.raises(ExecutorError, match="malformed PLE source tensor key"):
        _resolve_production_ple_source_layer(index)


def test_rejects_competing_scale_namespace(tmp_path: Path) -> None:
    weight_map = _production_map()
    weight_map[_scale_key(2)] = "other.safetensors"
    index = _write_index(tmp_path / "model.safetensors.index.json", weight_map)
    with pytest.raises(ExecutorError, match="scale layer mismatch"):
        _resolve_production_ple_source_layer(index)


def test_rejects_invalid_source_file_mapping(tmp_path: Path) -> None:
    weight_map = _production_map()
    weight_map[_key(1, 17)] = ""
    index = _write_index(tmp_path / "model.safetensors.index.json", weight_map)
    with pytest.raises(ExecutorError, match="invalid PLE source file mapping"):
        _resolve_production_ple_source_layer(index)


def test_ignores_unrelated_model_weights(tmp_path: Path) -> None:
    weight_map = _production_map()
    weight_map["model.language_model.layers.1.self_attn.q_proj.weight"] = "active.safetensors"
    index = _write_index(tmp_path / "model.safetensors.index.json", weight_map)
    assert _resolve_production_ple_source_layer(index) == 1
