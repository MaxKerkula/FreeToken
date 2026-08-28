from __future__ import annotations

import json
from pathlib import Path

import torch

from freetoken.checkpoint.convert import (
    _copy_metadata,
    _iter_qwen4_modular_dense_entries,
    convert_checkpoint,
)
from freetoken.checkpoint.ftw import FTWReader, FTWWriter, iter_ftw_weights


def test_copy_metadata_keeps_only_qwen4_host_mapped_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "model-plefp8-00000.safetensors").write_bytes(b"ple")
    (source / "model-00001.safetensors").write_bytes(b"dense")
    ple_name = (
        "model.language_model.layers.0.ple.ple_embedding."
        "ngram_embedding.shard_0.weight"
    )
    index = {
        "metadata": {"total_size": 8},
        "weight_map": {
            ple_name: "model-plefp8-00000.safetensors",
            "model.embed_tokens.weight": "model-00001.safetensors",
        },
    }
    (source / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )

    copied = _copy_metadata(str(source), str(output))

    assert (output / "config.json").is_file()
    assert (output / "model-plefp8-00000.safetensors").read_bytes() == b"ple"
    assert not (output / "model-00001.safetensors").exists()
    slim = json.loads((output / "model.safetensors.index.json").read_text())
    assert slim["weight_map"] == {ple_name: "model-plefp8-00000.safetensors"}
    assert slim["metadata"]["freetoken_host_mapped_only"] is True
    assert sorted(copied) == [
        "config.json",
        "model-plefp8-00000.safetensors",
        "model.safetensors.index.json",
    ]


def test_modular_metadata_does_not_copy_source_ple_payload(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "model-plefp8-00000.safetensors").write_bytes(b"ple")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({
            "weight_map": {
                "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight": "model-plefp8-00000.safetensors"
            }
        }),
        encoding="utf-8",
    )
    copied = _copy_metadata(str(source), str(output), include_host_mapped_weights=False)
    assert copied == ["config.json"]
    assert not (output / "model-plefp8-00000.safetensors").exists()


def test_modular_dense_stream_quantizes_map_and_excludes_vision() -> None:
    active = torch.ones((2, 16), dtype=torch.bfloat16)
    protected = torch.ones((4,), dtype=torch.bfloat16)
    names = [
        name
        for name, _tensor in _iter_qwen4_modular_dense_entries(
            [
                ("model.layers.0.self_attn.o_proj.weight", active),
                ("model.layers.0.input_layernorm.weight", protected),
                ("visual.blocks.0.weight", active),
            ]
        )
    ]
    assert names == [
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.o_proj.weight_scale",
        "model.layers.0.self_attn.o_proj.weight_global",
        "model.layers.0.input_layernorm.weight",
    ]


def test_convert_checkpoint_builds_marked_modular_active_ftw(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    output = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"architectures": ["Qwen4ExpForCausalLM"], "vision_config": {}}),
        encoding="utf-8",
    )

    class FakeEngineConfig:
        def __init__(self, **_kwargs):
            self.model_config = type(
                "ModelConfig",
                (),
                {
                    "architectures": ("Qwen4ExpForCausalLM",),
                    "expert_quant": "nvfp4",
                    "is_moe": True,
                },
            )()

    active = torch.ones((2, 16), dtype=torch.bfloat16)
    protected = torch.arange(4, dtype=torch.bfloat16)

    def fake_load_weight(_path, device, *, include_moe_experts):
        assert device.type == "cpu"
        assert include_moe_experts is False
        return iter(
            [
                ("model.layers.0.self_attn.o_proj.weight", active),
                ("model.layers.0.input_layernorm.weight", protected),
            ]
        )

    import freetoken.engine.config as engine_config
    import freetoken.models.weight as weight_module

    monkeypatch.setattr(engine_config, "EngineConfig", FakeEngineConfig)
    monkeypatch.setattr(weight_module, "load_weight", fake_load_weight)
    inventory = "a" * 64
    index = convert_checkpoint(
        str(source),
        str(output),
        artifact_format="qwen4_modular_v1",
        source_inventory_sha256=inventory,
        shard_limit=4096 * 16,
    )

    assert index["source_inventory_sha256"] == inventory
    assert index["counts"] == {"weight": 4, "experts_bank": 0}
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["freetoken_text_only"] == "qwen4_text_only_v1"
    assert config["freetoken_active_quant"] == "nvfp4_w4a16_v1"
    loaded = dict(iter_ftw_weights(str(output / "qwen4-active-v1.ftw"), workers=1))
    assert set(loaded) == {
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.o_proj.weight_scale",
        "model.layers.0.self_attn.o_proj.weight_global",
        "model.layers.0.input_layernorm.weight",
    }


def test_ftw_buffered_reader_works_on_the_current_platform(tmp_path: Path) -> None:
    output = tmp_path / "ftw"
    writer = FTWWriter(str(output), shard_limit=4096)
    expected = torch.arange(64, dtype=torch.int32).reshape(8, 8)
    writer.add_tensor("weight", expected, kind="weight")
    writer.finalize({})

    loaded = list(iter_ftw_weights(str(output), workers=2))

    assert len(loaded) == 1
    assert loaded[0][0] == "weight"
    assert torch.equal(loaded[0][1], expected)


def test_ftw_reader_can_drop_and_reopen_source_maps(tmp_path: Path) -> None:
    output = tmp_path / "ftw"
    writer = FTWWriter(str(output), shard_limit=4096)
    expected = torch.arange(64, dtype=torch.int32).reshape(8, 8)
    writer.add_tensor("weight", expected, kind="weight")
    writer.finalize({})

    reader = FTWReader(str(output))
    reader._direct = 0
    reader._probed = True
    entry = reader.entries("weight")[0]
    destination = bytearray(4096)
    reader.read_into(memoryview(destination), entry, workers=2)
    assert reader._maps

    reader.drop_maps()
    assert not reader._maps
    destination[:] = b"\0" * len(destination)
    reader.read_into(memoryview(destination), entry, workers=2)
    actual = torch.frombuffer(destination, dtype=torch.int32, count=64).reshape(8, 8)
    assert torch.equal(actual, expected)
    reader.close()
