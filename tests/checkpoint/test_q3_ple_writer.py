from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from uuid import uuid4
from pathlib import Path

import pytest
import torch

from freetoken.checkpoint.q3_ple import (
    ALIGN,
    BLOCK_BYTES,
    ROW_BYTES,
    ROW_VALUES,
    Q3PLEReader,
    PRODUCTION_SEGMENT_BYTES,
    PRODUCTION_SEGMENT_COUNT,
    PRODUCTION_ROWS_PER_SEGMENT,
    PRODUCTION_TOTAL_BYTES,
    PRODUCTION_TOTAL_ROWS,
    quantize_block,
    quantize_row,
    quantize_rows_batched,
    plan_q3_ple_production,
    write_q3_ple_from_safetensors,
    write_q3_ple_sidecar,
    write_q3_ple_segmented_sidecar,
)


def _rows(count: int):
    for row_index in range(count):
        yield [((row_index + 1) * 0.125) * ((column % 17) - 8) for column in range(ROW_VALUES)]


def _load_authoritative_reference():
    reference_path = Path(__file__).resolve().parents[4] / "scripts" / "q3_ple_32_reference.py"
    if not reference_path.exists():
        pytest.skip("authoritative Q3_PLE_32 reference script is not present in this checkout")
    spec = importlib.util.spec_from_file_location("q3_ple_32_authoritative_reference", reference_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def z_fixture_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / ".stage7-q3-writer-fixtures" / uuid4().hex
    assert root.drive.upper() == "Z:"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_writer_matches_reference_vectors_and_reader(z_fixture_dir: Path) -> None:
    # These vectors are the byte-for-byte values from the authoritative
    # scripts/q3_ple_32_reference.py codec (two refinement passes, BF16 scale).
    assert quantize_block([0.0] * 32).hex() == "0000244992244992244992244992"
    assert quantize_block([(index - 16) / 4.0 for index in range(32)]).hex() == (
        "9c3f492249dab69124dbb6b6edff"
    )

    data_path = z_fixture_dir / "ple-q3.bin"
    manifest_path = z_fixture_dir / "ple-q3.json"
    manifest = write_q3_ple_sidecar(
        _rows(5),
        data_path,
        manifest_path,
        source_fingerprint="A" * 64,
        weight_scale=1.25,
        segment_rows=2,
    )
    assert manifest["source_fingerprint"] == "a" * 64
    assert manifest["weight_scale"] == 1.25
    assert manifest["payload_bytes"] == 5 * ROW_BYTES
    assert manifest["file_bytes"] == data_path.stat().st_size
    assert [segment["first_row"] for segment in manifest["segments"]] == [0, 2, 4]
    assert manifest["storage_layout"] == "contiguous_rows_v1"
    assert manifest["file_bytes"] == manifest["payload_bytes"] == 5 * ROW_BYTES
    assert [segment["data_offset"] for segment in manifest["segments"]] == [0, 2 * ROW_BYTES, 4 * ROW_BYTES]
    assert all(
        segment["byte_length"] == (2 if segment["first_row"] < 4 else 1) * ROW_BYTES
        for segment in manifest["segments"]
    )

    raw = data_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["sha256"]
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded == manifest
    with Q3PLEReader(manifest_path) as reader:
        gathered = reader.gather([4, 0, 4])
        assert gathered.shape == (3, ROW_VALUES)
        assert gathered[0].equal(gathered[2])
        scaled = reader.gather([0], apply_weight_scale=True)
        assert scaled.equal(reader.gather([0]) * 1.25)


def test_writer_payload_is_byte_identical_to_authoritative_reference(z_fixture_dir: Path) -> None:
    reference = _load_authoritative_reference()
    rows = list(_rows(3))
    expected = reference.encode_table(rows, refinement_passes=2, scale_dtype="bf16")
    manifest = write_q3_ple_sidecar(
        iter(rows),
        z_fixture_dir / "reference.bin",
        z_fixture_dir / "reference.json",
        source_fingerprint="e" * 64,
        weight_scale=1.0,
        segment_rows=128,
    )
    assert (z_fixture_dir / "reference.bin").read_bytes() == expected
    assert manifest["file_bytes"] == len(expected)


def test_batched_quantizer_is_byte_identical_to_scalar_reference() -> None:
    generator = torch.Generator().manual_seed(20260829)
    random_rows = (torch.randn((8192, ROW_VALUES), generator=generator) * 12.0).to(
        torch.float8_e4m3fn
    )
    adversarial = torch.stack(
        [
            torch.zeros(ROW_VALUES),
            torch.arange(-80, 80, dtype=torch.float32) / 8.0,
            torch.tensor(([0.5, -0.5, 1.5, -1.5] * 40), dtype=torch.float32),
            torch.full((ROW_VALUES,), 448.0),
            torch.full((ROW_VALUES,), -448.0),
        ]
    ).to(torch.float8_e4m3fn)
    rows = torch.cat((random_rows, adversarial), dim=0)
    expected = b"".join(quantize_row(row.float()) for row in rows)

    assert quantize_rows_batched(rows, device="cpu") == expected
    if torch.cuda.is_available():
        assert quantize_rows_batched(rows, device="cuda") == expected


def test_batched_segmented_writer_matches_scalar_bytes(z_fixture_dir: Path) -> None:
    rows = (torch.arange(7 * ROW_VALUES, dtype=torch.float32).reshape(7, ROW_VALUES) / 32.0).to(
        torch.float8_e4m3fn
    )
    scalar_manifest = write_q3_ple_segmented_sidecar(
        ([row.float() for row in rows[:3]], [row.float() for row in rows[3:]]),
        z_fixture_dir / "scalar.bin",
        z_fixture_dir / "scalar.json",
        source_fingerprint="9" * 64,
        weight_scale=1.0,
        segment_count=2,
    )
    batched_manifest = write_q3_ple_segmented_sidecar(
        ((rows[:2], rows[2:3]), (rows[3:6], rows[6:])),
        z_fixture_dir / "batched.bin",
        z_fixture_dir / "batched.json",
        source_fingerprint="9" * 64,
        weight_scale=1.0,
        segment_count=2,
        batched=True,
        quantization_device="cuda" if torch.cuda.is_available() else "cpu",
    )

    assert (z_fixture_dir / "batched.bin").read_bytes() == (
        z_fixture_dir / "scalar.bin"
    ).read_bytes()
    for key in ("rows", "payload_bytes", "file_bytes", "sha256", "payload_sha256", "segments"):
        assert batched_manifest[key] == scalar_manifest[key]


def test_writer_consumes_rows_once_and_rejects_nonfinite_without_finalizing(z_fixture_dir: Path) -> None:
    data_path = z_fixture_dir / "ple-q3.bin"
    manifest_path = z_fixture_dir / "ple-q3.json"
    consumed = 0

    def source_rows():
        nonlocal consumed
        consumed += 1
        yield [0.0] * ROW_VALUES
        consumed += 1
        bad = [0.0] * ROW_VALUES
        bad[31] = float("nan")
        yield bad

    with pytest.raises(ValueError, match="non-finite"):
        write_q3_ple_sidecar(
            source_rows(),
            data_path,
            manifest_path,
            source_fingerprint="b" * 64,
            weight_scale=1.0,
            segment_rows=1,
        )
    assert consumed == 2
    assert not data_path.exists()
    assert not manifest_path.exists()
    assert not list(z_fixture_dir.glob(".*.partial-*"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_fingerprint": "short", "weight_scale": 1.0},
        {"source_fingerprint": "c" * 64, "weight_scale": float("inf")},
        {"source_fingerprint": "c" * 64, "weight_scale": 1.0, "segment_rows": 0},
    ],
)
def test_writer_rejects_bad_integrity_metadata(z_fixture_dir: Path, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        write_q3_ple_sidecar(
            _rows(1),
            z_fixture_dir / "ple-q3.bin",
            z_fixture_dir / "ple-q3.json",
            **kwargs,
        )


def test_writer_requires_z_backing() -> None:
    forbidden = Path("C:/stage7-q3-writer-must-not-create")
    with pytest.raises(ValueError, match="Z:"):
        write_q3_ple_sidecar(
            _rows(1),
            forbidden / "ple-q3.bin",
            forbidden / "ple-q3.json",
            source_fingerprint="d" * 64,
            weight_scale=1.0,
        )


def test_production_writer_streams_safetensor_shards_in_source_order(z_fixture_dir: Path) -> None:
    from safetensors.torch import save_file

    prefix = "model.language_model.layers.2.ple.ple_embedding.ngram_embedding"
    weight_map = {}
    source_rows = []
    for part in range(2):
        key = f"{prefix}.shard_{part}.weight"
        filename = f"model-plefp8-{part:05d}.safetensors"
        rows = (
            torch.tensor(list(_rows(2)), dtype=torch.float32) + float(part)
        ).to(torch.float8_e4m3fn).contiguous()
        tensors = {key: rows}
        if part == 0:
            tensors[prefix + ".weight_scale"] = torch.tensor(0.5, dtype=torch.bfloat16)
            weight_map[prefix + ".weight_scale"] = filename
        save_file(tensors, z_fixture_dir / filename)
        weight_map[key] = filename
        source_rows.extend(rows.float().tolist())
    (z_fixture_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    manifest = write_q3_ple_from_safetensors(
        z_fixture_dir,
        z_fixture_dir / "ple-q3.bin",
        z_fixture_dir / "ple-q3.json",
        layer_id=2,
        split_parts=2,
        source_fingerprint="f" * 64,
        rows_per_chunk=1,
        segment_rows=3,
    )
    reference = _load_authoritative_reference()
    assert (z_fixture_dir / "ple-q3.bin").read_bytes() == reference.encode_table(
        source_rows, refinement_passes=2, scale_dtype="bf16"
    )
    assert manifest["rows"] == 4
    assert manifest["file_bytes"] == 4 * ROW_BYTES
    assert manifest["weight_scale"] == 0.5


def _write_128_segment_source(root: Path, *, malformed: str | None = None) -> str:
    """Create a tiny source inventory with one deterministic row per shard."""
    from safetensors.torch import save_file

    prefix = "model.language_model.layers.2.ple.ple_embedding.ngram_embedding"
    tensors = {}
    weight_map = {}
    source_file = root / "model-plefp8-00000.safetensors"
    for index in range(PRODUCTION_SEGMENT_COUNT):
        suffix = f"shard_{index}.weight"
        if malformed == "duplicate" and index == 18:
            # A JSON object cannot contain a literal duplicate key.  A
            # different spelling of the same numeric suffix exercises the
            # production duplicate-index guard without relying on parser
            # behavior for duplicate object members.
            suffix = "shard_017.weight"
        if malformed == "outside" and index == 18:
            suffix = "shard_128.weight"
        if malformed == "malformed" and index == 18:
            suffix = "shard_bad.weight"
        key = f"{prefix}.{suffix}"
        values = torch.full((1, ROW_VALUES), float(index + 1), dtype=torch.float32)
        tensors[key] = values.to(torch.float8_e4m3fn)
        weight_map[key] = source_file.name
    if malformed == "missing":
        weight_map.pop(f"{prefix}.shard_63.weight")
        tensors.pop(f"{prefix}.shard_63.weight")
    if malformed == "shape":
        tensors[f"{prefix}.shard_7.weight"] = torch.zeros((2, ROW_VALUES), dtype=torch.float8_e4m3fn)
    tensors[f"{prefix}.weight_scale"] = torch.tensor(0.5, dtype=torch.bfloat16)
    weight_map[f"{prefix}.weight_scale"] = source_file.name
    save_file(tensors, source_file)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    return prefix


def test_q3_segmented_writer_has_stable_128_source_segments_and_boundaries(z_fixture_dir: Path) -> None:
    rows = [
        [[float(index + 1)] * ROW_VALUES]
        for index in range(PRODUCTION_SEGMENT_COUNT)
    ]
    first = write_q3_ple_segmented_sidecar(
        (iter(segment) for segment in rows),
        z_fixture_dir / "a.bin",
        z_fixture_dir / "a.json",
        source_fingerprint="1" * 64,
        weight_scale=0.5,
        segment_count=PRODUCTION_SEGMENT_COUNT,
    )
    second = write_q3_ple_segmented_sidecar(
        (iter(segment) for segment in rows),
        z_fixture_dir / "b.bin",
        z_fixture_dir / "b.json",
        source_fingerprint="1" * 64,
        weight_scale=0.5,
        segment_count=PRODUCTION_SEGMENT_COUNT,
    )
    assert len(first["segments"]) == PRODUCTION_SEGMENT_COUNT
    assert [item["first_row"] for item in first["segments"]] == list(range(128))
    assert first["segments"][0]["data_offset"] == 0
    assert first["segments"][1]["data_offset"] == ROW_BYTES
    assert first["segments"][-1]["first_row"] == 127
    assert first["segments"][-1]["end_row"] == 128
    # Data and semantic segment directory are byte-identical across runs.
    assert (z_fixture_dir / "a.bin").read_bytes() == (z_fixture_dir / "b.bin").read_bytes()
    assert {k: v for k, v in first.items() if k != "data_file"} == {
        k: v for k, v in second.items() if k != "data_file"
    }


def test_q3_safetensor_chunking_cannot_change_logical_segment_directory(z_fixture_dir: Path) -> None:
    source = z_fixture_dir / "source"
    source.mkdir()
    _write_128_segment_source(source)
    left = write_q3_ple_from_safetensors(
        source, z_fixture_dir / "left.bin", z_fixture_dir / "left.json",
        layer_id=2, split_parts=128, source_fingerprint="2" * 64,
        processing_chunk_rows=1,
    )
    right = write_q3_ple_from_safetensors(
        source, z_fixture_dir / "right.bin", z_fixture_dir / "right.json",
        layer_id=2, split_parts=128, source_fingerprint="2" * 64,
        processing_chunk_rows=2,
    )
    assert (z_fixture_dir / "left.bin").read_bytes() == (z_fixture_dir / "right.bin").read_bytes()
    assert left["segment_count"] == right["segment_count"] == 128
    assert left["segments"] == right["segments"]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "outside", "malformed", "shape"])
def test_q3_safetensor_source_rejects_bad_logical_segments(z_fixture_dir: Path, bad: str) -> None:
    source = z_fixture_dir / bad
    source.mkdir()
    _write_128_segment_source(source, malformed=bad)
    with pytest.raises(ValueError):
        write_q3_ple_from_safetensors(
            source, z_fixture_dir / f"{bad}.bin", z_fixture_dir / f"{bad}.json",
            layer_id=2, split_parts=128, source_fingerprint="3" * 64,
            rows_per_segment=1,
        )


@pytest.mark.parametrize("segment_count", [127, 129])
def test_q3_segmented_writer_rejects_wrong_segment_count(z_fixture_dir: Path, segment_count: int) -> None:
    segments = ([float(index)] * ROW_VALUES for index in range(128))
    with pytest.raises(ValueError):
        write_q3_ple_segmented_sidecar(
            ((row,) for row in segments),
            z_fixture_dir / f"{segment_count}.bin",
            z_fixture_dir / f"{segment_count}.json",
            source_fingerprint="4" * 64,
            weight_scale=1.0,
            segment_count=segment_count,
        )


def test_q3_production_planner_is_exact_without_allocating_payload() -> None:
    plan = plan_q3_ple_production()
    assert plan == {
        "segment_count": 128,
        "rows_per_segment": 2_500_012,
        "total_rows": 320_001_536,
        "row_bytes": ROW_BYTES,
        "segment_bytes": 175_000_840,
        "total_bytes": 22_400_107_520,
    }
    assert PRODUCTION_SEGMENT_COUNT * PRODUCTION_ROWS_PER_SEGMENT == PRODUCTION_TOTAL_ROWS
    assert PRODUCTION_SEGMENT_COUNT * PRODUCTION_SEGMENT_BYTES == PRODUCTION_TOTAL_BYTES
