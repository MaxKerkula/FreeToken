from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from freetoken.checkpoint.q3_ple import (
    ALIGN,
    BLOCK_BYTES,
    BLOCK_VALUES,
    Q3PLEReader,
    ROW_BYTES,
    ROW_VALUES,
)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent.parent / "scripts"))
from q3_ple_32_reference import dequantize_row, encode_table


FIXTURE_ROOT = ROOT / "artifacts" / "stage6_q3ple_fixture"
MANIFEST_PATH = FIXTURE_ROOT / "ple-q3.json"
DATA_PATH = FIXTURE_ROOT / "ple-q3-000.bin"


def _write_fixture() -> tuple[Path, Path, int]:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in range(12):
        rows.append([((row + 1) * 0.125) * ((i % 17) - 8) for i in range(ROW_VALUES)])
    encoded = encode_table(rows, refinement_passes=2, scale_dtype="bf16")
    split = 5 * ROW_BYTES
    second_offset = ALIGN
    payload = encoded[:split] + bytes(second_offset - split) + encoded[split:]
    DATA_PATH.write_bytes(payload)
    segments = []
    for first, end, offset in ((0, 5, 0), (5, 12, second_offset)):
        segment_bytes = encoded[first * ROW_BYTES : end * ROW_BYTES]
        segments.append(
            {
                "first_row": first,
                "end_row": end,
                "data_offset": offset,
                "byte_length": len(segment_bytes),
                "sha256": hashlib.sha256(segment_bytes).hexdigest(),
            }
        )
    manifest = {
        "format": "q3_ple_32",
        "version": 1,
        "endianness": "little",
        "block_values": BLOCK_VALUES,
        "block_bytes": BLOCK_BYTES,
        "row_values": ROW_VALUES,
        "row_bytes": ROW_BYTES,
        "rows": len(rows),
        "payload_bytes": len(encoded),
        "file_bytes": len(payload),
        "data_file": DATA_PATH.name,
        "weight_scale": 1.25,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "segments": segments,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return MANIFEST_PATH, DATA_PATH, len(rows)


@pytest.fixture(scope="module")
def fixture_paths():
    return _write_fixture()


def test_q3_reader_constants_and_ordered_gather(fixture_paths):
    manifest, _, row_count = fixture_paths
    with Q3PLEReader(manifest) as reader:
        assert reader.row_count == row_count
        assert reader.total_payload_bytes == row_count * 70
        assert reader.gather([11, 0, 11]).shape == (3, ROW_VALUES)
        scaled = reader.gather16(list(range(16)) if row_count >= 16 else [0] * 16, apply_weight_scale=True)
        assert scaled.dtype == torch.bfloat16
        assert scaled.shape == (16, ROW_VALUES)
        assert torch.equal(scaled[0], reader.gather([0], apply_weight_scale=True)[0])


def test_q3_reader_matches_authoritative_codec(fixture_paths):
    manifest, data, _ = fixture_paths
    # The fixture uses two segments and includes alignment padding between them.
    encoded = data.read_bytes()
    with Q3PLEReader(manifest) as reader:
        for row in (0, 1, 4, 5, 6, 11):
            raw_row = (encoded[row * ROW_BYTES : (row + 1) * ROW_BYTES]
                       if row < 5 else encoded[ALIGN + (row - 5) * ROW_BYTES : ALIGN + (row - 4) * ROW_BYTES])
            expected = torch.tensor(
                dequantize_row(raw_row),
                dtype=torch.bfloat16,
            )
            assert torch.equal(reader.gather([row])[0], expected)


@pytest.mark.parametrize(
    "field,value",
    [("version", 2), ("endianness", "big"), ("row_bytes", 69), ("sha256", "0" * 64)],
)
def test_q3_reader_rejects_bad_manifest(fixture_paths, field, value):
    manifest, _, _ = fixture_paths
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw[field] = value
    bad = manifest.with_name(f"bad-{field}.json")
    bad.write_text(json.dumps(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            Q3PLEReader(bad)
    finally:
        bad.unlink(missing_ok=True)


def test_q3_reader_rejects_gap_overlap_and_truncation(fixture_paths):
    manifest, data, _ = fixture_paths
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["segments"][1]["first_row"] = 6
    bad = manifest.with_name("bad-segments.json")
    bad.write_text(json.dumps(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            Q3PLEReader(bad)
    finally:
        bad.unlink(missing_ok=True)

    truncated = data.with_name("truncated.bin")
    truncated.write_bytes(data.read_bytes()[:-1])
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["data_file"] = truncated.name
    raw["file_bytes"] -= 1
    bad = manifest.with_name("bad-truncated.json")
    bad.write_text(json.dumps(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            Q3PLEReader(bad)
    finally:
        bad.unlink(missing_ok=True)
        truncated.unlink(missing_ok=True)


def test_q3_reader_requires_z_backing():
    with pytest.raises((ValueError, FileNotFoundError)):
        Q3PLEReader("C:\\q3-ple\\ple-q3.json")
