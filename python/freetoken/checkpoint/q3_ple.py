"""Native reader for the Qwen4 ``Q3_PLE_32`` lookup-table sidecar.

The reader deliberately knows only the PLE GET_ROWS format.  It does not expose a
matmul/dequant path and it never maps or allocates the complete table.  A small JSON
directory describes the logical rows and the byte ranges containing each segment;
the payload remains an ordinary read-only file on the project's required ``Z:``
volume.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


BLOCK_VALUES = 32
BLOCK_BYTES = 14
ROW_VALUES = 160
BLOCKS_PER_ROW = 5
ROW_BYTES = BLOCKS_PER_ROW * BLOCK_BYTES
FORMAT = "q3_ple_32"
VERSION = 1
ALIGN = 4096


def _z_path(path: str | os.PathLike[str]) -> Path:
    """Resolve *path* and fail closed unless its physical drive is ``Z:``."""

    resolved = Path(path).expanduser().resolve(strict=True)
    drive, _ = os.path.splitdrive(str(resolved))
    # On Windows splitdrive is authoritative.  The second clause keeps the
    # check useful in a POSIX test harness mounted as ``/z`` while still never
    # accepting an unqualified relative path.
    if drive.upper() != "Z:" and not str(resolved).lower().startswith("/z/"):
        raise ValueError(f"Q3_PLE_32 backing must resolve to Z:, got {resolved}")
    return resolved


def _unpack_codes(payload: bytes) -> list[int]:
    if len(payload) != 12:
        raise ValueError(f"Q3_PLE_32 code payload must be 12 bytes, got {len(payload)}")
    bits = int.from_bytes(payload, "little")
    return [(bits >> (3 * i)) & 0x7 for i in range(BLOCK_VALUES)]


def _decode_row(row: bytes) -> torch.Tensor:
    if len(row) != ROW_BYTES:
        raise ValueError(f"Q3_PLE_32 row must be {ROW_BYTES} bytes, got {len(row)}")
    values: list[float] = []
    for block_start in range(0, ROW_BYTES, BLOCK_BYTES):
        block = row[block_start : block_start + BLOCK_BYTES]
        # The format authority specifies a little-endian BF16 scalar.  Decode
        # through an integer bit pattern so host endianness cannot leak in.
        scale_bits = int.from_bytes(block[:2], "little")
        scale = struct.unpack("<f", struct.pack("<I", scale_bits << 16))[0]
        values.extend(scale * (code - 4) for code in _unpack_codes(block[2:]))
    return torch.tensor(values, dtype=torch.bfloat16)


@dataclass(frozen=True)
class Q3PLESegment:
    first_row: int
    end_row: int
    data_offset: int
    byte_length: int
    sha256: str

    @property
    def rows(self) -> int:
        return self.end_row - self.first_row


class Q3PLEReader:
    """Bounded random-row reader for a validated Q3_PLE_32 sidecar.

    ``manifest_path`` points to ``ple-q3.json`` and ``data_path`` may override
    its ``data_file``.  Opening validates the JSON schema, segment coverage,
    file length, and whole-file/segment hashes in bounded chunks.  ``gather``
    then reads only the requested 70-byte rows and returns BF16 values.
    """

    def __init__(self, manifest_path: str | os.PathLike[str], *, data_path: str | os.PathLike[str] | None = None):
        self.manifest_path = _z_path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.manifest = manifest
        if manifest.get("format") != FORMAT or int(manifest.get("version", -1)) != VERSION:
            raise ValueError("unsupported Q3_PLE_32 format/version")
        if manifest.get("endianness", "little") != "little":
            raise ValueError("Q3_PLE_32 requires little-endian metadata")
        if int(manifest.get("block_values", BLOCK_VALUES)) != BLOCK_VALUES:
            raise ValueError("Q3_PLE_32 block_values mismatch")
        if int(manifest.get("block_bytes", BLOCK_BYTES)) != BLOCK_BYTES:
            raise ValueError("Q3_PLE_32 block_bytes mismatch")
        if int(manifest.get("row_values", ROW_VALUES)) != ROW_VALUES:
            raise ValueError("Q3_PLE_32 row_values mismatch")
        if int(manifest.get("row_bytes", ROW_BYTES)) != ROW_BYTES:
            raise ValueError("Q3_PLE_32 row_bytes mismatch")

        candidate = data_path
        if candidate is None:
            candidate = self.manifest.get("data_file")
        if not candidate:
            raise ValueError("Q3_PLE_32 manifest has no data_file")
        data_candidate = Path(candidate)
        if not data_candidate.is_absolute():
            data_candidate = self.manifest_path.parent / data_candidate
        self.data_path = _z_path(data_candidate)
        self.row_count = int(manifest.get("rows", 0))
        self.total_payload_bytes = self.row_count * ROW_BYTES
        if self.row_count <= 0:
            raise ValueError("Q3_PLE_32 rows must be positive")
        declared_payload = int(manifest.get("payload_bytes", self.total_payload_bytes))
        if declared_payload != self.total_payload_bytes:
            raise ValueError("Q3_PLE_32 payload_bytes does not equal rows * row_bytes")

        raw_segments = manifest.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("Q3_PLE_32 segment directory is empty")
        self.segments: tuple[Q3PLESegment, ...] = tuple(self._parse_segment(item) for item in raw_segments)
        self._validate_segments()
        stat = self.data_path.stat()
        expected_file_bytes = int(manifest.get("file_bytes", stat.st_size))
        if stat.st_size != expected_file_bytes:
            raise ValueError(f"Q3_PLE_32 file length mismatch: {stat.st_size} != {expected_file_bytes}")
        self._handle = self.data_path.open("rb")
        self._io_lock = threading.Lock()
        try:
            self._verify_hashes()
        except Exception:
            self._handle.close()
            raise

        self.weight_scale = float(manifest.get("weight_scale", 1.0))
        if not math.isfinite(self.weight_scale):
            raise ValueError("Q3_PLE_32 weight_scale must be finite")

    def _parse_segment(self, item: object) -> Q3PLESegment:
        if not isinstance(item, dict):
            raise ValueError("Q3_PLE_32 segment must be an object")
        try:
            data_offset = item.get("data_offset")
            if data_offset is None:
                data_offset = item["offset"]
            byte_length = item.get("byte_length")
            if byte_length is None:
                byte_length = item["length"]
            segment = Q3PLESegment(
                first_row=int(item["first_row"]),
                end_row=int(item["end_row"]),
                data_offset=int(data_offset),
                byte_length=int(byte_length),
                sha256=str(item["sha256"]).lower(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed Q3_PLE_32 segment directory") from exc
        if len(segment.sha256) != 64 or any(c not in "0123456789abcdef" for c in segment.sha256):
            raise ValueError("malformed Q3_PLE_32 segment hash")
        return segment

    def _validate_segments(self) -> None:
        expected_row = 0
        previous_end = 0
        for segment in self.segments:
            if segment.first_row != expected_row or segment.end_row <= segment.first_row:
                raise ValueError("Q3_PLE_32 segment rows have a gap, overlap, or bad order")
            if segment.data_offset < 0 or segment.data_offset % ALIGN:
                raise ValueError("Q3_PLE_32 segment data offset is not 4 KiB aligned")
            if segment.byte_length != segment.rows * ROW_BYTES:
                raise ValueError("Q3_PLE_32 segment byte length does not match rows")
            if segment.data_offset < previous_end:
                raise ValueError("Q3_PLE_32 segment byte ranges overlap")
            expected_row = segment.end_row
            previous_end = segment.data_offset + segment.byte_length
        if expected_row != self.row_count:
            raise ValueError("Q3_PLE_32 segment rows do not cover the table")
        if previous_end > int(self.manifest.get("file_bytes", previous_end)):
            raise ValueError("Q3_PLE_32 segment exceeds file length")

    def _read_exact(self, offset: int, length: int) -> bytes:
        with self._io_lock:
            self._handle.seek(offset)
            payload = self._handle.read(length)
        if len(payload) != length:
            raise OSError(f"short Q3_PLE_32 read at {offset}: {len(payload)}/{length}")
        return payload

    def _verify_hashes(self) -> None:
        # Hashes are checked incrementally; this never allocates the table.
        whole = hashlib.sha256()
        with self.data_path.open("rb") as handle:
            while True:
                chunk = handle.read(8 << 20)
                if not chunk:
                    break
                whole.update(chunk)
        expected_whole = str(self.manifest.get("sha256", "")).lower()
        if len(expected_whole) != 64 or whole.hexdigest() != expected_whole:
            raise ValueError("Q3_PLE_32 whole-file hash mismatch")
        for segment in self.segments:
            digest_ctx = hashlib.sha256()
            remaining = segment.byte_length
            offset = segment.data_offset
            while remaining:
                take = min(8 << 20, remaining)
                digest_ctx.update(self._read_exact(offset, take))
                offset += take
                remaining -= take
            digest = digest_ctx.hexdigest()
            if digest != segment.sha256:
                raise ValueError(f"Q3_PLE_32 segment hash mismatch at row {segment.first_row}")

    def _segment_for_row(self, row: int) -> Q3PLESegment:
        if not 0 <= row < self.row_count:
            raise IndexError(f"Q3_PLE_32 row {row} outside 0..{self.row_count - 1}")
        # Segment count is small (128 in the production sidecar); linear search
        # keeps the directory representation transparent and deterministic.
        for segment in self.segments:
            if segment.first_row <= row < segment.end_row:
                return segment
        raise AssertionError("validated Q3_PLE_32 directory did not locate row")

    def read_row(self, row: int) -> torch.Tensor:
        segment = self._segment_for_row(int(row))
        offset = segment.data_offset + (int(row) - segment.first_row) * ROW_BYTES
        return _decode_row(self._read_exact(offset, ROW_BYTES))

    def gather(self, row_indices: Sequence[int], *, apply_weight_scale: bool = False) -> torch.Tensor:
        """Gather rows in the caller's order without deduplication or reordering."""

        rows = [self.read_row(int(index)) for index in row_indices]
        if not rows:
            output = torch.empty((0, ROW_VALUES), dtype=torch.bfloat16)
        else:
            output = torch.stack(rows, dim=0)
        if apply_weight_scale:
            output = output * self.weight_scale
        return output

    def gather16(self, row_indices: Sequence[int], *, apply_weight_scale: bool = False) -> torch.Tensor:
        if len(row_indices) != 16:
            raise ValueError(f"Qwen4 PLE requires exactly 16 logical rows, got {len(row_indices)}")
        return self.gather(row_indices, apply_weight_scale=apply_weight_scale)

    def close(self) -> None:
        handle, self._handle = getattr(self, "_handle", None), None
        if handle is not None:
            handle.close()

    def __enter__(self) -> "Q3PLEReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "ALIGN",
    "BLOCK_BYTES",
    "BLOCK_VALUES",
    "FORMAT",
    "Q3PLEReader",
    "Q3PLESegment",
    "ROW_BYTES",
    "ROW_VALUES",
    "VERSION",
]
