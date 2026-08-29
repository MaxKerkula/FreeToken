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
import operator
import struct
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


BLOCK_VALUES = 32
BLOCK_BYTES = 14
ROW_VALUES = 160
BLOCKS_PER_ROW = 5
ROW_BYTES = BLOCKS_PER_ROW * BLOCK_BYTES
FORMAT = "q3_ple_32"
VERSION = 1
ALIGN = 4096
REFINEMENT_PASSES = 2
DEFAULT_SEGMENT_ROWS = 128
DEFAULT_PROCESSING_CHUNK_ROWS = 131_072

# Production Qwen4 PLE geometry.  ``segment_count`` is a logical source
# tensor count; it is deliberately independent from the bounded row chunk used
# while reading a Safetensors tensor.
PRODUCTION_SEGMENT_COUNT = 128
PRODUCTION_ROWS_PER_SEGMENT = 2_500_012
PRODUCTION_TOTAL_ROWS = 320_001_536
PRODUCTION_SEGMENT_BYTES = 175_000_840
PRODUCTION_TOTAL_BYTES = 22_400_107_520


def plan_q3_ple_production() -> dict[str, int]:
    """Return the frozen real-table plan without allocating any payload."""

    return {
        "segment_count": PRODUCTION_SEGMENT_COUNT,
        "rows_per_segment": PRODUCTION_ROWS_PER_SEGMENT,
        "total_rows": PRODUCTION_TOTAL_ROWS,
        "row_bytes": ROW_BYTES,
        "segment_bytes": PRODUCTION_SEGMENT_BYTES,
        "total_bytes": PRODUCTION_TOTAL_BYTES,
    }


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


def _z_output_path(path: str | os.PathLike[str]) -> Path:
    """Resolve an output path and require an existing ``Z:`` parent directory.

    Unlike :func:`_z_path`, this helper permits the leaf file not to exist.  The
    writer intentionally does not create arbitrary parent directories: callers
    must choose an already-created, Z-backed fixture or checkpoint directory.
    """

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"Q3_PLE_32 output path must be absolute: {path}")
    lexical_drive, _ = os.path.splitdrive(str(candidate))
    if lexical_drive.upper() != "Z:" and not str(candidate).lower().startswith("/z/"):
        raise ValueError(f"Q3_PLE_32 output must resolve to Z:, got {candidate}")
    parent = candidate.parent.resolve(strict=True)
    resolved = parent / candidate.name
    drive, _ = os.path.splitdrive(str(resolved))
    if drive.upper() != "Z:" and not str(resolved).lower().startswith("/z/"):
        raise ValueError(f"Q3_PLE_32 output must resolve to Z:, got {resolved}")
    return resolved


def _pack_codes(codes: Sequence[int]) -> bytes:
    if len(codes) != BLOCK_VALUES:
        raise ValueError(f"expected {BLOCK_VALUES} codes, got {len(codes)}")
    packed = 0
    for index, code in enumerate(codes):
        if not 0 <= code <= 7:
            raise ValueError(f"code {index} is outside 0..7: {code}")
        packed |= int(code) << (3 * index)
    return packed.to_bytes(12, "little")


def _bf16_bits(value: float) -> int:
    """Round a finite Python float to an IEEE BF16 bit pattern."""

    try:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"Q3_PLE_32 scale is outside float32 range: {value!r}") from exc
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def _bf16_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) << 16))[0]


def _store_bf16_scale(value: float) -> tuple[bytes, float]:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"Q3_PLE_32 scale must be finite and non-negative: {value!r}")
    bits = _bf16_bits(value)
    # A positive source scale must remain representable after BF16 storage.  A
    # zero source scale is reserved for an all-zero block.
    if value > 0.0 and bits == 0:
        bits = 1
    stored = _bf16_from_bits(bits)
    if not math.isfinite(stored):
        raise ValueError("Q3_PLE_32 stored BF16 scale is not finite")
    return struct.pack("<H", bits), stored


def _codes_for_scale(values: Sequence[float], scale: float) -> list[int]:
    if scale == 0.0:
        return [4] * BLOCK_VALUES
    return [max(-4, min(3, int(round(value / scale)))) + 4 for value in values]


def quantize_block(values: Sequence[float], *, refinement_passes: int = REFINEMENT_PASSES) -> bytes:
    """Encode one 32-value block using the canonical Q3_PLE_32 recipe.

    This mirrors ``scripts/q3_ple_32_reference.py`` while keeping production
    conversion independent of the repository's executable reference script.
    """

    if len(values) != BLOCK_VALUES:
        raise ValueError(f"expected {BLOCK_VALUES} values, got {len(values)}")
    if refinement_passes < 0:
        raise ValueError("refinement_passes must be non-negative")
    try:
        source = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("Q3_PLE_32 values must be numeric") from exc
    if not all(math.isfinite(value) for value in source):
        raise ValueError("Q3_PLE_32 cannot encode non-finite values")

    minimum = min(source)
    maximum = max(source)
    scale = max(-minimum / 4.0, maximum / 3.0)
    if scale == 0.0:
        scale_bytes, _ = _store_bf16_scale(0.0)
        return scale_bytes + _pack_codes([4] * BLOCK_VALUES)

    codes = _codes_for_scale(source, scale)
    for _ in range(refinement_passes):
        quants = [code - 4 for code in codes]
        denominator = sum(quant * quant for quant in quants)
        if denominator == 0:
            break
        refined = sum(value * quant for value, quant in zip(source, quants)) / denominator
        if refined <= 0.0 or not math.isfinite(refined):
            break
        new_codes = _codes_for_scale(source, refined)
        scale = refined
        if new_codes == codes:
            codes = new_codes
            break
        codes = new_codes

    scale_bytes, stored_scale = _store_bf16_scale(scale)
    # Requantize once against the stored BF16 value so decoding exactly follows
    # runtime behavior rather than the pre-rounded Python scale.
    codes = _codes_for_scale(source, stored_scale)
    block = scale_bytes + _pack_codes(codes)
    if len(block) != BLOCK_BYTES:
        raise AssertionError(f"Q3_PLE_32 block has wrong size: {len(block)}")
    return block


def quantize_row(values: Sequence[float], *, refinement_passes: int = REFINEMENT_PASSES) -> bytes:
    """Encode a 160-value row as five canonical Q3_PLE_32 blocks."""

    if len(values) != ROW_VALUES:
        raise ValueError(f"expected {ROW_VALUES} values, got {len(values)}")
    return b"".join(
        quantize_block(values[offset : offset + BLOCK_VALUES], refinement_passes=refinement_passes)
        for offset in range(0, ROW_VALUES, BLOCK_VALUES)
    )


def quantize_rows_batched(
    rows: torch.Tensor,
    *,
    device: str | torch.device,
    refinement_passes: int = REFINEMENT_PASSES,
) -> bytes:
    """Encode a bounded ``[rows, 160]`` batch with scalar-codec-exact arithmetic.

    The production PLE source is FP8, so every source value and every product
    used by the least-squares refinement is exactly representable in float64.
    ``cumsum(...)[..., -1]`` deliberately preserves the reference encoder's
    left-to-right Python ``sum`` order.  The stored BF16 scale is rounded by the
    same integer-bit recipe as :func:`_bf16_bits`, followed by the same final
    requantization against that stored value.

    The returned bytes retain row-major, five-block-per-row order.  Choosing a
    CUDA device changes only where the bounded arithmetic executes; it does not
    change the on-disk format or logical segment identity.
    """

    if refinement_passes < 0:
        raise ValueError("refinement_passes must be non-negative")
    if not isinstance(rows, torch.Tensor) or rows.ndim != 2 or rows.shape[1] != ROW_VALUES:
        shape = tuple(rows.shape) if isinstance(rows, torch.Tensor) else None
        raise ValueError(f"expected a [rows, {ROW_VALUES}] tensor, got {shape}")
    if rows.shape[0] <= 0:
        raise ValueError("Q3_PLE_32 batch must contain at least one row")

    target = torch.device(device)
    # The historical source iterator converted FP8 to float32 before Python
    # materialized each value.  Keep that conversion boundary explicit before
    # moving to float64 so the batched path has identical source semantics.
    source = rows.to(dtype=torch.float32).to(device=target, dtype=torch.float64).reshape(
        -1, BLOCKS_PER_ROW, BLOCK_VALUES
    )
    if not bool(torch.isfinite(source).all().item()):
        raise ValueError("Q3_PLE_32 cannot encode non-finite values")

    minimum = source.amin(dim=-1)
    maximum = source.amax(dim=-1)
    scale = torch.maximum(-minimum / 4.0, maximum / 3.0)
    zero = scale == 0.0

    def codes_for_scale(candidate: torch.Tensor) -> torch.Tensor:
        safe = torch.where(candidate == 0.0, torch.ones_like(candidate), candidate)
        codes = torch.round(source / safe.unsqueeze(-1)).clamp_(-4, 3).to(torch.int16) + 4
        return torch.where((candidate == 0.0).unsqueeze(-1), 4, codes)

    codes = codes_for_scale(scale)
    done = zero.clone()
    for _ in range(refinement_passes):
        quants = codes.to(torch.int16) - 4
        denominator = (
            quants.to(torch.int64) * quants.to(torch.int64)
        ).cumsum(dim=-1)[..., -1]
        products = source * quants.to(torch.float64)
        numerator = products.cumsum(dim=-1)[..., -1]
        refined = torch.where(
            denominator != 0,
            numerator / denominator.to(torch.float64),
            torch.zeros_like(numerator),
        )
        valid = (~done) & (denominator != 0) & (refined > 0.0) & torch.isfinite(refined)
        candidate = torch.where(valid, refined, torch.ones_like(refined))
        new_codes = codes_for_scale(candidate)
        same = valid & (new_codes == codes).all(dim=-1)
        codes = torch.where(valid.unsqueeze(-1), new_codes, codes)
        scale = torch.where(valid, refined, scale)
        done |= (~valid) | same

    # Match struct.pack('<f') followed by the reference RN-even BF16 bit rule.
    float32_scale = scale.to(torch.float32).contiguous()
    float32_bits = float32_scale.view(torch.int32)
    scale_bits = (
        (float32_bits + 0x7FFF + ((float32_bits >> 16) & 1)) >> 16
    ) & 0xFFFF
    scale_bits = torch.where((scale > 0.0) & (scale_bits == 0), 1, scale_bits).to(torch.int32)
    stored_scale = (scale_bits << 16).contiguous().view(torch.float32).to(torch.float64)
    codes = codes_for_scale(stored_scale)

    output = torch.empty(
        (source.shape[0], BLOCKS_PER_ROW, BLOCK_BYTES),
        dtype=torch.uint8,
        device=target,
    )
    output[..., 0] = (scale_bits & 0xFF).to(torch.uint8)
    output[..., 1] = ((scale_bits >> 8) & 0xFF).to(torch.uint8)
    packed_codes = codes.to(torch.int64)
    for group in range(4):
        packed = torch.zeros_like(scale_bits, dtype=torch.int64)
        for code_index in range(8):
            packed |= packed_codes[..., group * 8 + code_index] << (3 * code_index)
        byte_offset = 2 + group * 3
        output[..., byte_offset] = (packed & 0xFF).to(torch.uint8)
        output[..., byte_offset + 1] = ((packed >> 8) & 0xFF).to(torch.uint8)
        output[..., byte_offset + 2] = ((packed >> 16) & 0xFF).to(torch.uint8)
    return output.contiguous().cpu().numpy().tobytes()


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
        # Older Stage 6 fixtures predate this field, so absence remains
        # readable.  A present fingerprint is always canonical SHA-256 hex;
        # malformed provenance must fail closed rather than being ignored.
        if "source_fingerprint" in manifest:
            _validate_source_fingerprint(manifest["source_fingerprint"])

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
        if "segment_count" in manifest and int(manifest["segment_count"]) != len(self.segments):
            raise ValueError("Q3_PLE_32 segment_count does not match the segment directory")
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
        contiguous = self.manifest.get("storage_layout") == "contiguous_rows_v1"
        for segment in self.segments:
            if segment.first_row != expected_row or segment.end_row <= segment.first_row:
                raise ValueError("Q3_PLE_32 segment rows have a gap, overlap, or bad order")
            if segment.data_offset < 0:
                raise ValueError("Q3_PLE_32 segment data offset is negative")
            if contiguous and segment.data_offset != previous_end:
                raise ValueError("Q3_PLE_32 contiguous segment directory has a gap or overlap")
            if not contiguous and segment.data_offset % ALIGN:
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


def _align_up(value: int, alignment: int = ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


def _validate_source_fingerprint(source_fingerprint: str) -> str:
    if not isinstance(source_fingerprint, str):
        raise ValueError("source_fingerprint must be a 64-character SHA-256 hex string")
    fingerprint = source_fingerprint.lower()
    if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
        raise ValueError("source_fingerprint must be a 64-character SHA-256 hex string")
    return fingerprint


def _materialize_row(row: object) -> list[float]:
    """Materialize one bounded row without retaining any other source rows."""

    if isinstance(row, torch.Tensor):
        if row.ndim != 1 or row.numel() != ROW_VALUES:
            raise ValueError(f"Q3_PLE_32 row must contain exactly {ROW_VALUES} values")
        try:
            values = row.detach().cpu().tolist()
        except Exception as exc:
            raise ValueError("Q3_PLE_32 row tensor could not be copied to CPU") from exc
    else:
        try:
            values = list(row)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Q3_PLE_32 row must contain exactly {ROW_VALUES} values") from exc
    if len(values) != ROW_VALUES:
        raise ValueError(f"Q3_PLE_32 row must contain exactly {ROW_VALUES} values, got {len(values)}")
    return values


def _partial_path(path: Path, token: str) -> Path:
    return path.with_name(f".{path.name}.partial-{os.getpid()}-{threading.get_ident()}-{token}")


def _validate_segment_directory(
    segments: Sequence[dict[str, int | str]], row_count: int, file_bytes: int
) -> None:
    expected_row = 0
    previous_end = 0
    for segment in segments:
        first_row = int(segment["first_row"])
        end_row = int(segment["end_row"])
        data_offset = int(segment["data_offset"])
        byte_length = int(segment["byte_length"])
        digest = str(segment["sha256"])
        if first_row != expected_row or end_row <= first_row:
            raise ValueError("Q3_PLE_32 writer generated a malformed segment directory")
        if data_offset != previous_end:
            raise ValueError("Q3_PLE_32 writer generated a non-contiguous segment")
        if byte_length != (end_row - first_row) * ROW_BYTES:
            raise ValueError("Q3_PLE_32 writer generated a segment length mismatch")
        if data_offset + byte_length > file_bytes:
            raise ValueError("Q3_PLE_32 writer generated overlapping/out-of-range segments")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Q3_PLE_32 writer generated a malformed segment hash")
        expected_row = end_row
        previous_end = data_offset + byte_length
    if not segments or expected_row != row_count:
        raise ValueError("Q3_PLE_32 writer generated incomplete segment coverage")


def _fsync(handle: object) -> None:
    # This helper exists to keep the finalize path explicit and easy to audit;
    # the writer only passes ordinary binary file handles here.
    file_handle = handle  # type narrowing for type checkers without a runtime dependency
    file_handle.flush()  # type: ignore[attr-defined]
    os.fsync(file_handle.fileno())  # type: ignore[attr-defined]


def write_q3_ple_sidecar(
    rows: Iterable[Sequence[float] | torch.Tensor],
    data_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    source_fingerprint: str,
    weight_scale: float,
    segment_rows: int = DEFAULT_SEGMENT_ROWS,
) -> dict:
    """Stream rows into an atomic, reader-compatible Q3_PLE_32 sidecar.

    ``rows`` is consumed exactly once and only the current 160-value row is held
    in memory.  Segments are logical hash/addressing ranges over one contiguous
    row stream.  There is no inter-row or inter-segment padding: the production
    table is exactly ``rows * 70`` bytes.  Both files are written under unique partial
    names, fsynced, and atomically renamed into place on successful completion.

    The converter rule is intentionally fixed at two least-squares refinement
    passes (the provisional Q3_PLE_32 recipe), and ``weight_scale`` is metadata
    applied by the runtime after row dequantization rather than folded into the
    per-block scales.
    """

    data_final = _z_output_path(data_path)
    manifest_final = _z_output_path(manifest_path)
    if data_final == manifest_final:
        raise ValueError("Q3_PLE_32 data_path and manifest_path must differ")
    source_digest = _validate_source_fingerprint(source_fingerprint)
    if isinstance(segment_rows, bool):
        raise ValueError("segment_rows must be a positive integer")
    try:
        segment_size = operator.index(segment_rows)
    except TypeError as exc:
        raise ValueError("segment_rows must be a positive integer") from exc
    if segment_size <= 0:
        raise ValueError("segment_rows must be a positive integer")
    try:
        global_scale = float(weight_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("weight_scale must be finite") from exc
    if not math.isfinite(global_scale):
        raise ValueError("weight_scale must be finite")

    # Unique tokens make concurrent conversion attempts independent and avoid
    # ever truncating a stale partial file left by an interrupted process.
    token = uuid.uuid4().hex
    data_partial = _partial_path(data_final, token)
    manifest_partial = _partial_path(manifest_final, token)
    segments: list[dict[str, int | str]] = []
    whole_digest = hashlib.sha256()
    payload_digest = hashlib.sha256()
    rows_written = 0
    file_offset = 0
    current_segment: dict[str, int | str] | None = None
    segment_digest: hashlib._Hash | None = None

    try:
        with data_partial.open("wb") as output:
            for source_row in rows:
                row_values = _materialize_row(source_row)
                encoded_row = quantize_row(row_values, refinement_passes=REFINEMENT_PASSES)
                if len(encoded_row) != ROW_BYTES:
                    raise AssertionError(f"Q3_PLE_32 row has wrong size: {len(encoded_row)}")

                if rows_written % segment_size == 0:
                    if current_segment is not None:
                        assert segment_digest is not None
                        current_segment["end_row"] = rows_written
                        current_segment["byte_length"] = (
                            rows_written * ROW_BYTES - int(current_segment["first_row"]) * ROW_BYTES
                        )
                        current_segment["sha256"] = segment_digest.hexdigest()
                        segments.append(current_segment)
                    current_segment = {
                        "first_row": rows_written,
                        "end_row": rows_written,
                        "data_offset": file_offset,
                        "byte_length": 0,
                        "sha256": "",
                    }
                    segment_digest = hashlib.sha256()

                assert current_segment is not None and segment_digest is not None
                output.write(encoded_row)
                whole_digest.update(encoded_row)
                payload_digest.update(encoded_row)
                segment_digest.update(encoded_row)
                file_offset += len(encoded_row)
                rows_written += 1

            if current_segment is None:
                raise ValueError("Q3_PLE_32 rows must contain at least one row")
            assert segment_digest is not None
            current_segment["end_row"] = rows_written
            current_segment["byte_length"] = (
                rows_written * ROW_BYTES - int(current_segment["first_row"]) * ROW_BYTES
            )
            current_segment["sha256"] = segment_digest.hexdigest()
            segments.append(current_segment)
            _fsync(output)
    except Exception:
        data_partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        raise

    try:
        file_bytes = data_partial.stat().st_size
    except Exception:
        data_partial.unlink(missing_ok=True)
        raise
    if file_bytes != file_offset:
        data_partial.unlink(missing_ok=True)
        raise OSError(f"Q3_PLE_32 partial length mismatch: {file_bytes} != {file_offset}")
    try:
        _validate_segment_directory(segments, rows_written, file_bytes)
    except Exception:
        data_partial.unlink(missing_ok=True)
        raise
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "endianness": "little",
        "block_values": BLOCK_VALUES,
        "block_bytes": BLOCK_BYTES,
        "row_values": ROW_VALUES,
        "row_bytes": ROW_BYTES,
        "rows": rows_written,
        "payload_bytes": rows_written * ROW_BYTES,
        "file_bytes": file_bytes,
        "storage_layout": "contiguous_rows_v1",
        "data_file": os.path.relpath(data_final, manifest_final.parent),
        "weight_scale": global_scale,
        "source_fingerprint": source_digest,
        "sha256": whole_digest.hexdigest(),
        "payload_sha256": payload_digest.hexdigest(),
        "segments": segments,
    }
    try:
        with manifest_partial.open("w", encoding="utf-8", newline="\n") as manifest_handle:
            json.dump(manifest, manifest_handle, ensure_ascii=False, indent=2, sort_keys=True)
            manifest_handle.write("\n")
            _fsync(manifest_handle)
        os.replace(data_partial, data_final)
        os.replace(manifest_partial, manifest_final)
    except Exception:
        data_partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        raise
    return manifest


def write_q3_ple_segmented_sidecar(
    segments: Iterable[Iterable[Sequence[float] | torch.Tensor]],
    data_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    source_fingerprint: str,
    weight_scale: float,
    segment_count: int,
    rows_per_segment: int | None = None,
    batched: bool = False,
    quantization_device: str | torch.device | None = None,
) -> dict:
    """Write Q3 data with explicit logical source-segment boundaries.

    Each item in ``segments`` represents one source tensor (for production,
    ``shard_0`` through ``shard_127``).  Segment identity is therefore stable
    regardless of the internal row-chunk size used by the caller.  The legacy
    :func:`write_q3_ple_sidecar` API remains row-count based for compatibility
    with small historical fixtures; production Safetensors conversion uses
    this explicit API instead.
    """

    data_final = _z_output_path(data_path)
    manifest_final = _z_output_path(manifest_path)
    if data_final == manifest_final:
        raise ValueError("Q3_PLE_32 data_path and manifest_path must differ")
    source_digest = _validate_source_fingerprint(source_fingerprint)
    if isinstance(segment_count, bool):
        raise ValueError("segment_count must be a positive integer")
    try:
        expected_segments = operator.index(segment_count)
    except TypeError as exc:
        raise ValueError("segment_count must be a positive integer") from exc
    if expected_segments <= 0:
        raise ValueError("segment_count must be a positive integer")
    expected_rows = None if rows_per_segment is None else operator.index(rows_per_segment)
    if expected_rows is not None and expected_rows <= 0:
        raise ValueError("rows_per_segment must be a positive integer")
    try:
        global_scale = float(weight_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("weight_scale must be finite") from exc
    if not math.isfinite(global_scale):
        raise ValueError("weight_scale must be finite")

    token = uuid.uuid4().hex
    data_partial = _partial_path(data_final, token)
    manifest_partial = _partial_path(manifest_final, token)
    segments_manifest: list[dict[str, int | str]] = []
    whole_digest = hashlib.sha256()
    payload_digest = hashlib.sha256()
    rows_written = 0
    file_offset = 0

    try:
        with data_partial.open("wb") as output:
            for segment_index, source_segment in enumerate(segments):
                if segment_index >= expected_segments:
                    raise ValueError(
                        f"Q3_PLE_32 expected {expected_segments} logical segments, got more"
                    )
                first_row = rows_written
                segment_offset = file_offset
                segment_digest = hashlib.sha256()
                segment_rows = 0
                for source_item in source_segment:
                    if batched:
                        if not isinstance(source_item, torch.Tensor):
                            raise ValueError("batched Q3_PLE_32 input items must be tensors")
                        batch_rows = int(source_item.shape[0]) if source_item.ndim == 2 else 0
                        encoded = quantize_rows_batched(
                            source_item,
                            device=quantization_device or "cpu",
                            refinement_passes=REFINEMENT_PASSES,
                        )
                        expected_bytes = batch_rows * ROW_BYTES
                    else:
                        row_values = _materialize_row(source_item)
                        encoded = quantize_row(row_values, refinement_passes=REFINEMENT_PASSES)
                        batch_rows = 1
                        expected_bytes = ROW_BYTES
                    if len(encoded) != expected_bytes:
                        raise AssertionError(
                            f"Q3_PLE_32 batch has wrong size: {len(encoded)} != {expected_bytes}"
                        )
                    output.write(encoded)
                    whole_digest.update(encoded)
                    payload_digest.update(encoded)
                    segment_digest.update(encoded)
                    file_offset += len(encoded)
                    rows_written += batch_rows
                    segment_rows += batch_rows
                if segment_rows <= 0:
                    raise ValueError(f"Q3_PLE_32 logical segment {segment_index} is empty")
                if expected_rows is not None and segment_rows != expected_rows:
                    raise ValueError(
                        f"Q3_PLE_32 logical segment {segment_index} has {segment_rows} rows, "
                        f"expected {expected_rows}"
                    )
                segments_manifest.append(
                    {
                        "first_row": first_row,
                        "end_row": rows_written,
                        "data_offset": segment_offset,
                        "byte_length": segment_rows * ROW_BYTES,
                        "sha256": segment_digest.hexdigest(),
                    }
                )
            if len(segments_manifest) != expected_segments:
                raise ValueError(
                    f"Q3_PLE_32 expected {expected_segments} logical segments, "
                    f"got {len(segments_manifest)}"
                )
            _fsync(output)
    except Exception:
        data_partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        raise

    file_bytes = data_partial.stat().st_size
    if file_bytes != file_offset:
        data_partial.unlink(missing_ok=True)
        raise OSError(f"Q3_PLE_32 partial length mismatch: {file_bytes} != {file_offset}")
    _validate_segment_directory(segments_manifest, rows_written, file_bytes)
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "endianness": "little",
        "block_values": BLOCK_VALUES,
        "block_bytes": BLOCK_BYTES,
        "row_values": ROW_VALUES,
        "row_bytes": ROW_BYTES,
        "rows": rows_written,
        "payload_bytes": rows_written * ROW_BYTES,
        "file_bytes": file_bytes,
        "storage_layout": "contiguous_rows_v1",
        "data_file": os.path.relpath(data_final, manifest_final.parent),
        "weight_scale": global_scale,
        "source_fingerprint": source_digest,
        "sha256": whole_digest.hexdigest(),
        "payload_sha256": payload_digest.hexdigest(),
        "segments": segments_manifest,
        "segment_count": expected_segments,
        "segment_identity": "source_tensor_numeric_suffix_v1",
    }
    try:
        with manifest_partial.open("w", encoding="utf-8", newline="\n") as manifest_handle:
            json.dump(manifest, manifest_handle, ensure_ascii=False, indent=2, sort_keys=True)
            manifest_handle.write("\n")
            _fsync(manifest_handle)
        os.replace(data_partial, data_final)
        os.replace(manifest_partial, manifest_final)
    except Exception:
        data_partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        raise
    return manifest


def write_q3_ple_from_safetensors(
    model_path: str | os.PathLike[str],
    data_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    layer_id: int,
    split_parts: int,
    source_fingerprint: str,
    rows_per_chunk: int = DEFAULT_PROCESSING_CHUNK_ROWS,
    segment_rows: int | None = None,
    processing_chunk_rows: int | None = None,
    segment_count: int | None = None,
    rows_per_segment: int | None = None,
    quantization_device: str | torch.device | None = None,
) -> dict:
    """Stream the official FP8 PLE shards into the native Q3 sidecar.

    Shards and rows are consumed in exact ``shard_0..shard_N`` order.  A
    Safetensors slice is read in bounded row chunks; the full 51.2-GiB table is
    never materialized.  The source per-model ``weight_scale`` remains a separate
    scalar in the Q3 manifest and is not folded into block scales.
    """

    folder = Path(model_path).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Q3 PLE source must be a local checkpoint directory: {folder}")
    if processing_chunk_rows is not None:
        if rows_per_chunk != DEFAULT_PROCESSING_CHUNK_ROWS:
            raise ValueError("specify only one of rows_per_chunk or processing_chunk_rows")
        rows_per_chunk = processing_chunk_rows
    if rows_per_chunk <= 0 or split_parts <= 0:
        raise ValueError("processing chunk rows and split_parts must be positive")
    if segment_rows is not None:
        # ``segment_rows`` was the legacy flat-stream API.  Keep it for small
        # fixtures, but production conversion must use explicit source tensor
        # segments so a value of 128 can never mean 128 rows per segment.
        if segment_count is not None or rows_per_segment is not None:
            raise ValueError("segment_rows cannot be combined with explicit segment geometry")
        if int(split_parts) == PRODUCTION_SEGMENT_COUNT and int(segment_rows) == DEFAULT_SEGMENT_ROWS:
            # Historical callers passed ``segment_rows=128`` intending the
            # production 128 logical tensors.  Treat that exact combination as
            # the explicit segmented contract; it must never create 128-row
            # chunks across the complete flattened table.
            segment_rows = None
    index_path = folder / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    prefix = (
        f"model.language_model.layers.{int(layer_id)}.ple.ple_embedding."
        "ngram_embedding"
    )
    shard_keys = [f"{prefix}.shard_{part}.weight" for part in range(int(split_parts))]
    shard_prefix = prefix + ".shard_"
    indexed_keys: dict[int, str] = {}
    for key in weight_map:
        if not isinstance(key, str) or not key.startswith(shard_prefix):
            continue
        suffix = key[len(shard_prefix) :]
        if not suffix.endswith(".weight"):
            raise ValueError(f"malformed PLE source tensor suffix: {key}")
        index_text = suffix[: -len(".weight")]
        if not index_text.isdigit():
            raise ValueError(f"malformed PLE source tensor suffix: {key}")
        index = int(index_text)
        if index in indexed_keys:
            raise ValueError(f"duplicate PLE source tensor index: {index}")
        if not 0 <= index < int(split_parts):
            raise ValueError(f"PLE source tensor index outside 0..{int(split_parts) - 1}: {index}")
        indexed_keys[index] = key
    if set(indexed_keys) != set(range(int(split_parts))):
        missing_indices = sorted(set(range(int(split_parts))) - set(indexed_keys))
        raise ValueError(f"missing PLE source tensor indices: {missing_indices}")
    missing = [key for key in shard_keys if key not in weight_map]
    scale_key = prefix + ".weight_scale"
    if missing or scale_key not in weight_map:
        raise ValueError(f"incomplete PLE source mapping under {prefix}")

    import safetensors

    scale_file = folder / weight_map[scale_key]
    with safetensors.safe_open(scale_file, framework="pt", device="cpu") as handle:
        scale = handle.get_tensor(scale_key).reshape(())
    weight_scale = float(scale.float().item())

    def iter_segment_batches(key: str):
        source_file = folder / weight_map[key]
        with safetensors.safe_open(source_file, framework="pt", device="cpu") as handle:
            sliced = handle.get_slice(key)
            shape = tuple(int(value) for value in sliced.get_shape())
            if len(shape) != 2 or shape[1] != ROW_VALUES:
                raise ValueError(f"unexpected PLE source shape for {key}: {shape}")
            if rows_per_segment is not None and shape[0] != int(rows_per_segment):
                raise ValueError(
                    f"unexpected PLE source row count for {key}: {shape[0]} != {rows_per_segment}"
                )
            for start in range(0, shape[0], int(rows_per_chunk)):
                chunk = sliced[start : min(start + int(rows_per_chunk), shape[0])]
                if chunk.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"unexpected PLE source dtype for {key}: {chunk.dtype}")
                yield chunk

    def iter_segment_rows(key: str):
        for chunk in iter_segment_batches(key):
            yield from chunk.float()

    # Explicit segmented mode is the production contract.  Legacy callers can
    # request flat row segmentation by passing ``segment_rows`` explicitly.
    if segment_rows is not None:
        def iter_rows():
            for key in shard_keys:
                yield from iter_segment_rows(key)

        return write_q3_ple_sidecar(
            iter_rows(), data_path, manifest_path,
            source_fingerprint=source_fingerprint,
            weight_scale=weight_scale,
            segment_rows=segment_rows,
        )

    logical_count = int(segment_count if segment_count is not None else split_parts)
    if logical_count != int(split_parts):
        raise ValueError("segment_count must equal split_parts for PLE Safetensors conversion")
    selected_device = quantization_device
    if selected_device is None:
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    ordered_segments = (iter_segment_batches(indexed_keys[index]) for index in range(logical_count))
    return write_q3_ple_segmented_sidecar(
        ordered_segments,
        data_path,
        manifest_path,
        source_fingerprint=source_fingerprint,
        weight_scale=weight_scale,
        segment_count=logical_count,
        rows_per_segment=rows_per_segment,
        batched=True,
        quantization_device=selected_device,
    )


__all__ = [
    "ALIGN",
    "BLOCK_BYTES",
    "BLOCK_VALUES",
    "DEFAULT_SEGMENT_ROWS",
    "DEFAULT_PROCESSING_CHUNK_ROWS",
    "FORMAT",
    "REFINEMENT_PASSES",
    "Q3PLEReader",
    "Q3PLESegment",
    "ROW_BYTES",
    "ROW_VALUES",
    "VERSION",
    "PRODUCTION_SEGMENT_COUNT",
    "PRODUCTION_ROWS_PER_SEGMENT",
    "PRODUCTION_TOTAL_ROWS",
    "PRODUCTION_SEGMENT_BYTES",
    "PRODUCTION_TOTAL_BYTES",
    "plan_q3_ple_production",
    "quantize_block",
    "quantize_row",
    "quantize_rows_batched",
    "write_q3_ple_sidecar",
    "write_q3_ple_segmented_sidecar",
    "write_q3_ple_from_safetensors",
]
