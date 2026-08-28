"""Bounded, file-backed routed-expert sources.

The normal :class:`~freetoken.moe.offload_cache.OffloadMoeCache` source is a
resident HostBank.  ``FileExpertSource`` is the explicit alternative used by
the Qwen4 text-only tier: one aligned record per expert and no materialised
full-layer tensor.  The format is deliberately small and boring so corruption
is detected before the first request is served.
"""

from __future__ import annotations

import hashlib
import os
import struct
import threading
from pathlib import Path
from typing import Iterable

import torch


MAGIC = b"FTEXNV4\0"
VERSION = 1
HEADER_BYTES = 4096
RECORD_BYTES = 2_772_992
RAW_RECORD_BYTES = 2_772_480
NUM_EXPERTS = 512
ALIGNMENT = 4096

# Native ModelOpt NVFP4 planes for Qwen4's local expert geometry (H=2560,
# I=640).  The six planes are kept in this order in every record.
PLANE_LAYOUT: tuple[tuple[str, int, str], ...] = (
    ("gate_up_packed", 1_638_400, "uint8"),
    ("gate_up_scale", 204_800, "float8_e4m3fn"),
    ("gate_up_global", 2_560, "float16"),
    ("down_packed", 819_200, "uint8"),
    ("down_scale", 102_400, "float8_e4m3fn"),
    ("down_global", 5_120, "float16"),
)
_PLANE_OFFSETS = {}
_cursor = 0
for _name, _size, _dtype in PLANE_LAYOUT:
    _PLANE_OFFSETS[_name] = _cursor
    _cursor += _size
assert _cursor == RAW_RECORD_BYTES

_HEADER_STRUCT = struct.Struct("<8sIIIIIQQ16s32s32s")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class ExpertSourceError(RuntimeError):
    """Raised when a tier cannot be trusted or read exactly."""


def _z_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path).resolve()
    drive, _ = os.path.splitdrive(str(resolved))
    # ``Path.drive`` is reliable on Windows; splitdrive also keeps tests clear
    # on environments where pathlib's Windows flavour is not selected.
    drive = (resolved.drive or drive).upper()
    if drive != "Z:":
        raise ExpertSourceError(f"file-backed experts must reside on Z:, got {resolved}")
    return resolved


def _dtype(name: str) -> torch.dtype:
    return {
        "uint8": torch.uint8,
        "float16": torch.float16,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }[name]


def _plane_shape(name: str) -> tuple[int, ...]:
    return {
        "gate_up_packed": (1280, 1280),
        "gate_up_scale": (1280, 160),
        "gate_up_global": (1280,),
        "down_packed": (2560, 320),
        "down_scale": (2560, 40),
        "down_global": (2560,),
    }[name]


class FileExpertSource:
    """Read fixed NVFP4 expert records from one layer sidecar.

    ``read_record`` returns independent CPU tensors, while ``read_into`` copies
    directly into the destination slot planes.  Calls are synchronous and each
    call owns at most one bounded record buffer.  ``read_records`` exposes an
    explicit queue-depth guard for future asynchronous readers; this reference
    implementation is intentionally serial (depth one) and therefore graph
    capture incompatible.
    """

    bank_schema = tuple(name for name, _, _ in PLANE_LAYOUT)
    record_bytes = RECORD_BYTES
    raw_record_bytes = RAW_RECORD_BYTES
    num_experts = NUM_EXPERTS
    max_queue_depth = 16
    plane_specs = {
        name: (_plane_shape(name), _dtype(dtype_name))
        for name, _size, dtype_name in PLANE_LAYOUT
    }

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        expected_sha256: str | None = None,
        expected_source_fingerprint: str | bytes | None = None,
        expected_layer_id: int | None = None,
        num_experts: int = NUM_EXPERTS,
        max_queue_depth: int = 1,
        verify_hash: bool = True,
    ) -> None:
        self.path = _z_path(path)
        if not self.path.is_file():
            raise ExpertSourceError(f"missing expert tier: {self.path}")
        if not 1 <= int(max_queue_depth) <= 16:
            raise ValueError("max_queue_depth must be in [1, 16]")
        if not 1 <= int(num_experts) <= NUM_EXPERTS:
            raise ValueError(f"num_experts must be in [1, {NUM_EXPERTS}]")
        self.num_experts = int(num_experts)
        self.requested_queue_depth = int(max_queue_depth)
        self.staging_record_bytes = RECORD_BYTES
        self.max_staging_records = self.requested_queue_depth
        self._lock = threading.Lock()
        self._closed = False
        self._fd = os.open(str(self.path), os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            self._validate_file(expected_source_fingerprint, expected_layer_id)
            self.sha256 = self._hash_file() if verify_hash else None
            if expected_sha256 is not None:
                expected_sha256 = expected_sha256.lower()
                if self.sha256 is None:
                    self.sha256 = self._hash_file()
                if self.sha256 != expected_sha256:
                    raise ExpertSourceError(
                        f"expert tier hash mismatch: expected {expected_sha256}, got {self.sha256}"
                    )
        except Exception:
            os.close(self._fd)
            raise
        self.read_count = 0
        self.bytes_read = 0
        self.max_inflight = 0
        self._inflight = 0

    @classmethod
    def create_synthetic(
        cls,
        path: str | os.PathLike[str],
        *,
        num_experts: int = NUM_EXPERTS,
        records: Iterable[bytes] | None = None,
        source_fingerprint: bytes | None = None,
        layer_id: int = 0,
    ) -> str:
        """Create a tiny deterministic sidecar for tests (never model data)."""
        path = _z_path(path)
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        source_fingerprint = source_fingerprint or hashlib.sha256(b"synthetic").digest()
        source_fingerprint = bytes(source_fingerprint[:32]).ljust(32, b"\0")
        rows = iter(records) if records is not None else None
        payload_hash = hashlib.sha256()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            header = bytearray(HEADER_BYTES)
            _HEADER_STRUCT.pack_into(
                header,
                0,
                MAGIC,
                VERSION,
                HEADER_BYTES,
                num_experts,
                len(PLANE_LAYOUT),
                int(layer_id),
                RAW_RECORD_BYTES,
                RECORD_BYTES,
                b"nvfp4-qwen4-v1\0\0",  # 16-byte layout tag
                source_fingerprint,
                b"\0" * 32,
            )
            fh.write(header)
            for expert_id in range(num_experts):
                raw = next(rows) if rows is not None else bytes([expert_id & 0xFF]) * RAW_RECORD_BYTES
                if len(raw) != RAW_RECORD_BYTES:
                    raise ValueError("synthetic record must contain exactly RAW_RECORD_BYTES")
                record = raw + bytes(RECORD_BYTES - RAW_RECORD_BYTES)
                payload_hash.update(record)
                fh.write(record)
        digest = payload_hash.digest()
        with path.open("r+b") as fh:
            fh.seek(0)
            header = bytearray(fh.read(HEADER_BYTES))
            header[_HEADER_STRUCT.size - 32 : _HEADER_STRUCT.size] = digest
            fh.seek(0)
            fh.write(header)
        # Keep synthetic native-geometry fixtures bounded: the real 512-record
        # shape is roughly 1.42 GB and must never be read into one Python bytes
        # object merely to produce its verification hash.
        return _sha256_path(path)

    def _validate_file(
        self, expected_source_fingerprint: str | bytes | None, expected_layer_id: int | None
    ) -> None:
        size = self.path.stat().st_size
        expected_size = HEADER_BYTES + self.num_experts * self.record_bytes
        header = self._read_exact(HEADER_BYTES, 0)
        if len(header) != HEADER_BYTES:
            raise ExpertSourceError("truncated expert tier header")
        try:
            magic, version, hbytes, experts, planes, layer_id, raw_bytes, rec_bytes, tag, fingerprint, payload_hash = _HEADER_STRUCT.unpack_from(header)
        except struct.error as exc:
            raise ExpertSourceError("malformed expert tier header") from exc
        if magic != MAGIC or version != VERSION or hbytes != HEADER_BYTES:
            raise ExpertSourceError("unsupported expert tier magic/version/header")
        if experts != self.num_experts or planes != len(PLANE_LAYOUT):
            raise ExpertSourceError("expert tier geometry mismatch")
        self.layer_id = int(layer_id)
        if expected_layer_id is not None and self.layer_id != int(expected_layer_id):
            raise ExpertSourceError(
                f"expert tier layer mismatch: {self.layer_id} != {int(expected_layer_id)}"
            )
        if raw_bytes != RAW_RECORD_BYTES or rec_bytes != RECORD_BYTES or tag.rstrip(b"\0") != b"nvfp4-qwen4-v1":
            raise ExpertSourceError("expert tier layout mismatch")
        if size != expected_size:
            raise ExpertSourceError(f"expert tier length mismatch: {size} != {expected_size}")
        if expected_source_fingerprint is not None:
            expected = bytes.fromhex(expected_source_fingerprint) if isinstance(expected_source_fingerprint, str) else bytes(expected_source_fingerprint)
            if fingerprint != expected[:32].ljust(32, b"\0"):
                raise ExpertSourceError("expert tier source fingerprint mismatch")
        self.payload_sha256 = payload_hash.hex()
        if self.payload_sha256 != "00" * 32:
            h = hashlib.sha256()
            offset = HEADER_BYTES
            with self.path.open("rb") as fh:
                fh.seek(offset)
                while True:
                    block = fh.read(8 << 20)
                    if not block:
                        break
                    h.update(block)
            if h.digest() != payload_hash:
                raise ExpertSourceError("expert tier payload hash mismatch")

    def _read_exact(self, size: int, offset: int) -> bytes:
        if self._closed:
            raise ExpertSourceError("expert tier is closed")
        with self._lock:
            if hasattr(os, "pread"):
                data = os.pread(self._fd, size, offset)
            else:  # pragma: no cover - Windows Python fallback
                os.lseek(self._fd, offset, os.SEEK_SET)
                data = os.read(self._fd, size)
        if len(data) != size:
            raise ExpertSourceError(f"short expert tier read at offset {offset}: {len(data)} != {size}")
        return data

    def _hash_file(self) -> str:
        return _sha256_path(self.path)

    def _record_bytes(self, expert_id: int) -> bytes:
        if self._closed:
            raise ExpertSourceError("expert tier is closed")
        expert_id = int(expert_id)
        if not 0 <= expert_id < self.num_experts:
            raise IndexError(f"expert_id {expert_id} outside [0, {self.num_experts})")
        self._inflight += 1
        self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            offset = HEADER_BYTES + expert_id * self.record_bytes
            if offset % ALIGNMENT != 0 or self.record_bytes % ALIGNMENT != 0:
                raise ExpertSourceError("expert tier record is not aligned to 4096 bytes")
            data = self._read_exact(self.record_bytes, offset)
            self.read_count += 1
            self.bytes_read += len(data)
            return data
        finally:
            self._inflight -= 1

    def read_record(self, expert_id: int) -> dict[str, torch.Tensor]:
        raw = self._record_bytes(expert_id)
        out: dict[str, torch.Tensor] = {}
        for name, size, dtype_name in PLANE_LAYOUT:
            offset = _PLANE_OFFSETS[name]
            dtype = _dtype(dtype_name)
            shape = _plane_shape(name)
            # bytearray supplies a writable, record-local staging buffer and
            # avoids exposing Python's immutable bytes through a writable tensor.
            staging = bytearray(raw[offset : offset + size])
            out[name] = torch.frombuffer(staging, dtype=dtype).reshape(shape)
        return out

    def read_records(self, expert_ids: Iterable[int], *, max_concurrency: int = 1) -> list[dict[str, torch.Tensor]]:
        if not 1 <= int(max_concurrency) <= self.requested_queue_depth:
            raise ValueError(f"max_concurrency must be in [1, {self.requested_queue_depth}]")
        # Serial is deliberate: it keeps staging bounded and is graph-safe only
        # outside CUDA capture.  A future async implementation may use up to 16.
        return [self.read_record(eid) for eid in expert_ids]

    def read_into(self, expert_id: int, destinations: dict[str, torch.Tensor], slot: int) -> int:
        """Read one record and copy its six planes into a GPU/CPU cache slot."""
        rows = self.read_record(expert_id)
        for name in self.bank_schema:
            dst = destinations[name]
            if dst.ndim < 1 or not 0 <= slot < dst.shape[0]:
                raise ValueError(f"destination slot {slot} invalid for {name}")
            if tuple(dst.shape[1:]) != tuple(rows[name].shape):
                raise ValueError(f"destination shape mismatch for {name}")
            dst[slot].copy_(rows[name], non_blocking=False)
        return self.record_bytes

    def close(self) -> None:
        if not self._closed:
            os.close(self._fd)
            self._closed = True

    def __enter__(self) -> "FileExpertSource":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


__all__ = ["FileExpertSource", "ExpertSourceError", "PLANE_LAYOUT", "HEADER_BYTES", "RECORD_BYTES", "RAW_RECORD_BYTES"]
