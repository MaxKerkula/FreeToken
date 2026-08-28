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
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch


MAGIC = b"FTEXPERT1"
# ``FTEXNV4`` was the private Stage 6 fixture format.  Keep the reader able to
# reopen those fixtures while making every new artifact unambiguously FTEXPERT1.
LEGACY_MAGIC = b"FTEXNV4\0"
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

_HEADER_STRUCT = struct.Struct("<9sIIIIIQQ16s32s32s32s")
_LEGACY_HEADER_STRUCT = struct.Struct("<8sIIIIIQQ16s32s32s")

# A fixed descriptor in the otherwise-reserved header makes reduced synthetic
# geometry self-describing.  Native artifacts retain the exact six-plane
# ModelOpt layout; tests can use tiny tensors without teaching the reader a
# second out-of-band schema.  The descriptor is deliberately binary and fixed
# width so two writes of the same inputs are byte-for-byte identical.
_GEOMETRY_MAGIC = b"GEO1"
_GEOMETRY_ENTRY = struct.Struct("<BBH4Q")
_GEOMETRY_BYTES = 4 + len(PLANE_LAYOUT) * _GEOMETRY_ENTRY.size
_PAYLOAD_HASH_OFFSET = _HEADER_STRUCT.size - 64
_WHOLE_HASH_OFFSET = _HEADER_STRUCT.size - 32

_DTYPE_CODES: dict[torch.dtype, int] = {
    torch.uint8: 1,
    torch.float8_e4m3fn: 2,
    torch.float16: 3,
    torch.float32: 4,
    torch.bfloat16: 5,
}
_CODE_DTYPES = {value: key for key, value in _DTYPE_CODES.items()}


def _dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.uint8: "uint8",
        torch.float8_e4m3fn: "float8_e4m3fn",
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
    }[dtype]


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


def _align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _normalise_fingerprint(value: str | bytes | bytearray | None) -> bytes:
    """Return the fixed 32-byte source fingerprint stored in the header.

    Fingerprints are normally SHA-256 bytes or their 64-character hexadecimal
    spelling.  Short byte strings are accepted for deterministic synthetic
    fixtures and zero padded; this mirrors the Stage 6 fixture contract while
    still rejecting an accidentally over-wide digest.
    """

    if value is None:
        return b"\0" * 32
    if isinstance(value, str):
        try:
            value = bytes.fromhex(value)
        except ValueError:
            # Human-readable fixture labels are useful in bounded tests; keep
            # them deterministic while documenting that production callers
            # should pass SHA-256 bytes/hex.
            value = value.encode("utf-8")
    raw = bytes(value)
    if len(raw) > 32:
        raise ValueError("source_fingerprint must be at most 32 bytes")
    return raw.ljust(32, b"\0")


def _normalise_geometry(
    geometry: Mapping[str, Any] | Sequence[tuple[str, Any, Any]] | None,
) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
    """Validate/normalise a six-plane geometry declaration.

    ``geometry`` may map plane names to ``(shape, dtype)`` pairs, or be a
    sequence of ``(name, shape, dtype)`` entries.  Names must appear exactly in
    :data:`PLANE_LAYOUT` order; accepting a mapping is convenient for callers,
    but serialization remains ordered and deterministic.
    """

    if geometry is None:
        return tuple((name, _plane_shape(name), _dtype(dtype_name)) for name, _size, dtype_name in PLANE_LAYOUT)
    if isinstance(geometry, Mapping):
        unknown = set(geometry) - set(name for name, _size, _dtype_name in PLANE_LAYOUT)
        missing = set(name for name, _size, _dtype_name in PLANE_LAYOUT) - set(geometry)
        if unknown or missing:
            raise ValueError(f"geometry must contain exactly six planes; missing={sorted(missing)}, unknown={sorted(unknown)}")
        entries = []
        for name, _size, _dtype_name in PLANE_LAYOUT:
            value = geometry[name]
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise TypeError(f"geometry[{name!r}] must be (shape, dtype)")
            shape, dtype = value
            entries.append((name, tuple(int(dim) for dim in shape), dtype))
    else:
        entries = []
        for item in geometry:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise TypeError("geometry entries must be (name, shape, dtype)")
            name, shape, dtype = item
            entries.append((str(name), tuple(int(dim) for dim in shape), dtype))
    expected_names = tuple(name for name, _size, _dtype_name in PLANE_LAYOUT)
    actual_names = tuple(name for name, _shape, _dtype in entries)
    if actual_names != expected_names:
        raise ValueError(f"geometry plane order must be {expected_names}, got {actual_names}")
    normalised = []
    for name, shape, dtype in entries:
        if not shape or any(dim <= 0 for dim in shape) or len(shape) > 4:
            raise ValueError(f"{name} shape must have 1-4 positive dimensions")
        if not isinstance(dtype, torch.dtype) or dtype not in _DTYPE_CODES:
            raise TypeError(f"unsupported dtype for {name}: {dtype!r}")
        normalised.append((name, shape, dtype))
    return tuple(normalised)


def _geometry_descriptor(
    specs: tuple[tuple[str, tuple[int, ...], torch.dtype], ...],
) -> bytes:
    descriptor = bytearray(_GEOMETRY_BYTES)
    descriptor[:4] = _GEOMETRY_MAGIC
    cursor = 4
    for _name, shape, dtype in specs:
        _GEOMETRY_ENTRY.pack_into(
            descriptor,
            cursor,
            _DTYPE_CODES[dtype],
            len(shape),
            0,
            *(tuple(shape) + (0,) * (4 - len(shape))),
        )
        cursor += _GEOMETRY_ENTRY.size
    return bytes(descriptor)


def _parse_geometry_descriptor(header: bytes) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...] | None:
    if len(header) < _GEOMETRY_BYTES:
        return None
    offset = _HEADER_STRUCT.size
    if header[offset : offset + 4] != _GEOMETRY_MAGIC:
        return None
    cursor = offset + 4
    specs = []
    try:
        for name, _size, _dtype_name in PLANE_LAYOUT:
            code, rank, _reserved, d0, d1, d2, d3 = _GEOMETRY_ENTRY.unpack_from(header, cursor)
            dtype = _CODE_DTYPES[code]
            if not 1 <= rank <= 4:
                return None
            dims = (d0, d1, d2, d3)[:rank]
            if any(dim <= 0 for dim in dims):
                return None
            specs.append((name, dims, dtype))
            cursor += _GEOMETRY_ENTRY.size
    except (KeyError, struct.error):
        return None
    return tuple(specs)


def _canonical_whole_sha256(path: Path) -> str:
    """Hash a finalized sidecar with the stored whole-hash field zeroed."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        offset = 0
        while chunk := handle.read(8 << 20):
            if offset <= _WHOLE_HASH_OFFSET < offset + len(chunk):
                begin = _WHOLE_HASH_OFFSET - offset
                chunk = chunk[:begin] + b"\0" * 32 + chunk[begin + 32 :]
            digest.update(chunk)
            offset += len(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(value: Any, *, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> bytes:
    """Validate one plane and return its native little-endian bytes."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        expected = int(torch.empty(shape, dtype=dtype).numel()) * torch.empty((), dtype=dtype).element_size()
        if len(raw) != expected:
            raise ValueError(f"{name} bytes length {len(raw)} != expected {expected}")
        return raw
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except Exception as exc:
        raise TypeError(f"{name} must be a torch tensor or bytes-like value") from exc
    # ModelOpt emits a scalar ``weight_scale_2`` for each source projection.
    # Adapters may pass that scalar directly for a global plane; expand only in
    # this explicit case and require the declared FP16 dtype afterward.
    if name.endswith("_global") and tensor.numel() == 1 and tuple(tensor.shape) != shape:
        if not tensor.dtype.is_floating_point:
            raise TypeError(f"{name} scalar expansion requires a floating source")
        tensor = tensor.to(dtype=dtype).expand(shape)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} != expected {shape}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} dtype {tensor.dtype} != expected {dtype}")
    tensor = tensor.detach().to(device="cpu").contiguous()
    return tensor.view(torch.uint8).numpy().tobytes()


def _record_from_planes(
    record: Mapping[str, Any],
    specs: tuple[tuple[str, tuple[int, ...], torch.dtype], ...],
) -> bytes:
    expected = tuple(name for name, _shape, _dtype in specs)
    keys = tuple(key for key in record if key not in {"expert_id", "id"})
    if set(keys) != set(expected):
        missing = sorted(set(expected) - set(keys))
        unknown = sorted(set(keys) - set(expected))
        raise ValueError(f"expert record planes mismatch; missing={missing}, unknown={unknown}")
    return b"".join(
        _tensor_bytes(record[name], name=name, shape=shape, dtype=dtype)
        for name, shape, dtype in specs
    )


_SOURCE_TENSOR_NAMES = tuple(
    f"{projection}.{suffix}"
    for projection in ("gate_proj", "up_proj", "down_proj")
    for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")
)


def _source_scalar(value: Any, *, name: str) -> torch.Tensor:
    """Validate one ModelOpt source scalar (F32, rank-zero or one element)."""

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} dtype {tensor.dtype} != expected torch.float32")
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be a scalar, got shape {tuple(tensor.shape)}")
    return tensor.detach().reshape(())


def _as_named_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    try:
        items = list(record)
    except TypeError as exc:
        raise TypeError("expert tensor record must be a mapping or (name, tensor) iterable") from exc
    mapped: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("expert tensor entries must be (name, tensor) pairs")
        name, value = item
        if name in mapped:
            raise ValueError(f"duplicate expert tensor name {name!r}")
        mapped[str(name)] = value
    return mapped


def adapt_expert_tensor_record(record: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Adapt one ModelOpt expert's twelve source tensors to six native planes.

    ModelOpt stores ``gate_proj``, ``up_proj`` and ``down_proj`` independently,
    each with ``weight``, ``weight_scale``, ``weight_scale_2`` and
    ``input_scale``.  The native runtime sidecar fuses gate/up along the output
    axis, ignores the activation ``input_scale`` (W4A16), and expands each F32
    ``weight_scale_2`` scalar to the FP16 per-output-row global plane.

    A six-plane mapping is returned unchanged (but copied), allowing callers
    that have already performed the adaptation to use the same writer API.
    """

    # ``_iter_record_items`` permits an explicit ID alongside the planes; the
    # ID is routing metadata, not one of the twelve source tensors.
    named_record = _as_named_mapping(record)
    clean_record = {key: value for key, value in named_record.items() if key not in {"expert_id", "id"}}
    keys = tuple(clean_record)
    native_names = tuple(name for name, _shape, _dtype in _normalise_geometry(None))
    if set(keys) == set(native_names):
        return {name: clean_record[name] for name in native_names}
    if set(keys) != set(_SOURCE_TENSOR_NAMES):
        missing = sorted(set(_SOURCE_TENSOR_NAMES) - set(keys))
        unknown = sorted(set(keys) - set(_SOURCE_TENSOR_NAMES))
        raise ValueError(f"expert source tensors mismatch; missing={missing}, unknown={unknown}")
    # Validate all twelve source names, including both metadata scalar kinds.
    for projection in ("gate_proj", "up_proj", "down_proj"):
        _source_scalar(clean_record[f"{projection}.input_scale"], name=f"{projection}.input_scale")
        _source_scalar(clean_record[f"{projection}.weight_scale_2"], name=f"{projection}.weight_scale_2")
    gate_weight = clean_record["gate_proj.weight"]
    up_weight = clean_record["up_proj.weight"]
    gate_scale = clean_record["gate_proj.weight_scale"]
    up_scale = clean_record["up_proj.weight_scale"]
    down_weight = clean_record["down_proj.weight"]
    down_scale = clean_record["down_proj.weight_scale"]
    # Let the regular six-plane validator provide exact dtype/shape diagnostics
    # after these bounded concatenations.  Concatenation happens per expert and
    # therefore never materialises a layer or model-sized tensor.
    gate_weight_t = gate_weight if isinstance(gate_weight, torch.Tensor) else torch.as_tensor(gate_weight)
    up_weight_t = up_weight if isinstance(up_weight, torch.Tensor) else torch.as_tensor(up_weight)
    gate_scale_t = gate_scale if isinstance(gate_scale, torch.Tensor) else torch.as_tensor(gate_scale)
    up_scale_t = up_scale if isinstance(up_scale, torch.Tensor) else torch.as_tensor(up_scale)
    down_weight_t = down_weight if isinstance(down_weight, torch.Tensor) else torch.as_tensor(down_weight)
    down_scale_t = down_scale if isinstance(down_scale, torch.Tensor) else torch.as_tensor(down_scale)
    if gate_weight_t.ndim != up_weight_t.ndim or gate_weight_t.ndim < 1:
        raise ValueError("gate_proj.weight and up_proj.weight must have matching rank")
    if gate_scale_t.ndim != up_scale_t.ndim or gate_scale_t.ndim < 1:
        raise ValueError("gate_proj.weight_scale and up_proj.weight_scale must have matching rank")
    gate_rows = int(gate_weight_t.shape[0])
    up_rows = int(up_weight_t.shape[0])
    gate_global = _source_scalar(clean_record["gate_proj.weight_scale_2"], name="gate_proj.weight_scale_2").to(torch.float16).expand(gate_rows)
    up_global = _source_scalar(clean_record["up_proj.weight_scale_2"], name="up_proj.weight_scale_2").to(torch.float16).expand(up_rows)
    down_global = _source_scalar(clean_record["down_proj.weight_scale_2"], name="down_proj.weight_scale_2").to(torch.float16).expand(int(down_weight_t.shape[0]))
    return {
        "gate_up_packed": torch.cat((gate_weight_t, up_weight_t), dim=0),
        "gate_up_scale": torch.cat((gate_scale_t, up_scale_t), dim=0),
        "gate_up_global": torch.cat((gate_global, up_global), dim=0),
        "down_packed": down_weight_t,
        "down_scale": down_scale_t,
        "down_global": down_global,
    }


def _iter_record_items(
    records_or_planes: Any,
    *,
    num_experts: int,
) -> Iterable[tuple[int, Any]]:
    """Yield ``(expert_id, record)`` without materialising the expert bank."""

    expected_names = {name for name, _size, _dtype_name in PLANE_LAYOUT}
    if isinstance(records_or_planes, Mapping):
        keys = set(records_or_planes)
        source_names = set(_SOURCE_TENSOR_NAMES)
        if keys & (expected_names | source_names):
            if keys.issubset(source_names):
                bank_names = tuple(_SOURCE_TENSOR_NAMES)
            elif keys == expected_names:
                bank_names = tuple(name for name, _s, _d in PLANE_LAYOUT)
            else:
                raise ValueError("plane-bank input must contain exactly the six native or twelve source tensor names")
            # Stacked tensors/sequences are indexed lazily one expert at a time.
            banks = records_or_planes
            for eid in range(num_experts):
                row = {}
                for name in bank_names:
                    bank = banks[name]
                    if isinstance(bank, Mapping):
                        bank_ids = set(bank)
                        expected_ids = set(range(num_experts))
                        if bank_ids != expected_ids:
                            raise ValueError(
                                f"plane bank {name} IDs mismatch; missing={sorted(expected_ids - bank_ids)}, "
                                f"unknown={sorted(bank_ids - expected_ids)}"
                            )
                        if eid not in bank:
                            raise ValueError(f"missing expert id {eid} in plane bank {name}")
                        row[name] = bank[eid]
                    else:
                        try:
                            bank_len = len(bank)
                        except TypeError:
                            bank_len = None
                        if bank_len is not None and bank_len != num_experts:
                            raise ValueError(f"plane bank {name} length {bank_len} != num_experts {num_experts}")
                        try:
                            row[name] = bank[eid]
                        except (IndexError, KeyError, TypeError) as exc:
                            raise ValueError(f"missing expert id {eid} in plane bank {name}") from exc
                yield eid, row
            return
        for key in sorted(records_or_planes):
            if not isinstance(key, int):
                raise TypeError("expert mapping keys must be integer expert IDs")
            yield key, records_or_planes[key]
        return

    for index, item in enumerate(records_or_planes):
        expert_id = index
        record = item
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
            expert_id, record = item
        elif isinstance(item, Mapping):
            explicit = item.get("expert_id", item.get("id", None))
            if explicit is not None:
                expert_id = explicit
        yield int(expert_id), record


def write_expert_sidecar(
    path: str | os.PathLike[str],
    records_or_planes: Any,
    *,
    layer_id: int,
    source_fingerprint: str | bytes,
    num_experts: int = NUM_EXPERTS,
    geometry: Mapping[str, Any] | Sequence[tuple[str, Any, Any]] | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Stream a deterministic FTEXPERT1 routed-expert sidecar to ``path``.

    ``records_or_planes`` can be an iterable of six-plane mappings (or the
    twelve ModelOpt source-tensor mappings, optionally ``(expert_id, mapping)``),
    an ``{expert_id: mapping}`` mapping, or a mapping of six/twelve plane names
    to stacked tensors/sequences.  Exactly one record for every ID
    ``0..num_experts-1`` is required.  The destination is written to a
    ``.partial`` sibling and atomically replaced only after all validation,
    payload hashing, and header hashes complete.
    """

    destination = _z_path(path)
    if int(num_experts) < 1 or int(num_experts) > NUM_EXPERTS:
        raise ValueError(f"num_experts must be in [1, {NUM_EXPERTS}]")
    num_experts = int(num_experts)
    if not 0 <= int(layer_id) <= 0xFFFFFFFF:
        raise ValueError("layer_id must fit an unsigned 32-bit field")
    layer_id = int(layer_id)
    fingerprint = _normalise_fingerprint(source_fingerprint)
    specs = _normalise_geometry(geometry)
    sizes = tuple(int(torch.empty(shape, dtype=dtype).numel()) * torch.empty((), dtype=dtype).element_size() for _name, shape, dtype in specs)
    raw_record_bytes = sum(sizes)
    record_bytes = _align_up(raw_record_bytes)
    descriptor = _geometry_descriptor(specs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(destination) + ".partial")
    if partial.exists():
        partial.unlink()
    payload_hash = hashlib.sha256()
    seen: set[int] = set()
    try:
        with partial.open("wb") as handle:
            header = bytearray(HEADER_BYTES)
            _HEADER_STRUCT.pack_into(
                header,
                0,
                MAGIC,
                VERSION,
                HEADER_BYTES,
                num_experts,
                len(specs),
                layer_id,
                raw_record_bytes,
                record_bytes,
                b"nvfp4-qwen4-v1\0\0",
                fingerprint,
                b"\0" * 32,
                b"\0" * 32,
            )
            header[_HEADER_STRUCT.size : _HEADER_STRUCT.size + len(descriptor)] = descriptor
            handle.write(header)
            for expert_id, record in _iter_record_items(records_or_planes, num_experts=num_experts):
                if not 0 <= expert_id < num_experts:
                    raise ValueError(f"expert_id {expert_id} outside [0, {num_experts})")
                if expert_id in seen:
                    raise ValueError(f"duplicate expert_id {expert_id}")
                seen.add(expert_id)
                if isinstance(record, (bytes, bytearray, memoryview)):
                    if len(record) != raw_record_bytes:
                        raise ValueError(f"expert {expert_id} raw bytes length {len(record)} != {raw_record_bytes}")
                    raw = bytes(record)
                elif isinstance(record, Mapping) or isinstance(record, Iterable):
                    raw = _record_from_planes(adapt_expert_tensor_record(record), specs)
                else:
                    raise TypeError(f"expert {expert_id} must be a plane mapping or raw bytes")
                if len(raw) != raw_record_bytes:
                    raise ValueError(f"expert {expert_id} serialized length {len(raw)} != {raw_record_bytes}")
                padded = raw + b"\0" * (record_bytes - raw_record_bytes)
                payload_hash.update(padded)
                handle.write(padded)
            missing = sorted(set(range(num_experts)) - seen)
            if missing:
                raise ValueError(f"missing expert IDs: {missing}")
            handle.flush()
            os.fsync(handle.fileno())
        digest = payload_hash.digest()
        # Patch payload hash first, then derive a canonical whole hash over the
        # finalized header with only the whole-hash field zeroed.
        with partial.open("r+b") as handle:
            handle.seek(_PAYLOAD_HASH_OFFSET)
            handle.write(digest)
            handle.flush()
            os.fsync(handle.fileno())
        canonical = _canonical_whole_sha256(partial)
        with partial.open("r+b") as handle:
            handle.seek(_WHOLE_HASH_OFFSET)
            handle.write(bytes.fromhex(canonical))
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and destination.exists():
            raise FileExistsError(destination)
        os.replace(partial, destination)
    except Exception:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise
    whole = _sha256_path(destination)
    return {
        "path": str(destination),
        "format": "FTEXPERT1",
        "version": VERSION,
        "layer_id": layer_id,
        "num_experts": num_experts,
        "planes": tuple(name for name, _shape, _dtype in specs),
        "raw_record_bytes": raw_record_bytes,
        "record_bytes": record_bytes,
        "source_fingerprint": fingerprint.hex(),
        "payload_sha256": digest.hex(),
        "canonical_sha256": canonical,
        "whole_sha256": whole,
        "sha256": whole,
        "sample_ids": (0, num_experts - 1),
    }


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
            self.staging_record_bytes = self.record_bytes
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
        rows = iter(records) if records is not None else None

        def _rows() -> Iterable[bytes]:
            for expert_id in range(num_experts):
                raw = next(rows) if rows is not None else bytes([expert_id & 0xFF]) * RAW_RECORD_BYTES
                if len(raw) != RAW_RECORD_BYTES:
                    raise ValueError("synthetic record must contain exactly RAW_RECORD_BYTES")
                yield raw

        result = write_expert_sidecar(
            path,
            _rows(),
            layer_id=layer_id,
            source_fingerprint=source_fingerprint,
            num_experts=num_experts,
        )
        # Keep synthetic native-geometry fixtures bounded: the real 512-record
        # shape is roughly 1.42 GB and must never be read into one Python bytes
        # object merely to produce its verification hash.
        return str(result["sha256"])

    def _validate_file(
        self, expected_source_fingerprint: str | bytes | None, expected_layer_id: int | None
    ) -> None:
        size = self.path.stat().st_size
        header = self._read_exact(HEADER_BYTES, 0)
        if len(header) != HEADER_BYTES:
            raise ExpertSourceError("truncated expert tier header")
        try:
            if header[: len(MAGIC)] == MAGIC:
                (
                    magic,
                    version,
                    hbytes,
                    experts,
                    planes,
                    layer_id,
                    raw_bytes,
                    rec_bytes,
                    tag,
                    fingerprint,
                    payload_hash,
                    whole_hash,
                ) = _HEADER_STRUCT.unpack_from(header)
                whole_offset = _WHOLE_HASH_OFFSET
                specs = _parse_geometry_descriptor(header)
                if specs is None:
                    specs = tuple((name, _plane_shape(name), _dtype(dtype_name)) for name, _size, dtype_name in PLANE_LAYOUT)
            elif header[: len(LEGACY_MAGIC)] == LEGACY_MAGIC:
                (
                    magic,
                    version,
                    hbytes,
                    experts,
                    planes,
                    layer_id,
                    raw_bytes,
                    rec_bytes,
                    tag,
                    fingerprint,
                    payload_hash,
                ) = _LEGACY_HEADER_STRUCT.unpack_from(header)
                whole_hash = b"\0" * 32
                whole_offset = None
                specs = tuple((name, _plane_shape(name), _dtype(dtype_name)) for name, _size, dtype_name in PLANE_LAYOUT)
            else:
                raise ExpertSourceError("unsupported expert tier magic/version/header")
        except struct.error as exc:
            raise ExpertSourceError("malformed expert tier header") from exc
        if magic not in (MAGIC, LEGACY_MAGIC) or version != VERSION or hbytes != HEADER_BYTES:
            raise ExpertSourceError("unsupported expert tier magic/version/header")
        if experts != self.num_experts or planes != len(PLANE_LAYOUT):
            raise ExpertSourceError("expert tier geometry mismatch")
        if tag.rstrip(b"\0") != b"nvfp4-qwen4-v1":
            raise ExpertSourceError("expert tier layout mismatch")
        self.layer_id = int(layer_id)
        if expected_layer_id is not None and self.layer_id != int(expected_layer_id):
            raise ExpertSourceError(f"expert tier layer mismatch: {self.layer_id} != {int(expected_layer_id)}")
        self.plane_layout = tuple(
            (name, int(torch.empty(shape, dtype=dtype).numel()) * torch.empty((), dtype=dtype).element_size(), _dtype_name(dtype))
            for name, shape, dtype in specs
        )
        self.plane_specs = {name: (shape, dtype) for name, shape, dtype in specs}
        self._plane_offsets = {}
        cursor = 0
        for name, shape, dtype in specs:
            self._plane_offsets[name] = cursor
            cursor += int(torch.empty(shape, dtype=dtype).numel()) * torch.empty((), dtype=dtype).element_size()
        if cursor != int(raw_bytes) or int(rec_bytes) != _align_up(cursor):
            raise ExpertSourceError("expert tier layout mismatch")
        self.raw_record_bytes = int(raw_bytes)
        self.record_bytes = int(rec_bytes)
        expected_size = HEADER_BYTES + self.num_experts * self.record_bytes
        if size != expected_size:
            raise ExpertSourceError(f"expert tier length mismatch: {size} != {expected_size}")
        if expected_source_fingerprint is not None:
            try:
                expected = _normalise_fingerprint(expected_source_fingerprint)
            except ValueError as exc:
                raise ExpertSourceError("invalid expected source fingerprint") from exc
            if fingerprint != expected:
                raise ExpertSourceError("expert tier source fingerprint mismatch")
        self.source_fingerprint = fingerprint.hex()
        self.payload_sha256 = payload_hash.hex()
        if magic == MAGIC and payload_hash == b"\0" * 32:
            raise ExpertSourceError("expert tier missing payload hash")
        if self.payload_sha256 != "00" * 32:
            h = hashlib.sha256()
            with self.path.open("rb") as fh:
                fh.seek(HEADER_BYTES)
                while block := fh.read(8 << 20):
                    h.update(block)
            if h.digest() != payload_hash:
                raise ExpertSourceError("expert tier payload hash mismatch")
        self.whole_sha256 = _sha256_path(self.path)
        self.canonical_sha256 = None
        if magic == MAGIC and whole_hash == b"\0" * 32:
            raise ExpertSourceError("expert tier missing whole hash")
        if whole_offset is not None and whole_hash != b"\0" * 32:
            self.canonical_sha256 = _canonical_whole_sha256(self.path)
            if self.canonical_sha256 != whole_hash.hex():
                raise ExpertSourceError("expert tier whole hash mismatch")

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
        for name, (shape, dtype) in self.plane_specs.items():
            size = int(torch.empty(shape, dtype=dtype).numel()) * torch.empty((), dtype=dtype).element_size()
            offset = self._plane_offsets[name]
            # Clone the record-local staging slice so returned tensors remain
            # independent after this bounded read buffer is released.
            staging = raw[offset : offset + size]
            out[name] = torch.frombuffer(bytearray(staging), dtype=dtype).clone().reshape(shape)
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


__all__ = [
    "FileExpertSource",
    "ExpertSourceError",
    "PLANE_LAYOUT",
    "HEADER_BYTES",
    "RECORD_BYTES",
    "RAW_RECORD_BYTES",
    "MAGIC",
    "write_expert_sidecar",
    "adapt_expert_tensor_record",
]
