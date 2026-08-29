"""Contract-complete, restartable Step 9B acquisition controller.

This module deliberately owns orchestration and transport only.  The accepted
Q3, FTEXPERT1 and FTW writers remain the byte-level authorities.  In particular,
the default mode is a manifest-only dry run: a body GET is impossible unless
both ``execute`` and ``allow_network_body`` are explicitly enabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Protocol


PINNED_REPOSITORY = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
PINNED_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
MAX_TRANSFER_BYTES = 135_252_480_565
MIN_DISK_FREE_BYTES = 309_257_827_893
MIN_DISK_RESERVE_BYTES = 68_719_476_736
COMMON_PEAK_ALLOWANCE_BYTES = 9_989_288_586
MIN_HOST_FREE_BYTES = 6_442_450_944
KNOWN_TARGET_BYTES = 95_353_758_720
Q3_BYTES = 22_400_107_520
ACTIVE_BYTES = 4_804_403_200
EXPERT_BYTES = 1_419_776_000
MAX_DOWNLOADS = 2
MAX_SAFETENSORS_HEADER_BYTES = 256 << 20
PRODUCTION_PLE_SOURCE_LAYER_ID = 1
PRODUCTION_PLE_SEGMENT_COUNT = 128
ACCEPTED_SOURCE_INVENTORY = "8572d200e31b344faff0fda f0dc72aa4726c1f062443d4109531b62ca63f66eb".replace(" ", "")
ACQUISITION_MANIFEST_V1 = "freetoken-step9-acquisition-v1"
ACQUISITION_MANIFEST_V2 = "freetoken-step9-acquisition-v2"
SOURCE_IDENTITY_VERSION = 2
SOURCE_RECEIPT_VERSION = 2


class ExecutorError(RuntimeError):
    """A stop-gate or validation failure; callers must preserve evidence."""


class BodyTransferDisabled(ExecutorError):
    """Raised before a network body request when explicit authorization is absent."""


class ResumeRejected(ExecutorError):
    """Raised when the server cannot prove an identity-safe range response."""


class AtomicJsonReplaceError(ExecutorError):
    """A bounded Windows atomic-publication retry exhausted its deadline."""


_PLE_SHARD_KEY = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.ple\.ple_embedding\."
    r"ngram_embedding\.shard_(?P<index>\d+)\.weight$"
)
_PLE_SHARD_PREFIX = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.ple\.ple_embedding\."
    r"ngram_embedding\.shard_"
)
_PLE_SCALE_KEY = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.ple\.ple_embedding\."
    r"ngram_embedding\.weight_scale$"
)


def _resolve_production_ple_source_layer(index_path: Path) -> int:
    """Validate and return the frozen production PLE source layer.

    The target model exposes one PLE table under source layer 1.  Its 128
    logical tensors are distinct from the ten physical Safetensors files.  Do
    not infer a different layer from whichever matching key happens to appear:
    require the exact frozen namespace and reject competing PLE shard sets.
    """
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        weight_map = document["weight_map"]
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ExecutorError(f"cannot read PLE source index: {index_path}") from exc
    if not isinstance(weight_map, Mapping):
        raise ExecutorError("source index weight_map must be an object")

    by_layer: dict[int, set[int]] = {}
    scale_layers: set[int] = set()
    for key, source_file in weight_map.items():
        if not isinstance(key, str):
            continue
        match = _PLE_SHARD_KEY.fullmatch(key)
        if match is None:
            malformed = _PLE_SHARD_PREFIX.match(key)
            if malformed is not None:
                raise ExecutorError(f"malformed PLE source tensor key: {key}")
            scale = _PLE_SCALE_KEY.fullmatch(key)
            if scale is not None:
                scale_layers.add(int(scale.group("layer")))
            continue
        if not isinstance(source_file, str) or not source_file:
            raise ExecutorError(f"invalid PLE source file mapping for {key}")
        layer = int(match.group("layer"))
        index = int(match.group("index"))
        if index in by_layer.setdefault(layer, set()):
            raise ExecutorError(f"duplicate PLE source tensor index {index} for layer {layer}")
        by_layer[layer].add(index)

    expected_indices = set(range(PRODUCTION_PLE_SEGMENT_COUNT))
    if set(by_layer) != {PRODUCTION_PLE_SOURCE_LAYER_ID}:
        raise ExecutorError(
            "production PLE source layer mismatch: "
            f"expected only layer {PRODUCTION_PLE_SOURCE_LAYER_ID}, found {sorted(by_layer)}"
        )
    observed = by_layer[PRODUCTION_PLE_SOURCE_LAYER_ID]
    if observed != expected_indices:
        missing = sorted(expected_indices - observed)
        unexpected = sorted(observed - expected_indices)
        raise ExecutorError(
            "production PLE logical segment mismatch: "
            f"missing={missing}; unexpected={unexpected}"
        )

    prefix = (
        f"model.language_model.layers.{PRODUCTION_PLE_SOURCE_LAYER_ID}."
        "ple.ple_embedding.ngram_embedding"
    )
    scale_key = prefix + ".weight_scale"
    if not scale_layers:
        raise ExecutorError(f"production PLE global scale is missing: {scale_key}")
    if scale_layers != {PRODUCTION_PLE_SOURCE_LAYER_ID}:
        raise ExecutorError(
            "production PLE scale layer mismatch: "
            f"expected only layer {PRODUCTION_PLE_SOURCE_LAYER_ID}, found {sorted(scale_layers)}"
        )
    if not isinstance(weight_map[scale_key], str) or not weight_map[scale_key]:
        raise ExecutorError(f"invalid PLE source file mapping for {scale_key}")
    return PRODUCTION_PLE_SOURCE_LAYER_ID


def _sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        file_digest = bytes.fromhex(_sha256(path))
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "little"))
        digest.update(file_digest)
        total += size
    return total, digest.hexdigest()


def _z_path(path: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve(strict=must_exist)
    drive = (resolved.drive or os.path.splitdrive(str(resolved))[0]).upper()
    # Tests may use a POSIX-mounted /z volume; production Windows requires Z:.
    if drive != "Z:" and not str(resolved).lower().startswith("/z/"):
        raise ExecutorError(f"{path} must resolve physically to Z:, got {resolved}")
    return resolved


_WINDOWS_REPLACE_TRANSIENT_ERRORS = frozenset({5, 32, 33})


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    replace_attempts: int = 8,
    replace_deadline_seconds: float = 2.0,
    replace_backoff_seconds: float = 0.025,
) -> None:
    """Durably publish JSON with bounded Windows share-lock recovery.

    The temporary file is closed and fsynced before ``os.replace``.  On
    Windows, a transient share/access lock can make replace fail with WinError
    5, 32, or 33.  Retry only those errors for a short bounded window; all
    other errors fail immediately and the temporary file is intentionally
    preserved as recovery evidence.  The existing canonical destination is
    never removed before a successful replace.
    """
    if replace_attempts < 1:
        raise ValueError("replace_attempts must be positive")
    if replace_deadline_seconds < 0 or replace_backoff_seconds < 0:
        raise ValueError("replace retry timing must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}-{threading.get_ident()}")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            os.replace(partial, path)
            return
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            transient = os.name == "nt" and winerror in _WINDOWS_REPLACE_TRANSIENT_ERRORS
            within_budget = attempt < replace_attempts and (time.monotonic() - started) < replace_deadline_seconds
            if not transient or not within_budget:
                if transient:
                    raise AtomicJsonReplaceError(
                        f"atomic JSON replace failed after {attempt} attempts; "
                        f"destination={path}; preserved_orphan={partial}; winerror={winerror}"
                    ) from exc
                raise
            delay = min(replace_backoff_seconds * (2 ** (attempt - 1)), max(0.0, replace_deadline_seconds - (time.monotonic() - started)))
            if delay:
                time.sleep(delay)


def _publish_component_receipt(receipt_path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    final = dict(value)
    final["completion"] = "COMPONENT_COMPLETE"
    precommit = receipt_path.with_suffix(receipt_path.suffix + ".precommit")
    _atomic_json(precommit, {**final, "completion": "VALIDATED_PRECOMMIT"})
    _atomic_json(receipt_path, final)
    return final


def _receipt_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return value.get("completion") == "COMPONENT_COMPLETE" and all(value.get(key) == item for key, item in expected.items())


def _clean_identity(value: Any) -> str | None:
    """Return a normalized hex identity, retaining non-hex v1 fixtures."""
    if value is None:
        return None
    text = str(value).strip().strip('"')
    return text.lower() or None


def _identity_values(*values: Any) -> tuple[str, ...]:
    """Deduplicate identity values while preserving their declared order."""
    result: list[str] = []
    for value in values:
        text = _clean_identity(value)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _validate_hex_identity(name: str, value: str | None, length: int) -> None:
    if value is None:
        return
    if len(value) != length or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ExecutorError(f"{name} must be a {length}-character hexadecimal digest")


@dataclass(frozen=True)
class SourceEntry:
    """Normalized source row from ``source_weight_shards`` or metadata."""

    filename: str
    byte_length: int
    source_class: str
    acquisition_order: int
    repository: str
    revision: str
    accepted_etag: str | None = None
    lfs_oid_sha256: str | None = None
    accepted_header_length: int | None = None
    accepted_header_sha256: str | None = None
    layer_id: int | None = None
    tensor_payload_bytes: int | None = None
    git_blob_id: str | None = None
    xet_file_hash: str | None = None
    allowed_body_etags: tuple[str, ...] = ()
    identity_version: int = 1

    def __post_init__(self) -> None:
        # v1 fixtures intentionally use short synthetic ETags.  Strict length
        # and source-kind checks are applied only to normalized v2 rows.
        if int(self.identity_version) >= SOURCE_IDENTITY_VERSION:
            _validate_hex_identity("git_blob_id", self.git_blob_id, 40)
            _validate_hex_identity("lfs_oid_sha256", self.lfs_oid_sha256, 64)
            _validate_hex_identity("xet_file_hash", self.xet_file_hash, 64)
            if not self.git_blob_id:
                raise ExecutorError(f"{self.filename}: v2 row requires git_blob_id provenance")
            if self.source_class.upper() != "METADATA" and (not self.lfs_oid_sha256 or not self.xet_file_hash):
                raise ExecutorError(f"{self.filename}: v2 weight row requires LFS OID and Xet hash")
            if self.lfs_oid_sha256 is not None and self.xet_file_hash is None:
                raise ExecutorError(f"{self.filename}: v2 LFS row requires xet_file_hash")
            if self.lfs_oid_sha256 is None and self.xet_file_hash is not None:
                raise ExecutorError(f"{self.filename}: Xet hash requires an LFS OID")
            if not self.allowed_body_etags:
                raise ExecutorError(f"{self.filename}: v2 row requires allowed_body_etags")
            for etag in self.allowed_body_etags:
                if not str(etag).strip():
                    raise ExecutorError(f"{self.filename}: body ETag cannot be empty")
            expected_body = (
                {self.git_blob_id.lower()}
                if self.lfs_oid_sha256 is None and self.git_blob_id
                else {self.lfs_oid_sha256.lower(), self.xet_file_hash.lower()}
            )
            if {str(etag).strip('"').lower() for etag in self.allowed_body_etags} != expected_body:
                raise ExecutorError(f"{self.filename}: allowed_body_etags must match its Git or LFS/Xet identities")

    @property
    def body_etags(self) -> tuple[str, ...]:
        """Canonical body ETag allow-list (v1 alias retained for callers)."""
        return tuple(self.allowed_body_etags)

    @property
    def metadata_etag(self) -> str | None:
        """The immutable metadata ETag, never the transport-body ETag."""
        return self.lfs_oid_sha256 or self.git_blob_id

    @property
    def semantic_kind(self) -> str:
        if self.lfs_oid_sha256 and self.xet_file_hash:
            return "LFS_XET"
        if self.lfs_oid_sha256:
            return "LFS"
        if self.git_blob_id:
            return "GIT"
        return "LEGACY"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, order: int | None = None, metadata: bool = False, schema_version: int = 1, legacy_migration: bool = False) -> "SourceEntry":
        filename = str(raw["filename"])
        if not filename or Path(filename).name != filename or filename in {".", ".."}:
            raise ExecutorError(f"unsafe manifest filename: {filename!r}")
        repository = str(raw.get("repository", PINNED_REPOSITORY))
        revision = str(raw.get("revision", PINNED_REVISION))
        if repository != PINNED_REPOSITORY or revision != PINNED_REVISION:
            raise ExecutorError(f"source identity mismatch for {filename}")
        raw_git = _clean_identity(raw.get("git_blob_id"))
        raw_lfs = _clean_identity(raw.get("lfs_oid_sha256"))
        raw_xet = _clean_identity(raw.get("xet_file_hash"))
        legacy_etag = raw.get("accepted_etag")
        # v1 used accepted_etag for Xet (and occasionally for Git).  It is
        # migrated into an explicit xet_file_hash/body allow-list but never
        # consulted by v2 execution paths.
        if raw_xet is None and raw_lfs and legacy_etag and "xet_file_hash" not in raw and (int(schema_version) < SOURCE_IDENTITY_VERSION or legacy_migration):
            # Migration of a v1 row: accepted_etag was the Xet hash.  Once a
            # v2 row carries an explicit xet_file_hash (including null), the
            # legacy field is intentionally ignored.
            raw_xet = _clean_identity(legacy_etag)
        body_values = raw.get("allowed_body_etags", raw.get("body_etags", raw.get("accepted_body_etags")))
        if body_values is None:
            if int(schema_version) >= SOURCE_IDENTITY_VERSION:
                body_values = _identity_values(raw_lfs, raw_xet) if raw_lfs else _identity_values(raw_git)
            else:
                body_values = _identity_values(legacy_etag, raw_lfs, raw_git)
        elif isinstance(body_values, str):
            body_values = (body_values,)
        body_etags = _identity_values(*tuple(body_values))
        return cls(
            filename=filename,
            byte_length=int(raw["byte_length"]),
            source_class=("METADATA" if metadata else str(raw["source_class"])),
            acquisition_order=int(raw.get("acquisition_order", order or 0)),
            repository=repository,
            revision=revision,
            accepted_etag=(raw.get("accepted_etag") or (raw.get("git_blob_id") if metadata else None)),
            lfs_oid_sha256=raw_lfs,
            accepted_header_length=raw.get("accepted_header_length"),
            accepted_header_sha256=raw.get("accepted_header_sha256"),
            layer_id=(None if raw.get("layer_id") is None else int(raw["layer_id"])),
            tensor_payload_bytes=(None if raw.get("tensor_payload_bytes") is None else int(raw["tensor_payload_bytes"])),
            git_blob_id=raw_git,
            xet_file_hash=raw_xet,
            allowed_body_etags=body_etags,
            identity_version=int(schema_version),
        )


@dataclass(frozen=True)
class AcquisitionManifest:
    repository: str
    revision: str
    entries: tuple[SourceEntry, ...]
    metadata: tuple[SourceEntry, ...]
    source_inventory_fingerprint: str
    expected_weight_bytes: int
    transfer_cap: int = MAX_TRANSFER_BYTES
    schema_version: int = 1

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, require_v2: bool = False) -> "AcquisitionManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        schema = str(raw.get("schema") or "")
        if schema not in {ACQUISITION_MANIFEST_V1, ACQUISITION_MANIFEST_V2}:
            raise ExecutorError("unsupported acquisition manifest schema")
        schema_version = 2 if schema == ACQUISITION_MANIFEST_V2 else 1
        if schema_version >= SOURCE_IDENTITY_VERSION and raw.get("identity_schema") != "source_identity_v2":
            raise ExecutorError("acquisition manifest v2 requires identity_schema=source_identity_v2")
        if require_v2 and schema_version < SOURCE_IDENTITY_VERSION:
            raise ExecutorError("canonical Step 9B execution requires acquisition manifest v2")
        repo, revision = str(raw.get("repository")), str(raw.get("revision"))
        if repo != PINNED_REPOSITORY or revision != PINNED_REVISION:
            raise ExecutorError("manifest source pin does not match the frozen revision")
        rows = tuple(SourceEntry.from_mapping(row, schema_version=schema_version) for row in raw.get("source_weight_shards", ()))
        metadata = tuple(SourceEntry.from_mapping(row, order=i + 1, metadata=True, schema_version=schema_version) for i, row in enumerate(raw.get("required_small_metadata", ())))
        if len(rows) != 206 or len(metadata) != 9:
            raise ExecutorError(f"manifest requires 206 weights and 9 metadata files, got {len(rows)} / {len(metadata)}")
        orders = [row.acquisition_order for row in rows]
        if orders != list(range(1, len(rows) + 1)):
            raise ExecutorError("weight acquisition order is not contiguous")
        expected = int(raw["reconciliation"]["expected_source_file_bytes"])
        if sum(row.byte_length for row in rows) != expected:
            raise ExecutorError("source weight byte reconciliation failed")
        class_contract = {
            "BF16": (4, 16_007_756_462),
            "PLE": (10, 51_200_267_901),
            "EXPERT": (192, 67_987_279_488),
        }
        for source_class, (count, byte_length) in class_contract.items():
            selected = tuple(row for row in rows if row.source_class.upper() == source_class)
            if len(selected) != count or sum(row.byte_length for row in selected) != byte_length:
                raise ExecutorError(f"{source_class} source inventory mismatch")
        for layer in range(48):
            selected = tuple(row for row in rows if row.source_class.upper() == "EXPERT" and row.layer_id == layer)
            if len(selected) != 4:
                raise ExecutorError(f"expert layer {layer} must have exactly four source files")
        if sum(row.byte_length for row in metadata) != 57_176_714:
            raise ExecutorError("metadata inventory byte reconciliation failed")
        inventory = str(raw.get("source_inventory_sha256") or ACCEPTED_SOURCE_INVENTORY).lower()
        if inventory != ACCEPTED_SOURCE_INVENTORY:
            raise ExecutorError("source tensor inventory fingerprint mismatch")
        return cls(repo, revision, rows, metadata, inventory, expected, MAX_TRANSFER_BYTES, schema_version)

    @property
    def all_entries(self) -> tuple[SourceEntry, ...]:
        return self.metadata + self.entries

    def rows_for_stage(self, stage: str) -> tuple[SourceEntry, ...]:
        key = stage.upper()
        if key == "B1":
            return self.metadata
        if key == "B2":
            return tuple(row for row in self.entries if row.source_class.upper() == "PLE")
        if key == "B4":
            return tuple(row for row in self.entries if row.source_class.upper() == "BF16")
        return self.entries


def normalize_source_entry_mapping(raw: Mapping[str, Any], *, metadata: bool = False, order: int | None = None, schema_version: int = SOURCE_IDENTITY_VERSION, identity_overrides: Mapping[str, Mapping[str, Any]] | None = None, legacy_migration: bool = False) -> dict[str, Any]:
    """Normalize one v1/v2 manifest row into explicit source identities.

    ``accepted_etag`` is retained solely as a legacy audit field.  v2 callers
    must use ``git_blob_id``, ``lfs_oid_sha256``, ``xet_file_hash`` and the
    explicit ``allowed_body_etags`` list.
    """
    normalized_raw = dict(raw)
    override = dict((identity_overrides or {}).get(str(raw.get("filename")), {}))
    for key in ("git_blob_id", "lfs_oid_sha256", "xet_file_hash", "allowed_body_etags"):
        if key in override:
            normalized_raw[key] = override[key]
    entry = SourceEntry.from_mapping(normalized_raw, metadata=metadata, order=order, schema_version=schema_version, legacy_migration=legacy_migration)
    value = dict(normalized_raw)
    value["filename"] = entry.filename
    value["byte_length"] = entry.byte_length
    value["source_class"] = entry.source_class
    value["acquisition_order"] = entry.acquisition_order
    value["repository"] = entry.repository
    value["revision"] = entry.revision
    value["git_blob_id"] = entry.git_blob_id
    value["lfs_oid_sha256"] = entry.lfs_oid_sha256
    value["xet_file_hash"] = entry.xet_file_hash
    value["allowed_body_etags"] = list(entry.allowed_body_etags)
    # Preserve accepted_etag for v1 audit/replay only; no v2 execution path
    # reads it as an identity.
    return value


def generate_acquisition_manifest_v2(raw: Mapping[str, Any], *, identity_overrides: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Generate a v2 manifest from a v1-shaped mapping without payload I/O."""
    if str(raw.get("schema") or "") not in {ACQUISITION_MANIFEST_V1, ACQUISITION_MANIFEST_V2}:
        raise ExecutorError("unsupported acquisition manifest schema")
    result = dict(raw)
    legacy_migration = str(raw.get("schema") or "") == ACQUISITION_MANIFEST_V1
    result["schema"] = ACQUISITION_MANIFEST_V2
    result["identity_schema"] = "source_identity_v2"
    result["required_small_metadata"] = [
        normalize_source_entry_mapping(row, metadata=True, order=index + 1, schema_version=SOURCE_IDENTITY_VERSION, identity_overrides=identity_overrides, legacy_migration=legacy_migration)
        for index, row in enumerate(raw.get("required_small_metadata", ()))
    ]
    result["source_weight_shards"] = [
        normalize_source_entry_mapping(row, schema_version=SOURCE_IDENTITY_VERSION, identity_overrides=identity_overrides, legacy_migration=legacy_migration)
        for row in raw.get("source_weight_shards", ())
    ]
    return result


def migrate_acquisition_manifest_v1_to_v2(path: str | os.PathLike[str], output_path: str | os.PathLike[str] | None = None, *, identity_overrides: Mapping[str, Mapping[str, Any]] | None = None) -> Path:
    """Write a durable v2 manifest beside the v1 source (metadata-only)."""
    source = Path(path)
    destination = Path(output_path) if output_path is not None else source.with_name(f"{source.stem}.v2{source.suffix}")
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    converted = generate_acquisition_manifest_v2(raw, identity_overrides=identity_overrides)
    _atomic_json(destination, converted)
    return destination


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def migrate_source_receipt_v1_to_v2(
    receipt_path: str | os.PathLike[str],
    *,
    row: SourceEntry,
    source_inventory_fingerprint: str,
    predecessor_path: str | os.PathLike[str] | None = None,
    observed_metadata_etag: str | None = None,
    observed_xet_file_hash: str | None = None,
    observed_body_etag: str | None = None,
    body_bytes: int = 0,
) -> dict[str, Any]:
    """Migrate one completed v1 source receipt without transferring bytes.

    The original JSON bytes are copied to an immutable ``.v1`` sibling and
    hashed into the v2 receipt.  Callers must validate the final source and
    immutable metadata before invoking this helper.
    """
    path = Path(receipt_path)
    predecessor = Path(predecessor_path) if predecessor_path is not None else path.with_suffix(path.suffix + ".v1")
    if not path.is_file():
        raise ExecutorError(f"source receipt missing: {path}")
    original = path.read_bytes()
    predecessor_sha = _bytes_sha256(original)
    try:
        prior = json.loads(original.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ExecutorError("invalid v1 source receipt") from exc
    if prior.get("completion") != "SOURCE_COMPLETE":
        raise ExecutorError("only completed v1 source receipts can be migrated")
    if prior.get("receipt_version") == SOURCE_RECEIPT_VERSION:
        predecessor_hash = _bytes_sha256(predecessor.read_bytes()) if predecessor.exists() else None
        if prior.get("predecessor_receipt_sha256") and predecessor_hash != prior["predecessor_receipt_sha256"]:
            raise ExecutorError("v2 receipt predecessor hash changed")
        return prior
    if predecessor.exists():
        if predecessor.read_bytes() != original:
            raise ExecutorError("existing predecessor receipt differs from current v1 bytes")
    else:
        predecessor.parent.mkdir(parents=True, exist_ok=True)
        temporary = predecessor.with_name(f".{predecessor.name}.partial-{os.getpid()}-{threading.get_ident()}")
        temporary.write_bytes(original)
        try:
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            pass
        os.replace(temporary, predecessor)
    metadata_etag = observed_metadata_etag or prior.get("observed_metadata_etag") or prior.get("observed_etag") or row.metadata_etag
    xet = observed_xet_file_hash or prior.get("observed_xet_file_hash")
    if row.xet_file_hash:
        if not xet:
            raise ExecutorError("LFS/Xet v1 receipt migration requires explicitly validated Xet metadata")
        if _clean_identity(xet) != _clean_identity(row.xet_file_hash):
            raise ExecutorError("v1 receipt Xet identity does not match manifest")
    body_value = observed_body_etag or prior.get("observed_body_etag") or prior.get("observed_etag") or (row.allowed_body_etags[0] if row.allowed_body_etags else None)
    body = str(body_value) if body_value is not None else None
    receipt_entry = asdict(row)
    receipt_entry["allowed_body_etags"] = list(row.allowed_body_etags)
    migrated = {
        **prior,
        "receipt_version": SOURCE_RECEIPT_VERSION,
        "identity_version": SOURCE_IDENTITY_VERSION,
        "entry": receipt_entry,
        "source_inventory_fingerprint": source_inventory_fingerprint,
        "resolved_commit": row.revision,
        "expected_git_blob_id": row.git_blob_id,
        "expected_lfs_oid_sha256": row.lfs_oid_sha256,
        "expected_xet_file_hash": row.xet_file_hash,
        "allowed_body_etags": list(row.allowed_body_etags),
        "observed_metadata_etag": metadata_etag,
        "observed_xet_file_hash": xet,
        "observed_body_etag": body,
        "body_bytes": int(body_bytes),
        "original_body_bytes": int(prior.get("body_bytes", 0) or 0),
        "original_completion": prior.get("completion"),
        "new_body_bytes": 0,
        "lifetime_transfer_accounting": "unchanged",
        "migration_reason": "storage_identity_v2",
        "predecessor_receipt_sha256": predecessor_sha,
        "predecessor_receipt_path": str(predecessor),
        "completion": "SOURCE_COMPLETE",
    }
    _atomic_json(path, migrated)
    return migrated


@dataclass
class TransferBudget:
    cap: int
    transferred: int = 0
    state_path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _inflight_reserved: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.state_path and self.state_path.is_file():
            try:
                with self.state_path.open("r", encoding="utf-8") as handle:
                    prior = json.load(handle)
                if int(prior.get("cap", self.cap)) != self.cap:
                    raise ExecutorError("transfer budget cap changed across restart")
                self.transferred = int(prior.get("transferred", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ExecutorError("invalid persisted transfer budget") from exc

    def _persist(self) -> None:
        if self.state_path:
            _atomic_json(self.state_path, {"cap": self.cap, "transferred": self.transferred})

    def reserve(self, amount: int) -> None:
        amount = int(amount)
        if amount < 0:
            raise ValueError("transfer amount cannot be negative")
        with self._lock:
            self.transferred += amount
            self._persist()
            if self.transferred > self.cap:
                raise ExecutorError(f"transfer cap exceeded: {self.transferred} > {self.cap}")

    def admit(self, maximum_response_bytes: int) -> None:
        """Atomically reserve room before a body request is opened."""
        amount = int(maximum_response_bytes)
        if amount < 0:
            raise ValueError("admission amount cannot be negative")
        with self._lock:
            if self.transferred + self._inflight_reserved + amount > self.cap:
                raise ExecutorError("transfer budget cannot admit response body")
            self._inflight_reserved += amount

    def record_received(self, amount: int) -> None:
        """Persist actual bytes received and consume their inflight reservation."""
        amount = int(amount)
        if amount < 0:
            raise ValueError("received amount cannot be negative")
        with self._lock:
            admitted = min(amount, self._inflight_reserved)
            self._inflight_reserved -= admitted
            self.transferred += amount
            self._persist()
            if amount > admitted:
                raise ExecutorError("received bytes exceed admitted response budget")
            if self.transferred > self.cap:
                raise ExecutorError(f"transfer cap exceeded: {self.transferred} > {self.cap}")

    def release_admission(self, unused_bytes: int) -> None:
        amount = int(unused_bytes)
        if amount < 0:
            raise ValueError("unused admission cannot be negative")
        with self._lock:
            if amount > self._inflight_reserved:
                raise ExecutorError("released admission exceeds inflight reservation")
            self._inflight_reserved -= amount

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.cap - self.transferred


class TransportResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    def iter_bytes(self, chunk_bytes: int = 8 << 20) -> Iterable[bytes]: ...
    def close(self) -> None: ...


class Transport(Protocol):
    def head(self, url: str, *, headers: Mapping[str, str] | None = None) -> TransportResponse: ...
    def get(self, url: str, *, headers: Mapping[str, str] | None = None, allow_body: bool = False) -> TransportResponse: ...


class _UrllibResponse:
    def __init__(self, response: Any):
        self._response = response
        self.status = int(response.status)
        self.headers = {str(k): str(v) for k, v in response.headers.items()}

    def iter_bytes(self, chunk_bytes: int = 8 << 20) -> Iterable[bytes]:
        while data := self._response.read(chunk_bytes):
            yield data

    def close(self) -> None:
        self._response.close()


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    _allowed_hosts = ("huggingface.co", "hf.co")

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = (urllib.parse.urlparse(newurl).hostname or "").lower()
        if not any(host == suffix or host.endswith("." + suffix) for suffix in self._allowed_hosts):
            raise ExecutorError(f"refusing model redirect to unapproved host: {host or '<missing>'}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class UrllibTransport:
    """Small dependency-free HTTP transport; body permission is explicit."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(_SafeRedirectHandler())

    def _request(self, method: str, url: str, headers: Mapping[str, str] | None) -> _UrllibResponse:
        request = urllib.request.Request(url, method=method, headers=dict(headers or {}))
        try:
            return _UrllibResponse(self._opener.open(request, timeout=60))
        except urllib.error.HTTPError as exc:
            return _UrllibResponse(exc)

    def head(self, url: str, *, headers: Mapping[str, str] | None = None) -> _UrllibResponse:
        return self._request("HEAD", url, headers)

    def get(self, url: str, *, headers: Mapping[str, str] | None = None, allow_body: bool = False) -> _UrllibResponse:
        if not allow_body:
            raise BodyTransferDisabled("GET body blocked; require --execute and --allow-network-body")
        return self._request("GET", url, headers)


class JsonlLogger:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = _z_path(path)
        self._lock = threading.Lock()

    def event(self, event: str, **fields: Any) -> None:
        # Never persist credentials, signed URLs, or cookies.
        safe = {k: ("<redacted>" if any(x in k.lower() for x in ("token", "cookie", "authorization", "url")) else v) for k, v in fields.items()}
        safe.update(event=event, timestamp=time.time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(safe, sort_keys=True) + "\n")


class Downloader:
    def __init__(self, source_root: str | os.PathLike[str], manifest: AcquisitionManifest, *, receipt_root: str | os.PathLike[str] | None = None, transport: Transport | None = None, execute: bool = False, allow_network_body: bool = False, max_concurrent: int = MAX_DOWNLOADS, budget: TransferBudget | None = None, logger: JsonlLogger | None = None):
        self.root = _z_path(source_root)
        self.receipt_root = _z_path(receipt_root or (self.root / ".step9b-receipts"))
        self.manifest = manifest
        self.transport = transport or UrllibTransport()
        self.execute = bool(execute)
        self.allow_network_body = bool(allow_network_body)
        if max_concurrent < 1 or max_concurrent > MAX_DOWNLOADS:
            raise ValueError("max_concurrent must be between 1 and 2")
        self.max_concurrent = max_concurrent
        self.budget = budget or TransferBudget(manifest.transfer_cap)
        self.logger = logger
        self.active = 0
        self.max_active = 0
        self._active_lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._cancel = threading.Event()
        self._file_locks: dict[str, threading.Lock] = {}
        self._file_locks_guard = threading.Lock()
        # Last body-disabled/resume plans are retained for audit and tests;
        # they contain no payload bytes.
        self.resume_plans: dict[str, dict[str, Any]] = {}

    def _url(self, row: SourceEntry) -> str:
        return f"https://huggingface.co/{row.repository}/resolve/{row.revision}/{row.filename}"

    def cancel(self) -> None:
        self._cancel.set()

    def _identity(self, row: SourceEntry, length: int, remote: Mapping[str, Any] | None = None, *, observed_body_etag: str | None = None) -> dict[str, Any]:
        remote = dict(remote or {})
        metadata_etag = remote.get("metadata_etag") or remote.get("etag")
        body_etag = observed_body_etag if observed_body_etag is not None else remote.get("observed_body_etag")
        value = {
            "identity_version": SOURCE_IDENTITY_VERSION if self.manifest.schema_version >= SOURCE_IDENTITY_VERSION else 1,
            "repository": row.repository,
            "revision": row.revision,
            "resolved_commit": remote.get("commit") or remote.get("resolved_commit") or row.revision,
            "filename": row.filename,
            "expected_length": row.byte_length,
            "expected_git_blob_id": row.git_blob_id,
            "expected_lfs_oid_sha256": row.lfs_oid_sha256,
            "expected_xet_file_hash": row.xet_file_hash,
            "allowed_body_etags": list(row.allowed_body_etags),
            # Legacy aliases remain readable during v1->v2 migration.
            "expected_etag": row.accepted_etag,
            "expected_lfs_oid": row.lfs_oid_sha256,
            "expected_header_length": row.accepted_header_length,
            "expected_header_sha256": row.accepted_header_sha256,
            "observed_metadata_etag": metadata_etag,
            "observed_etag": metadata_etag,
            "observed_xet_file_hash": remote.get("xet_file_hash"),
            "observed_body_etag": body_etag,
            "partial_length": length,
            "source_inventory_fingerprint": self.manifest.source_inventory_fingerprint,
            "acquisition_order": row.acquisition_order,
        }
        return value

    def validate_metadata(self, row: SourceEntry, response: TransportResponse) -> dict[str, Any]:
        headers = {k.lower(): v for k, v in response.headers.items()}
        observed_length = int(headers.get("content-length", "-1"))
        observed_etag = headers.get("etag")
        observed_xet = headers.get("x-xet-hash") or headers.get("xet-file-hash") or headers.get("xet_file_hash")
        observed_commit = headers.get("x-repo-commit") or headers.get("x-linked-commit")
        if row.identity_version >= SOURCE_IDENTITY_VERSION:
            if not observed_commit:
                raise ExecutorError(f"{row.filename}: metadata response omitted immutable commit identity")
            if observed_commit.strip() != row.revision:
                raise ExecutorError(f"{row.filename}: metadata commit identity mismatch")
        elif observed_commit and observed_commit.strip() != row.revision:
            raise ExecutorError(f"{row.filename}: metadata commit identity mismatch")
        if observed_length != row.byte_length:
            raise ExecutorError(f"{row.filename}: content length mismatch")
        # Hugging Face uses two identities for Xet/LFS files.  The manifest's
        # accepted_etag is the Xet file hash (often quoted), while lfs_oid is
        # surfaced as HfFileMetadata.etag and may be returned as HTTP ETag by
        # the CDN.  A transport may expose either, so accept only either exact
        # frozen identity and never a merely non-empty header.
        if row.identity_version >= SOURCE_IDENTITY_VERSION:
            expected_metadata_etag = row.metadata_etag
            if expected_metadata_etag and not observed_etag:
                raise ExecutorError(f"{row.filename}: metadata response omitted required metadata ETag")
            if expected_metadata_etag and observed_etag and observed_etag.strip('"').lower() != expected_metadata_etag.strip('"').lower():
                raise ExecutorError(f"{row.filename}: metadata ETag identity mismatch")
            if row.xet_file_hash and observed_xet and observed_xet.strip('"').lower() != row.xet_file_hash.lower():
                raise ExecutorError(f"{row.filename}: metadata Xet identity mismatch")
            if row.xet_file_hash and not observed_xet:
                raise ExecutorError(f"{row.filename}: metadata response omitted required Xet identity")
            if not row.xet_file_hash and observed_xet:
                raise ExecutorError(f"{row.filename}: Git-backed metadata unexpectedly exposed Xet identity")
            return {"resolved_commit": row.revision, "length": observed_length, "etag": observed_etag, "metadata_etag": observed_etag, "xet_file_hash": observed_xet, "body_bytes": 0}
        accepted = {str(value).strip('"') for value in (row.accepted_etag, row.lfs_oid_sha256) if value}
        if accepted and not observed_etag:
            raise ExecutorError(f"{row.filename}: metadata response omitted required ETag")
        if observed_etag and accepted and observed_etag.strip('"') not in accepted:
            raise ExecutorError(f"{row.filename}: ETag/Xet identity mismatch")
        return {"resolved_commit": row.revision, "length": observed_length, "etag": observed_etag}

    def resolve_hf_metadata(self, row: SourceEntry) -> dict[str, Any]:
        """Resolve immutable HF metadata without requesting a response body.

        ``huggingface_hub`` is imported lazily so synthetic transports and the
        dry-run controller remain usable in minimal environments.  For Xet
        entries, ``xet_file_data.file_hash`` is checked against the manifest's
        accepted_etag and ``etag`` against the LFS OID.  Metadata files use the
        Git blob etag instead.
        """
        try:
            from huggingface_hub import get_hf_file_metadata
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ExecutorError("huggingface_hub is required for HF metadata identity") from exc
        url = self._url(row)
        try:
            meta = get_hf_file_metadata(url=url, token=None)
        except TypeError:
            # Older hub releases use ``filename``/``repo_id`` but retain the
            # immutable resolve URL contract; only this metadata-only fallback
            # is allowed.
            meta = get_hf_file_metadata(url)
        commit = str(getattr(meta, "commit_hash", "") or "")
        size = int(getattr(meta, "size", -1) or -1)
        etag = str(getattr(meta, "etag", "") or "").strip('"')
        if commit != row.revision:
            raise ExecutorError(f"{row.filename}: resolved commit mismatch")
        if size != row.byte_length:
            raise ExecutorError(f"{row.filename}: metadata length mismatch")
        xet = getattr(meta, "xet_file_data", None)
        xet_hash = str(getattr(xet, "file_hash", "") or "").strip('"')
        if row.identity_version >= SOURCE_IDENTITY_VERSION:
            expected_metadata_etag = (row.lfs_oid_sha256 or row.git_blob_id or "").lower()
            if expected_metadata_etag and etag != expected_metadata_etag:
                raise ExecutorError(f"{row.filename}: metadata {'LFS OID' if row.lfs_oid_sha256 else 'Git blob'} identity mismatch")
            if row.xet_file_hash and xet_hash != row.xet_file_hash.lower():
                raise ExecutorError(f"{row.filename}: metadata Xet hash mismatch")
            if not row.lfs_oid_sha256 and xet_hash:
                raise ExecutorError(f"{row.filename}: Git metadata unexpectedly carries Xet identity")
        elif row.lfs_oid_sha256:
            if etag != row.lfs_oid_sha256.lower():
                raise ExecutorError(f"{row.filename}: metadata LFS OID mismatch")
            if row.accepted_etag and xet_hash != row.accepted_etag.strip('"').lower():
                raise ExecutorError(f"{row.filename}: metadata Xet hash mismatch")
        elif row.accepted_etag:
            # Git-backed metadata has no Xet data; accepted_etag is the blob id.
            if etag != row.accepted_etag.strip('"').lower():
                raise ExecutorError(f"{row.filename}: metadata Git identity mismatch")
        return {"url": url, "commit": commit, "size": size, "etag": etag, "metadata_etag": etag, "xet_file_hash": xet_hash or None, "body_bytes": 0}

    def _validate_partial_identity(self, row: SourceEntry, meta: Path, length: int, remote: Mapping[str, Any]) -> None:
        if not meta.is_file():
            raise ResumeRejected(f"partial identity sidecar missing for {row.filename}")
        with meta.open("r", encoding="utf-8") as handle:
            identity = json.load(handle)
        expected = self._identity(row, length, remote)
        keys = ("repository", "revision", "resolved_commit", "filename", "expected_length", "source_inventory_fingerprint", "acquisition_order")
        if row.identity_version >= SOURCE_IDENTITY_VERSION:
            keys += ("identity_version", "expected_git_blob_id", "expected_lfs_oid_sha256", "expected_xet_file_hash", "allowed_body_etags", "observed_metadata_etag", "observed_xet_file_hash")
        else:
            keys += ("expected_etag", "expected_lfs_oid", "observed_etag", "observed_xet_file_hash")
        for key in keys:
            if identity.get(key) != expected.get(key):
                raise ResumeRejected(f"partial identity mismatch: {key}")
        if int(identity.get("partial_length", -1)) != length:
            raise ResumeRejected("partial length identity mismatch")
        if row.identity_version >= SOURCE_IDENTITY_VERSION:
            observed_body = _clean_identity(identity.get("observed_body_etag"))
            allowed = {_clean_identity(value) for value in row.allowed_body_etags}
            if length > 0 and observed_body not in allowed:
                raise ResumeRejected("partial identity missing allowed observed_body_etag")

    @staticmethod
    def _atomic_identity_candidates(identity: Path) -> tuple[Path, ...]:
        """Return only this executor's atomic-temp siblings, deterministically."""
        prefix = f".{identity.name}.partial-"
        return tuple(sorted((item for item in identity.parent.iterdir() if item.is_file() and item.name.startswith(prefix)), key=lambda item: item.name))

    def _completed_source_bytes(self, *, excluding: str | None = None) -> int:
        """Count exact-length final source files for ledger consistency checks."""
        total = 0
        for item in self.manifest.all_entries:
            if item.filename == excluding:
                continue
            final = self.root / item.filename
            if not final.is_file():
                continue
            if final.stat().st_size != item.byte_length:
                raise ResumeRejected(f"completed source has wrong length: {item.filename}")
            try:
                self.validate_existing(item, final)
            except ExecutorError as exc:
                raise ResumeRejected(f"completed source failed validation: {item.filename}") from exc
            total += item.byte_length
        return total

    def _validate_orphan_identity(
        self,
        row: SourceEntry,
        candidate: Path,
        partial: Path,
        remote: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate an atomic-temp checkpoint against the complete v2 contract."""
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ResumeRejected(f"invalid orphan identity JSON: {candidate.name}") from exc
        if not isinstance(value, Mapping):
            raise ResumeRejected(f"orphan identity is not an object: {candidate.name}")
        if int(value.get("identity_version", -1)) != SOURCE_IDENTITY_VERSION:
            raise ResumeRejected("orphan identity schema is not v2")
        actual_length = partial.stat().st_size
        expected = self._identity(row, actual_length, remote)
        required = (
            "repository", "revision", "resolved_commit", "filename", "expected_length",
            "source_inventory_fingerprint", "acquisition_order", "identity_version",
            "expected_git_blob_id", "expected_lfs_oid_sha256", "expected_xet_file_hash",
            "observed_metadata_etag", "observed_xet_file_hash",
        )
        for key in required:
            if value.get(key) != expected.get(key):
                raise ResumeRejected(f"orphan identity mismatch: {key}")
        if int(value.get("partial_length", -1)) != actual_length:
            raise ResumeRejected("orphan partial length does not match physical partial")
        declared_allowed = {_clean_identity(item) for item in value.get("allowed_body_etags", ())}
        expected_allowed = {_clean_identity(item) for item in row.allowed_body_etags}
        if declared_allowed != expected_allowed:
            raise ResumeRejected("orphan allowed body ETag set mismatch")
        observed_body = _clean_identity(value.get("observed_body_etag"))
        if actual_length and observed_body not in expected_allowed:
            raise ResumeRejected("orphan observed body ETag is not allowed")
        # The ledger must account for every validated final source plus this
        # partial.  It may include retransmitted bytes, hence the >= relation.
        required_ledger = self._completed_source_bytes(excluding=row.filename) + actual_length
        if self.budget.transferred < required_ledger:
            raise ResumeRejected(
                f"orphan transfer ledger is inconsistent: {self.budget.transferred} < {required_ledger}"
            )
        return dict(value)

    def recover_partial_identity_checkpoint(
        self,
        row: SourceEntry,
        *,
        partial: Path | None = None,
        identity: Path | None = None,
        remote: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Fail-closed adoption of a valid executor atomic-temp sidecar.

        Only siblings matching ``.<identity>.partial-*`` are inspected.  A
        candidate is adopted when it exactly describes the current physical
        partial and the canonical sidecar is absent or strictly behind it.
        Ambiguous or invalid candidates are rejected rather than guessed.
        """
        partial = partial or (self.root / f"{row.filename}.partial")
        identity = identity or partial.with_name(partial.name + ".meta.json")
        if not partial.is_file():
            return None
        candidates = self._atomic_identity_candidates(identity)
        if not candidates:
            return None
        if remote is None:
            if isinstance(self.transport, UrllibTransport):
                remote = self.resolve_hf_metadata(row)
            else:
                head = self.transport.head(self._url(row), headers={})
                try:
                    remote = self.validate_metadata(row, head)
                finally:
                    head.close()
        validated: list[tuple[Path, dict[str, Any]]] = []
        for candidate in candidates:
            validated.append((candidate, self._validate_orphan_identity(row, candidate, partial, remote)))
        # All candidates must be semantically identical; otherwise fail closed.
        baseline = validated[0][1]
        identity_keys = (
            "identity_version", "repository", "revision", "resolved_commit", "filename",
            "expected_length", "expected_git_blob_id", "expected_lfs_oid_sha256",
            "expected_xet_file_hash", "allowed_body_etags", "observed_metadata_etag",
            "observed_xet_file_hash", "observed_body_etag", "partial_length",
            "source_inventory_fingerprint", "acquisition_order",
        )
        for _, value in validated[1:]:
            if any(value.get(key) != baseline.get(key) for key in identity_keys):
                raise ResumeRejected("AMBIGUOUS ORPHAN CHECKPOINT")
        canonical_value: dict[str, Any] | None = None
        if identity.exists():
            try:
                with identity.open("r", encoding="utf-8") as handle:
                    parsed = json.load(handle)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ResumeRejected("canonical partial identity is malformed") from exc
            if not isinstance(parsed, Mapping):
                raise ResumeRejected("canonical partial identity is not an object")
            canonical_value = dict(parsed)
            canonical_length = int(canonical_value.get("partial_length", -1))
            if canonical_length > partial.stat().st_size:
                raise ResumeRejected("canonical partial identity is ahead of physical partial")
            for key in identity_keys:
                if key == "partial_length":
                    continue
                if canonical_value.get(key) != baseline.get(key):
                    raise ResumeRejected(f"canonical/orphan checkpoint conflict: {key}")
            if canonical_length == partial.stat().st_size:
                return {"state": "ALREADY_CURRENT", "candidate": str(validated[0][0]), "partial_length": partial.stat().st_size}
            if canonical_length >= int(baseline.get("partial_length", -1)):
                raise ResumeRejected("canonical partial identity is not strictly behind orphan")
        # Publish the validated orphan contents through the hardened helper.
        _atomic_json(identity, baseline)
        with identity.open("r", encoding="utf-8") as handle:
            adopted = json.load(handle)
        if int(adopted.get("partial_length", -1)) != partial.stat().st_size:
            raise ResumeRejected("adopted canonical partial identity changed unexpectedly")
        return {
            "state": "ADOPTED",
            "candidate": str(validated[0][0]),
            "partial_length": partial.stat().st_size,
            "canonical_was_present": canonical_value is not None,
            "body_bytes": 0,
        }

    def _validate_safetensors_header(self, row: SourceEntry, path: Path) -> None:
        if row.accepted_header_length is None or row.accepted_header_sha256 is None:
            return
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ExecutorError(f"{row.filename}: missing Safetensors framing")
            (header_length,) = struct.unpack("<Q", prefix)
            if int(header_length) != int(row.accepted_header_length):
                raise ExecutorError(f"{row.filename}: Safetensors header length mismatch")
            if header_length > MAX_SAFETENSORS_HEADER_BYTES or header_length > row.byte_length - 8:
                raise ExecutorError(f"{row.filename}: unsafe Safetensors header length")
            header = handle.read(header_length)
        if hashlib.sha256(header).hexdigest() != row.accepted_header_sha256:
            raise ExecutorError(f"{row.filename}: Safetensors header hash mismatch")

    def validate_existing(self, row: SourceEntry, final: Path) -> dict[str, Any]:
        if not final.is_file() or final.stat().st_size != row.byte_length:
            raise ExecutorError(f"{row.filename}: final source is missing or wrong length")
        digest = _sha256(final)
        if row.lfs_oid_sha256 and digest.lower() != row.lfs_oid_sha256.lower():
            raise ExecutorError(f"{row.filename}: source SHA/LFS OID mismatch")
        if row.git_blob_id and not row.lfs_oid_sha256:
            git = hashlib.sha1(f"blob {row.byte_length}\0".encode())
            with final.open("rb") as source:
                while data := source.read(8 << 20):
                    git.update(data)
            git_digest = git.hexdigest()
            if git_digest.lower() != row.git_blob_id.lower():
                raise ExecutorError(f"{row.filename}: Git blob identity mismatch")
        self._validate_safetensors_header(row, final)
        return {"state": "VALID", "bytes": row.byte_length, "sha256": digest}

    def _validate_promote_partial(
        self,
        row: SourceEntry,
        partial: Path,
        identity: Path,
        final: Path,
        receipt: Path,
        remote_meta: Mapping[str, Any],
        *,
        resumed_from: int,
        body_bytes_this_run: int,
        observed_body_etag: str | None = None,
        recovered_complete_partial: bool = False,
    ) -> dict[str, Any]:
        if partial.stat().st_size != row.byte_length:
            raise ExecutorError("partial is not complete enough to promote")
        self._validate_safetensors_header(row, partial)
        digest = _sha256(partial)
        if row.lfs_oid_sha256 and digest.lower() != row.lfs_oid_sha256.lower():
            raise ExecutorError("source SHA/LFS OID mismatch")
        if row.git_blob_id and not row.lfs_oid_sha256:
            git = hashlib.sha1()
            git.update(f"blob {row.byte_length}\0".encode())
            with partial.open("rb") as source:
                while data := source.read(8 << 20):
                    git.update(data)
            if git.hexdigest().lower() != row.git_blob_id.lower():
                raise ExecutorError("Git blob identity mismatch")
        os.replace(partial, final)
        identity.unlink(missing_ok=True)
        result = {
            "state": "SOURCE_COMPLETE",
            "final_path": str(final),
            "expected_bytes": row.byte_length,
            "bytes": row.byte_length,
            "sha256": digest,
            "resumed_from": resumed_from,
            "body_bytes": row.byte_length,
            "body_bytes_this_run": body_bytes_this_run,
            "resolved_commit": row.revision,
            "expected_etag": row.accepted_etag,
            "expected_git_blob_id": row.git_blob_id,
            "expected_lfs_oid_sha256": row.lfs_oid_sha256,
            "expected_xet_file_hash": row.xet_file_hash,
            "allowed_body_etags": list(row.allowed_body_etags),
            "observed_etag": remote_meta.get("etag"),
            "observed_metadata_etag": remote_meta.get("metadata_etag") or remote_meta.get("etag"),
            "observed_xet_file_hash": remote_meta.get("xet_file_hash"),
            "observed_body_etag": observed_body_etag,
            "expected_lfs_oid": row.lfs_oid_sha256,
            "observed_lfs_oid": row.lfs_oid_sha256,
            "observed_header_length": row.accepted_header_length,
            "observed_header_sha256": row.accepted_header_sha256,
            "recovered_complete_partial": recovered_complete_partial,
        }
        _atomic_json(receipt, {"entry": asdict(row), "receipt_version": SOURCE_RECEIPT_VERSION, "identity_version": row.identity_version, **result, "source_inventory_fingerprint": self.manifest.source_inventory_fingerprint, "completion": "SOURCE_COMPLETE"})
        return result

    def acquire(self, row: SourceEntry) -> dict[str, Any]:
        with self._file_locks_guard:
            lock = self._file_locks.setdefault(row.filename, threading.Lock())
        with lock:
            return self._acquire_locked(row)

    def _acquire_locked(self, row: SourceEntry) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        final = self.root / row.filename
        partial = self.root / f"{row.filename}.partial"
        identity = partial.with_name(partial.name + ".meta.json")
        receipt = self.receipt_root / f"{row.acquisition_order:03d}-{row.filename}.receipt.json"
        if final.exists():
            result = self.validate_existing(row, final)
            receipt_entry = asdict(row)
            # JSON persists tuples as arrays.  Normalize the expected binding
            # to that durable representation so a migrated v2 receipt is
            # accepted on the next restart instead of comparing list vs tuple.
            receipt_entry["allowed_body_etags"] = list(row.allowed_body_etags)
            receipt_binding = {
                "entry": receipt_entry,
                "final_path": str(final),
                "expected_bytes": row.byte_length,
                "bytes": row.byte_length,
                "resolved_commit": row.revision,
                "expected_etag": row.accepted_etag,
                "expected_git_blob_id": row.git_blob_id,
                "expected_lfs_oid_sha256": row.lfs_oid_sha256,
                "expected_xet_file_hash": row.xet_file_hash,
                "allowed_body_etags": list(row.allowed_body_etags),
                "expected_lfs_oid": row.lfs_oid_sha256,
                "observed_lfs_oid": row.lfs_oid_sha256,
                "observed_header_length": row.accepted_header_length,
                "observed_header_sha256": row.accepted_header_sha256,
                "source_inventory_fingerprint": self.manifest.source_inventory_fingerprint,
                "sha256": result["sha256"],
            }
            if receipt.exists():
                try:
                    with receipt.open("r", encoding="utf-8") as handle:
                        prior = json.load(handle)
                    if prior.get("receipt_version") == SOURCE_RECEIPT_VERSION:
                        prior_binding = dict(receipt_binding)
                        if prior.get("completion") != "SOURCE_COMPLETE" or any(prior.get(key) != value for key, value in prior_binding.items()):
                            raise ExecutorError(f"{row.filename}: source receipt binding mismatch")
                        accepted_observed = {_clean_identity(value) for value in row.allowed_body_etags}
                        if _clean_identity(prior.get("observed_body_etag") or prior.get("observed_etag")) not in accepted_observed:
                            raise ExecutorError(f"{row.filename}: source receipt body ETag binding mismatch")
                        if _clean_identity(prior.get("observed_metadata_etag") or prior.get("metadata_etag")) != _clean_identity(row.metadata_etag):
                            raise ExecutorError(f"{row.filename}: source receipt metadata identity mismatch")
                        if row.xet_file_hash and _clean_identity(prior.get("observed_xet_file_hash")) != _clean_identity(row.xet_file_hash):
                            raise ExecutorError(f"{row.filename}: source receipt Xet binding mismatch")
                    else:
                        # v1 receipt migration is metadata-only.  Resolve the
                        # immutable source metadata before rewriting its receipt.
                        if self.execute:
                            if isinstance(self.transport, UrllibTransport):
                                remote = self.resolve_hf_metadata(row)
                            else:
                                head = self.transport.head(self._url(row), headers={})
                                try:
                                    remote = self.validate_metadata(row, head)
                                finally:
                                    head.close()
                        else:
                            remote = {}
                        migrate_source_receipt_v1_to_v2(
                            receipt,
                            row=row,
                            source_inventory_fingerprint=self.manifest.source_inventory_fingerprint,
                            observed_metadata_etag=remote.get("metadata_etag") or remote.get("etag"),
                            observed_xet_file_hash=remote.get("xet_file_hash"),
                            observed_body_etag=prior.get("observed_body_etag") or prior.get("observed_etag"),
                            body_bytes=int(prior.get("body_bytes", 0) or 0),
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ExecutorError(f"{row.filename}: invalid source receipt") from exc
            else:
                _atomic_json(receipt, {**receipt_binding, "receipt_version": SOURCE_RECEIPT_VERSION, "identity_version": row.identity_version, "observed_metadata_etag": row.metadata_etag, "observed_etag": row.metadata_etag, "observed_xet_file_hash": row.xet_file_hash, "observed_body_etag": (row.allowed_body_etags[0] if row.allowed_body_etags else None), "resumed_from": None, "body_bytes": 0, "completion": "SOURCE_COMPLETE", "recovered_after_promotion": True})
            return {"filename": row.filename, **result, "state": "SKIP_VALID_FINAL"}
        if not self.execute:
            if partial.exists() or identity.exists():
                raise BodyTransferDisabled("dry run cannot inspect/resume body partials")
            return {"filename": row.filename, "state": "PLANNED", "bytes": row.byte_length}
        if self._cancel.is_set():
            raise ExecutorError("source acquisition cancelled")
        if isinstance(self.transport, UrllibTransport):
            remote_meta = self.resolve_hf_metadata(row)
        else:
            head = self.transport.head(self._url(row), headers={})
            try:
                remote_meta = self.validate_metadata(row, head)
            finally:
                head.close()
        current = partial.stat().st_size if partial.exists() else 0
        if partial.exists() != identity.exists():
            # A process crash can leave a valid atomic-temp identity sidecar
            # beside the body partial while the canonical replace is pending.
            # Recover it before enforcing the pair invariant.
            if partial.exists() and not identity.exists():
                self.recover_partial_identity_checkpoint(row, partial=partial, identity=identity, remote=remote_meta)
            if partial.exists() != identity.exists():
                raise ResumeRejected(f"partial and identity sidecar must exist together for {row.filename}")
        elif partial.exists():
            # The canonical sidecar may lag the body after a sharing-lock
            # failure.  This call is a no-op when no orphan is present.
            self.recover_partial_identity_checkpoint(row, partial=partial, identity=identity, remote=remote_meta)
            current = partial.stat().st_size
        if current or partial.exists():
            self._validate_partial_identity(row, identity, current, remote_meta)
        if current == row.byte_length:
            result = self._validate_promote_partial(
                row,
                partial,
                identity,
                final,
                receipt,
                remote_meta,
                resumed_from=current,
                body_bytes_this_run=0,
                recovered_complete_partial=True,
            )
            return {"filename": row.filename, **result}
        headers: dict[str, str] = {}
        persisted_body_etag: str | None = None
        if current:
            try:
                with identity.open("r", encoding="utf-8") as handle:
                    persisted = json.load(handle)
                persisted_value = persisted.get("observed_body_etag") or (persisted.get("observed_etag") if row.identity_version < SOURCE_IDENTITY_VERSION else None)
                persisted_body_etag = str(persisted_value) if persisted_value is not None else None
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ResumeRejected("invalid partial identity sidecar") from exc
            if row.identity_version >= SOURCE_IDENTITY_VERSION and not persisted_body_etag:
                raise ResumeRejected("validated observed_body_etag unavailable for If-Range")
            if row.identity_version >= SOURCE_IDENTITY_VERSION:
                headers = {"Range": f"bytes={current}-", "If-Range": persisted_body_etag}
            else:
                # v1 compatibility: its observed_etag represented metadata
                # identity, so retain the historical If-Range value while v2
                # uses the exact transport-body ETag above.
                headers = {"Range": f"bytes={current}-", "If-Range": str(remote_meta.get("etag") or persisted_body_etag or "")}
            plan = {
                "filename": row.filename,
                "range_start": current,
                "remaining_bytes": row.byte_length - current,
                "if_range": persisted_body_etag,
                "body_request_authorized": bool(self.allow_network_body),
            }
            self.resume_plans[row.filename] = plan
            if self.logger:
                self.logger.event("resume_plan", **plan)
        if not self.allow_network_body:
            raise BodyTransferDisabled("execution mode requires explicit network-body authorization")
        self._semaphore.acquire()
        with self._active_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        response = None
        received = 0
        admitted = 0
        try:
            # Reserve the advertised maximum before opening the body; this
            # prevents a response from starting after the transfer cap is spent.
            remaining = row.byte_length - current
            if remaining < 0 or remaining > self.budget.remaining:
                raise ExecutorError(f"transfer budget cannot admit {row.filename}")
            self.budget.admit(remaining)
            admitted = remaining
            if self._cancel.is_set():
                raise ExecutorError("source acquisition cancelled")
            response = self.transport.get(self._url(row), headers=headers, allow_body=True)
            rh = {k.lower(): v for k, v in response.headers.items()}
            if current:
                if response.status != 206:
                    raise ResumeRejected("server ignored Range; refusing to append a 200 response")
                content_range = rh.get("content-range", "")
                expected_prefix = f"bytes {current}-"
                try:
                    span, total = content_range.removeprefix("bytes ").split("/")
                    start, end = (int(value) for value in span.split("-"))
                except (ValueError, TypeError):
                    raise ResumeRejected("malformed Content-Range")
                content_length = int(rh.get("content-length", "-1"))
                if (not content_range.startswith(expected_prefix) or total != str(row.byte_length) or start != current or end != row.byte_length - 1 or content_length != remaining):
                    raise ResumeRejected("malformed Content-Range")
            elif response.status != 200:
                raise ExecutorError(f"unexpected initial body status {response.status}")
            elif int(rh.get("content-length", "-1")) != row.byte_length:
                raise ExecutorError("initial Content-Length mismatch")
            etag = rh.get("etag")
            accepted = {_clean_identity(value) for value in (row.allowed_body_etags if row.identity_version >= SOURCE_IDENTITY_VERSION else (row.accepted_etag, row.lfs_oid_sha256)) if value}
            if accepted and not etag:
                raise ExecutorError("body response omitted required ETag identity")
            if etag and accepted and etag.strip('"').lower() not in accepted:
                raise ExecutorError("body ETag identity changed")
            if current and row.identity_version >= SOURCE_IDENTITY_VERSION and persisted_body_etag and _clean_identity(etag) != _clean_identity(persisted_body_etag):
                raise ResumeRejected("body ETag changed across resume")
            body_etag = str(etag) if etag is not None else None
            mode = "ab" if current else "wb"
            with partial.open(mode) as target:
                if not current:
                    _atomic_json(identity, self._identity(row, 0, remote_meta, observed_body_etag=body_etag))
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    self.budget.record_received(len(chunk))
                    admitted -= len(chunk)
                    if self._cancel.is_set():
                        raise ExecutorError("source acquisition cancelled")
                    if received + len(chunk) > remaining or current + received + len(chunk) > row.byte_length:
                        raise ExecutorError("response body exceeds expected length")
                    target.write(chunk)
                    received += len(chunk)
                    _atomic_json(identity, self._identity(row, current + received, remote_meta, observed_body_etag=body_etag))
                target.flush()
                os.fsync(target.fileno())
            if current + received != row.byte_length:
                raise ExecutorError("undersized final body")
            result = self._validate_promote_partial(
                row,
                partial,
                identity,
                final,
                receipt,
                remote_meta,
                resumed_from=current,
                body_bytes_this_run=received,
                observed_body_etag=body_etag,
            )
            return {"filename": row.filename, **result}
        finally:
            if response is not None:
                response.close()
            if admitted:
                self.budget.release_admission(admitted)
            with self._active_lock:
                self.active -= 1
            self._semaphore.release()


class Step9BExecutor:
    """Explicit B1-B5 state machine; conversion methods delegate to accepted writers."""

    def __init__(self, manifest_path: str | os.PathLike[str], source_root: str | os.PathLike[str], target_root: str | os.PathLike[str], scratch_root: str | os.PathLike[str], logs_root: str | os.PathLike[str], *, builder_commit: str, runtime_worktree: str | os.PathLike[str], runtime_commit: str, source_inventory_fingerprint: str | None = None, source_revision: str = PINNED_REVISION, transfer_cap: int = MAX_TRANSFER_BYTES, toolchain_root: str | os.PathLike[str] | None = None, execute: bool = False, allow_network_body: bool = False, source_retirement_authorized: bool = False, min_disk_free: int = MIN_DISK_RESERVE_BYTES, min_host_free: int = MIN_HOST_FREE_BYTES, max_concurrent_downloads: int = MAX_DOWNLOADS, transport: Transport | None = None):
        self.manifest_path = Path(manifest_path)
        # Payload execution is canonical only with explicit v2 identities;
        # dry-run remains able to inspect the frozen v1 plan for compatibility
        # and migration planning.
        self.manifest = AcquisitionManifest.load(manifest_path, require_v2=bool(execute))
        if str(source_revision) != self.manifest.revision:
            raise ExecutorError("source revision does not match acquisition manifest")
        if int(transfer_cap) != self.manifest.transfer_cap:
            raise ExecutorError("transfer cap does not match frozen authorization")
        self.source_root = _z_path(source_root)
        self.target_root = _z_path(target_root)
        self.scratch_root = _z_path(scratch_root)
        self.logs_root = _z_path(logs_root)
        self.toolchain_root = _z_path(toolchain_root or (self.logs_root.resolve().parents[2] / "artifacts" / "toolchain" / "step9b-pr257"))
        self.builder_commit = str(builder_commit).lower()
        self.source_inventory_fingerprint = str(source_inventory_fingerprint or self.manifest.source_inventory_fingerprint).lower()
        if len(self.source_inventory_fingerprint) != 64 or any(char not in "0123456789abcdef" for char in self.source_inventory_fingerprint):
            raise ExecutorError("source inventory fingerprint must be a SHA-256 digest")
        if self.source_inventory_fingerprint != self.manifest.source_inventory_fingerprint:
            raise ExecutorError("source inventory fingerprint does not match frozen manifest")
        self.runtime_worktree = _z_path(runtime_worktree, must_exist=True)
        self.runtime_commit = str(runtime_commit).lower()
        self.execute = bool(execute)
        self.allow_network_body = bool(allow_network_body)
        self.source_retirement_authorized = bool(source_retirement_authorized)
        self.min_disk_free = int(min_disk_free)
        self.min_host_free = int(min_host_free)
        self.logger = JsonlLogger(self.logs_root / "events.jsonl")
        self.budget = TransferBudget(self.manifest.transfer_cap, state_path=self.scratch_root / "transfer-budget.json")
        self.downloader = Downloader(self.source_root, self.manifest, receipt_root=self.scratch_root / "receipts" / "sources", transport=transport, execute=execute, allow_network_body=allow_network_body, max_concurrent=max_concurrent_downloads, budget=self.budget, logger=self.logger)
        self.state_path = self.scratch_root / "executor-state.json"
        self.state: dict[str, Any] = {"mode": "EXECUTE" if execute else "DRY_RUN", "source_retirement_authorized": self.source_retirement_authorized, "stages": {}}

    def preflight(self) -> dict[str, Any]:
        if self.source_retirement_authorized:
            # Real Step 9 handoff intentionally sets this false.  A caller may
            # test true only with an explicit, separately reviewed controller.
            raise ExecutorError("source retirement is disabled for this executor invocation")
        self._validate_source_workspace()
        source_payload_exists = self.source_root.exists() and any(
            path.is_file() and (path.name in {row.filename for row in self.manifest.all_entries} or path.name.endswith(".partial"))
            for path in self.source_root.iterdir()
        )
        if source_payload_exists and not (self.scratch_root / "transfer-budget.json").is_file():
            raise ExecutorError("source payload exists without durable transfer-budget provenance")
        usage = shutil.disk_usage(self.source_root.anchor or self.source_root)
        required_free = self._projected_required_free()
        if usage.free < required_free:
            raise ExecutorError(f"Z: free space below projected retained-source gate: {usage.free} < {required_free}")
        # Builder/runtime authority checks happen before any irreversible work.
        builder_repo = Path(__file__).resolve().parents[3]
        try:
            builder_head = subprocess.check_output(["git", "-C", str(builder_repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip().lower()
            dirty = bool(subprocess.check_output(["git", "-C", str(builder_repo), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ExecutorError("cannot inspect builder Git authority") from exc
        if builder_head != self.builder_commit or dirty:
            raise ExecutorError(f"builder authority mismatch/dirty: {builder_head} dirty={dirty}")
        try:
            runtime_head = subprocess.check_output(["git", "-C", str(self.runtime_worktree), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip().lower()
            runtime_dirty = bool(subprocess.check_output(["git", "-C", str(self.runtime_worktree), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ExecutorError("cannot inspect runtime Git authority") from exc
        if runtime_head != self.runtime_commit or runtime_dirty:
            raise ExecutorError(f"runtime authority mismatch/dirty: {runtime_head} dirty={runtime_dirty}")
        # Keep every JIT/toolchain cache on Z for this execution.
        cache_root = self.toolchain_root
        cache_root.mkdir(parents=True, exist_ok=True)
        for name in ("TRITON_CACHE_DIR", "TRITON_HOME", "TRITON_DUMP_DIR", "TRITON_OVERRIDE_DIR", "FREETOKEN_KERNEL_CACHE_DIR", "TVM_FFI_CACHE_DIR", "XDG_CACHE_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TEMP", "TMP", "TMPDIR"):
            os.environ[name] = str(cache_root)
        extension = cache_root / "build-lib" / "pinned-only" / "freetoken" / "kernel" / "_pinned_tensor.cp312-win_amd64.pyd"
        extension_sha = "58c2ea0b7f74e457a48707eeb88012dfd28c943786e43cfa2896970e01f26b14"
        extension_ok = extension.is_file() and _sha256(extension).lower() == extension_sha
        if self.execute and not extension_ok:
            raise ExecutorError("resident HostBank pinning extension missing or hash mismatch")
        pin_probe = {"executed": False, "ok": extension_ok}
        if self.execute:
            probe_env = os.environ.copy()
            probe_env["PYTHONPATH"] = str(self.runtime_worktree / "python")
            probe_env["FREETOKEN_PINNED_EXTENSION_DIR"] = str(extension.parent)
            probe = subprocess.run(
                [sys.executable, "-c", "import os,torch,freetoken.kernel; freetoken.kernel.__path__.append(os.environ['FREETOKEN_PINNED_EXTENSION_DIR']); from freetoken.moe.host_banks import HostBank,HostResidency; from freetoken.kernel.pinned import device_ptr; b=HostBank((64,32),torch.bfloat16,backing='cuda'); assert b.residency is HostResidency.PINNED and b.addr%4096==0 and device_ptr(b.tensor)>0; del b; torch.cuda.synchronize()"],
                cwd=self.runtime_worktree,
                env=probe_env,
                text=True,
                capture_output=True,
                timeout=120,
            )
            if probe.returncode != 0:
                raise ExecutorError(f"resident HostBank pinning probe failed: {probe.stderr[-1000:]}")
            pin_probe = {"executed": True, "ok": True}
        host = self._host_gate()
        host_available, pagefile_used = host["available_bytes"], host["pagefile_used_bytes"]
        if pagefile_used > 0 and self.execute:
            # Swap use is evidence, not capacity; continue only if reserve is
            # still measured physically, but record the condition for audit.
            self.logger.event("pagefile_observed", bytes=pagefile_used)
        self.state["preflight"] = {"free_bytes": usage.free, "payload_bytes": 0, "manifest": str(self.manifest_path)}
        self.state["preflight"].update({"builder_head": builder_head, "runtime_head": runtime_head, "host_available": host_available, "pagefile_used": pagefile_used, "pinned_extension": str(extension), "pinned_extension_ok": extension_ok, "pin_probe": pin_probe, "cache_root": str(cache_root), "projected_required_free": required_free})
        _atomic_json(self.state_path, self.state)
        self.logger.event("preflight", free_bytes=usage.free, transfer_cap=self.manifest.transfer_cap)
        return self.state["preflight"]

    def _download_stage(self, stage: str, rows: Iterable[SourceEntry]) -> list[dict[str, Any]]:
        results = []
        for row in rows:
            self._disk_gate()
            self.logger.event("file_acquisition_start", filename=row.filename, stage=stage, planned_bytes=row.byte_length)
            result = self.downloader.acquire(row)
            results.append(result)
            self.logger.event("file_acquisition_end", filename=row.filename, stage=stage, state=result["state"], body_bytes=result.get("body_bytes", 0))
        self.state["stages"].setdefault(stage, {})["sources"] = results
        _atomic_json(self.state_path, self.state)
        return results

    def _remaining_weight_bytes(self) -> int:
        remaining = 0
        for row in self.manifest.entries:
            final = self.source_root / row.filename
            if final.is_file() and final.stat().st_size == row.byte_length:
                continue
            partial = self.source_root / f"{row.filename}.partial"
            present = partial.stat().st_size if partial.is_file() else 0
            remaining += max(0, row.byte_length - present)
        return remaining

    def _validate_source_workspace(self) -> None:
        if not self.source_root.exists():
            return
        allowed = {row.filename for row in self.manifest.all_entries}
        for path in self.source_root.iterdir():
            if path.is_dir() and path.name == ".step9b-receipts":
                continue
            if not path.is_file():
                raise ExecutorError(f"unexpected source workspace entry: {path.name}")
            name = path.name
            if name in allowed:
                continue
            if name.endswith(".partial") and name[:-8] in allowed:
                continue
            if name.endswith(".partial.meta.json") and name[:-18] in allowed:
                continue
            # Atomic JSON publication uses an executor-owned hidden sibling
            # (``.<destination>.partial-<pid>-<thread>``).  Keep these
            # restartable checkpoints in scope for orphan recovery; arbitrary
            # hidden JSON remains rejected below.
            if any(name.startswith(f".{item}.partial.meta.json.partial-") for item in allowed):
                continue
            raise ExecutorError(f"unexpected source workspace file: {name}")

    def _remaining_target_bytes(self) -> int:
        remaining = 0
        q3 = self.target_root / "ple-q3-000.bin"
        if not q3.is_file() or q3.stat().st_size != Q3_BYTES:
            remaining += Q3_BYTES
        for layer in range(48):
            sidecar = self.target_root / f"experts-L{layer:02d}.nvfp4"
            if not sidecar.is_file() or sidecar.stat().st_size != EXPERT_BYTES:
                remaining += EXPERT_BYTES
        active_index = self.target_root / "qwen4-active-v1.ftw"
        if not active_index.is_dir():
            remaining += ACTIVE_BYTES
        return remaining

    def _projected_required_free(self) -> int:
        return self._remaining_weight_bytes() + self._remaining_target_bytes() + COMMON_PEAK_ALLOWANCE_BYTES + self.min_disk_free

    def _disk_gate(self) -> int:
        usage = shutil.disk_usage(self.source_root.anchor or self.source_root)
        required = self._projected_required_free()
        if usage.free < required:
            raise ExecutorError(f"Z: projected reserve would be threatened: free={usage.free} required={required}")
        self.logger.event("disk_gate", free_bytes=usage.free, projected_required_bytes=required, reserve_bytes=self.min_disk_free)
        return usage.free

    def _host_gate(self) -> dict[str, int]:
        try:
            import psutil
        except ImportError:
            if os.name != "nt":
                raise ExecutorError("physical host reserve probe unavailable without psutil")
            command = (
                "$m=Get-CimInstance Win32_OperatingSystem;"
                "$pf=(Get-CimInstance Win32_PageFileUsage|Measure-Object CurrentUsage -Sum).Sum;"
                f"$p=Get-Process -Id {os.getpid()};"
                "[pscustomobject]@{total_bytes=[int64]$m.TotalVisibleMemorySize*1024;"
                "available_bytes=[int64]$m.FreePhysicalMemory*1024;"
                "process_rss_bytes=[int64]$p.WorkingSet64;"
                "pagefile_used_bytes=[int64]$pf*1MB}|ConvertTo-Json -Compress"
            )
            try:
                completed = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=True,
                )
                result = {key: int(value) for key, value in json.loads(completed.stdout).items()}
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ExecutorError("Windows physical host reserve probe failed") from exc
            available = result["available_bytes"]
        else:
            process = psutil.Process()
            available = int(psutil.virtual_memory().available)
            result = {"total_bytes": int(psutil.virtual_memory().total), "available_bytes": available, "process_rss_bytes": int(process.memory_info().rss), "pagefile_used_bytes": int(psutil.swap_memory().used)}
        if available < self.min_host_free:
            raise ExecutorError(f"physical host reserve below gate: {available} < {self.min_host_free}")
        self.logger.event("host_gate", **result, minimum_available_bytes=self.min_host_free)
        return result

    def acquire_metadata(self) -> list[dict[str, Any]]:
        rows = self._download_stage("B1", self.manifest.metadata)
        expected = sum(row.byte_length for row in self.manifest.metadata)
        if expected != 57_176_714:
            raise ExecutorError("metadata byte cap mismatch")
        if self.execute:
            receipt_path = self.scratch_root / "receipts" / "B1-metadata.json"
            source_hashes = {row.filename: self.downloader.validate_existing(row, self.source_root / row.filename)["sha256"] for row in self.manifest.metadata}
            binding = {"stage": "B1", "files": len(rows), "bytes": expected, "source_inventory_fingerprint": self.source_inventory_fingerprint, "source_revision": self.manifest.revision, "builder_commit": self.builder_commit, "source_hashes": source_hashes}
            if not _receipt_matches(receipt_path, binding):
                _publish_component_receipt(receipt_path, {**binding, "validation": {"all_source_receipts": True, "metadata_total": expected}})
        return rows

    def acquire_ple(self) -> list[dict[str, Any]]:
        rows = self.manifest.rows_for_stage("B2")
        if len(rows) != 10 or sum(row.byte_length for row in rows) != 51_200_267_901:
            raise ExecutorError("PLE source inventory mismatch")
        return self._download_stage("B2", rows)

    def acquire_expert_layer(self, layer: int) -> list[dict[str, Any]]:
        rows = tuple(row for row in self.manifest.entries if row.source_class.upper() == "EXPERT" and row.layer_id == layer)
        if len(rows) != 4:
            raise ExecutorError(f"layer {layer} requires exactly four source rows")
        return self._download_stage(f"B3-L{layer:02d}", rows)

    def acquire_active(self) -> list[dict[str, Any]]:
        rows = self.manifest.rows_for_stage("B4")
        if len(rows) != 4 or sum(row.byte_length for row in rows) != 16_007_756_462:
            raise ExecutorError("BF16 source inventory mismatch")
        return self._download_stage("B4", rows)

    def _source_bindings(self, rows: Iterable[SourceEntry]) -> list[dict[str, Any]]:
        """Return receipt-backed source hashes for a component transaction."""
        bindings: list[dict[str, Any]] = []
        for row in rows:
            current = self.downloader.validate_existing(row, self.source_root / row.filename)
            receipt = self.scratch_root / "receipts" / "sources" / f"{row.acquisition_order:03d}-{row.filename}.receipt.json"
            try:
                with receipt.open("r", encoding="utf-8") as handle:
                    value = json.load(handle)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ExecutorError(f"{row.filename}: source receipt unavailable for component binding") from exc
            receipt_entry = dict(value.get("entry") or {})
            if isinstance(receipt_entry.get("allowed_body_etags"), list):
                receipt_entry["allowed_body_etags"] = tuple(receipt_entry["allowed_body_etags"])
            if (value.get("completion") != "SOURCE_COMPLETE"
                    or receipt_entry != asdict(row)
                    or value.get("source_inventory_fingerprint") != self.source_inventory_fingerprint
                    or value.get("resolved_commit") != self.manifest.revision
                    or int(value.get("bytes", -1)) != row.byte_length
                    or str(value.get("sha256", "")).lower() != str(current["sha256"]).lower()):
                raise ExecutorError(f"{row.filename}: source receipt cannot bind component input")
            bindings.append({"filename": row.filename, "bytes": row.byte_length, "sha256": str(current["sha256"]).lower()})
        return bindings

    def dry_run_plan(self) -> dict[str, Any]:
        staged_rows = [*self.manifest.metadata, *self.manifest.rows_for_stage("B2")]
        for layer in range(48):
            staged_rows.extend(row for row in self.manifest.entries if row.source_class.upper() == "EXPERT" and row.layer_id == layer)
        staged_rows.extend(self.manifest.rows_for_stage("B4"))
        source_receipt_root = self.scratch_root / "receipts" / "sources"
        component_receipts = [
            self.scratch_root / "receipts" / "B1-metadata.json",
            self.scratch_root / "receipts" / "B2-q3.json",
            *(self.scratch_root / "receipts" / f"B3-L{layer:02d}.json" for layer in range(48)),
            self.scratch_root / "receipts" / "B4-active.json",
            self.scratch_root / "receipts" / "B5-manifest.json",
            self.scratch_root / "receipts" / "C6-static.json",
        ]
        return {
            "mode": "DRY_RUN",
            "body_requests": 0,
            "real_model_payload_bytes": 0,
            "metadata_files": len(self.manifest.metadata),
            "weight_files": len(self.manifest.entries),
            "weight_bytes": self.manifest.expected_weight_bytes,
            "metadata_bytes": sum(row.byte_length for row in self.manifest.metadata),
            "transfer_cap": self.manifest.transfer_cap,
            "known_target_bytes": KNOWN_TARGET_BYTES,
            "projected_required_free": self._projected_required_free(),
            "disk_reserve_bytes": self.min_disk_free,
            "host_reserve_bytes": self.min_host_free,
            "max_concurrent_downloads": self.downloader.max_concurrent,
            "stage_order": ["B1", "B2", "B3-L00..L47", "B4", "B5", "C6"],
            "expert_layers": list(range(48)),
            "planned_files": [row.filename for row in staged_rows],
            "source_receipts": [str(source_receipt_root / f"{row.acquisition_order:03d}-{row.filename}.receipt.json") for row in staged_rows],
            "component_receipts": [str(path) for path in component_receipts],
        }

    def convert_and_validate_q3(self) -> dict[str, Any]:
        from freetoken.checkpoint.q3_ple import plan_q3_ple_production
        plan = plan_q3_ple_production()
        if plan["segment_count"] != 128 or plan["total_bytes"] != Q3_BYTES:
            raise ExecutorError("Q3 production plan mismatch")
        if not self.execute:
            return {
                "format": "q3_ple_32",
                **plan,
                "source_layer_id": PRODUCTION_PLE_SOURCE_LAYER_ID,
                "state": "PLANNED",
            }
        self._disk_gate()
        self._host_gate()
        self.target_root.mkdir(parents=True, exist_ok=True)
        from freetoken.checkpoint.q3_ple import (
            DEFAULT_PROCESSING_CHUNK_ROWS,
            Q3PLEReader,
            write_q3_ple_from_safetensors,
        )
        data_path = self.target_root / "ple-q3-000.bin"
        manifest_path = self.target_root / "ple-q3.json"
        receipt_path = self.scratch_root / "receipts" / "B2-q3.json"
        result: dict[str, Any] = {}
        if data_path.exists() != manifest_path.exists():
            raise ExecutorError("incomplete Q3 final target pair")
        if not data_path.exists():
            ple_source_layer_id = _resolve_production_ple_source_layer(
                self.source_root / "model.safetensors.index.json"
            )
            result = write_q3_ple_from_safetensors(
                self.source_root,
                data_path,
                manifest_path,
                layer_id=ple_source_layer_id,
                split_parts=PRODUCTION_PLE_SEGMENT_COUNT,
                source_fingerprint=self.source_inventory_fingerprint,
                rows_per_segment=2_500_012,
                processing_chunk_rows=DEFAULT_PROCESSING_CHUNK_ROWS,
            )
        if data_path.stat().st_size != Q3_BYTES:
            raise ExecutorError("Q3 target extent mismatch")
        with Q3PLEReader(manifest_path) as reader:
            if int(reader.manifest.get("segment_count", -1)) != 128:
                raise ExecutorError("Q3 manifest must contain exactly 128 logical segments")
            reader.gather([0, 2_500_011, 2_500_012, 160_000_768, 317_501_524, 320_001_535])
        source_inputs = self._source_bindings(self.manifest.rows_for_stage("B2"))
        expected = {"stage": "B2", "format": "q3_ple_32", "target": str(data_path), "target_bytes": Q3_BYTES, "target_sha256": _sha256(data_path), "manifest_sha256": _sha256(manifest_path), "source_inventory_fingerprint": self.source_inventory_fingerprint, "source_revision": self.manifest.revision, "builder_commit": self.builder_commit, "source_inputs": source_inputs}
        recovered = not bool(result)
        if not _receipt_matches(receipt_path, expected):
            _publish_component_receipt(receipt_path, {**expected, "validation": {"reopen": True, "segment_count": 128, "sample_rows": [0, 2_500_011, 2_500_012, 160_000_768, 317_501_524, 320_001_535]}, "recovered_after_promotion": recovered})
        return {**expected, "state": "COMPLETE", "recovered_after_promotion": recovered}

    def convert_and_validate_expert(self, layer: int) -> dict[str, Any]:
        if not 0 <= int(layer) < 48:
            raise ValueError("expert layer outside 0..47")
        if not self.execute:
            return {"layer": int(layer), "format": "ftexpert1_nvfp4_v1", "target_bytes": EXPERT_BYTES, "state": "PLANNED"}
        self._disk_gate()
        self._host_gate()
        self.target_root.mkdir(parents=True, exist_ok=True)
        from freetoken.moe.expert_source import FileExpertSource, write_expert_sidecar_from_safetensors
        path = self.target_root / f"experts-L{int(layer):02d}.nvfp4"
        receipt_path = self.scratch_root / "receipts" / f"B3-L{int(layer):02d}.json"
        created = False
        if not path.exists():
            write_expert_sidecar_from_safetensors(self.source_root, path, layer_id=int(layer), source_fingerprint=self.source_inventory_fingerprint)
            created = True
        if path.stat().st_size != EXPERT_BYTES:
            raise ExecutorError(f"expert layer {layer} target extent mismatch")
        with FileExpertSource(path, expected_source_fingerprint=self.manifest.source_inventory_fingerprint, expected_layer_id=int(layer), verify_hash=True) as source:
            source.read_records([0, 511])
        layer_rows = tuple(row for row in self.manifest.entries if row.source_class.upper() == "EXPERT" and row.layer_id == int(layer))
        expected = {"stage": f"B3-L{int(layer):02d}", "layer": int(layer), "format": "ftexpert1_nvfp4_v1", "target": str(path), "target_bytes": EXPERT_BYTES, "target_sha256": _sha256(path), "source_inventory_fingerprint": self.source_inventory_fingerprint, "source_revision": self.manifest.revision, "builder_commit": self.builder_commit, "source_inputs": self._source_bindings(layer_rows)}
        if not _receipt_matches(receipt_path, expected):
            _publish_component_receipt(receipt_path, {**expected, "validation": {"reopen": True, "sample_experts": [0, 511]}, "recovered_after_promotion": not created})
        return {**expected, "state": "COMPLETE", "recovered_after_promotion": not created}

    def convert_and_validate_active(self) -> dict[str, Any]:
        if not self.execute:
            return {"format": "nvfp4_w4a16_v1", "target_bytes": ACTIVE_BYTES, "state": "PLANNED"}
        self._disk_gate()
        self._host_gate()
        self.target_root.mkdir(parents=True, exist_ok=True)
        from freetoken.checkpoint.convert import convert_checkpoint
        active = self.target_root / "qwen4-active-v1.ftw"
        receipt_path = self.scratch_root / "receipts" / "B4-active.json"
        result: dict[str, Any] = {}
        if not active.exists():
            result = convert_checkpoint(str(self.source_root), str(self.target_root), artifact_format="qwen4_modular_v1", source_inventory_sha256=self.source_inventory_fingerprint)
        from freetoken.checkpoint.ftw import INDEX_NAME
        index = active / INDEX_NAME
        if not active.is_dir() or not index.is_file():
            raise ExecutorError("active FTW was not created")
        with index.open("r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        if int(index_data.get("total_bytes", -1)) != ACTIVE_BYTES:
            raise ExecutorError("active FTW extent mismatch")
        tree_bytes, tree_sha = _tree_sha256(active)
        expected = {"stage": "B4", "format": "nvfp4_w4a16_v1", "target": str(active), "target_bytes": ACTIVE_BYTES, "target_tree_bytes": tree_bytes, "target_sha256": tree_sha, "source_inventory_fingerprint": self.source_inventory_fingerprint, "source_revision": self.manifest.revision, "builder_commit": self.builder_commit, "runtime_commit": self.runtime_commit, "source_inputs": self._source_bindings(self.manifest.rows_for_stage("B4"))}
        if not _receipt_matches(receipt_path, expected):
            _publish_component_receipt(receipt_path, {**expected, "copied_metadata": list(result.get("copied_metadata", ())), "validation": {"ftw_index": True}, "recovered_after_promotion": not bool(result)})
        return {**expected, "state": "COMPLETE", "recovered_after_promotion": not bool(result)}

    def finalize_artifact(self) -> dict[str, Any]:
        if not self.execute:
            return {"known_target_bytes": KNOWN_TARGET_BYTES, "reconciliation_error": 0, "state": "PLANNED"}
        self._disk_gate()
        self._host_gate()
        self.target_root.mkdir(parents=True, exist_ok=True)
        from freetoken.checkpoint.qwen4_artifact import finalize_qwen4_modular_manifest
        manifest_path = self.target_root / "manifest.json"
        receipt_path = self.scratch_root / "receipts" / "B5-manifest.json"
        expert_paths = {layer: f"experts-L{layer:02d}.nvfp4" for layer in range(48)}
        metadata = []
        active_receipt = self.scratch_root / "receipts" / "B4-active.json"
        if active_receipt.is_file():
            with active_receipt.open("r", encoding="utf-8") as handle:
                copied = (json.load(handle).get("result") or {}).get("copied_metadata", ())
            metadata = [str(value) for value in copied if value and (self.target_root / str(value)).is_file()]
        if not metadata:
            metadata = [str(path.relative_to(self.target_root)) for path in sorted(self.target_root.iterdir()) if path.is_file() and path.name not in {"manifest.json", "ple-q3.json", "ple-q3-000.bin", "expert-placement.json"}]
        created = False
        if not manifest_path.exists():
            finalize_qwen4_modular_manifest(self.target_root, source_repository=self.manifest.repository, source_revision=self.manifest.revision, source_inventory_sha256=self.source_inventory_fingerprint, minimum_freetoken_commit=self.builder_commit, tvm_ffi_patch_sha256="889310b8152a147a6552a3e451b3251a7df70cdc8e6e4c1c87c7adf3854182ec", expert_paths=expert_paths, file_tier_layers=(0, 1, 2, 3, 4, 5, 42, 43, 44, 45, 46, 47), metadata_paths=metadata)
            created = True
        from freetoken.checkpoint.qwen4_artifact import load_qwen4_artifact_manifest
        artifact = load_qwen4_artifact_manifest(self.target_root, require=True)
        artifact.verify_active()
        expected = {"stage": "B5", "known_target_bytes": KNOWN_TARGET_BYTES, "reconciliation_error": 0, "builder_commit": self.builder_commit, "runtime_commit": self.runtime_commit, "manifest_sha256": _sha256(manifest_path), "manifest_fingerprint": artifact.raw.get("complete_artifact_fingerprint"), "source_inventory_fingerprint": self.source_inventory_fingerprint}
        if not _receipt_matches(receipt_path, expected):
            _publish_component_receipt(receipt_path, {**expected, "source_inputs": ["B2", *[f"B3-L{layer:02d}" for layer in range(48)], "B4"], "validation": {"manifest_reopen": True, "active": True, "experts": 48, "q3": True}, "recovered_after_promotion": not created})
        return {**expected, "state": "COMPLETE", "recovered_after_promotion": not created}

    def run_c6_static_reopen(self) -> dict[str, Any]:
        if not self.runtime_worktree.is_dir():
            raise ExecutorError("runtime worktree missing")
        if not self.execute:
            return {"state": "PLANNED", "runtime_worktree": str(self.runtime_worktree), "runtime_commit": self.runtime_commit, "inference": False}
        self._disk_gate()
        self._host_gate()
        runtime_python = self.runtime_worktree / "python"
        extension_root = self.toolchain_root / "build-lib" / "pinned-only"
        script = r"""
import json, os
from pathlib import Path
from types import SimpleNamespace
import torch, freetoken.kernel
freetoken.kernel.__path__.append(os.environ['FREETOKEN_PINNED_EXTENSION_DIR'])
from freetoken.checkpoint.qwen4_artifact import load_qwen4_artifact_manifest, load_qwen4_expert_placement_policy
from freetoken.models.qwen4_exp.ple import Q3PLEFileTable
from freetoken.moe.expert_source import FileExpertSource
from freetoken.moe.host_banks import HostBank, HostResidency
from freetoken.kernel.pinned import device_ptr
from freetoken.engine.engine import _apply_pr257_hardware_fit_policy
root = Path(os.environ['FREETOKEN_C6_ARTIFACT'])
manifest = load_qwen4_artifact_manifest(root, require=True)
if not manifest.is_pr257_hardware_fit or not manifest.production_geometry:
    raise RuntimeError('marked runtime foundation not recognized')
if manifest.raw.get('runtime_foundation') != 'pr257_hardware_fit_v1' or not manifest.text_only or manifest.active_format != 'nvfp4_w4a16_v1':
    raise RuntimeError('hardware-fit manifest markers mismatch')
manifest.verify_active()
policy = load_qwen4_expert_placement_policy(manifest)
expected_tier = (0,1,2,3,4,5,42,43,44,45,46,47)
if policy.file_tier_layers != expected_tier or len(policy.resident_layers) != 36 or policy.file_expert_queue_depth != 4:
    raise RuntimeError('placement policy mismatch')
table = Q3PLEFileTable(str(manifest.ple_manifest_path), expected_sha256=manifest.ple_sha256, expected_source_fingerprint=manifest.source['inventory_sha256'])
table.close()
for layer in policy.file_tier_layers:
    entry = manifest.file_for_layer(layer)
    source = FileExpertSource(entry.path, expected_source_fingerprint=manifest.source['inventory_sha256'], expected_layer_id=layer, max_queue_depth=4, verify_hash=False)
    if source.requested_queue_depth != 4:
        raise RuntimeError('file source queue depth mismatch')
    source.close()
config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
if config.get('freetoken_runtime_foundation') != 'pr257_hardware_fit_v1' or config.get('freetoken_text_only') != 'qwen4_text_only_v1' or config.get('freetoken_active_quant') != 'nvfp4_w4a16_v1':
    raise RuntimeError('config markers mismatch')
cfg = SimpleNamespace(model_config=SimpleNamespace(freetoken_runtime_foundation='pr257_hardware_fit_v1'), moe_prefill_overlap=True, cuda_graph_bs=None, cuda_graph_max_bs=None)
if not _apply_pr257_hardware_fit_policy(cfg, graph_requested=False) or cfg.cuda_graph_bs != [] or cfg.cuda_graph_max_bs != 0 or cfg.moe_prefill_overlap is not False:
    raise RuntimeError('eager graph policy mismatch')
try:
    _apply_pr257_hardware_fit_policy(cfg, graph_requested=True)
except ValueError:
    pass
else:
    raise RuntimeError('forced graph policy did not fail closed')
bank = HostBank((64, 32), torch.bfloat16, backing='cuda')
if bank.residency is not HostResidency.PINNED or bank.addr % 4096 or device_ptr(bank.tensor) <= 0:
    raise RuntimeError('resident pin capability unavailable')
del bank
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(json.dumps({'status':'C6_STATIC_OK','experts':len(manifest.expert_files),'file_tier':len(policy.file_tier_layers),'resident':len(policy.resident_layers),'queue_depth':policy.file_expert_queue_depth,'graphs':cfg.cuda_graph_bs,'prefill_overlap':cfg.moe_prefill_overlap,'inference':False}, sort_keys=True))
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(runtime_python), str(extension_root)))
        env["FREETOKEN_C6_ARTIFACT"] = str(self.target_root)
        env["FREETOKEN_C6_NO_INFERENCE"] = "1"
        env["FREETOKEN_PINNED_EXTENSION_DIR"] = str(extension_root / "freetoken" / "kernel")
        proc = subprocess.run([sys.executable, "-c", script], env=env, cwd=str(self.runtime_worktree), text=True, capture_output=True, timeout=600)
        if proc.returncode != 0 or "C6_STATIC_OK" not in proc.stdout:
            raise ExecutorError(f"isolated C6 static reopen failed: {proc.stderr[-1000:]}")
        receipt = {"state": "COMPLETE", "runtime_worktree": str(self.runtime_worktree), "runtime_commit": self.runtime_commit, "artifact": str(self.target_root), "inference": False, "stdout": proc.stdout.strip()}
        _atomic_json(self.scratch_root / "receipts" / "C6-static.json", receipt)
        return receipt

    def closeout(self) -> dict[str, Any]:
        result = {"real_model_payload_bytes": 0 if not self.execute else self.budget.transferred, "source_retirement_authorized": False, "transfer_bytes": self.budget.transferred, "state": "CLOSED"}
        self.state["closeout"] = result
        _atomic_json(self.state_path, self.state)
        return result

    def run(self) -> dict[str, Any]:
        """Run explicit stage boundaries.  ``DRY_RUN`` never opens a body GET."""
        if not self.execute:
            self.preflight()
            return self.dry_run_plan()
        self.preflight()
        self.acquire_metadata()
        self.acquire_ple()
        self.convert_and_validate_q3()
        for layer in range(48):
            self.acquire_expert_layer(layer)
            self.convert_and_validate_expert(layer)
        self.acquire_active()
        self.convert_and_validate_active()
        result = self.finalize_artifact()
        result["c6"] = self.run_c6_static_reopen()
        return self.closeout() | result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe Step 9B staged acquisition/conversion controller")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--logs-root", required=True)
    parser.add_argument("--builder-commit", required=True)
    parser.add_argument("--runtime-worktree", required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-inventory-fingerprint", required=True)
    parser.add_argument("--transfer-cap", required=True, type=int)
    parser.add_argument("--toolchain-root", required=True)
    parser.add_argument("--dry-run", action="store_true", help="plan only (default)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-network-body", action="store_true")
    parser.add_argument("--max-concurrent-downloads", type=int, default=2)
    parser.add_argument("--min-disk-reserve", type=int, required=True)
    parser.add_argument("--min-host-reserve", type=int, required=True)
    parser.add_argument("--source-retirement-policy", choices=("false", "true"), default="false")
    return parser


def main(argv: list[str] | None = None) -> int:
    ns = _parser().parse_args(argv)
    execute = bool(ns.execute and not ns.dry_run)
    executor = Step9BExecutor(ns.manifest, ns.source_root, ns.target_root, ns.scratch_root, ns.logs_root, builder_commit=ns.builder_commit, runtime_worktree=ns.runtime_worktree, runtime_commit=ns.runtime_commit, source_revision=ns.source_revision, source_inventory_fingerprint=ns.source_inventory_fingerprint, transfer_cap=ns.transfer_cap, toolchain_root=ns.toolchain_root, execute=execute, allow_network_body=ns.allow_network_body, source_retirement_authorized=(ns.source_retirement_policy == "true"), min_disk_free=ns.min_disk_reserve, min_host_free=ns.min_host_reserve, max_concurrent_downloads=ns.max_concurrent_downloads)
    result = executor.run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ACQUISITION_MANIFEST_V1",
    "ACQUISITION_MANIFEST_V2",
    "AcquisitionManifest",
    "BodyTransferDisabled",
    "Downloader",
    "ExecutorError",
    "JsonlLogger",
    "MAX_TRANSFER_BYTES",
    "SourceEntry",
    "Step9BExecutor",
    "TransferBudget",
    "UrllibTransport",
    "generate_acquisition_manifest_v2",
    "migrate_acquisition_manifest_v1_to_v2",
    "migrate_source_receipt_v1_to_v2",
    "normalize_source_entry_mapping",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
