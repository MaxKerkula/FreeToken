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
ACCEPTED_SOURCE_INVENTORY = "8572d200e31b344faff0fda f0dc72aa4726c1f062443d4109531b62ca63f66eb".replace(" ", "")


class ExecutorError(RuntimeError):
    """A stop-gate or validation failure; callers must preserve evidence."""


class BodyTransferDisabled(ExecutorError):
    """Raised before a network body request when explicit authorization is absent."""


class ResumeRejected(ExecutorError):
    """Raised when the server cannot prove an identity-safe range response."""


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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
    os.replace(partial, path)


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

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, order: int | None = None, metadata: bool = False) -> "SourceEntry":
        filename = str(raw["filename"])
        if not filename or Path(filename).name != filename or filename in {".", ".."}:
            raise ExecutorError(f"unsafe manifest filename: {filename!r}")
        repository = str(raw.get("repository", PINNED_REPOSITORY))
        revision = str(raw.get("revision", PINNED_REVISION))
        if repository != PINNED_REPOSITORY or revision != PINNED_REVISION:
            raise ExecutorError(f"source identity mismatch for {filename}")
        return cls(
            filename=filename,
            byte_length=int(raw["byte_length"]),
            source_class=("METADATA" if metadata else str(raw["source_class"])),
            acquisition_order=int(raw.get("acquisition_order", order or 0)),
            repository=repository,
            revision=revision,
            accepted_etag=(raw.get("accepted_etag") or (raw.get("git_blob_id") if metadata else None)),
            lfs_oid_sha256=raw.get("lfs_oid_sha256"),
            accepted_header_length=raw.get("accepted_header_length"),
            accepted_header_sha256=raw.get("accepted_header_sha256"),
            layer_id=(None if raw.get("layer_id") is None else int(raw["layer_id"])),
            tensor_payload_bytes=(None if raw.get("tensor_payload_bytes") is None else int(raw["tensor_payload_bytes"])),
            git_blob_id=(None if raw.get("git_blob_id") is None else str(raw["git_blob_id"])),
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

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "AcquisitionManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("schema") != "freetoken-step9-acquisition-v1":
            raise ExecutorError("unsupported acquisition manifest schema")
        repo, revision = str(raw.get("repository")), str(raw.get("revision"))
        if repo != PINNED_REPOSITORY or revision != PINNED_REVISION:
            raise ExecutorError("manifest source pin does not match the frozen revision")
        rows = tuple(SourceEntry.from_mapping(row) for row in raw.get("source_weight_shards", ()))
        metadata = tuple(SourceEntry.from_mapping(row, order=i + 1, metadata=True) for i, row in enumerate(raw.get("required_small_metadata", ())))
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
        return cls(repo, revision, rows, metadata, inventory, expected, MAX_TRANSFER_BYTES)

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

    def _url(self, row: SourceEntry) -> str:
        return f"https://huggingface.co/{row.repository}/resolve/{row.revision}/{row.filename}"

    def cancel(self) -> None:
        self._cancel.set()

    def _identity(self, row: SourceEntry, length: int, remote: Mapping[str, Any] | None = None) -> dict[str, Any]:
        remote = dict(remote or {})
        return {"repository": row.repository, "revision": row.revision, "resolved_commit": remote.get("commit") or remote.get("resolved_commit") or row.revision, "filename": row.filename, "expected_length": row.byte_length, "expected_etag": row.accepted_etag, "expected_lfs_oid": row.lfs_oid_sha256, "expected_header_length": row.accepted_header_length, "expected_header_sha256": row.accepted_header_sha256, "observed_etag": remote.get("etag"), "observed_xet_file_hash": remote.get("xet_file_hash"), "partial_length": length, "source_inventory_fingerprint": self.manifest.source_inventory_fingerprint, "acquisition_order": row.acquisition_order}

    def validate_metadata(self, row: SourceEntry, response: TransportResponse) -> dict[str, Any]:
        headers = {k.lower(): v for k, v in response.headers.items()}
        observed_length = int(headers.get("content-length", "-1"))
        observed_etag = headers.get("etag")
        if observed_length != row.byte_length:
            raise ExecutorError(f"{row.filename}: content length mismatch")
        # Hugging Face uses two identities for Xet/LFS files.  The manifest's
        # accepted_etag is the Xet file hash (often quoted), while lfs_oid is
        # surfaced as HfFileMetadata.etag and may be returned as HTTP ETag by
        # the CDN.  A transport may expose either, so accept only either exact
        # frozen identity and never a merely non-empty header.
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
        if row.lfs_oid_sha256:
            if etag != row.lfs_oid_sha256.lower():
                raise ExecutorError(f"{row.filename}: metadata LFS OID mismatch")
            if row.accepted_etag and xet_hash != row.accepted_etag.strip('"').lower():
                raise ExecutorError(f"{row.filename}: metadata Xet hash mismatch")
        elif row.accepted_etag:
            # Git-backed metadata has no Xet data; accepted_etag is the blob id.
            if etag != row.accepted_etag.strip('"').lower():
                raise ExecutorError(f"{row.filename}: metadata Git identity mismatch")
        return {"url": url, "commit": commit, "size": size, "etag": etag, "xet_file_hash": xet_hash or None, "body_bytes": 0}

    def _validate_partial_identity(self, row: SourceEntry, meta: Path, length: int, remote: Mapping[str, Any]) -> None:
        if not meta.is_file():
            raise ResumeRejected(f"partial identity sidecar missing for {row.filename}")
        with meta.open("r", encoding="utf-8") as handle:
            identity = json.load(handle)
        expected = self._identity(row, length, remote)
        for key in ("repository", "revision", "resolved_commit", "filename", "expected_length", "expected_etag", "expected_lfs_oid", "observed_etag", "observed_xet_file_hash", "source_inventory_fingerprint", "acquisition_order"):
            if identity.get(key) != expected.get(key):
                raise ResumeRejected(f"partial identity mismatch: {key}")
        if int(identity.get("partial_length", -1)) != length:
            raise ResumeRejected("partial length identity mismatch")

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
            "observed_etag": remote_meta.get("etag"),
            "observed_xet_file_hash": remote_meta.get("xet_file_hash"),
            "expected_lfs_oid": row.lfs_oid_sha256,
            "observed_lfs_oid": row.lfs_oid_sha256,
            "observed_header_length": row.accepted_header_length,
            "observed_header_sha256": row.accepted_header_sha256,
            "recovered_complete_partial": recovered_complete_partial,
        }
        _atomic_json(receipt, {"entry": asdict(row), **result, "source_inventory_fingerprint": self.manifest.source_inventory_fingerprint, "completion": "SOURCE_COMPLETE"})
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
            receipt_binding = {
                "entry": asdict(row),
                "final_path": str(final),
                "expected_bytes": row.byte_length,
                "bytes": row.byte_length,
                "resolved_commit": row.revision,
                "expected_etag": row.accepted_etag,
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
                    if prior.get("completion") != "SOURCE_COMPLETE" or any(prior.get(key) != value for key, value in receipt_binding.items()):
                        raise ExecutorError(f"{row.filename}: source receipt binding mismatch")
                    accepted_observed = {str(value).strip('"').lower() for value in (row.accepted_etag, row.lfs_oid_sha256) if value}
                    if str(prior.get("observed_etag", "")).strip('"').lower() not in accepted_observed:
                        raise ExecutorError(f"{row.filename}: source receipt ETag binding mismatch")
                    if row.lfs_oid_sha256 and str(prior.get("observed_xet_file_hash", "")).strip('"').lower() != str(row.accepted_etag).strip('"').lower():
                        raise ExecutorError(f"{row.filename}: source receipt Xet binding mismatch")
                    resumed = prior.get("resumed_from")
                    if resumed is not None and not 0 <= int(resumed) <= row.byte_length:
                        raise ExecutorError(f"{row.filename}: invalid source receipt resume offset")
                    if int(prior.get("body_bytes", -1)) < 0:
                        raise ExecutorError(f"{row.filename}: invalid source receipt body byte count")
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ExecutorError(f"{row.filename}: invalid source receipt") from exc
            else:
                _atomic_json(receipt, {**receipt_binding, "observed_etag": row.lfs_oid_sha256 or row.accepted_etag, "observed_xet_file_hash": row.accepted_etag, "resumed_from": None, "body_bytes": 0, "completion": "SOURCE_COMPLETE", "recovered_after_promotion": True})
            return {"filename": row.filename, **result, "state": "SKIP_VALID_FINAL"}
        if self.execute and not self.allow_network_body:
            raise BodyTransferDisabled("execution mode requires explicit network-body authorization")
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
            raise ResumeRejected(f"partial and identity sidecar must exist together for {row.filename}")
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
        if current:
            validated_etag = str(remote_meta.get("etag") or "")
            if not validated_etag:
                raise ResumeRejected("validated remote ETag unavailable for If-Range")
            headers = {"Range": f"bytes={current}-", "If-Range": validated_etag}
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
            accepted = {str(value).strip('"').lower() for value in (row.accepted_etag, row.lfs_oid_sha256) if value}
            if accepted and not etag:
                raise ExecutorError("body response omitted required ETag identity")
            if etag and accepted and etag.strip('"').lower() not in accepted:
                raise ExecutorError("body ETag/Xet identity changed")
            mode = "ab" if current else "wb"
            with partial.open(mode) as target:
                if not current:
                    _atomic_json(identity, self._identity(row, 0, remote_meta))
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
                    _atomic_json(identity, self._identity(row, current + received, remote_meta))
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
        self.manifest = AcquisitionManifest.load(manifest_path)
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
        if self.execute and not self.allow_network_body:
            raise BodyTransferDisabled("--execute requires --allow-network-body")
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
            if (value.get("completion") != "SOURCE_COMPLETE"
                    or value.get("entry") != asdict(row)
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
            return {"format": "q3_ple_32", **plan, "state": "PLANNED"}
        self._disk_gate()
        self._host_gate()
        self.target_root.mkdir(parents=True, exist_ok=True)
        from freetoken.checkpoint.q3_ple import Q3PLEReader, write_q3_ple_from_safetensors
        data_path = self.target_root / "ple-q3-000.bin"
        manifest_path = self.target_root / "ple-q3.json"
        receipt_path = self.scratch_root / "receipts" / "B2-q3.json"
        result: dict[str, Any] = {}
        if data_path.exists() != manifest_path.exists():
            raise ExecutorError("incomplete Q3 final target pair")
        if not data_path.exists():
            result = write_q3_ple_from_safetensors(self.source_root, data_path, manifest_path, layer_id=2, split_parts=128, source_fingerprint=self.source_inventory_fingerprint, rows_per_segment=2_500_012, processing_chunk_rows=8192)
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


__all__ = ["AcquisitionManifest", "BodyTransferDisabled", "Downloader", "ExecutorError", "JsonlLogger", "MAX_TRANSFER_BYTES", "SourceEntry", "Step9BExecutor", "TransferBudget", "UrllibTransport", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
