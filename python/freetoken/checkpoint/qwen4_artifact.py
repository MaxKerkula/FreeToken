"""Validation and runtime wiring for the Qwen4 modular artifact.

The modular artifact is intentionally a small manifest around three independently
validated pieces: native active weights, a Q3 PLE sidecar, and a mixed resident/file
expert tier.  This module owns only the manifest contract and the wiring seam.  The
large writers and the byte-level readers live in their existing modules.

An absent ``manifest.json`` is not an error and keeps the normal Qwen4 checkpoint path
unchanged.  Once the marker is present, malformed or unknown values fail closed rather
than silently falling back to a different representation.
"""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FORMAT = "freetoken-qwen4-modular-v1"
VERSION = 1
TEXT_ONLY_MARKER = "qwen4_text_only_v1"
ARTIFACT_FORMAT = "qwen4_modular_v1"
ACTIVE_FORMAT = "nvfp4_w4a16_v1"
PLE_FORMAT = "q3_ple_32"
EXPERT_FORMAT = "ftexpert1_nvfp4_v1"
REQUIRED_VOLUME = "Z:"
MANIFEST_NAME = "manifest.json"
ACTIVE_TARGET_BYTES = 4_804_403_200
PLE_TARGET_BYTES = 22_400_107_520
EXPERT_FILE_BYTES = 1_419_776_000
EXPERT_LAYERS = 48
EXPERT_NUM_EXPERTS = 512
KNOWN_TARGET_BYTES = 95_353_758_720
FILE_TIER_LAYERS = (0, 1, 2, 3, 4, 5, 42, 43, 44, 45, 46, 47)
PINNED_SOURCE_REPOSITORY = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
PINNED_SOURCE_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
TVM_FFI_PATCH_SHA256 = "889310b8152a147a6552a3e451b3251a7df70cdc8e6e4c1c87c7adf3854182ec"
RUNTIME_FOUNDATION_MARKER = "pr257_hardware_fit_v1"


class Qwen4ArtifactError(ValueError):
    """Raised when a Qwen4 modular manifest cannot be trusted."""


def _resolve_z(path: str | os.PathLike[str], *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    drive = (resolved.drive or os.path.splitdrive(str(resolved))[0]).upper()
    if drive != REQUIRED_VOLUME:
        raise Qwen4ArtifactError(f"{label} must resolve to Z:, got {resolved}")
    return resolved


def _path_from(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise Qwen4ArtifactError(f"{label} must be a non-empty path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return _resolve_z(candidate, label=label)


def _path_within_root(root: Path, value: object, *, label: str) -> Path:
    path = _path_from(root, value, label=label)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Qwen4ArtifactError(f"{label} resolves outside artifact root") from exc
    return path


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Qwen4ArtifactError(f"{label} must be an object")
    return value


def _require_sha(value: object, *, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise Qwen4ArtifactError(f"{label} must be a SHA-256 hex digest")
    return text


@dataclass(frozen=True)
class ExpertFile:
    layer: int
    path: Path
    bytes: int
    sha256: str
    source_fingerprint: str


@dataclass(frozen=True)
class ComponentFile:
    path: Path
    bytes: int
    sha256: str


def _sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_component_file(entry: ComponentFile, *, label: str) -> None:
    if not entry.path.is_file():
        raise Qwen4ArtifactError(f"{label} is missing: {entry.path}")
    actual_bytes = entry.path.stat().st_size
    if actual_bytes != entry.bytes:
        raise Qwen4ArtifactError(
            f"{label} length mismatch: {actual_bytes} != {entry.bytes}"
        )
    actual_sha = _sha256_file(entry.path)
    if actual_sha != entry.sha256:
        raise Qwen4ArtifactError(f"{label} SHA-256 mismatch")


def _component_file(root: Path, path: Path) -> dict[str, Any]:
    resolved = _resolve_z(path, label="artifact component")
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise Qwen4ArtifactError(f"artifact component resolves outside root: {resolved}") from exc
    return {
        "path": relative.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


@dataclass(frozen=True)
class Qwen4ArtifactManifest:
    """Validated view of ``manifest.json``.

    ``raw`` is retained for provenance and future additive fields.  Paths are absolute
    Z: paths so callers never accidentally resolve a relative sidecar against cwd.
    """

    path: Path
    raw: Mapping[str, Any]
    active_path: Path
    active_files: tuple[ComponentFile, ...]
    ple_manifest_path: Path
    ple_data_bytes: int
    ple_sha256: str
    expert_files: tuple[ExpertFile, ...]
    metadata_files: tuple[ComponentFile, ...]
    file_tier_layers: tuple[int, ...]
    resident_layers: tuple[int, ...]
    production_geometry: bool

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def manifest_path(self) -> Path:
        return self.path

    def __getitem__(self, key: str):
        return self.raw[key]

    def get(self, key: str, default=None):
        return self.raw.get(key, default)

    @property
    def text_only(self) -> bool:
        return bool(self.raw.get("text_only", False))

    @property
    def active_format(self) -> str:
        return str(_require_mapping(self.raw.get("active"), label="active")["format"])

    @property
    def source(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw.get("source", {}), label="source")

    @property
    def active(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw.get("active"), label="active")

    @property
    def ple(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw.get("ple"), label="ple")

    @property
    def experts(self) -> Mapping[str, Any]:
        return _require_mapping(self.raw.get("experts"), label="experts")

    @property
    def expert_format(self) -> str:
        return str(_require_mapping(self.raw.get("experts"), label="experts")["format"])

    @property
    def num_layers(self) -> int:
        layers = set(self.file_tier_layers) | set(self.resident_layers)
        return max(layers) + 1 if layers else 0

    def file_for_layer(self, layer_id: int) -> ExpertFile:
        for entry in self.expert_files:
            if entry.layer == int(layer_id):
                return entry
        raise Qwen4ArtifactError(f"manifest has no expert file for layer {layer_id}")

    def verify_active(self) -> None:
        for index, entry in enumerate(self.active_files):
            _verify_component_file(entry, label=f"active.files[{index}]")
        from freetoken.checkpoint.ftw import INDEX_NAME

        with (self.active_path / INDEX_NAME).open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        expected = str(self.source["inventory_sha256"]).lower()
        if str(index.get("source_inventory_sha256", "")).lower() != expected:
            raise Qwen4ArtifactError("active FTW source inventory fingerprint mismatch")


def _validate_layers(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise Qwen4ArtifactError(f"experts.{label} must be a list")
    result = []
    for item in value:
        if isinstance(item, bool):
            raise Qwen4ArtifactError(f"experts.{label} contains a non-integer layer")
        try:
            layer = int(item)
        except (TypeError, ValueError) as exc:
            raise Qwen4ArtifactError(f"experts.{label} contains a non-integer layer") from exc
        if layer < 0 or layer in result:
            raise Qwen4ArtifactError(f"experts.{label} contains an invalid/duplicate layer")
        result.append(layer)
    return tuple(result)


def _read_manifest_path(model_path: str | os.PathLike[str]) -> Path | None:
    candidate = Path(model_path)
    if candidate.name == MANIFEST_NAME and candidate.is_file():
        return _resolve_z(candidate, label="Qwen4 modular manifest")
    if candidate.is_dir():
        path = candidate / MANIFEST_NAME
        if path.is_file():
            return _resolve_z(path, label="Qwen4 modular manifest")
    return None


def load_qwen4_artifact_manifest(
    model_path: str | os.PathLike[str], *, require: bool = False,
    allow_synthetic_geometry: bool = False,
) -> Qwen4ArtifactManifest | None:
    """Load and validate a Qwen4 modular manifest.

    ``None`` means no manifest is present.  This is the compatibility path for all
    unmarked checkpoints.  ``require=True`` is useful at a marked call site where a
    missing manifest must not silently fall back to source weights.
    """

    path = _read_manifest_path(model_path)
    if path is None:
        if require:
            raise Qwen4ArtifactError(f"Qwen4 modular manifest missing under {model_path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen4ArtifactError(f"cannot read Qwen4 modular manifest {path}") from exc
    root = _require_mapping(raw, label="Qwen4 modular manifest")
    if root.get("format") != FORMAT or int(root.get("version", -1)) != VERSION:
        raise Qwen4ArtifactError("unsupported Qwen4 modular manifest format/version")
    if root.get("artifact_schema") != FORMAT:
        raise Qwen4ArtifactError("unsupported Qwen4 modular artifact_schema")
    if root.get("text_only") is not True:
        raise Qwen4ArtifactError("Qwen4 modular manifest must declare text_only=true")
    if root.get("runtime_foundation") != RUNTIME_FOUNDATION_MARKER:
        raise Qwen4ArtifactError("Qwen4 modular manifest lacks the accepted runtime foundation")
    source = _require_mapping(root.get("source"), label="source")
    if not str(source.get("repository", "")).strip() or not str(source.get("revision", "")).strip():
        raise Qwen4ArtifactError("source repository and revision are required")
    source_inventory_sha256 = _require_sha(
        source.get("inventory_sha256"), label="source.inventory_sha256"
    )
    minimum_commit = str(root.get("minimum_freetoken_commit", "")).lower()
    if len(minimum_commit) != 40 or any(char not in "0123456789abcdef" for char in minimum_commit):
        raise Qwen4ArtifactError("minimum_freetoken_commit must be a 40-character Git OID")
    _require_sha(root.get("tvm_ffi_patch_sha256"), label="tvm_ffi_patch_sha256")
    if not allow_synthetic_geometry:
        if source.get("repository") != PINNED_SOURCE_REPOSITORY:
            raise Qwen4ArtifactError("source repository does not match the pinned production source")
        if source.get("revision") != PINNED_SOURCE_REVISION:
            raise Qwen4ArtifactError("source revision does not match the pinned production revision")
        if str(root.get("tvm_ffi_patch_sha256", "")).lower() != TVM_FFI_PATCH_SHA256:
            raise Qwen4ArtifactError("TVM-FFI patch does not match the frozen contract")
    declared_fingerprint = _require_sha(
        root.get("complete_artifact_fingerprint"), label="complete_artifact_fingerprint"
    )
    unsigned = dict(root)
    unsigned.pop("complete_artifact_fingerprint", None)
    actual_fingerprint = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if actual_fingerprint != declared_fingerprint:
        raise Qwen4ArtifactError("complete artifact fingerprint mismatch")
    metadata = _require_mapping(root.get("metadata"), label="metadata")
    if not isinstance(metadata.get("files"), list) or not metadata["files"]:
        raise Qwen4ArtifactError("metadata.files must be a non-empty list")
    metadata_files: list[ComponentFile] = []
    for index, item in enumerate(metadata["files"]):
        entry = _require_mapping(item, label=f"metadata.files[{index}]")
        component = ComponentFile(
            path=_path_within_root(
                path.parent, entry.get("path"), label=f"metadata.files[{index}].path"
            ),
            bytes=int(entry.get("bytes", -1)),
            sha256=_require_sha(
                entry.get("sha256"), label=f"metadata.files[{index}].sha256"
            ),
        )
        if component.bytes < 0:
            raise Qwen4ArtifactError("metadata file bytes must be non-negative")
        _verify_component_file(component, label=f"metadata.files[{index}]")
        metadata_files.append(component)

    active = _require_mapping(root.get("active"), label="active")
    if active.get("format") != ACTIVE_FORMAT:
        raise Qwen4ArtifactError(f"unsupported active format {active.get('format')!r}")
    active_path = _path_within_root(path.parent, active.get("path"), label="active.path")
    if "bytes" in active and int(active["bytes"]) < 0:
        raise Qwen4ArtifactError("active.bytes must be non-negative")
    raw_active_files = active.get("files")
    if not isinstance(raw_active_files, list) or not raw_active_files:
        raise Qwen4ArtifactError("active.files must be a non-empty list")
    active_files: list[ComponentFile] = []
    for index, item in enumerate(raw_active_files):
        entry = _require_mapping(item, label=f"active.files[{index}]")
        try:
            size = int(entry["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Qwen4ArtifactError(f"active.files[{index}].bytes must be an integer") from exc
        if size < 0:
            raise Qwen4ArtifactError(f"active.files[{index}].bytes must be non-negative")
        component = ComponentFile(
            path=_path_within_root(path.parent, entry.get("path"), label=f"active.files[{index}].path"),
            bytes=size,
            sha256=_require_sha(entry.get("sha256"), label=f"active.files[{index}].sha256"),
        )
        try:
            component.path.relative_to(active_path)
        except ValueError as exc:
            raise Qwen4ArtifactError("active file resolves outside active.path") from exc
        active_files.append(component)

    ple = _require_mapping(root.get("ple"), label="ple")
    if ple.get("format") != PLE_FORMAT:
        raise Qwen4ArtifactError(f"unsupported PLE format {ple.get('format')!r}")
    if str(ple.get("required_volume", REQUIRED_VOLUME)).upper() != REQUIRED_VOLUME:
        raise Qwen4ArtifactError("Qwen4 PLE sidecar must reside on Z:")
    ple_manifest_path = _path_within_root(path.parent, ple.get("manifest"), label="ple.manifest")
    ple_data_bytes = int(ple.get("data_bytes", 0))
    if ple_data_bytes < 0:
        raise Qwen4ArtifactError("ple.data_bytes must be non-negative")
    ple_sha256 = _require_sha(ple.get("sha256"), label="ple.sha256")
    if not ple_manifest_path.is_file():
        raise Qwen4ArtifactError(f"PLE manifest is missing: {ple_manifest_path}")
    try:
        with ple_manifest_path.open("r", encoding="utf-8") as handle:
            ple_sidecar_manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen4ArtifactError("cannot read Q3 PLE manifest") from exc
    if str(ple_sidecar_manifest.get("source_fingerprint", "")).lower() != source_inventory_sha256:
        raise Qwen4ArtifactError("Q3 PLE source fingerprint mismatch")

    experts = _require_mapping(root.get("experts"), label="experts")
    if experts.get("format") != EXPERT_FORMAT:
        raise Qwen4ArtifactError(f"unsupported expert format {experts.get('format')!r}")
    if str(experts.get("required_volume", REQUIRED_VOLUME)).upper() != REQUIRED_VOLUME:
        raise Qwen4ArtifactError("Qwen4 expert sidecars must reside on Z:")
    file_layers = _validate_layers(experts.get("file_tier_layers"), label="file_tier_layers")
    resident_layers = _validate_layers(experts.get("resident_layers"), label="resident_layers")
    if set(file_layers) & set(resident_layers):
        raise Qwen4ArtifactError("experts file_tier_layers and resident_layers overlap")
    files_raw = experts.get("files")
    if not isinstance(files_raw, list) or not files_raw:
        raise Qwen4ArtifactError("experts.files must be a non-empty list")
    files: list[ExpertFile] = []
    seen: set[int] = set()
    for item in files_raw:
        entry = _require_mapping(item, label="experts.files[]")
        try:
            layer = int(entry["layer"])
            size = int(entry["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Qwen4ArtifactError("expert file requires integer layer and bytes") from exc
        if layer < 0 or layer in seen or size < 0:
            raise Qwen4ArtifactError("expert file has invalid/duplicate layer or bytes")
        seen.add(layer)
        files.append(
            ExpertFile(
                layer=layer,
                path=_path_within_root(path.parent, entry.get("path"), label=f"experts.files[{layer}].path"),
                bytes=size,
                sha256=_require_sha(entry.get("sha256"), label=f"experts.files[{layer}].sha256"),
                source_fingerprint=_require_sha(
                    entry.get("source_fingerprint"),
                    label=f"experts.files[{layer}].source_fingerprint",
                ),
            )
        )
    if any(item.source_fingerprint != source_inventory_sha256 for item in files):
        raise Qwen4ArtifactError("expert sidecar source fingerprint mismatch")
    declared_layers = set(file_layers) | set(resident_layers)
    if declared_layers != seen:
        raise Qwen4ArtifactError(
            "experts.files layers must exactly match file_tier_layers + resident_layers"
        )
    production_geometry = not allow_synthetic_geometry
    if production_geometry:
        active_payload = int(active.get("payload_bytes", active.get("bytes", -1)))
        if active_payload != ACTIVE_TARGET_BYTES:
            raise Qwen4ArtifactError("active payload does not match frozen production bytes")
        if ple_data_bytes != PLE_TARGET_BYTES:
            raise Qwen4ArtifactError("Q3 PLE extent does not match frozen production bytes")
        if len(files) != EXPERT_LAYERS or any(item.bytes != EXPERT_FILE_BYTES for item in files):
            raise Qwen4ArtifactError("expert sidecars do not match frozen production geometry")
        if file_layers != FILE_TIER_LAYERS:
            raise Qwen4ArtifactError("file-tier layers do not match the frozen production policy")
        expected_resident = tuple(layer for layer in range(EXPERT_LAYERS) if layer not in FILE_TIER_LAYERS)
        if resident_layers != expected_resident:
            raise Qwen4ArtifactError("resident layers do not match the frozen production policy")
        if active_payload + ple_data_bytes + sum(item.bytes for item in files) != KNOWN_TARGET_BYTES:
            raise Qwen4ArtifactError("known target component byte reconciliation failed")
    return Qwen4ArtifactManifest(
        path=path,
        raw=root,
        active_path=active_path,
        active_files=tuple(active_files),
        ple_manifest_path=ple_manifest_path,
        ple_data_bytes=ple_data_bytes,
        ple_sha256=ple_sha256,
        expert_files=tuple(sorted(files, key=lambda item: item.layer)),
        metadata_files=tuple(metadata_files),
        file_tier_layers=file_layers,
        resident_layers=resident_layers,
        production_geometry=production_geometry,
    )


def qwen4_text_only_marker(config: Any) -> bool:
    """Validate and return the explicit target config marker.

    The marker is deliberately target-specific.  A typo or future marker is an error,
    not permission to disable vision under an unknown policy.
    """

    marker = getattr(config, "freetoken_text_only", None)
    if marker is None:
        return False
    if marker != TEXT_ONLY_MARKER:
        raise Qwen4ArtifactError(f"unsupported freetoken_text_only marker {marker!r}")
    return True


def build_mixed_expert_sources(
    manifest: Qwen4ArtifactManifest,
    *,
    num_experts: int = 512,
    resident_residency: list[str] | None = None,
    allocator=None,
    verify_hash: bool = True,
):
    """Materialize only resident layers from bounded expert sidecars.

    Each resident layer is allocated as six independent :class:`HostBank` buffers and
    filled one record at a time through ``FileExpertSource.read_record``.  File-tier
    layers remain ``None`` in the returned bank lists and retain an open
    ``FileExpertSource`` for demand paging.  ``allocator`` is an injectable
    ``(shape, dtype) -> buffer`` callback for CPU-only tests; a production call uses
    the normal HostBank allocator and settles each completed layer to the requested
    residency class.
    """

    if not isinstance(manifest, Qwen4ArtifactManifest):
        raise TypeError("manifest must be a validated Qwen4ArtifactManifest")
    num_experts = int(num_experts)
    from freetoken.moe.expert_source import FileExpertSource
    from freetoken.moe.host_banks import HostBank, HostResidency

    if not 1 <= num_experts <= FileExpertSource.num_experts:
        raise ValueError(
            f"Qwen4 modular expert sidecars support num_experts in [1, {FileExpertSource.num_experts}]"
        )

    layers = manifest.num_layers
    residency = resident_residency or [HostResidency.PINNED.value] * layers
    if len(residency) != layers:
        raise ValueError(f"resident_residency has {len(residency)} layers, expected {layers}")
    unknown_residency = set(residency) - {item.value for item in HostResidency}
    if unknown_residency:
        raise ValueError(f"unknown host residency values: {sorted(unknown_residency)}")
    if allocator is None:
        allocator = lambda shape, dtype: HostBank(shape, dtype)

    sources = {name: [None] * layers for name in FileExpertSource.bank_schema}
    file_sources = {}
    for layer in sorted(manifest.file_tier_layers):
        entry = manifest.file_for_layer(layer)
        file_sources[layer] = FileExpertSource(
            entry.path,
            expected_sha256=entry.sha256,
            expected_source_fingerprint=entry.source_fingerprint,
            expected_layer_id=layer,
            num_experts=num_experts,
            verify_hash=verify_hash,
        )

    # Resident layers are bounded by one six-plane record at a time.  We do not
    # retain a second full-layer staging tensor and close each source after fill.
    # If construction fails, close already-open tier sources so partial startup
    # cannot retain file handles or make fixture cleanup impossible.
    try:
        for layer in sorted(manifest.resident_layers):
            entry = manifest.file_for_layer(layer)
            with FileExpertSource(
                entry.path,
                expected_sha256=entry.sha256,
                expected_source_fingerprint=entry.source_fingerprint,
                expected_layer_id=layer,
                num_experts=num_experts,
                verify_hash=verify_hash,
            ) as source:
                buffers = {
                    name: allocator(shape=(num_experts, *shape), dtype=dtype)
                    for name, (shape, dtype) in source.plane_specs.items()
                }
                for expert_id in range(num_experts):
                    row = source.read_record(expert_id)
                    for name, buffer in buffers.items():
                        destination = getattr(buffer, "tensor", buffer)
                        destination[expert_id].copy_(row[name])
                settle = residency[layer]
                for buffer in buffers.values():
                    if settle == HostResidency.PINNED.value and hasattr(buffer, "pin"):
                        buffer.pin()
                    elif settle == HostResidency.LOCKED.value and hasattr(buffer, "lock"):
                        buffer.lock()
                for name, buffer in buffers.items():
                    sources[name][layer] = getattr(buffer, "tensor", buffer)
    except Exception:
        for source in file_sources.values():
            source.close()
        raise
    return sources, file_sources


def configure_mixed_expert_sources(
    cache,
    manifest: Qwen4ArtifactManifest | Mapping[str, Any],
    resident_sources,
    *,
    file_sources: Mapping[int, object] | None = None,
    layer_residency: list[str] | None = None,
):
    """Wire resident bank entries and file-backed layers into an offload cache.

    ``resident_sources`` is injected by the caller (normally the existing FTW bank
    loader).  This keeps the helper unit-testable with tiny synthetic tensors and avoids
    implementing another expert writer here.  File tiers use only the public
    :class:`FileExpertSource` reader API.
    """

    if not isinstance(manifest, Qwen4ArtifactManifest):
        raise TypeError("manifest must be a validated Qwen4ArtifactManifest")
    if set(manifest.file_tier_layers) & set(cache.cpu_layer_ids):
        raise ValueError("Qwen4 modular file-tier layers are GPU-only")
    if cache.prefill_overlap:
        raise ValueError("Qwen4 modular file tiers require prefill_overlap=False")
    # The artifact's native expert geometry is 512 slots.  Enforce this independently
    # of a malformed/synthetic model config so rebuilds cannot create an undersized cache.
    if int(cache.cache_size) < 512:
        raise ValueError("Qwen4 modular expert cache requires at least 512 slots")
    if not isinstance(resident_sources, Mapping):
        raise TypeError("resident_sources must be a bank-name -> per-layer mapping")
    layers = manifest.num_layers
    if layers <= 0 or int(cache.num_layers) != layers:
        raise ValueError(
            f"manifest layer geometry ({layers}) does not match cache ({cache.num_layers})"
        )
    file_layers = set(manifest.file_tier_layers)
    resident_layers = set(manifest.resident_layers)
    if file_layers | resident_layers != set(range(layers)):
        raise ValueError("manifest expert layer sets must cover a contiguous model")
    if set(resident_sources) != set(cache.bank_schema):
        raise ValueError("resident bank schema does not match cache quant_format")
    per_layer: dict[str, list[Any]] = {}
    for name in cache.bank_schema:
        values = list(resident_sources[name])
        if len(values) != layers:
            raise ValueError(f"resident bank {name!r} has {len(values)} layers, expected {layers}")
        for layer in file_layers:
            if values[layer] is not None:
                # A file tier must be represented explicitly as a None resident source.
                values[layer] = None
        for layer in resident_layers:
            if values[layer] is None:
                raise ValueError(f"resident layer {layer} has no resident source for {name}")
        per_layer[name] = values
    cache.set_bank_sources(per_layer, layer_residency=layer_residency)
    from freetoken.moe.expert_source import FileExpertSource

    if file_sources is None:
        file_sources = {}
        for layer in sorted(file_layers):
            entry = manifest.file_for_layer(layer)
            source = FileExpertSource(
                entry.path,
                expected_sha256=entry.sha256,
                expected_source_fingerprint=entry.source_fingerprint,
                expected_layer_id=layer,
                num_experts=cache.num_experts,
            )
            file_sources[layer] = source
    else:
        file_sources = {int(layer): source for layer, source in file_sources.items()}
        if set(file_sources) != file_layers:
            raise ValueError("file_sources keys do not match manifest file_tier_layers")
    cache.set_file_sources(dict(file_sources))
    return dict(file_sources)


def build_qwen4_modular_artifact(
    source_path: str | os.PathLike[str],
    artifact_root: str | os.PathLike[str],
    *,
    source_repository: str = PINNED_SOURCE_REPOSITORY,
    source_revision: str = PINNED_SOURCE_REVISION,
    source_inventory_sha256: str,
    minimum_freetoken_commit: str,
    ple_layer_id: int = 2,
    ple_split_parts: int = 128,
    expert_layers: tuple[int, ...] = tuple(range(EXPERT_LAYERS)),
    file_tier_layers: tuple[int, ...] = FILE_TIER_LAYERS,
    expert_num_experts: int = EXPERT_NUM_EXPERTS,
    expert_geometry=None,
    allow_synthetic_geometry: bool = False,
) -> dict[str, Any]:
    """Run the canonical C1-C5 modular conversion sequence.

    C0 (full source-file identity verification) remains a mandatory caller gate.
    This function performs no network I/O and accepts only an already-local Z:-backed
    source snapshot.  Each component writer owns its bounded streaming/atomic contract;
    the final manifest is published only after all components reopen successfully.
    """

    source = _resolve_z(source_path, label="source checkpoint")
    root = _resolve_z(artifact_root, label="artifact root")
    if not source.is_dir():
        raise Qwen4ArtifactError(f"source checkpoint is not a directory: {source}")
    root.mkdir(parents=True, exist_ok=True)
    inventory = _require_sha(source_inventory_sha256, label="source_inventory_sha256")

    from freetoken.checkpoint.convert import convert_checkpoint
    from freetoken.checkpoint.q3_ple import write_q3_ple_from_safetensors
    from freetoken.moe.expert_source import write_expert_sidecar_from_safetensors

    active_index = convert_checkpoint(
        str(source),
        str(root),
        artifact_format=ARTIFACT_FORMAT,
        source_inventory_sha256=inventory,
    )
    write_q3_ple_from_safetensors(
        source,
        root / "ple-q3-000.bin",
        root / "ple-q3.json",
        layer_id=int(ple_layer_id),
        split_parts=int(ple_split_parts),
        source_fingerprint=inventory,
        rows_per_segment=(None if allow_synthetic_geometry else 2_500_012),
    )
    expert_paths: dict[int, str] = {}
    for layer in expert_layers:
        name = f"experts-L{int(layer):02d}.nvfp4"
        write_expert_sidecar_from_safetensors(
            source,
            root / name,
            layer_id=int(layer),
            source_fingerprint=inventory,
            num_experts=int(expert_num_experts),
            geometry=expert_geometry,
        )
        expert_paths[int(layer)] = name
    metadata_paths = list(active_index.get("copied_metadata", ()))
    if not metadata_paths:
        raise Qwen4ArtifactError("active conversion copied no target metadata")
    return finalize_qwen4_modular_manifest(
        root,
        source_repository=source_repository,
        source_revision=source_revision,
        source_inventory_sha256=inventory,
        minimum_freetoken_commit=minimum_freetoken_commit,
        tvm_ffi_patch_sha256=TVM_FFI_PATCH_SHA256,
        expert_paths=expert_paths,
        file_tier_layers=file_tier_layers,
        metadata_paths=metadata_paths,
        expert_num_experts=expert_num_experts,
        allow_synthetic_geometry=allow_synthetic_geometry,
    )


def finalize_qwen4_modular_manifest(
    artifact_root: str | os.PathLike[str],
    *,
    source_repository: str,
    source_revision: str,
    source_inventory_sha256: str,
    minimum_freetoken_commit: str,
    tvm_ffi_patch_sha256: str,
    active_dir: str | os.PathLike[str] = "qwen4-active-v1.ftw",
    ple_manifest: str | os.PathLike[str] = "ple-q3.json",
    expert_paths: Mapping[int, str | os.PathLike[str]],
    file_tier_layers: list[int] | tuple[int, ...],
    metadata_paths: list[str | os.PathLike[str]],
    expert_num_experts: int = 512,
    allow_synthetic_geometry: bool = False,
) -> dict[str, Any]:
    """Validate completed components and atomically publish ``manifest.json``.

    This is the final C5 orchestration seam.  It never creates weight payloads;
    the C2/C3/C4 writers must already have atomically finalized their components.
    Every file is length/hash inventoried here, the Q3 and FTEXPERT1 readers reopen
    their formats, and only then is the complete manifest promoted.
    """

    root = _resolve_z(artifact_root, label="artifact root")
    root.mkdir(parents=True, exist_ok=True)
    active_root = _path_from(root, active_dir, label="active_dir")
    from freetoken.checkpoint.ftw import INDEX_NAME, is_ftw_checkpoint

    if not active_root.is_dir() or not is_ftw_checkpoint(str(active_root)):
        raise Qwen4ArtifactError(f"active component is not an FTW checkpoint: {active_root}")
    with (active_root / INDEX_NAME).open("r", encoding="utf-8") as handle:
        active_index = json.load(handle)
    inventory_digest = _require_sha(
        source_inventory_sha256, label="source_inventory_sha256"
    )
    if str(active_index.get("source_inventory_sha256", "")).lower() != inventory_digest:
        raise Qwen4ArtifactError("active FTW source inventory fingerprint mismatch")
    active_payload_bytes = int(active_index.get("total_bytes", -1))
    if active_payload_bytes < 0:
        raise Qwen4ArtifactError("active FTW index has no valid total_bytes")
    active_files = [
        _component_file(root, item)
        for item in sorted(path for path in active_root.rglob("*") if path.is_file())
    ]

    ple_path = _path_within_root(root, ple_manifest, label="ple_manifest")
    from freetoken.checkpoint.q3_ple import Q3PLEReader

    with Q3PLEReader(ple_path) as ple_reader:
        ple_data = ple_reader.data_path
        ple_data_bytes = ple_data.stat().st_size
        ple_sha256 = _sha256_file(ple_data)
        ple_source_fingerprint = str(ple_reader.manifest.get("source_fingerprint", ""))
        if ple_source_fingerprint.lower() != inventory_digest:
            raise Qwen4ArtifactError("Q3 PLE source fingerprint mismatch")

    tiered = tuple(sorted(_validate_layers(list(file_tier_layers), label="file_tier_layers")))
    expert_items: list[dict[str, Any]] = []
    from freetoken.moe.expert_source import FileExpertSource

    for layer, value in sorted((int(layer), path) for layer, path in expert_paths.items()):
        expert_path = _path_within_root(root, value, label=f"expert layer {layer}")
        with FileExpertSource(
            expert_path,
            expected_source_fingerprint=inventory_digest,
            expected_layer_id=layer,
            num_experts=int(expert_num_experts),
            verify_hash=True,
        ) as source:
            if source.layer_id != layer:
                raise Qwen4ArtifactError(f"expert sidecar layer mismatch for {expert_path}")
            expert_items.append(
                {
                    "layer": layer,
                    **_component_file(root, expert_path),
                    "source_fingerprint": source.source_fingerprint,
                }
            )
    all_layers = tuple(item["layer"] for item in expert_items)
    if all_layers != tuple(range(len(all_layers))):
        raise Qwen4ArtifactError("expert sidecars must cover contiguous layers from zero")
    if not set(tiered) <= set(all_layers):
        raise Qwen4ArtifactError("file tier contains a layer without an expert sidecar")
    resident = sorted(set(all_layers) - set(tiered))
    if not allow_synthetic_geometry:
        if source_repository != PINNED_SOURCE_REPOSITORY or source_revision != PINNED_SOURCE_REVISION:
            raise Qwen4ArtifactError("production artifact source pin mismatch")
        commit = str(minimum_freetoken_commit).lower()
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise Qwen4ArtifactError("production artifact requires a concrete FreeToken commit")
        if str(tvm_ffi_patch_sha256).lower() != TVM_FFI_PATCH_SHA256:
            raise Qwen4ArtifactError("production artifact TVM-FFI patch mismatch")
        if int(expert_num_experts) != EXPERT_NUM_EXPERTS:
            raise Qwen4ArtifactError("production artifact requires 512 experts per layer")
        if active_payload_bytes != ACTIVE_TARGET_BYTES:
            raise Qwen4ArtifactError("active FTW does not match frozen production bytes")
        if ple_data_bytes != PLE_TARGET_BYTES:
            raise Qwen4ArtifactError("Q3 PLE does not match frozen production extent")
        if tuple(all_layers) != tuple(range(EXPERT_LAYERS)):
            raise Qwen4ArtifactError("production artifact requires 48 expert sidecars")
        if any(item["bytes"] != EXPERT_FILE_BYTES for item in expert_items):
            raise Qwen4ArtifactError("expert sidecar does not match frozen production bytes")
        if tiered != FILE_TIER_LAYERS:
            raise Qwen4ArtifactError("file tier does not match the frozen production policy")
        if active_payload_bytes + ple_data_bytes + sum(item["bytes"] for item in expert_items) != KNOWN_TARGET_BYTES:
            raise Qwen4ArtifactError("known target component byte reconciliation failed")

    metadata_files = [
        _component_file(root, _path_within_root(root, value, label="metadata file"))
        for value in metadata_paths
    ]
    if not any(item["path"] == "config.json" for item in metadata_files):
        raise Qwen4ArtifactError("modular artifact metadata must include config.json")
    config_path = root / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("freetoken_text_only") != TEXT_ONLY_MARKER:
        raise Qwen4ArtifactError("config.json lacks the accepted text-only marker")
    if config.get("freetoken_active_quant") != ACTIVE_FORMAT:
        raise Qwen4ArtifactError("config.json lacks the accepted active-quant marker")
    config_runtime_foundation = config.get("freetoken_runtime_foundation")
    if config_runtime_foundation != RUNTIME_FOUNDATION_MARKER:
        raise Qwen4ArtifactError("config.json lacks the accepted runtime-foundation marker")

    manifest: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "artifact_schema": FORMAT,
        "text_only": True,
        "runtime_foundation": RUNTIME_FOUNDATION_MARKER,
        "source": {
            "repository": str(source_repository),
            "revision": str(source_revision),
            "inventory_sha256": inventory_digest,
        },
        "minimum_freetoken_commit": str(minimum_freetoken_commit),
        "tvm_ffi_patch_sha256": _require_sha(
            tvm_ffi_patch_sha256, label="tvm_ffi_patch_sha256"
        ),
        "active": {
            "format": ACTIVE_FORMAT,
            "path": active_root.relative_to(root).as_posix(),
            "payload_bytes": active_payload_bytes,
            "physical_file_bytes": sum(item["bytes"] for item in active_files),
            "files": active_files,
        },
        "ple": {
            "format": PLE_FORMAT,
            "manifest": ple_path.relative_to(root).as_posix(),
            "data_bytes": ple_data_bytes,
            "sha256": ple_sha256,
            "source_fingerprint": ple_source_fingerprint,
            "required_volume": REQUIRED_VOLUME,
        },
        "experts": {
            "format": EXPERT_FORMAT,
            "files": expert_items,
            "file_tier_layers": list(tiered),
            "resident_layers": resident,
            "required_volume": REQUIRED_VOLUME,
        },
        "metadata": {"files": metadata_files},
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["complete_artifact_fingerprint"] = hashlib.sha256(canonical).hexdigest()
    partial = root / f".{MANIFEST_NAME}.partial-{os.getpid()}"
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, root / MANIFEST_NAME)
    return manifest


__all__ = [
    "ACTIVE_FORMAT",
    "ARTIFACT_FORMAT",
    "EXPERT_FORMAT",
    "FORMAT",
    "MANIFEST_NAME",
    "PLE_FORMAT",
    "Qwen4ArtifactError",
    "Qwen4ArtifactManifest",
    "TEXT_ONLY_MARKER",
    "RUNTIME_FOUNDATION_MARKER",
    "VERSION",
    "build_qwen4_modular_artifact",
    "configure_mixed_expert_sources",
    "build_mixed_expert_sources",
    "finalize_qwen4_modular_manifest",
    "load_qwen4_artifact_manifest",
    "qwen4_text_only_marker",
]
