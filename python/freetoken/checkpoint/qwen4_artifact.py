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


@dataclass(frozen=True)
class Qwen4ArtifactManifest:
    """Validated view of ``manifest.json``.

    ``raw`` is retained for provenance and future additive fields.  Paths are absolute
    Z: paths so callers never accidentally resolve a relative sidecar against cwd.
    """

    path: Path
    raw: Mapping[str, Any]
    active_path: Path
    ple_manifest_path: Path
    ple_data_bytes: int
    ple_sha256: str
    expert_files: tuple[ExpertFile, ...]
    file_tier_layers: tuple[int, ...]
    resident_layers: tuple[int, ...]

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
    model_path: str | os.PathLike[str], *, require: bool = False
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
    if root.get("text_only") is not True:
        raise Qwen4ArtifactError("Qwen4 modular manifest must declare text_only=true")

    active = _require_mapping(root.get("active"), label="active")
    if active.get("format") != ACTIVE_FORMAT:
        raise Qwen4ArtifactError(f"unsupported active format {active.get('format')!r}")
    active_path = _path_from(path.parent, active.get("path"), label="active.path")
    if "bytes" in active and int(active["bytes"]) < 0:
        raise Qwen4ArtifactError("active.bytes must be non-negative")
    if "sha256" in active:
        _require_sha(active["sha256"], label="active.sha256")

    ple = _require_mapping(root.get("ple"), label="ple")
    if ple.get("format") != PLE_FORMAT:
        raise Qwen4ArtifactError(f"unsupported PLE format {ple.get('format')!r}")
    if str(ple.get("required_volume", REQUIRED_VOLUME)).upper() != REQUIRED_VOLUME:
        raise Qwen4ArtifactError("Qwen4 PLE sidecar must reside on Z:")
    ple_manifest_path = _path_from(path.parent, ple.get("manifest"), label="ple.manifest")
    ple_data_bytes = int(ple.get("data_bytes", 0))
    if ple_data_bytes < 0:
        raise Qwen4ArtifactError("ple.data_bytes must be non-negative")
    ple_sha256 = _require_sha(ple.get("sha256"), label="ple.sha256")

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
                path=_path_from(path.parent, entry.get("path"), label=f"experts.files[{layer}].path"),
                bytes=size,
                sha256=_require_sha(entry.get("sha256"), label=f"experts.files[{layer}].sha256"),
            )
        )
    declared_layers = set(file_layers) | set(resident_layers)
    if declared_layers != seen:
        raise Qwen4ArtifactError(
            "experts.files layers must exactly match file_tier_layers + resident_layers"
        )
    return Qwen4ArtifactManifest(
        path=path,
        raw=root,
        active_path=active_path,
        ple_manifest_path=ple_manifest_path,
        ple_data_bytes=ple_data_bytes,
        ple_sha256=ple_sha256,
        expert_files=tuple(sorted(files, key=lambda item: item.layer)),
        file_tier_layers=file_layers,
        resident_layers=resident_layers,
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
            expected_layer_id=layer,
            num_experts=num_experts,
            verify_hash=verify_hash,
        )

    # Resident layers are bounded by one six-plane record at a time.  We do not
    # retain a second full-layer staging tensor and close each source after fill.
    for layer in sorted(manifest.resident_layers):
        entry = manifest.file_for_layer(layer)
        with FileExpertSource(
            entry.path,
            expected_sha256=entry.sha256,
            expected_layer_id=layer,
            num_experts=num_experts,
            verify_hash=verify_hash,
        ) as source:
            buffers = {
                name: allocator(shape=(num_experts, *shape), dtype=dtype)
                for name, (shape, dtype) in FileExpertSource.plane_specs.items()
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
    return sources, file_sources


def configure_mixed_expert_sources(
    cache,
    manifest: Qwen4ArtifactManifest | Mapping[str, Any],
    resident_sources,
    *,
    file_sources: Mapping[int, object] | None = None,
):
    """Wire resident bank entries and file-backed layers into an offload cache.

    ``resident_sources`` is injected by the caller (normally the existing FTW bank
    loader).  This keeps the helper unit-testable with tiny synthetic tensors and avoids
    implementing another expert writer here.  File tiers use only the public
    :class:`FileExpertSource` reader API.
    """

    if not isinstance(manifest, Qwen4ArtifactManifest):
        raise TypeError("manifest must be a validated Qwen4ArtifactManifest")
    if cache.decode_target != "gpu":
        raise ValueError("Qwen4 modular file tiers are GPU-only")
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
    cache.set_bank_sources(per_layer)
    from freetoken.moe.expert_source import FileExpertSource

    if file_sources is None:
        file_sources = {}
        for layer in sorted(file_layers):
            entry = manifest.file_for_layer(layer)
            source = FileExpertSource(
                entry.path,
                expected_sha256=entry.sha256,
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
    "VERSION",
    "configure_mixed_expert_sources",
    "build_mixed_expert_sources",
    "load_qwen4_artifact_manifest",
    "qwen4_text_only_marker",
]
