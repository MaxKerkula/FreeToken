from __future__ import annotations

from collections.abc import Iterable
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.checkpoint.nvfp4 import encode_bf16_nvfp4
from freetoken.models.loader import iter_weight_files
from tqdm import tqdm

from freetoken.models.qwen3_5_moe.weight import (
    iter_weights_parallel,
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
    setup_offload_expert_banks,
)


_FUSIONS = {
    ".self_attn.qkv_proj.weight": (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
    ),
    # Canonical runtime-state names match the explicit Qwen4 GDN split: native
    # NVFP4 qkv|z and a separate BF16 b|a projection.  Keeping these as two
    # entries avoids a load-time dequant/re-fusion ambiguity.
    ".linear_attn.in_proj_qkvz.weight": (
        ".linear_attn.in_proj_qkv.weight",
        ".linear_attn.in_proj_z.weight",
    ),
    ".linear_attn.in_proj_ba.weight": (
        ".linear_attn.in_proj_b.weight",
        ".linear_attn.in_proj_a.weight",
    ),
    ".mlp.shared_expert.gate_up_proj.weight": (
        ".mlp.shared_expert.gate_proj.weight",
        ".mlp.shared_expert.up_proj.weight",
    ),
}

ACTIVE_NVFP4_FORMAT = "nvfp4_w4a16_v1"
_ACTIVE_NVFP4_WEIGHT_SUFFIXES = (
    ".self_attn.qkv_proj.weight",
    ".self_attn.o_proj.weight",
    ".linear_attn.in_proj_qkvz.weight",
    ".linear_attn.out_proj.weight",
    ".attn_hyper_connection.input_mix_weight_down.weight",
    ".attn_hyper_connection.input_mix_weight_up.weight",
    ".mlp_hyper_connection.input_mix_weight_down.weight",
    ".mlp_hyper_connection.input_mix_weight_up.weight",
    ".hyper_connection_mixer.input_mix_weight_down.weight",
    ".hyper_connection_mixer.input_mix_weight_up.weight",
    ".mlp.shared_expert.gate_up_proj.weight",
    ".mlp.shared_expert.down_proj.weight",
)


def is_active_nvfp4_weight(name: str) -> bool:
    """Whether a fused runtime-state weight belongs to the frozen Qwen4 map."""

    return name.endswith(_ACTIVE_NVFP4_WEIGHT_SUFFIXES)


def iter_active_nvfp4_runtime_entries(
    entries: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream fused BF16 state into canonical native NVFP4 FTW entries."""

    for name, tensor in entries:
        if not is_active_nvfp4_weight(name):
            yield name, tensor
            continue
        packed, scale, global_scale = encode_bf16_nvfp4(tensor)
        prefix = name.removesuffix(".weight")
        yield name, packed
        yield prefix + ".weight_scale", scale
        yield prefix + ".weight_global", global_scale


def _rename(raw_name: str) -> str | None:
    if raw_name.startswith("mtp."):
        return None
    if raw_name.startswith("model.visual."):
        return "visual." + raw_name[len("model.visual.") :]
    if raw_name.startswith("visual."):
        return raw_name
    if ".ple.ple_embedding.ngram_embedding." in raw_name:
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _try_fuse(name: str, tensor: torch.Tensor, buffers: dict):
    for fused_suffix, parts in _FUSIONS.items():
        for index, part in enumerate(parts):
            if name.endswith(part):
                fused_name = name[: -len(part)] + fused_suffix
                slots = buffers.setdefault(fused_name, {})
                slots[index] = tensor
                if len(slots) == len(parts):
                    del buffers[fused_name]
                    return fused_name, torch.cat([slots[i] for i in range(len(parts))], dim=0)
                return ()
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if get_tp_info().size > 1:
        raise NotImplementedError("Qwen4-Exp currently supports TP=1 only")
    if include_moe_experts:
        raise ValueError("Qwen4-Exp requires --moe-backend offload, cpu, or hybrid")
    if not include_non_moe:
        return

    buffers = {}
    for filename in tqdm(
        iter_weight_files(model_path),
        desc="Loading Qwen4-Exp resident weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(filename, framework="pt", device=str(device)) as handle:
            for raw_name in handle.keys():
                name = _rename(raw_name)
                if (
                    name is None
                    or ".mlp.experts." in name
                    or raw_name.endswith(".weight_scale_inv")
                ):
                    continue
                tensor = handle.get_tensor(raw_name)
                fused = _try_fuse(name, tensor, buffers)
                if fused is not None:
                    if fused:
                        yield fused
                    continue
                yield name, tensor
    if buffers:
        raise RuntimeError(f"Incomplete Qwen4-Exp projection fusions: {sorted(buffers)}")


__all__ = [
    "iter_weights",
    "encode_bf16_nvfp4",
    "ACTIVE_NVFP4_FORMAT",
    "is_active_nvfp4_weight",
    "iter_active_nvfp4_runtime_entries",
    "iter_weights_parallel",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "setup_offload_expert_banks",
]
