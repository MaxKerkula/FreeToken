"""Deterministic host-side NVFP4 W4A16 encoding helpers.

The native FreeToken dense NVFP4 operators consume three row-major tensors:

* packed E2M1 codes (two low-bit-first nibbles per byte),
* one positive E4M3 scale for every 16 input values, and
* one FP16 positive global scale per output row.

This module is intentionally CPU-safe and does not retain a BF16 copy.  It is
used by metadata/conversion code and by synthetic component tests; runtime
operators continue to live in :mod:`freetoken.kernel.triton.nvfp4_linear`.
"""

from __future__ import annotations

import torch


# Keep this table in lock-step with the Triton/native dequant implementations.
# The unsigned codes are magnitudes; bit 3 is the sign bit.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAGNITUDES = torch.tensor(E2M1_VALUES, dtype=torch.float32)
E2M1_SIGNED = torch.tensor(
    E2M1_VALUES + tuple(-v for v in E2M1_VALUES), dtype=torch.float32
)
_FP16_MAX = float(torch.finfo(torch.float16).max)
_FP16_MIN_SUBNORMAL = 2.0 ** -24
_E4M3_MAX = 448.0
_E4M3_MIN_SUBNORMAL = 2.0 ** -9


def _round_e2m1_rne(magnitude: torch.Tensor) -> torch.Tensor:
    """Round non-negative values to E2M1 using the shared tie-to-even rule.

    Ties are resolved by the parity of the integer E2M1 code (for example,
    0.5/1.0 resolves to code 2, while 1.0/1.5 resolves to code 2).  The
    comparison is carried out in float64 so all BF16 inputs have deterministic
    behavior at exact midpoints.
    """

    if torch.any(~torch.isfinite(magnitude)) or torch.any(magnitude < 0):
        raise ValueError("E2M1 rounding expects finite non-negative values")
    grid = E2M1_MAGNITUDES.to(device=magnitude.device, dtype=torch.float64)
    x = magnitude.to(torch.float64).unsqueeze(-1)
    distance = (x - grid).abs()
    minimum = distance.min(dim=-1, keepdim=True).values
    candidates = distance == minimum
    # Prefer the even code among exact ties.  Since candidates are at most two
    # adjacent codes, selecting the last even candidate gives the desired rule.
    codes = torch.arange(8, device=magnitude.device).expand_as(distance)
    even = candidates & ((codes & 1) == 0)
    picked = torch.where(even, codes, torch.full_like(codes, -1)).amax(dim=-1)
    # Non-ties have no even candidate only for an impossible malformed grid;
    # retain the nearest code as a defensive total fallback.
    nearest = distance.argmin(dim=-1)
    return torch.where(picked >= 0, picked, nearest).to(torch.uint8)


def _round_e4m3_positive(values: torch.Tensor) -> torch.Tensor:
    """Encode finite non-negative values to E4M3 bytes with explicit bounds.

    E4M3's finite range is [0, 448].  Values above the finite range saturate
    to 448 before the PyTorch cast (which otherwise produces the NaN sentinel),
    and values below the representable subnormal range round to zero.  This is
    the documented, deterministic total policy for synthetic conversion.
    """

    if torch.any(~torch.isfinite(values)) or torch.any(values < 0):
        raise ValueError("E4M3 scale encoding expects finite non-negative values")
    bounded = values.to(torch.float32).clamp_(0.0, _E4M3_MAX)
    # torch's CPU float8 conversion is round-to-nearest-even on the E4M3 grid.
    return bounded.to(torch.float8_e4m3fn)


def encode_bf16_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a BF16/FP16/FP32 matrix into the native row-major NVFP4 triple.

    Args:
        weight: ``[out_features, in_features]`` finite real matrix.  The input
            width must be divisible by 16, matching ``Nvfp4DenseLinear``.

    Returns:
        ``(packed, block_scale, global_scale)`` where packed is uint8
        ``[N,K//2]``, block_scale is native ``torch.float8_e4m3fn``
        ``[N,K//16]`` and
        global_scale is FP16 ``[N]``.  The returned tensors are newly allocated
        and no copy of ``weight`` is retained.

    Scale rule:
        ``global = round_fp16(max_abs / 6)`` (clamped to the finite FP16 range,
        with the smallest FP16 subnormal used when a positive value would round
        to zero); ``block = round_e4m3(max_abs_block / (6*global))``.  A zero
        row uses global=1 and zero block scales.  Quantization then rounds each
        value to E2M1 after dividing by ``global*block``.  Zero block scales
        produce zero codes.  These explicit bounds make conversion total for
        every finite input, including under/overflow extrema.
    """

    if weight.ndim != 2:
        raise ValueError(f"NVFP4 encoder expects a rank-2 matrix, got {tuple(weight.shape)}")
    if weight.shape[1] % 16:
        raise ValueError(f"NVFP4 input width must be divisible by 16, got {weight.shape[1]}")
    if not weight.dtype.is_floating_point:
        raise TypeError(f"NVFP4 encoder expects a floating tensor, got {weight.dtype}")
    if not torch.isfinite(weight).all():
        raise ValueError("NVFP4 encoder rejects NaN and infinity inputs")

    source = weight.to(dtype=torch.float32)
    n_rows, width = source.shape
    abs_source = source.abs()
    row_max = abs_source.amax(dim=1)
    nonzero = row_max > 0

    # Rounding through one float16 conversion is intentional.  Explicitly clamp
    # the target first because a large BF16 row otherwise converts to inf.
    global_target = (row_max / 6.0).clamp(_FP16_MIN_SUBNORMAL, _FP16_MAX)
    global_target = torch.where(nonzero, global_target, torch.ones_like(global_target))
    global_scale = global_target.to(torch.float16)
    # A positive target below the FP16 subnormal can still become zero on some
    # CPU implementations; repair it explicitly and deterministically.
    global_scale = torch.where(
        nonzero & (global_scale == 0),
        torch.full_like(global_scale, _FP16_MIN_SUBNORMAL, dtype=torch.float16),
        global_scale,
    )

    blocks = source.view(n_rows, width // 16, 16)
    block_max = blocks.abs().amax(dim=-1)
    denom = global_scale.float().unsqueeze(-1) * 6.0
    block_target = torch.where(block_max > 0, block_max / denom, torch.zeros_like(block_max))
    block_scale = _round_e4m3_positive(block_target)
    block_real = block_scale.view(torch.float8_e4m3fn).float()

    # Quantize against the *rounded* scales consumed by the kernel.  Saturating
    # normalized values to +/-6 is the finite E2M1 endpoint policy.
    scale_real = global_scale.float().unsqueeze(-1).unsqueeze(-1) * block_real.unsqueeze(-1)
    normalized = torch.where(scale_real > 0, blocks / scale_real, torch.zeros_like(blocks))
    magnitude = normalized.abs().clamp_(0.0, 6.0)
    mag_code = _round_e2m1_rne(magnitude.reshape(-1)).view(n_rows, width // 16, 16)
    sign = (normalized < 0).to(torch.uint8)
    code = mag_code | (sign << 3)
    # Two values per byte, low nibble first, as required by the native kernels.
    packed = code.reshape(n_rows, width // 2, 2)
    packed = packed[..., 0] | (packed[..., 1] << 4)
    return packed.contiguous(), block_scale.contiguous(), global_scale.contiguous()


def decode_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference dequantization for the native row-major NVFP4 triple."""

    if packed.dtype != torch.uint8 or block_scale.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise TypeError("packed must be uint8 and block_scale must be uint8-view or float8_e4m3fn")
    if packed.ndim != 2 or block_scale.ndim != 2 or global_scale.ndim != 1:
        raise ValueError("NVFP4 tensors must be packed[N,K/2], scale[N,K/16], global[N]")
    rows, packed_width = packed.shape
    width = packed_width * 2
    if block_scale.shape != (rows, width // 16) or global_scale.shape != (rows,):
        raise ValueError("NVFP4 tensor shapes do not agree")
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(rows, width).to(torch.long)
    values = E2M1_SIGNED.to(device=packed.device)[codes]
    scales = block_scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return (values * scales * global_scale.float().unsqueeze(-1)).to(dtype)


__all__ = ["E2M1_VALUES", "encode_bf16_nvfp4", "decode_nvfp4"]
