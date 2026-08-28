"""Synthetic Qwen4 active-weight NVFP4 component coverage.

No model files are used.  These tests exercise the deterministic host encoder,
canonical runtime fusion names, and the explicit GDN split without constructing
the full Qwen4 model.
"""

from types import SimpleNamespace

import pytest
import torch


def test_encoder_is_deterministic_and_round_trips_layout():
    from freetoken.checkpoint.nvfp4 import decode_nvfp4, encode_bf16_nvfp4

    torch.manual_seed(38038)
    source = torch.randn(3, 32, dtype=torch.bfloat16)
    first = encode_bf16_nvfp4(source)
    second = encode_bf16_nvfp4(source.clone())
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    packed, scales, globals_ = first
    assert packed.shape == (3, 16) and packed.dtype == torch.uint8
    assert scales.shape == (3, 2) and scales.dtype == torch.float8_e4m3fn
    assert globals_.shape == (3,) and globals_.dtype == torch.float16
    assert decode_nvfp4(packed, scales, globals_).shape == source.shape


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        2.0**-133,  # smallest finite BF16 subnormal
        2.0**-126,  # smallest finite BF16 normal
        448.0,
        -448.0,
        3.38953139e38,  # largest finite BF16 (saturating policy is defined)
    ],
)
def test_encoder_total_for_finite_extrema(value):
    from freetoken.checkpoint.nvfp4 import encode_bf16_nvfp4

    source = torch.tensor([[value] * 16], dtype=torch.bfloat16)
    packed, scales, globals_ = encode_bf16_nvfp4(source)
    assert torch.isfinite(globals_).all()
    assert torch.isfinite(scales.view(torch.float8_e4m3fn).float()).all()
    assert packed.numel() == 8


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_encoder_rejects_nonfinite(bad):
    from freetoken.checkpoint.nvfp4 import encode_bf16_nvfp4

    with pytest.raises(ValueError, match="rejects NaN"):
        encode_bf16_nvfp4(torch.full((1, 16), bad, dtype=torch.float32))


def test_encoder_e2m1_midpoint_tie_to_even():
    from freetoken.checkpoint.nvfp4 import encode_bf16_nvfp4, decode_nvfp4

    # Select a single block with global=1 and block=1; row max=6 establishes
    # that scale, then the midpoint pairs exercise each E2M1 tie.
    values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] + [6.0] * 9
    source = torch.tensor([values], dtype=torch.float32).to(torch.bfloat16)
    packed, scales, globals_ = encode_bf16_nvfp4(source)
    decoded = decode_nvfp4(packed, scales, globals_)[0]
    # E2M1 even-code tie choices: 0, 1, 1, 2, 2, 4, 4.
    assert decoded[:7].tolist() == pytest.approx([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])


def test_qwen4_weight_fusions_use_runtime_state_names():
    from freetoken.models.qwen4_exp.weight import _try_fuse

    base = "model.layers.0.linear_attn."
    buf = {}
    assert _try_fuse(base + "in_proj_qkv.weight", torch.ones(2, 16), buf) == ()
    name, merged = _try_fuse(base + "in_proj_z.weight", torch.full((1, 16), 2.0), buf)
    assert name == base + "in_proj_qkvz.weight"
    assert merged.shape == (3, 16)

    buf = {}
    assert _try_fuse(base + "in_proj_b.weight", torch.ones(1, 16), buf) == ()
    name, merged = _try_fuse(base + "in_proj_a.weight", torch.full((1, 16), 3.0), buf)
    assert name == base + "in_proj_ba.weight"
    assert merged[:, 0].tolist() == [1.0, 3.0]


def test_gdn_nvfp4_qkvz_is_explicit_opt_in():
    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged, Nvfp4DenseLinear
    from freetoken.layers import LinearColParallelMerged
    from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    kwargs = dict(
        hidden_size=32,
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=8,
        head_v_dim=8,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
        layer_id=0,
        expert_quant="none",
        attn_quant="nvfp4",
    )
    legacy = Qwen3_5GatedDeltaNet(**kwargs)
    assert isinstance(legacy.in_proj, LinearColParallelMerged)
    explicit = Qwen3_5GatedDeltaNet(**kwargs, nvfp4_qkvz=True)
    assert isinstance(explicit.in_proj_qkvz, Nvfp4DenseColMerged)
    assert isinstance(explicit.in_proj_ba, LinearColParallelMerged)
    assert isinstance(explicit.out_proj, Nvfp4DenseLinear)
