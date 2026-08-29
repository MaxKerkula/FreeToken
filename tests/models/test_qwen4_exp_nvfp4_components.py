"""Synthetic Qwen4 active-weight NVFP4 component coverage.

No model files are used.  These tests exercise the deterministic host encoder,
canonical runtime fusion names, and the explicit GDN split without constructing
the full Qwen4 model.
"""

from types import SimpleNamespace
import math

import pytest
import torch


def _independent_nvfp4_reference(source: torch.Tensor):
    """Tiny scalar oracle implementing the documented format, not FreeToken helpers."""

    grid = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    packed_rows, scale_rows, globals_out = [], [], []
    for row in source.float().tolist():
        row_max = max(abs(value) for value in row)
        if row_max == 0:
            global_scale = 1.0
        else:
            target = min(max(row_max / 6.0, 2.0**-24), 65504.0)
            global_scale = float(torch.tensor(target, dtype=torch.float16))
            if global_scale == 0:
                global_scale = 2.0**-24
        globals_out.append(global_scale)
        codes, block_scales = [], []
        for start in range(0, len(row), 16):
            block = row[start : start + 16]
            block_max = max(abs(value) for value in block)
            target = 0.0 if block_max == 0 else min(block_max / (6.0 * global_scale), 448.0)
            scale = float(torch.tensor(target, dtype=torch.float8_e4m3fn))
            block_scales.append(scale)
            for value in block:
                normalized = 0.0 if scale == 0 else max(-6.0, min(6.0, value / (scale * global_scale)))
                magnitude = abs(normalized)
                code = min(range(8), key=lambda item: (abs(grid[item] - magnitude), item & 1))
                codes.append(code | (8 if normalized < 0 else 0))
        packed_rows.append([codes[i] | (codes[i + 1] << 4) for i in range(0, len(codes), 2)])
        scale_rows.append(block_scales)
    return (
        torch.tensor(packed_rows, dtype=torch.uint8),
        torch.tensor(scale_rows, dtype=torch.float8_e4m3fn),
        torch.tensor(globals_out, dtype=torch.float16),
    )


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


def test_encoder_bytes_match_independent_scalar_oracle():
    from freetoken.checkpoint.nvfp4 import encode_bf16_nvfp4

    torch.manual_seed(38038)
    source = torch.randn(4, 32, dtype=torch.bfloat16)
    actual = encode_bf16_nvfp4(source)
    expected = _independent_nvfp4_reference(source)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


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


def test_active_converter_emits_canonical_runtime_triple_and_preserves_slices():
    from freetoken.checkpoint.nvfp4 import encode_bf16_nvfp4
    from freetoken.models.qwen4_exp.weight import iter_active_nvfp4_runtime_entries

    q = torch.full((2, 16), 1.0, dtype=torch.bfloat16)
    k = torch.full((1, 16), 2.0, dtype=torch.bfloat16)
    v = torch.full((1, 16), -3.0, dtype=torch.bfloat16)
    name = "model.layers.0.self_attn.qkv_proj.weight"
    emitted = dict(iter_active_nvfp4_runtime_entries([(name, torch.cat((q, k, v), dim=0))]))
    prefix = name.removesuffix(".weight")
    assert list(emitted) == [name, prefix + ".weight_scale", prefix + ".weight_global"]
    packed, scales, globals_ = emitted[name], emitted[prefix + ".weight_scale"], emitted[prefix + ".weight_global"]
    cursor = 0
    for constituent in (q, k, v):
        expected = encode_bf16_nvfp4(constituent)
        end = cursor + constituent.shape[0]
        assert torch.equal(packed[cursor:end], expected[0])
        assert torch.equal(scales[cursor:end], expected[1])
        assert torch.equal(globals_[cursor:end], expected[2])
        cursor = end


def test_active_converter_protects_non_map_tensors():
    from freetoken.models.qwen4_exp.weight import iter_active_nvfp4_runtime_entries

    protected = torch.randn(3, 16, dtype=torch.bfloat16)
    rows = list(iter_active_nvfp4_runtime_entries([
        ("model.layers.0.self_attn.index_qk_proj.weight", protected),
    ]))
    assert len(rows) == 1 and rows[0][0].endswith("index_qk_proj.weight")
    assert rows[0][1] is protected


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


def test_qwen4_frozen_operator_map_and_protected_linears():
    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged, Nvfp4DenseLinear
    from freetoken.layers import LinearReplicated
    from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention
    from freetoken.models.qwen4_exp.model import _GatedResidual, _SharedExpert

    rotary = SimpleNamespace(rotary_dim=128, max_position=128, base=10_000.0, scaling=None)
    attention_config = SimpleNamespace(
        head_dim=128, num_qo_heads=2, num_kv_heads=1, hidden_size=256,
        rms_norm_eps=1e-6, rotary_config=rotary, expert_quant="none", attn_quant="nvfp4",
    )
    attention = Qwen3_5Attention(attention_config, 0)
    assert isinstance(attention.qkv_proj, Nvfp4DenseColMerged)
    assert isinstance(attention.o_proj, Nvfp4DenseLinear)

    qwen_config = SimpleNamespace(
        hidden_size=32, rms_norm_eps=1e-6, dense_quant="nvfp4",
        shared_expert_intermediate_size=16,
        qwen4_args=SimpleNamespace(hc_count=4, hc_lowrank=16),
    )
    residual = _GatedResidual(qwen_config, combine=True)
    assert isinstance(residual.input_mix_weight_down, Nvfp4DenseLinear)
    assert isinstance(residual.input_mix_weight_up, Nvfp4DenseLinear)
    assert isinstance(residual.block_inject_weight, LinearReplicated)
    shared = _SharedExpert(qwen_config)
    assert isinstance(shared.gate_up_proj, Nvfp4DenseColMerged)
    assert isinstance(shared.down_proj, Nvfp4DenseLinear)


def test_legacy_gdn_quant_modes_keep_their_original_dispatch():
    from freetoken.kernel.triton.fp8_block_linear import Fp8BlockColMerged
    from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged
    from freetoken.layers import LinearColParallelMerged
    from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet

    base = dict(
        hidden_size=256, num_k_heads=1, num_v_heads=1, head_k_dim=128,
        head_v_dim=128, conv_kernel_size=4, rms_norm_eps=1e-6, layer_id=0,
    )
    bf16 = Qwen3_5GatedDeltaNet(**base, expert_quant="none", attn_quant="none")
    assert isinstance(bf16.in_proj, LinearColParallelMerged)
    block = Qwen3_5GatedDeltaNet(**base, expert_quant="fp8_block", attn_quant="none")
    assert isinstance(block.in_proj_qkvz, Fp8BlockColMerged)
    pertensor = Qwen3_5GatedDeltaNet(**base, expert_quant="none", attn_quant="fp8_pertensor")
    assert isinstance(pertensor.in_proj_qkvz, Fp8PerTensorColMerged)
