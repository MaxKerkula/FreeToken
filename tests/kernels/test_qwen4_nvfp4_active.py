"""Bounded synthetic NVFP4 W4A16 differential tests (no model payloads)."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "out_features,in_features",
    [
        (320, 10240),       # mHC down
        (10240, 320),       # mHC up
        (1280, 2560),       # shared gate|up
        (2560, 640),        # shared down
        (13312, 2560),      # QSA q|k|v
        (2560, 6144),       # QSA/GDN output
        (16384, 2560),      # GDN qkv|z
    ],
)
@pytest.mark.parametrize("rows", [1, 2, 64, 65])
def test_native_nvfp4_dense_matches_dequantized_reference(rows, out_features, in_features):
    from freetoken.checkpoint.nvfp4 import encode_bf16_nvfp4
    from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

    torch.manual_seed(38038 + rows + out_features)
    source = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    packed, scales, globals_ = encode_bf16_nvfp4(source)
    x = torch.randn(rows, in_features, dtype=torch.bfloat16, device="cuda")
    packed_cuda, scales_cuda, globals_cuda = packed.cuda(), scales.cuda(), globals_.cuda()
    out = nvfp4_dense_linear(x, packed_cuda, scales_cuda, globals_cuda)
    weight = dequant_nvfp4(
        packed_cuda.unsqueeze(0), scales_cuda.unsqueeze(0), globals_cuda.unsqueeze(0),
        torch.zeros(1, dtype=torch.int32, device="cuda"), dtype=torch.bfloat16,
    )[0]
    reference = x @ weight.t()
    # Reuse the pre-existing native NVFP4 backend tolerance verbatim: BF16
    # grouped/dense GEMMs accumulate large reductions, so the absolute bound is
    # relative to this fixture's output magnitude.
    atol = 0.03 * float(reference.abs().max())
    torch.testing.assert_close(out.float(), reference.float(), rtol=3e-2, atol=atol)
    assert torch.cuda.max_memory_allocated() < 6 * (1 << 30)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_native_nvfp4_state_dict_repackages_without_bf16_weight_copy():
    from freetoken.checkpoint.nvfp4 import encode_bf16_nvfp4
    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear

    torch.manual_seed(38039)
    source = torch.randn(17, 32, dtype=torch.bfloat16)
    packed, scales, globals_ = encode_bf16_nvfp4(source)
    op = Nvfp4DenseLinear(32, 17)
    state = {"weight": packed, "weight_scale": scales, "weight_global": globals_}
    op.load_state_dict(state)
    assert not state
    assert op.weight.dtype == torch.int32 and op.weight.shape == (4, 17)
    assert op.weight_scale.shape == (2, 17)
    assert op.weight_global.dtype == torch.float16
