"""Bounded synthetic NVFP4 W4A16 differential tests (no model payloads)."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("rows", [1, 2, 64, 65])
def test_native_nvfp4_dense_matches_dequantized_reference(rows):
    from freetoken.checkpoint.nvfp4 import decode_nvfp4, encode_bf16_nvfp4
    from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear

    torch.manual_seed(38038 + rows)
    source = torch.randn(13, 32, dtype=torch.bfloat16)
    packed, scales, globals_ = encode_bf16_nvfp4(source)
    x = torch.randn(rows, 32, dtype=torch.bfloat16, device="cuda")
    out = nvfp4_dense_linear(x, packed.cuda(), scales.cuda(), globals_.cuda())
    reference = x.float() @ decode_nvfp4(packed, scales, globals_).cuda().float().t()
    torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=2e-2)
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
