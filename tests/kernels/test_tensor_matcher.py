from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import sys

import pytest
import torch


_SOURCE = r"""
#include <freetoken/tensor.h>
#include <tvm/ffi/container/tensor.h>

int symbolic_cuda_same(tvm::ffi::TensorView first, tvm::ffi::TensorView second) {
  auto device = host::SymbolicDevice{};
  host::TensorMatcher({-1})
      .with_device<kDLCUDA>(device)
      .verify(first)
      .verify(second);
  return device.unwrap().device_id;
}

int symbolic_cuda_same_typed(tvm::ffi::TensorView first,
                             tvm::ffi::TensorView second) {
  auto length = host::SymbolicSize{"length"};
  auto dtype = host::SymbolicDType{};
  auto device = host::SymbolicDevice{};
  host::TensorMatcher({length})
      .with_dtype<int32_t, int64_t>(dtype)
      .with_device<kDLCUDA>(device)
      .verify(first)
      .verify(second);
  return device.unwrap().device_id;
}

template <int Tag> struct SymbolicCudaTemplate {
  static int run(tvm::ffi::TensorView first, tvm::ffi::TensorView second) {
    auto length = host::SymbolicSize{"length"};
    auto dtype = host::SymbolicDType{};
    auto device = host::SymbolicDevice{};
    host::TensorMatcher({length})
        .with_dtype<int32_t, int64_t>(dtype)
        .template with_device<kDLCUDA>(device)
        .verify(first)
        .verify(second);
    return device.unwrap().device_id + Tag - Tag;
  }
};

int symbolic_cuda_same_templated(tvm::ffi::TensorView first,
                                 tvm::ffi::TensorView second) {
  return SymbolicCudaTemplate<1>::run(first, second);
}

int symbolic_unrestricted(tvm::ffi::TensorView value) {
  auto device = host::SymbolicDevice{};
  host::TensorMatcher({-1}).with_device(device).verify(value);
  return static_cast<int>(device.unwrap().device_type);
}

void fixed_cpu(tvm::ffi::TensorView value) {
  host::TensorMatcher({-1}).with_device<kDLCPU>().verify(value);
}

void explicit_cpu(tvm::ffi::TensorView value) {
  host::TensorMatcher({-1}).with_device({{kDLCPU, 0}}).verify(value);
}

void reject_different_cuda_device_ids() {
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();
  device.verify({kDLCUDA, 0});
  device.verify({kDLCUDA, 1});
}
"""

_FUNCTIONS = [
    "symbolic_cuda_same",
    "symbolic_cuda_same_typed",
    "symbolic_cuda_same_templated",
    "symbolic_unrestricted",
    "fixed_cpu",
    "explicit_cpu",
    "reject_different_cuda_device_ids",
]


@lru_cache(maxsize=1)
def _cpu_module():
    from freetoken.kernel.utils import DEFAULT_CFLAGS, DEFAULT_INCLUDE
    from tvm_ffi.cpp import load_inline

    return load_inline(
        "freetoken_tensor_matcher_cpu_test_v7",
        cpp_sources=_SOURCE,
        functions=_FUNCTIONS,
        extra_cflags=DEFAULT_CFLAGS,
        extra_include_paths=DEFAULT_INCLUDE,
    )


@lru_cache(maxsize=1)
def _cuda_module():
    from freetoken.kernel.utils import DEFAULT_INCLUDE, _cuda_cflags
    from tvm_ffi.cpp import load_inline

    extra_ldflags = []
    if sys.platform == "win32":
        cuda_home = Path(os.environ["CUDA_HOME"])
        extra_ldflags = [f"/LIBPATH:{cuda_home / 'lib' / 'x64'}", "cudart.lib"]

    return load_inline(
        "freetoken_tensor_matcher_cuda_test_v7",
        cuda_sources=_SOURCE,
        functions=_FUNCTIONS,
        extra_cuda_cflags=_cuda_cflags([]),
        extra_ldflags=extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE,
        backend="cuda",
    )


def test_symbolic_device_restrictions_and_existing_cpu_paths():
    module = _cpu_module()
    value = torch.ones(4)

    assert module.symbolic_unrestricted(value) == 1  # DLPack kDLCPU
    module.fixed_cpu(value)
    module.explicit_cpu(value)
    with pytest.raises(Exception, match="Device"):
        module.symbolic_cuda_same(value, value)
    with pytest.raises(Exception, match="Device"):
        module.reject_different_cuda_device_ids()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_symbolic_cuda_binding_compiles_and_checks_same_device():
    module = _cuda_module()
    first = torch.ones(4, device="cuda")
    second = torch.zeros(4, device="cuda")

    assert module.symbolic_cuda_same(first, second) == torch.cuda.current_device()
    first_int = torch.ones(4, dtype=torch.int32, device="cuda")
    second_int = torch.zeros(4, dtype=torch.int32, device="cuda")
    assert module.symbolic_cuda_same_typed(first_int, second_int) == torch.cuda.current_device()
    assert module.symbolic_cuda_same_templated(first_int, second_int) == torch.cuda.current_device()
    with pytest.raises(Exception, match="Device"):
        module.symbolic_cuda_same(torch.ones(4), torch.zeros(4))
