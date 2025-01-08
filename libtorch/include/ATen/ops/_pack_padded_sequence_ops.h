#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API _pack_padded_sequence {
  using schema = ::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_pack_padded_sequence")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)")
  static ::std::tuple<at::Tensor,at::Tensor> call(const at::Tensor & input, const at::Tensor & lengths, bool batch_first);
  static ::std::tuple<at::Tensor,at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & lengths, bool batch_first);
};

struct TORCH_API _pack_padded_sequence_out {
  using schema = ::std::tuple<at::Tensor &,at::Tensor &> (const at::Tensor &, const at::Tensor &, bool, at::Tensor &, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_pack_padded_sequence")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_pack_padded_sequence.out(Tensor input, Tensor lengths, bool batch_first, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))")
  static ::std::tuple<at::Tensor &,at::Tensor &> call(const at::Tensor & input, const at::Tensor & lengths, bool batch_first, at::Tensor & out0, at::Tensor & out1);
  static ::std::tuple<at::Tensor &,at::Tensor &> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & lengths, bool batch_first, at::Tensor & out0, at::Tensor & out1);
};

}} // namespace at::_ops