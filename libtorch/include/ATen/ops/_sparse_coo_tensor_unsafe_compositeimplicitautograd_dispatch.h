#pragma once
// @generated by torchgen/gen.py from DispatchKeyFunction.h

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {

namespace compositeimplicitautograd {

TORCH_API at::Tensor _sparse_coo_tensor_unsafe(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, at::TensorOptions options={}, c10::optional<bool> is_coalesced=c10::nullopt);
TORCH_API at::Tensor _sparse_coo_tensor_unsafe(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<bool> is_coalesced);
TORCH_API at::Tensor _sparse_coo_tensor_unsafe_symint(const at::Tensor & indices, const at::Tensor & values, c10::SymIntArrayRef size, at::TensorOptions options={}, c10::optional<bool> is_coalesced=c10::nullopt);
TORCH_API at::Tensor _sparse_coo_tensor_unsafe_symint(const at::Tensor & indices, const at::Tensor & values, c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<bool> is_coalesced);

} // namespace compositeimplicitautograd
} // namespace at
