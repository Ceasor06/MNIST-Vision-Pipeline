#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>
#include <ATen/ops/clamp_max_meta.h>

namespace at {
namespace native {
struct TORCH_API structured_clamp_max_out : public at::meta::structured_clamp_max {
void impl(const at::Tensor & self, const at::Scalar & max, const at::Tensor & out);
};
struct TORCH_API structured_clamp_max_out_mps : public at::meta::structured_clamp_max {
void impl(const at::Tensor & self, const at::Scalar & max, const at::Tensor & out);
};
struct TORCH_API structured_clamp_max_Tensor_out : public at::meta::structured_clamp_max_Tensor {
void impl(const at::Tensor & self, const at::Tensor & max, const at::Tensor & out);
};
struct TORCH_API structured_clamp_max_Tensor_out_mps : public at::meta::structured_clamp_max_Tensor {
void impl(const at::Tensor & self, const at::Tensor & max, const at::Tensor & out);
};
} // namespace native
} // namespace at