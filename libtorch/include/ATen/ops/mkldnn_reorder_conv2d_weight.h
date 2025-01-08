#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/mkldnn_reorder_conv2d_weight_ops.h>

namespace at {


// aten::mkldnn_reorder_conv2d_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1, int[]? input_size=None) -> Tensor
inline at::Tensor mkldnn_reorder_conv2d_weight(const at::Tensor & self, at::IntArrayRef padding=0, at::IntArrayRef stride=1, at::IntArrayRef dilation=1, int64_t groups=1, at::OptionalIntArrayRef input_size=c10::nullopt) {
    return at::_ops::mkldnn_reorder_conv2d_weight::call(self, padding, stride, dilation, groups, input_size);
}

// aten::mkldnn_reorder_conv2d_weight.out(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1, int[]? input_size=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & mkldnn_reorder_conv2d_weight_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef padding=0, at::IntArrayRef stride=1, at::IntArrayRef dilation=1, int64_t groups=1, at::OptionalIntArrayRef input_size=c10::nullopt) {
    return at::_ops::mkldnn_reorder_conv2d_weight_out::call(self, padding, stride, dilation, groups, input_size, out);
}
// aten::mkldnn_reorder_conv2d_weight.out(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1, int[]? input_size=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & mkldnn_reorder_conv2d_weight_outf(const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, at::OptionalIntArrayRef input_size, at::Tensor & out) {
    return at::_ops::mkldnn_reorder_conv2d_weight_out::call(self, padding, stride, dilation, groups, input_size, out);
}

}
