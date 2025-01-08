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



#include <ATen/ops/_trilinear_ops.h>

namespace at {


// aten::_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor
inline at::Tensor _trilinear(const at::Tensor & i1, const at::Tensor & i2, const at::Tensor & i3, at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3, at::IntArrayRef sumdim, int64_t unroll_dim=1) {
    return at::_ops::_trilinear::call(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
}

// aten::_trilinear.out(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _trilinear_out(at::Tensor & out, const at::Tensor & i1, const at::Tensor & i2, const at::Tensor & i3, at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3, at::IntArrayRef sumdim, int64_t unroll_dim=1) {
    return at::_ops::_trilinear_out::call(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim, out);
}
// aten::_trilinear.out(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _trilinear_outf(const at::Tensor & i1, const at::Tensor & i2, const at::Tensor & i3, at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3, at::IntArrayRef sumdim, int64_t unroll_dim, at::Tensor & out) {
    return at::_ops::_trilinear_out::call(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim, out);
}

}
