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



#include <ATen/ops/randperm_ops.h>

namespace at {


// aten::randperm(SymInt n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randperm(int64_t n, at::TensorOptions options=at::kLong) {
    return at::_ops::randperm::call(n, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randperm(int64_t n, at::TensorOptions options=at::kLong) {
    return at::_ops::randperm::call(n, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::randperm(SymInt n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randperm(int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randperm::call(n, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randperm(int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randperm::call(n, dtype, layout, device, pin_memory);
  }
}

// aten::randperm(SymInt n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randperm_symint(c10::SymInt n, at::TensorOptions options=at::kLong) {
    return at::_ops::randperm::call(n, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randperm(c10::SymInt n, at::TensorOptions options=at::kLong) {
    return at::_ops::randperm::call(n, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::randperm(SymInt n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randperm_symint(c10::SymInt n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randperm::call(n, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randperm(c10::SymInt n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randperm::call(n, dtype, layout, device, pin_memory);
  }
}

// aten::randperm.generator(SymInt n, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randperm(int64_t n, c10::optional<at::Generator> generator, at::TensorOptions options=at::kLong) {
    return at::_ops::randperm_generator::call(n, generator, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randperm(int64_t n, c10::optional<at::Generator> generator, at::TensorOptions options=at::kLong) {
    return at::_ops::randperm_generator::call(n, generator, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::randperm.generator(SymInt n, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randperm(int64_t n, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randperm_generator::call(n, generator, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randperm(int64_t n, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randperm_generator::call(n, generator, dtype, layout, device, pin_memory);
  }
}

// aten::randperm.generator(SymInt n, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randperm_symint(c10::SymInt n, c10::optional<at::Generator> generator, at::TensorOptions options=at::kLong) {
    return at::_ops::randperm_generator::call(n, generator, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randperm(c10::SymInt n, c10::optional<at::Generator> generator, at::TensorOptions options=at::kLong) {
    return at::_ops::randperm_generator::call(n, generator, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::randperm.generator(SymInt n, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randperm_symint(c10::SymInt n, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randperm_generator::call(n, generator, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randperm(c10::SymInt n, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randperm_generator::call(n, generator, dtype, layout, device, pin_memory);
  }
}

// aten::randperm.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randperm_out(at::Tensor & out, int64_t n) {
    return at::_ops::randperm_out::call(n, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randperm_out(at::Tensor & out, int64_t n) {
    return at::_ops::randperm_out::call(n, out);
  }
}

// aten::randperm.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randperm_outf(int64_t n, at::Tensor & out) {
    return at::_ops::randperm_out::call(n, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randperm_outf(int64_t n, at::Tensor & out) {
    return at::_ops::randperm_out::call(n, out);
  }
}

// aten::randperm.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randperm_symint_out(at::Tensor & out, c10::SymInt n) {
    return at::_ops::randperm_out::call(n, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randperm_out(at::Tensor & out, c10::SymInt n) {
    return at::_ops::randperm_out::call(n, out);
  }
}

// aten::randperm.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randperm_symint_outf(c10::SymInt n, at::Tensor & out) {
    return at::_ops::randperm_out::call(n, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randperm_outf(c10::SymInt n, at::Tensor & out) {
    return at::_ops::randperm_out::call(n, out);
  }
}

// aten::randperm.generator_out(SymInt n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randperm_out(at::Tensor & out, int64_t n, c10::optional<at::Generator> generator) {
    return at::_ops::randperm_generator_out::call(n, generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randperm_out(at::Tensor & out, int64_t n, c10::optional<at::Generator> generator) {
    return at::_ops::randperm_generator_out::call(n, generator, out);
  }
}

// aten::randperm.generator_out(SymInt n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randperm_outf(int64_t n, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::randperm_generator_out::call(n, generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randperm_outf(int64_t n, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::randperm_generator_out::call(n, generator, out);
  }
}

// aten::randperm.generator_out(SymInt n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randperm_symint_out(at::Tensor & out, c10::SymInt n, c10::optional<at::Generator> generator) {
    return at::_ops::randperm_generator_out::call(n, generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randperm_out(at::Tensor & out, c10::SymInt n, c10::optional<at::Generator> generator) {
    return at::_ops::randperm_generator_out::call(n, generator, out);
  }
}

// aten::randperm.generator_out(SymInt n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randperm_symint_outf(c10::SymInt n, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::randperm_generator_out::call(n, generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randperm_outf(c10::SymInt n, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::randperm_generator_out::call(n, generator, out);
  }
}

}
