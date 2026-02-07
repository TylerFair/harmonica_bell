#include "core/bindings.hpp"

#ifdef HARMONICA_ENABLE_CUDA

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr int64_t kMaxTimes = 10000000;
constexpr int64_t kMaxBatch = 4096;
constexpr int64_t kMaxRs = 4096;

inline bool cuda_ok(cudaError_t err, const char* expr) {
  if (err == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "[harmonica cuda custom-call] %s failed: %s\\n", expr,
               cudaGetErrorString(err));
  return false;
}

inline bool copy_i64_from_device(void* ptr, int64_t* out) {
  return cuda_ok(
      cudaMemcpy(out, ptr, sizeof(int64_t), cudaMemcpyDeviceToHost),
      "cudaMemcpy(device->host scalar)");
}

inline bool copy_doubles_from_device(void* ptr, int64_t n,
                                     std::vector<double>* out) {
  out->resize(static_cast<size_t>(n));
  if (n == 0) {
    return true;
  }
  return cuda_ok(cudaMemcpy(out->data(), ptr, static_cast<size_t>(n) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(device->host array)");
}

inline bool copy_doubles_to_device(const std::vector<double>& in, void* ptr) {
  if (in.empty()) {
    return true;
  }
  return cuda_ok(cudaMemcpy(ptr, in.data(), in.size() * sizeof(double),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(host->device array)");
}

inline bool plausible_single_meta(int64_t n_times, int64_t n_rs) {
  return n_times > 0 && n_times <= kMaxTimes && n_rs > 0 && n_rs <= kMaxRs &&
         (n_rs % 2 == 1);
}

inline bool plausible_batch_meta(int64_t batch_size, int64_t n_times,
                                 int64_t n_rs) {
  return batch_size > 0 && batch_size <= kMaxBatch &&
         plausible_single_meta(n_times, n_rs);
}

struct SingleLayout {
  int input_base;
  int output_base;
  int64_t n_times;
  int64_t n_rs;
  bool ok;
};

struct BatchLayout {
  int input_base;
  int output_base;
  int64_t batch_size;
  int64_t n_times;
  int64_t n_rs;
  bool ok;
};

SingleLayout detect_single_layout(void** buffers, int n_params_without_rs,
                                  int output_count) {
  SingleLayout layout = {0, 0, 0, 0, false};

  int64_t a0 = 0, a1 = 0, b0 = 0, b1 = 0;
  if (!copy_i64_from_device(buffers[0], &a0) ||
      !copy_i64_from_device(buffers[1], &a1) ||
      !copy_i64_from_device(buffers[2], &b0) ||
      !copy_i64_from_device(buffers[3], &b1)) {
    return layout;
  }

  bool a_plausible = plausible_single_meta(a0, a1);
  bool b_plausible = plausible_single_meta(b0, b1);

  if (a_plausible) {
    int64_t n_rs = a1;
    int input_count = 3 + n_params_without_rs + static_cast<int>(n_rs);
    layout.input_base = 0;
    layout.output_base = input_count;
    layout.n_times = a0;
    layout.n_rs = n_rs;
    layout.ok = true;
    return layout;
  }

  if (b_plausible) {
    layout.input_base = output_count;
    layout.output_base = 0;
    layout.n_times = b0;
    layout.n_rs = b1;
    layout.ok = true;
    return layout;
  }

  std::fprintf(stderr,
               "[harmonica cuda custom-call] unable to infer single-call buffer "
               "layout\\n");
  return layout;
}

BatchLayout detect_batch_layout(void** buffers, int n_params_without_rs,
                                int output_count) {
  BatchLayout layout = {0, 0, 0, 0, 0, false};

  int64_t a0 = 0, a1 = 0, a2 = 0, b0 = 0, b1 = 0, b2 = 0;
  if (!copy_i64_from_device(buffers[0], &a0) ||
      !copy_i64_from_device(buffers[1], &a1) ||
      !copy_i64_from_device(buffers[2], &a2) ||
      !copy_i64_from_device(buffers[2], &b0) ||
      !copy_i64_from_device(buffers[3], &b1) ||
      !copy_i64_from_device(buffers[4], &b2)) {
    return layout;
  }

  bool a_plausible = plausible_batch_meta(a0, a1, a2);
  bool b_plausible = plausible_batch_meta(b0, b1, b2);

  if (a_plausible) {
    int64_t n_rs = a2;
    int input_count = 4 + n_params_without_rs + static_cast<int>(n_rs);
    layout.input_base = 0;
    layout.output_base = input_count;
    layout.batch_size = a0;
    layout.n_times = a1;
    layout.n_rs = a2;
    layout.ok = true;
    return layout;
  }

  if (b_plausible) {
    layout.input_base = output_count;
    layout.output_base = 0;
    layout.batch_size = b0;
    layout.n_times = b1;
    layout.n_rs = b2;
    layout.ok = true;
    return layout;
  }

  std::fprintf(stderr,
               "[harmonica cuda custom-call] unable to infer batched-call buffer "
               "layout\\n");
  return layout;
}

bool validate_i32_range(int64_t v, const char* name) {
  if (v < 0 || v > std::numeric_limits<int>::max()) {
    std::fprintf(stderr,
                 "[harmonica cuda custom-call] %s out of int32 range: %lld\\n",
                 name, static_cast<long long>(v));
    return false;
  }
  return true;
}

}  // namespace

void jax_light_curve_quad_ld_cuda(void* /*stream*/, void** buffers,
                                  const char* /*opaque*/,
                                  std::size_t /*opaque_len*/,
                                  void* /*status*/) {
  constexpr int kBaseParams = 6 + 2;
  constexpr int kOutputCount = 2;

  SingleLayout layout = detect_single_layout(buffers, kBaseParams, kOutputCount);
  if (!layout.ok) {
    return;
  }

  if (!validate_i32_range(layout.n_times, "n_times") ||
      !validate_i32_range(layout.n_rs, "n_rs")) {
    return;
  }

  const int n_times = static_cast<int>(layout.n_times);
  const int n_rs = static_cast<int>(layout.n_rs);
  const int n_params = kBaseParams + n_rs;

  std::vector<std::vector<double>> host_inputs(static_cast<size_t>(1 + n_params));

  // times
  if (!copy_doubles_from_device(buffers[layout.input_base + 2], layout.n_times,
                                &host_inputs[0])) {
    return;
  }

  // params
  for (int i = 0; i < n_params; ++i) {
    if (!copy_doubles_from_device(buffers[layout.input_base + 3 + i], layout.n_times,
                                  &host_inputs[1 + i])) {
      return;
    }
  }

  std::vector<const void*> in_ptrs(static_cast<size_t>(3 + n_params));
  in_ptrs[0] = &n_times;
  in_ptrs[1] = &n_rs;
  in_ptrs[2] = host_inputs[0].data();
  for (int i = 0; i < n_params; ++i) {
    in_ptrs[3 + i] = host_inputs[1 + i].data();
  }

  std::vector<double> f_host(static_cast<size_t>(n_times));
  std::vector<double> jac_host(static_cast<size_t>(n_times) *
                               static_cast<size_t>(kBaseParams + n_rs));

  void* out_tuple[kOutputCount] = {f_host.data(), jac_host.data()};
  jax_light_curve_quad_ld(out_tuple, in_ptrs.data(), nullptr);

  if (!copy_doubles_to_device(f_host, buffers[layout.output_base]) ||
      !copy_doubles_to_device(jac_host, buffers[layout.output_base + 1])) {
    return;
  }
}

void jax_light_curve_nonlinear_ld_cuda(void* /*stream*/, void** buffers,
                                       const char* /*opaque*/,
                                       std::size_t /*opaque_len*/,
                                       void* /*status*/) {
  constexpr int kBaseParams = 6 + 4;
  constexpr int kOutputCount = 2;

  SingleLayout layout = detect_single_layout(buffers, kBaseParams, kOutputCount);
  if (!layout.ok) {
    return;
  }

  if (!validate_i32_range(layout.n_times, "n_times") ||
      !validate_i32_range(layout.n_rs, "n_rs")) {
    return;
  }

  const int n_times = static_cast<int>(layout.n_times);
  const int n_rs = static_cast<int>(layout.n_rs);
  const int n_params = kBaseParams + n_rs;

  std::vector<std::vector<double>> host_inputs(static_cast<size_t>(1 + n_params));

  if (!copy_doubles_from_device(buffers[layout.input_base + 2], layout.n_times,
                                &host_inputs[0])) {
    return;
  }

  for (int i = 0; i < n_params; ++i) {
    if (!copy_doubles_from_device(buffers[layout.input_base + 3 + i], layout.n_times,
                                  &host_inputs[1 + i])) {
      return;
    }
  }

  std::vector<const void*> in_ptrs(static_cast<size_t>(3 + n_params));
  in_ptrs[0] = &n_times;
  in_ptrs[1] = &n_rs;
  in_ptrs[2] = host_inputs[0].data();
  for (int i = 0; i < n_params; ++i) {
    in_ptrs[3 + i] = host_inputs[1 + i].data();
  }

  std::vector<double> f_host(static_cast<size_t>(n_times));
  std::vector<double> jac_host(static_cast<size_t>(n_times) *
                               static_cast<size_t>(kBaseParams + n_rs));

  void* out_tuple[kOutputCount] = {f_host.data(), jac_host.data()};
  jax_light_curve_nonlinear_ld(out_tuple, in_ptrs.data(), nullptr);

  if (!copy_doubles_to_device(f_host, buffers[layout.output_base]) ||
      !copy_doubles_to_device(jac_host, buffers[layout.output_base + 1])) {
    return;
  }
}

void jax_light_curve_quad_ld_batch_cuda(void* /*stream*/, void** buffers,
                                        const char* /*opaque*/,
                                        std::size_t /*opaque_len*/,
                                        void* /*status*/) {
  constexpr int kBaseParams = 6 + 2;
  constexpr int kOutputCount = 2;

  BatchLayout layout = detect_batch_layout(buffers, kBaseParams, kOutputCount);
  if (!layout.ok) {
    return;
  }

  if (!validate_i32_range(layout.batch_size, "batch_size") ||
      !validate_i32_range(layout.n_times, "n_times") ||
      !validate_i32_range(layout.n_rs, "n_rs")) {
    return;
  }

  const int batch_size = static_cast<int>(layout.batch_size);
  const int n_times = static_cast<int>(layout.n_times);
  const int n_rs = static_cast<int>(layout.n_rs);
  const int n_params = kBaseParams + n_rs;

  std::vector<std::vector<double>> host_inputs(static_cast<size_t>(1 + n_params));

  // times[T]
  if (!copy_doubles_from_device(buffers[layout.input_base + 3], layout.n_times,
                                &host_inputs[0])) {
    return;
  }

  // params[B]
  for (int i = 0; i < n_params; ++i) {
    if (!copy_doubles_from_device(buffers[layout.input_base + 4 + i],
                                  layout.batch_size, &host_inputs[1 + i])) {
      return;
    }
  }

  std::vector<const void*> in_ptrs(static_cast<size_t>(4 + n_params));
  in_ptrs[0] = &batch_size;
  in_ptrs[1] = &n_times;
  in_ptrs[2] = &n_rs;
  in_ptrs[3] = host_inputs[0].data();
  for (int i = 0; i < n_params; ++i) {
    in_ptrs[4 + i] = host_inputs[1 + i].data();
  }

  const size_t flux_size = static_cast<size_t>(batch_size) * static_cast<size_t>(n_times);
  const size_t jac_size = flux_size * static_cast<size_t>(kBaseParams + n_rs);
  std::vector<double> f_host(flux_size);
  std::vector<double> jac_host(jac_size);

  void* out_tuple[kOutputCount] = {f_host.data(), jac_host.data()};
  jax_light_curve_quad_ld_batch(out_tuple, in_ptrs.data(), nullptr);

  if (!copy_doubles_to_device(f_host, buffers[layout.output_base]) ||
      !copy_doubles_to_device(jac_host, buffers[layout.output_base + 1])) {
    return;
  }
}

void jax_light_curve_nonlinear_ld_batch_cuda(void* /*stream*/, void** buffers,
                                             const char* /*opaque*/,
                                             std::size_t /*opaque_len*/,
                                             void* /*status*/) {
  constexpr int kBaseParams = 6 + 4;
  constexpr int kOutputCount = 2;

  BatchLayout layout = detect_batch_layout(buffers, kBaseParams, kOutputCount);
  if (!layout.ok) {
    return;
  }

  if (!validate_i32_range(layout.batch_size, "batch_size") ||
      !validate_i32_range(layout.n_times, "n_times") ||
      !validate_i32_range(layout.n_rs, "n_rs")) {
    return;
  }

  const int batch_size = static_cast<int>(layout.batch_size);
  const int n_times = static_cast<int>(layout.n_times);
  const int n_rs = static_cast<int>(layout.n_rs);
  const int n_params = kBaseParams + n_rs;

  std::vector<std::vector<double>> host_inputs(static_cast<size_t>(1 + n_params));

  // times[T]
  if (!copy_doubles_from_device(buffers[layout.input_base + 3], layout.n_times,
                                &host_inputs[0])) {
    return;
  }

  // params[B]
  for (int i = 0; i < n_params; ++i) {
    if (!copy_doubles_from_device(buffers[layout.input_base + 4 + i],
                                  layout.batch_size, &host_inputs[1 + i])) {
      return;
    }
  }

  std::vector<const void*> in_ptrs(static_cast<size_t>(4 + n_params));
  in_ptrs[0] = &batch_size;
  in_ptrs[1] = &n_times;
  in_ptrs[2] = &n_rs;
  in_ptrs[3] = host_inputs[0].data();
  for (int i = 0; i < n_params; ++i) {
    in_ptrs[4 + i] = host_inputs[1 + i].data();
  }

  const size_t flux_size = static_cast<size_t>(batch_size) * static_cast<size_t>(n_times);
  const size_t jac_size = flux_size * static_cast<size_t>(kBaseParams + n_rs);
  std::vector<double> f_host(flux_size);
  std::vector<double> jac_host(jac_size);

  void* out_tuple[kOutputCount] = {f_host.data(), jac_host.data()};
  jax_light_curve_nonlinear_ld_batch(out_tuple, in_ptrs.data(), nullptr);

  if (!copy_doubles_to_device(f_host, buffers[layout.output_base]) ||
      !copy_doubles_to_device(jac_host, buffers[layout.output_base + 1])) {
    return;
  }
}

#endif  // HARMONICA_ENABLE_CUDA
