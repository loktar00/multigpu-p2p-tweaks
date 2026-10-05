// SPDX-License-Identifier: Apache-2.0
// Derived from vLLM's gdn_decode_post_conv_mtp_kernel
// (csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu, Copyright contributors to the vLLM project),
// modified: value rows split over several CTAs per head. See ../LICENSE.Apache-2.0.
//
// The stock kernel runs one CTA per (request, value head): at TP8 single-stream that is 6 CTAs on an 82-SM
// GPU walking a 128x128 state through 5 MTP tokens. Every value row of the state is independent, so this
// variant gives each CTA 16 rows (grid z = 8). The per-row math is unchanged; the raw per-row outputs are
// rounded to bf16 exactly where the stock kernel rounds them (shared_out), parked in a global scratch, and the
// last CTA of each (request, head) applies the same RMSNorm + output gate in the same order. Expected
// bit-identical outputs and states (checked by gdn_split_test.py).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;      // 4
constexpr int kRowsPerWarp = 4;
constexpr int kChunkV = kWarps * kRowsPerWarp;  // 16 rows per CTA
constexpr int kSplits = kDimV / kChunkV;        // 8
constexpr int kMaxMtpTokens = 8;
constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;

struct GdnDecodeStrides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t gate_row;
  int64_t state_slot;
};

__device__ __forceinline__ float sigmoid_fast(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu_fast(float x) { return x * sigmoid_fast(x); }
__device__ __forceinline__ float softplus_fast(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head, int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) return __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
  if (dt_bias_type == kDtBiasFloat16) return __half2float(static_cast<const __half*>(dt_bias)[head]);
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

struct Sum2 { float x; float y; };

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

template <typename StateT> __device__ __forceinline__ float4 load_state4(const StateT* s);
template <> __device__ __forceinline__ float4 load_state4<float>(const float* s) {
  return *reinterpret_cast<const float4*>(s);
}
template <> __device__ __forceinline__ float4 load_state4<__nv_bfloat16>(const __nv_bfloat16* s) {
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(s);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(s + 2);
  return make_float4(__bfloat162float(lo.x), __bfloat162float(lo.y), __bfloat162float(hi.x), __bfloat162float(hi.y));
}
template <typename StateT> __device__ __forceinline__ void store_state4(StateT* s, float4 v);
template <> __device__ __forceinline__ void store_state4<float>(float* s, float4 v) {
  *reinterpret_cast<float4*>(s) = v;
}
template <> __device__ __forceinline__ void store_state4<__nv_bfloat16>(__nv_bfloat16* s, float4 v) {
  *reinterpret_cast<__nv_bfloat162*>(s) = __floats2bfloat162_rn(v.x, v.y);
  *reinterpret_cast<__nv_bfloat162*>(s + 2) = __floats2bfloat162_rn(v.z, v.w);
}

template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate>
__global__ __launch_bounds__(kThreads) void gdn_split_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv, const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, StateT* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate, const void* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ raw, int* __restrict__ counters,
    int H, int HV, int state_indices_width, int dt_bias_type, bool norm_weight_is_bf16, float scale,
    float norm_eps, GdnDecodeStrides strides) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int split = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) return;

  const int accepted = num_accepted_tokens[request];
  const int source_slot = accepted > 0 && accepted <= state_indices_width
                              ? state_indices[request * state_indices_width + accepted - 1]
                              : 0;
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    // same zero fill as the stock kernel, each split writes its own rows
    for (int linear = tid; linear < num_tokens * kChunkV; linear += kThreads) {
      const int token = bos + linear / kChunkV;
      const int value = split * kChunkV + linear % kChunkV;
      out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + value] = __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / ValueHeadsPerKeyHead;
  __shared__ __align__(16) StateT shared_state[kChunkV][kDimK];
  __shared__ __align__(16) float shared_q[kMaxMtpTokens][kDimK];
  __shared__ __align__(16) float shared_k[kMaxMtpTokens][kDimK];
  __shared__ __nv_bfloat16 shared_v[kMaxMtpTokens][kChunkV];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];
  __shared__ int is_last;

  StateT* source_state =
      state + static_cast<int64_t>(source_slot) * strides.state_slot + value_head * kDimV * kDimK;
  {
    constexpr int kElems = 16 / sizeof(StateT);
    constexpr int kCopies = kChunkV * kDimK / kElems;
    const StateT* src = source_state + split * kChunkV * kDimK;
    for (int c = tid; c < kCopies; c += kThreads)
      reinterpret_cast<int4*>(&shared_state[0][0])[c] = reinterpret_cast<const int4*>(src)[c];
  }

  for (int t = warp; t < num_tokens; t += kWarps) {
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4], k_values[4];
    float q_square = 0.0f, k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] = __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = __bfloat162float(mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    if (lane < kChunkV)
      shared_v[t][lane] = mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + split * kChunkV + lane];
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head, dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) rows[row] = warp + row * kWarps;

  float h[kRowsPerWarp][4];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    const float4 sv = load_state4(&shared_state[rows[row]][k_base]);
    h[row][0] = sv.x; h[row][1] = sv.y; h[row][2] = sv.z; h[row][3] = sv.w;
  }

  for (int t = 0; t < num_tokens; ++t) {
    const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
    const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
    const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
    const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};
    float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        h[row][i] *= shared_decay[t];
        dot_hk[row] += h[row][i] * k_values[i];
      }
    }
    const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
    const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
    const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y, dot_hk_23.x, dot_hk_23.y};
    float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float delta = (__bfloat162float(shared_v[t][rows[row]]) - reduced_hk[row]) * shared_beta[t];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        h[row][i] += k_values[i] * delta;
        dot_hq[row] += h[row][i] * q_values[i];
      }
    }
    const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
    const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
    if (lane == 0) {
      __nv_bfloat16* r = raw + (static_cast<int64_t>(bos + t) * HV + value_head) * kDimV + split * kChunkV;
      r[rows[0]] = __float2bfloat16(dot_hq_01.x);
      r[rows[1]] = __float2bfloat16(dot_hq_01.y);
      r[rows[2]] = __float2bfloat16(dot_hq_23.x);
      r[rows[3]] = __float2bfloat16(dot_hq_23.y);
    }
    const int destination_slot = state_indices[request * state_indices_width + t];
    if (destination_slot > 0) {
      StateT* dst = state + static_cast<int64_t>(destination_slot) * strides.state_slot + value_head * kDimV * kDimK;
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = split * kChunkV + rows[row];
        store_state4(dst + value * kDimK + k_base, make_float4(h[row][0], h[row][1], h[row][2], h[row][3]));
      }
    }
  }

  // last CTA of this (request, head) normalizes; counter resets itself (CUDA-graph safe)
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    const int old = atomicAdd(&counters[request * HV + value_head], 1);
    is_last = (old == kSplits - 1);
  }
  __syncthreads();
  if (!is_last) return;
  __threadfence();

  for (int t = warp; t < num_tokens; t += kWarps) {
    const int token = bos + t;
    const __nv_bfloat16* r = raw + (static_cast<int64_t>(token) * HV + value_head) * kDimV;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const unsigned short bits = __ldcg(reinterpret_cast<const unsigned short*>(r + value));
      output_values[i] = __bfloat162float(__ushort_as_bfloat16(bits));
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd = rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate_input =
          __bfloat162float(output_gate[static_cast<int64_t>(token) * strides.gate_row + value_head * kDimV + value]);
      const float gate = SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight = norm_weight_is_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(norm_weight)[value])
                                               : static_cast<const float*>(norm_weight)[value];
      out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + value] =
          __float2bfloat16(output_values[i] * rstd * weight * gate);
    }
  }
  if (tid == 0) counters[request * HV + value_head] = 0;
}

}  // namespace

void gdn_split(torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias,
               torch::Tensor state_indices, torch::Tensor cu_seqlens, torch::Tensor num_accepted_tokens,
               torch::Tensor state, torch::Tensor output_gate, torch::Tensor norm_weight, torch::Tensor out,
               torch::Tensor raw, torch::Tensor counters, double scale, double norm_eps, bool sigmoid_gate) {
  const c10::cuda::CUDAGuard guard(mixed_qkv.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int num_value_heads = static_cast<int>(state.size(1));
  const int num_key_heads = static_cast<int>((mixed_qkv.size(1) - num_value_heads * kDimV) / (2 * kDimK));
  const int vpk = num_value_heads / num_key_heads;
  const int num_requests = static_cast<int>(state_indices.size(0));
  const auto dtb = dt_bias.scalar_type();
  const int dt_bias_type = dtb == at::kFloat ? kDtBiasFloat32 : (dtb == at::kBFloat16 ? kDtBiasBFloat16 : kDtBiasFloat16);
  TORCH_CHECK(counters.numel() >= num_requests * num_value_heads, "counters too small");
  TORCH_CHECK(raw.numel() >= out.numel(), "raw scratch too small");
  const GdnDecodeStrides strides{mixed_qkv.stride(0), a.stride(0), b.stride(0), output_gate.stride(0), state.stride(0)};
  const dim3 grid(num_requests, num_value_heads, kSplits);
  const bool nw_bf16 = norm_weight.scalar_type() == at::kBFloat16;
#define GDN_LAUNCH(ST, VPK, SG)                                                                                  \
  gdn_split_kernel<ST, VPK, SG><<<grid, kThreads, 0, stream>>>(                                                 \
      reinterpret_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),                                              \
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()), \
      a_log.data_ptr<float>(), dt_bias.data_ptr(), state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),    \
      num_accepted_tokens.data_ptr<int>(), reinterpret_cast<ST*>(state.data_ptr()),                              \
      reinterpret_cast<const __nv_bfloat16*>(output_gate.data_ptr()), norm_weight.data_ptr(),                    \
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), reinterpret_cast<__nv_bfloat16*>(raw.data_ptr()),       \
      counters.data_ptr<int>(), num_key_heads, num_value_heads, static_cast<int>(state_indices.size(1)),        \
      dt_bias_type, nw_bf16, static_cast<float>(scale), static_cast<float>(norm_eps), strides)
#define GDN_VPK(ST, SG)                                    \
  switch (vpk) {                                           \
    case 1: GDN_LAUNCH(ST, 1, SG); break;                  \
    case 2: GDN_LAUNCH(ST, 2, SG); break;                  \
    case 3: GDN_LAUNCH(ST, 3, SG); break;                  \
    case 4: GDN_LAUNCH(ST, 4, SG); break;                  \
    default: GDN_LAUNCH(ST, 8, SG); break;                 \
  }
  if (state.scalar_type() == at::kFloat) {
    if (sigmoid_gate) { GDN_VPK(float, true) } else { GDN_VPK(float, false) }
  } else {
    if (sigmoid_gate) { GDN_VPK(__nv_bfloat16, true) } else { GDN_VPK(__nv_bfloat16, false) }
  }
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "gdn_split launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gdn_split", &gdn_split); }
