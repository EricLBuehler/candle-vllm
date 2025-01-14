#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

__global__ void gemm_half_q_half_alt_4bit_kernel(
    const half2* __restrict__ vec, const uint32_t* __restrict__ mat,
    half* __restrict__ mul, const half* __restrict__ scales,
    const uint32_t* __restrict__ zeros, const int* __restrict__ g_idx,
    int batch, int height, int width) {
  int zero_width = width / 8;
  int vec_height = height * 4;
  const int blockwidth2 = BLOCK_KN_SIZE / 2;
  int b = blockIdx.y * BLOCK_M_SIZE_MAX;
  int b_end = min(BLOCK_M_SIZE_MAX, batch - b);
  int h = BLOCK_KN_SIZE * blockIdx.z / 8;
  int h_end = min(BLOCK_KN_SIZE / 8, height - h) * 4;
  int w = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;

  __shared__ half2 blockvec[BLOCK_M_SIZE_MAX][blockwidth2];
  if (threadIdx.x < h_end) {
    for (int m = 0; m < b_end; ++m) {
      blockvec[m][threadIdx.x] =
          vec[(m + b) * vec_height + blockIdx.z * BLOCK_KN_SIZE / 2 +
              threadIdx.x];
    }
  }

  __shared__ half2 deq2[256][8];
  int val = threadIdx.x / 8;
  int off = threadIdx.x % 8;
  for (; val < 256; val += BLOCK_KN_SIZE / 8) {
    deq2[val][off] =
        __halves2half2(__int2half_rn(val & 0xF), __int2half_rn(val >> 4));
  }

  if (blockIdx.z == 0) {
    for (int m = 0; m < b_end; m++) mul[(b + m) * width + w] = __int2half_rn(0);
  }
  __syncthreads();

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;
  int z_w = w / 8;
  int z_mod = (w % 8) * 4;
  half2 res2;
  half res[BLOCK_M_SIZE_MAX] = {};

  unsigned int tmp;
  while (k < h_end) {
    tmp = mat[i];
    half2 scales_tmp[4];
    half2 zeros_tmp[4];
    for (int tmp_k = 0; tmp_k < 4; tmp_k++) {
      int g = g_idx[g_h + (k + tmp_k) * 2];
      int g2 = g_idx[g_h + (k + tmp_k) * 2 + 1];
      half scale_f = scales[g * width + w];
      half scale_f2 = scales[g2 * width + w];
      half2 scale = __halves2half2(scale_f, scale_f2);
      half2 zero = __halves2half2(
          __hmul(scale_f,
                 __int2half_rn(-((zeros[g * zero_width + z_w] >> z_mod) & 0xF) -
                               1)),
          __hmul(scale_f2,
                 __int2half_rn(
                     -((zeros[g2 * zero_width + z_w] >> z_mod) & 0xF) - 1)));
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
    }
    for (int m = 0; m < b_end; m++) {
#ifndef USE_ROCM
      res2 = {};
#else
      res2.x = __half_as_ushort(__float2half(0));
      res2.y = __half_as_ushort(__float2half(0));
#endif
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 0) & 0xff][off], scales_tmp[0], zeros_tmp[0]),
          blockvec[m][k + 0], res2);
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 8) & 0xff][off], scales_tmp[1], zeros_tmp[1]),
          blockvec[m][k + 1], res2);
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 16) & 0xff][off], scales_tmp[2], zeros_tmp[2]),
          blockvec[m][k + 2], res2);
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 24) & 0xff][off], scales_tmp[3], zeros_tmp[3]),
          blockvec[m][k + 3], res2);
#ifndef USE_ROCM
      res[m] = __hadd(res[m], __hadd(res2.x, res2.y));
#else
      res[m] = __hadd(
          res[m], __hadd(__ushort_as_half(res2.x), __ushort_as_half(res2.y)));
#endif
    }
    i += width;
    k += 4;
  }
  for (int m = 0; m < b_end; m++) {
    atomicAdd(&mul[(b + m) * width + w], res[m]);
  }
}

__global__ void gemm_half_q_half_alt_8bit_kernel(
    const half2* __restrict__ vec, const uint32_t* __restrict__ mat,
    half* __restrict__ mul, const half* __restrict__ scales,
    const uint32_t* __restrict__ zeros, const int* __restrict__ g_idx,
    int batch, int height, int width) {
  int zero_width = width / 4;
  int vec_height = height * 2;
  const int blockwidth2 = BLOCK_KN_SIZE / 2;
  int b = blockIdx.y * BLOCK_M_SIZE_MAX;
  int b_end = min(BLOCK_M_SIZE_MAX, batch - b);
  int h = BLOCK_KN_SIZE * blockIdx.z / 4;
  int h_end = min(BLOCK_KN_SIZE / 4, height - h) * 2;
  int w = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;

  __shared__ half2 blockvec[BLOCK_M_SIZE_MAX][blockwidth2];
  if (threadIdx.x < h_end) {
    for (int m = 0; m < b_end; ++m) {
      blockvec[m][threadIdx.x] =
          vec[(m + b) * vec_height + blockIdx.z * BLOCK_KN_SIZE / 2 +
              threadIdx.x];
    }
  }

  if (blockIdx.z == 0) {
    for (int m = 0; m < b_end; m++) mul[(b + m) * width + w] = __int2half_rn(0);
  }
  __syncthreads();

  int i = width * h + w;
  int g_h = h * 4;
  int k = 0;
  int z_w = w / 4;
  int z_mod = (w % 4) * 8;
  half2 res2;
  half res[BLOCK_M_SIZE_MAX] = {};

  unsigned int tmp;
  while (k < h_end) {
    tmp = mat[i];
    half2 scales_tmp[2];
    half2 zeros_tmp[2];
    for (int tmp_k = 0; tmp_k < 2; tmp_k++) {
      int g = g_idx[g_h + (k + tmp_k) * 2];
      int g2 = g_idx[g_h + (k + tmp_k) * 2 + 1];
      half scale_f = scales[g * width + w];
      half scale_f2 = scales[g2 * width + w];
      half2 scale = __halves2half2(scale_f, scale_f2);
      half2 zero = __halves2half2(
          __hmul(scale_f,
                 __int2half_rn(
                     -((zeros[g * zero_width + z_w] >> z_mod) & 0xff) - 1)),
          __hmul(scale_f2,
                 __int2half_rn(
                     -((zeros[g2 * zero_width + z_w] >> z_mod) & 0xff) - 1)));
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
    }
    for (int m = 0; m < b_end; m++) {
#ifndef USE_ROCM
      res2 = {};
#else
      res2.x = __half_as_ushort(__float2half(0));
      res2.y = __half_as_ushort(__float2half(0));
#endif
      half2 v12 = __halves2half2(__int2half_rn(tmp & 0xFF),
                                 __int2half_rn((tmp >> 8) & 0xFF));
      res2 = __hfma2(__hfma2(v12, scales_tmp[0], zeros_tmp[0]),
                     blockvec[m][k + 0], res2);
      half2 v34 = __halves2half2(__int2half_rn((tmp >> 16) & 0xFF),
                                 __int2half_rn((tmp >> 24) & 0xFF));
      res2 = __hfma2(__hfma2(v34, scales_tmp[1], zeros_tmp[1]),
                     blockvec[m][k + 1], res2);
#ifndef USE_ROCM
      res[m] = __hadd(res[m], __hadd(res2.x, res2.y));
#else
      res[m] = __hadd(
          res[m], __hadd(__ushort_as_half(res2.x), __ushort_as_half(res2.y)));
#endif
    }
    i += width;
    k += 2;
  }
  for (int m = 0; m < b_end; m++) {
    atomicAdd(&mul[(b + m) * width + w], res[m]);
  }
}

extern "C" void gemm_half_q_half_alt(const void* a, const uint32_t* b_q_weight,
                          const uint32_t* b_gptq_qzeros,
                          const void* b_gptq_scales, const int* b_g_idx,
                          void* c, int size_m, int size_n, int size_k,
                          int bit, int64_t stream_) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE);
  gridDim.y = DIVIDE(size_m, BLOCK_M_SIZE_MAX);
  gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);

  auto kernel = gemm_half_q_half_alt_4bit_kernel;
  if (bit == 8) {
    kernel = gemm_half_q_half_alt_8bit_kernel;
  }

  const cudaStream_t stream = (cudaStream_t)stream_;
  kernel<<<gridDim, blockDim, 0, stream>>>(
      (const half2*)(const half*)a, b_q_weight, (half*)c, (const half*)b_gptq_scales, b_gptq_qzeros, b_g_idx,
      size_m, size_k / 32 * bit, size_n);
}