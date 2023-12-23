#include <stdint.h>

template<typename scalar_t>
__device__ void reshape_and_cache_internal_kernel(
  const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  scalar_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, blosudo ck_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                + head_idx * (head_size / x) * block_size * x
                                + x_idx * block_size * x
                                + block_offset * x
                                + x_offset;
    const int64_t tgt_value_idx = block_idx * num_heads * head_size * block_size
                                  + head_idx * head_size * block_size
                                  + head_offset * block_size
                                  + block_offset;
    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

// Monomorphize the generics ourselves
extern "C" __global__ void reshape_and_cache_kernel_u8(
  const uint8_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const uint8_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  uint8_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  uint8_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  reshape_and_cache_internal_kernel<uint8_t>(key, value, key_cache, value_cache, slot_mapping, key_stride, value_stride, num_heads, head_size, block_size, x);
}

extern "C" __global__ void reshape_and_cache_kernel_u32(
  const uint32_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const uint32_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  uint32_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  uint32_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  reshape_and_cache_internal_kernel<uint32_t>(key, value, key_cache, value_cache, slot_mapping, key_stride, value_stride, num_heads, head_size, block_size, x);
}

extern "C" __global__ void reshape_and_cache_kernel_i64(
  const int64_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const int64_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  int64_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  int64_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  reshape_and_cache_internal_kernel<int64_t>(key, value, key_cache, value_cache, slot_mapping, key_stride, value_stride, num_heads, head_size, block_size, x);
}

extern "C" __global__ void reshape_and_cache_kernel_f32(
  const float* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const float* __restrict__ value,         // [num_tokens, num_heads, head_size]
  float* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  float* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  reshape_and_cache_internal_kernel<float>(key, value, key_cache, value_cache, slot_mapping, key_stride, value_stride, num_heads, head_size, block_size, x);
}

extern "C" __global__ void reshape_and_cache_kernel_f64(
  const double* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const double* __restrict__ value,         // [num_tokens, num_heads, head_size]
  double* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  double* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  reshape_and_cache_internal_kernel<double>(key, value, key_cache, value_cache, slot_mapping, key_stride, value_stride, num_heads, head_size, block_size, x);
}

extern "C" __global__ void reshape_and_cache_kernel_f16(
  const int16_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const int16_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  int16_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  int16_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  reshape_and_cache_internal_kernel<int16_t>(key, value, key_cache, value_cache, slot_mapping, key_stride, value_stride, num_heads, head_size, block_size, x);
}

extern "C" __global__ void reshape_and_cache_kernel_bf16(
  const int16_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const int16_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  int16_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  int16_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  reshape_and_cache_internal_kernel<int16_t>(key, value, key_cache, value_cache, slot_mapping, key_stride, value_stride, num_heads, head_size, block_size, x);
}
