
#include <cuda.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <iostream>
#include "marlin/marlin.cuh"
#include "marlin/marlin_dtypes.cuh"
using namespace marlin;

__global__ void gptq_repack_kernel(
  uint32_t* in,
  uint32_t* out,
  int m,
  int n
) {
  uint32_t row = blockIdx.x * 2;
  uint32_t col = blockIdx.y * 64;
  uint32_t t = threadIdx.x;

  // marlin packs 4 16x16 blocks one time;
  const int pad_len = 18;
  __shared__ uint8_t block[4][16][pad_len];

  // unpack
  int block_idx = t / 8;
  int block_offset = t % 8;
  for (int offset = block_offset; offset < 16; offset += 8) {
    uint32_t v1 = in[row * n + col + block_idx * 16 + offset];
    uint32_t v2 = in[(row + 1) * n + col + block_idx * 16 + offset];
#pragma unroll
    for (int i = 0; i < 8; i += 1) {
      block[block_idx][i][offset] = v1 & 0xf;
      v1 >>= 4;
      block[block_idx][i + 8][offset] = v2 & 0xf;
      v2 >>= 4;
    }
  }

  // repack
  uint32_t srow = (t % 4) * 2;
  uint32_t scol = t / 4;

  uint32_t idx[8][2];
  idx[0][0] = srow;     idx[0][1] = scol;
  idx[1][0] = srow + 8; idx[1][1] = scol;
  idx[2][0] = srow;     idx[2][1] = scol + 8;
  idx[3][0] = srow + 8; idx[3][1] = scol + 8;

  idx[4][0] = srow + 1; idx[4][1] = scol;
  idx[5][0] = srow + 9; idx[5][1] = scol;
  idx[6][0] = srow + 1; idx[6][1] = scol + 8;
  idx[7][0] = srow + 9; idx[7][1] = scol + 8;

#pragma unroll
  for (int i = 0; i < 4; i += 1) {
    uint32_t v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      v[j] = block[i][idx[j][0]][idx[j][1]];
    }

    uint32_t pack = (v[7] << 28) | (v[6] << 24) | (v[5] << 20) | (v[4] << 16) |
        (v[3] << 12) | (v[2] << 8) | (v[1] << 4) | v[0];

    out[blockIdx.x * n * 2 + blockIdx.y * 128 + t * 4 + i] = pack;
  }
}

extern "C" void gptq_repack(
  void* in,
  void* out,
  int m,
  int n,
  int64_t stream_
) {

  assert(m % 2 == 0);
  assert(n % 64 == 0);
  const dim3 threads(32);
  // marlin packs 16 x 64 block and gptq packs 8 x 1
  const dim3 blocks(m / 2, n / 64);
  cudaStream_t stream = (cudaStream_t)stream_;
  gptq_repack_kernel<<<blocks, threads, 0, stream>>>(
    (uint32_t*)in,
    (uint32_t*)out,
    m,
    n
  );
}

template <int const num_threads, int const num_bits>
__global__ void awq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr, uint32_t* __restrict__ out_ptr,
    int size_k, int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  int k_tiles = size_k / tile_k_size;
  int n_tiles = size_n / tile_n_size;
  int block_k_tiles = ceildiv(k_tiles, gridDim.x);

  auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  constexpr int tile_n_ints = tile_n_size / pack_factor;

  constexpr int stage_n_threads = tile_n_ints / 4;
  constexpr int stage_k_threads = tile_k_size;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * tile_n_size;
    int first_n_packed = first_n / pack_factor;

    int4* sh_ptr = sh + stage_size * pipe;

    if (threadIdx.x < stage_size) {
      auto k_id = threadIdx.x / stage_n_threads;
      auto n_id = threadIdx.x % stage_n_threads;

      int first_k = k_tile_id * tile_k_size;

      cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                reinterpret_cast<int4 const*>(
                    &(b_q_weight_ptr[(first_k + k_id) * (size_n / pack_factor) +
                                     first_n_packed + (n_id * 4)])));
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    auto warp_id = threadIdx.x / 32;
    auto th_id = threadIdx.x % 32;

    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * 2;

    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int cur_n = warp_id * 16 + tc_col;
    int cur_n_packed = cur_n / pack_factor;
    int cur_n_pos = cur_n % pack_factor;

    constexpr int sh_stride = tile_n_ints;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    // Undo interleaving
    int cur_n_pos_unpacked;
    if constexpr (num_bits == 4) {
      constexpr int undo_pack[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      cur_n_pos_unpacked = undo_pack[cur_n_pos];
    } else {
      constexpr int undo_pack[4] = {0, 2, 1, 3};
      cur_n_pos_unpacked = undo_pack[cur_n_pos];
    }

    uint32_t vals[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int cur_elem = tc_row + tc_offsets[i];

      int packed_src_0 = sh_stage_int_ptr[cur_n_packed + sh_stride * cur_elem];
      int packed_src_1 = sh_stage_int_ptr[cur_n_packed + (8 / pack_factor) +
                                          sh_stride * cur_elem];

      vals[i] = (packed_src_0 >> (cur_n_pos_unpacked * num_bits)) & mask;
      vals[4 + i] = (packed_src_1 >> (cur_n_pos_unpacked * num_bits)) & mask;
    }

    constexpr int tile_size = tile_k_size * tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;

    // Result of:
    // https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
    if constexpr (num_bits == 4) {
      constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else {
      constexpr int pack_idx[4] = {0, 2, 1, 3};

      uint32_t res1 = 0;
      uint32_t res2 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        res1 |= vals[pack_idx[i]] << (i * 8);
        res2 |= vals[4 + pack_idx[i]] << (i * 8);
      }

      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };
#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id,
                        n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }
}

#define CALL_IF2(NUM_BITS)                                                   \
  else if (num_bits == NUM_BITS) {                                          \
    cudaFuncSetAttribute(                                                   \
        awq_marlin_repack_kernel<repack_threads, NUM_BITS>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);       \
    awq_marlin_repack_kernel<repack_threads, NUM_BITS>      \
        <<<blocks, repack_threads, max_shared_mem, stream>>>(       \
            weight_ptr, out_ptr, size_k, size_n);                       \
  }

extern "C" void awq_repack(void* in, void* out, int k,
                                int n, int num_bits, int64_t stream_) {

  //in_dim 4096, out_dim 1024 (/pack_factor)
  //ws shape [4096, 128]
  //out_shape [256, 2048]

   //recover original size_k and size_n
   int const pack_factor = 32 / num_bits;
   int size_k = k;
   int size_n = n * pack_factor;

  // Verify compatibility with marlin tile of 16x64
  CHECK(size_k % marlin::tile_k_size == 0, "size_k = ", size_k,
              " is not divisible by tile_k_size = ", marlin::tile_k_size);
  CHECK(size_n % marlin::tile_n_size == 0, "size_n = ", size_n,
              " is not divisible by tile_n_size = ", marlin::tile_n_size);

  CHECK(num_bits == 4 || num_bits == 8,
              "num_bits must be 4 or 8. Got = ", num_bits);
  cudaStream_t stream = (cudaStream_t)stream_;

  uint32_t const* weight_ptr =
      reinterpret_cast<uint32_t const*>(in);
  uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out);

  // cudaPointerAttributes attributes;
  // cudaPointerGetAttributes(
  //   &attributes,
  //   in
  // );

  int blocks;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, 0);
  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  CHECK(max_shared_mem > 0, "error max_shared_mem");

  if (false) {
  }
  CALL_IF2(4)
  CALL_IF2(8)
  else {
    CHECK(false, "Unsupported repack config: num_bits = ", num_bits);
  }
}