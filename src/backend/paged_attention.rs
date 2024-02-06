use candle_core::{DType, Tensor};

use crate::openai::responses::APIError;

#[allow(clippy::too_many_arguments)]
fn paged_attention_v1_launcher(
    query: Tensor,            // [num_seqs, num_heads, head_size]
    key_cache: Tensor,        // [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache: Tensor,      // [num_blocks, num_heads, head_size, block_size]
    num_key_value_heads: i32, // [num_heads]
    scale: f32,
    block_tables: Tensor, // [num_seqs, max_num_blocks_per_seq]
    context_lens: Tensor, // [num_seqs]
    block_size: usize,
    max_context_len: usize,
    alibi_slopes: Option<Tensor>,
    dtype: DType,
    is_fp8_e5m2_kv_cache: bool,
) -> Tensor {
    todo!()
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v1(
    query: Tensor,            // [num_seqs, num_heads, head_size]
    key_cache: Tensor,        // [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache: Tensor,      // [num_blocks, num_heads, head_size, block_size]
    num_key_value_heads: i32, // [num_heads]
    scale: f32,
    block_tables: Tensor, // [num_seqs, max_num_blocks_per_seq]
    context_lens: Tensor, // [num_seqs]
    block_size: usize,
    max_context_len: usize,
    alibi_slopes: Option<Tensor>,
    kv_cache_dtype: &str,
) -> Result<Tensor, APIError> {
    let query_dtype = query.dtype();
    if kv_cache_dtype == "auto" {
        match query_dtype {
            DType::F32 | DType::F16 | DType::BF16 => Ok(paged_attention_v1_launcher(
                query,
                key_cache,
                value_cache,
                num_key_value_heads,
                scale,
                block_tables,
                context_lens,
                block_size,
                max_context_len,
                alibi_slopes,
                query_dtype,
                false,
            )),
            _ => Err(APIError::new(format!(
                "Unsupported data type {:?}",
                query_dtype
            ))),
        }
    } else if kv_cache_dtype == "fp8_e5m2" {
        match query_dtype {
            DType::F32 | DType::F16 | DType::BF16 => Ok(paged_attention_v1_launcher(
                query,
                key_cache,
                value_cache,
                num_key_value_heads,
                scale,
                block_tables,
                context_lens,
                block_size,
                max_context_len,
                alibi_slopes,
                query_dtype,
                true,
            )),
            _ => Err(APIError::new(format!(
                "Unsupported data type {:?}",
                query_dtype
            ))),
        }
    } else {
        Err(APIError::new(format!(
            "Unsupported data type {:?}",
            query_dtype
        )))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v2(
    _exp_sums: Tensor,
    _max_logits: Tensor,
    _query: Tensor,
    _key_cache: Tensor,
    _value_cache: Tensor,
    _num_key_value_heads: i32,
    _scale: f32,
    _block_tables: Tensor,
    _context_lens: Tensor,
    _block_size: usize,
    _max_context_len: usize,
    _alibi_slopes: Option<Tensor>,
) -> Tensor {
    todo!();
}

/*
#ifndef USE_ROCM
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)
#else
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#endif

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                                  \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                                       \
    ((void*)vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,   \
      IS_FP8_E5M2_KV_CACHE>), shared_mem_size);                                               \
  vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,             \
  IS_FP8_E5M2_KV_CACHE><<<grid, block, shared_mem_size, stream>>>(                            \
    out_ptr,                                                                                  \
    query_ptr,                                                                                \
    key_cache_ptr,                                                                            \
    value_cache_ptr,                                                                          \
    num_kv_heads,                                                                             \
    scale,                                                                                    \
    block_tables_ptr,                                                                         \
    context_lens_ptr,                                                                         \
    max_num_blocks_per_seq,                                                                   \
    alibi_slopes_ptr,                                                                         \
    q_stride,                                                                                 \
    kv_block_stride,                                                                          \
    kv_head_stride);

// TODO(woosuk): Tune NUM_THREADS.
template<
  typename T,
  typename CACHE_T,
  int BLOCK_SIZE,
  bool IS_FP8_E5M2_KV_CACHE,
  int NUM_THREADS = 128>
void paged_attention_v1_launcher(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr = alibi_slopes ?
    reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
    : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_E5M2_KV_CACHE)       \
  paged_attention_v1_launcher<T, CACHE_T, BLOCK_SIZE, IS_FP8_E5M2_KV_CACHE>( \
    out,                                                                     \
    query,                                                                   \
    key_cache,                                                               \
    value_cache,                                                             \
    num_kv_heads,                                                            \
    scale,                                                                   \
    block_tables,                                                            \
    context_lens,                                                            \
    max_context_len,                                                         \
    alibi_slopes);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V1_LAUNCHER_BLOCK_SIZE(T, CACHE_T, IS_FP8_E5M2_KV_CACHE) \
  switch (block_size) {                                               \
    case 8:                                                           \
      CALL_V1_LAUNCHER(T, CACHE_T, 8, IS_FP8_E5M2_KV_CACHE);          \
      break;                                                          \
    case 16:                                                          \
      CALL_V1_LAUNCHER(T, CACHE_T, 16, IS_FP8_E5M2_KV_CACHE);         \
      break;                                                          \
    case 32:                                                          \
      CALL_V1_LAUNCHER(T, CACHE_T, 32, IS_FP8_E5M2_KV_CACHE);         \
      break;                                                          \
    default:                                                          \
      TORCH_CHECK(false, "Unsupported block size: ", block_size);     \
      break;                                                          \
  }

*/
