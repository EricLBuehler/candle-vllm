use candle_core::{DType, Device, Tensor};
use tch::Kind;

use crate::{
    openai::responses::APIError,
    paged_attention::bindings::{
        paged_attention_v1, paged_attention_v2, reshape_and_cache, Optional, Storage,
    },
};

use self::input_metadata::InputMetadata;
mod attn_bias;
mod bindings;
pub(crate) mod input_metadata;
mod memory_efficient_attention;
use memory_efficient_attention::_memory_efficient_attention;
mod cache_engine;
mod scheduler;
pub(crate) mod utils;

const _PARTITION_SIZE: usize = 512;

#[allow(dead_code)]
pub struct PagedAttention {
    num_attention_heads: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
    num_queries_per_kv: usize,
    head_mapping: Tensor,
    alibi_slopes: Option<Tensor>,
}

fn convert_candle_to_tch(candle: &mut Tensor) -> tch::Tensor {
    let output_kind = match candle.dtype() {
        DType::BF16 => Kind::BFloat16,
        DType::F16 => Kind::Float,
        DType::F32 => Kind::Float,
        DType::F64 => Kind::Float,
        DType::I64 => Kind::Int64,
        DType::U8 => Kind::Uint8,
        DType::U32 => Kind::Int,
    };

    let mut dims = Vec::new();
    for dim in candle.dims() {
        dims.push(*dim as i64);
    }

    tch::Tensor::from_data_size(
        &candle
            .to_vec3::<u8>()
            .unwrap()
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>()[..],
        &dims[..],
        output_kind,
    )
}

fn convert_tch_to_ptr(
    tch: &mut tch::Tensor,
) -> (*mut torch_sys::C_tensor, &mut torch_sys::C_tensor) {
    (tch.as_mut_ptr(), unsafe { &mut *tch.as_mut_ptr() })
}

fn convert_option_to_optional(option: Option<Tensor>) -> Optional<torch_sys::C_tensor> {
    let storage: Storage<torch_sys::C_tensor> = if let Some(mut value) = option {
        let mut slopes_tchtensor = convert_candle_to_tch(&mut value);
        let (_, slopes_ctensor) = convert_tch_to_ptr(&mut slopes_tchtensor);
        Storage {
            value_: *slopes_ctensor,
        }
    } else {
        Storage {
            dummy_: std::os::raw::c_uchar::from(0),
        }
    };

    Optional {
        init_: true,
        storage_: storage,
    }
}

impl PagedAttention {
    pub fn new(
        num_attention_heads: usize,
        head_dim: usize,
        scale: f32,
        num_key_value_heads: Option<usize>,
        sliding_window: Option<usize>,
        device: Device,
        alibi_slopes: Option<Vec<f64>>,
    ) -> Result<Self, APIError> {
        let num_key_value_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let num_queries_per_kv = num_attention_heads / num_key_value_heads;
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            Some(Tensor::new(alibi_slopes, &device).map_err(APIError::from)?)
        } else {
            None
        };
        Ok(Self {
            num_attention_heads,
            head_dim,
            num_key_value_heads,
            scale,
            sliding_window,
            num_queries_per_kv,
            head_mapping: Tensor::arange(0u32, num_key_value_heads as u32, &device)
                .map_err(APIError::from)?
                .repeat(num_queries_per_kv)
                .map_err(APIError::from)?,
            alibi_slopes,
        })
    }

    /// Args:
    /// output: shape = [num_generation_tokens, num_heads, head_size]
    ///
    /// query: shape = [num_generation_tokens, num_heads, head_size]
    ///
    /// key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
    ///     block_size, x]
    ///
    /// value_cache: shape = [num_blocks, num_kv_heads, head_size,
    ///     block_size]
    ///
    /// input_metadata: metadata for paged attention.
    ///
    /// alibi_slopes: shape = [num_heads]
    pub fn _paged_attention(
        &mut self,
        mut query: Tensor,
        mut key_cache: Tensor,
        mut value_cache: Tensor,
        input_metadata: &mut InputMetadata,
        alibi_slopes: Option<Tensor>,
    ) -> Result<Tensor, APIError> {
        //use Tensor::empty, huggingface/candle#1374
        let mut output = query.zeros_like().map_err(APIError::from)?;

        let block_size = *value_cache.shape().dims().get(3).unwrap();
        let (num_seqs, num_heads, head_size) = query.shape().dims3().map_err(APIError::from)?;
        let max_num_partitions =
            (input_metadata.max_context_len.unwrap() + _PARTITION_SIZE - 1) / _PARTITION_SIZE;

        let use_v1 = input_metadata.max_context_len.unwrap() <= 8192
            && (max_num_partitions == 1 || num_seqs * num_heads > 512);
        if use_v1 {
            //Run PagedAttention V1
            let mut output_tch = convert_candle_to_tch(&mut output);
            let (output_ptr, _) = convert_tch_to_ptr(&mut output_tch);

            let mut query_tch = convert_candle_to_tch(&mut query);
            let (query_ptr, _) = convert_tch_to_ptr(&mut query_tch);

            let mut keycache_tch = convert_candle_to_tch(&mut key_cache);
            let (keycache_ptr, _) = convert_tch_to_ptr(&mut keycache_tch);

            let mut valuecache_tch = convert_candle_to_tch(&mut value_cache);
            let (valuecache_ptr, _) = convert_tch_to_ptr(&mut valuecache_tch);

            let mut head_mapping_tch = convert_candle_to_tch(&mut self.head_mapping);
            let (head_mapping_ptr, _) = convert_tch_to_ptr(&mut head_mapping_tch);

            let mut block_tbl_tch =
                convert_candle_to_tch(input_metadata.block_tables.as_mut().unwrap());
            let (block_tbl_ptr, _) = convert_tch_to_ptr(&mut block_tbl_tch);

            let mut ctxt_lens_tch =
                convert_candle_to_tch(input_metadata.context_lens.as_mut().unwrap());
            let (ctxt_lens_ptr, _) = convert_tch_to_ptr(&mut ctxt_lens_tch);

            let optional = convert_option_to_optional(alibi_slopes);

            unsafe {
                paged_attention_v1(
                    output_ptr,
                    query_ptr,
                    keycache_ptr,
                    valuecache_ptr,
                    head_mapping_ptr,
                    self.scale,
                    block_tbl_ptr,
                    ctxt_lens_ptr,
                    block_size.try_into().unwrap(),
                    input_metadata.max_context_len.unwrap().try_into().unwrap(),
                    &optional as *const Optional<torch_sys::C_tensor>,
                )
            };
        } else {
            //Run PagedAttention V2
            assert_eq!(_PARTITION_SIZE % block_size, 0);

            let mut tmp_output = Tensor::zeros(
                //use Tensor::empty, huggingface/candle#1374
                (num_seqs, num_heads, max_num_partitions, head_size),
                output.dtype(),
                output.device(),
            )
            .map_err(APIError::from)?;
            let mut exp_sums = Tensor::zeros(
                //use Tensor::empty, huggingface/candle#1374
                (num_seqs, num_heads, max_num_partitions),
                DType::F32,
                output.device(),
            )
            .map_err(APIError::from)?;
            let mut max_logits = exp_sums.zeros_like().map_err(APIError::from)?; //use Tensor::empty, huggingface/candle#1374

            let mut output_tch = convert_candle_to_tch(&mut output);
            let (output_ptr, _) = convert_tch_to_ptr(&mut output_tch);

            let mut exp_sums_tch = convert_candle_to_tch(&mut exp_sums);
            let (exp_sums_ptr, _) = convert_tch_to_ptr(&mut exp_sums_tch);

            let mut max_logits_tch = convert_candle_to_tch(&mut max_logits);
            let (max_logits_ptr, _) = convert_tch_to_ptr(&mut max_logits_tch);

            let mut tmp_output_tch = convert_candle_to_tch(&mut tmp_output);
            let (tmp_output_ptr, _) = convert_tch_to_ptr(&mut tmp_output_tch);

            let mut query_tch = convert_candle_to_tch(&mut query);
            let (query_ptr, _) = convert_tch_to_ptr(&mut query_tch);

            let mut keycache_tch = convert_candle_to_tch(&mut key_cache);
            let (keycache_ptr, _) = convert_tch_to_ptr(&mut keycache_tch);

            let mut valuecache_tch = convert_candle_to_tch(&mut value_cache);
            let (valuecache_ptr, _) = convert_tch_to_ptr(&mut valuecache_tch);

            let mut head_mapping_tch = convert_candle_to_tch(&mut self.head_mapping);
            let (head_mapping_ptr, _) = convert_tch_to_ptr(&mut head_mapping_tch);

            let mut block_tbl_tch =
                convert_candle_to_tch(input_metadata.block_tables.as_mut().unwrap());
            let (block_tbl_ptr, _) = convert_tch_to_ptr(&mut block_tbl_tch);

            let mut ctxt_lens_tch =
                convert_candle_to_tch(input_metadata.context_lens.as_mut().unwrap());
            let (ctxt_lens_ptr, _) = convert_tch_to_ptr(&mut ctxt_lens_tch);

            let optional = convert_option_to_optional(alibi_slopes);

            unsafe {
                paged_attention_v2(
                    output_ptr,
                    exp_sums_ptr,
                    max_logits_ptr,
                    tmp_output_ptr,
                    query_ptr,
                    keycache_ptr,
                    valuecache_ptr,
                    head_mapping_ptr,
                    self.scale,
                    block_tbl_ptr,
                    ctxt_lens_ptr,
                    block_size.try_into().unwrap(),
                    input_metadata.max_context_len.unwrap().try_into().unwrap(),
                    &optional as *const Optional<torch_sys::C_tensor>,
                );
            };
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn _normal_attention(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        input_metadata: &mut InputMetadata,
        seq_len: usize,
        batch_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor, APIError> {
        _memory_efficient_attention(
            self,
            query,
            key,
            value,
            input_metadata,
            seq_len,
            batch_size,
            device,
            dtype,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    /// query: shape = [batch_size, seq_len, num_heads * head_size]
    /// key: shape = [batch_size, seq_len, num_kv_heads * head_size]
    /// value: shape = [batch_size, num_kv_heads * head_size]
    /// key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
    ///     block_size, x]
    /// value_cache: shape = [num_blocks, num_kv_heads, head_size,
    ///     block_size]
    /// input_metadata: metadata for paged attention.
    pub fn forward(
        &mut self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mut key_cache: Option<Tensor>,
        mut value_cache: Option<Tensor>,
        input_metadata: &mut InputMetadata,
        dtype: DType,
        device: Device,
    ) -> Result<Tensor, APIError> {
        let (batch_size, seq_len, hidden_size) = query.shape().dims3().map_err(APIError::from)?;
        let query = query
            .reshape(((), self.num_attention_heads, self.head_dim))
            .map_err(APIError::from)?;
        let mut key = key
            .reshape(((), self.num_key_value_heads, self.head_dim))
            .map_err(APIError::from)?;
        let mut value = value
            .reshape(((), self.num_key_value_heads, self.head_dim))
            .map_err(APIError::from)?;
        let mut slot_mapping = input_metadata
            .slot_mappinng
            .flatten(0, input_metadata.slot_mappinng.dims().len())
            .map_err(APIError::from)?;

        if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            let mut key_tch = convert_candle_to_tch(&mut key);
            let (key_ptr, _) = convert_tch_to_ptr(&mut key_tch);

            let mut value_tch = convert_candle_to_tch(&mut value);
            let (value_ptr, _) = convert_tch_to_ptr(&mut value_tch);

            let mut key_cache_tch = convert_candle_to_tch(key_cache.as_mut().unwrap());
            let (key_cache_ptr, _) = convert_tch_to_ptr(&mut key_cache_tch);

            let mut value_cache_tch = convert_candle_to_tch(value_cache.as_mut().unwrap());
            let (value_cache_ptr, _) = convert_tch_to_ptr(&mut value_cache_tch);

            let mut slot_mapping_tch = convert_candle_to_tch(&mut slot_mapping);
            let (slot_mapping_ptr, _) = convert_tch_to_ptr(&mut slot_mapping_tch);

            unsafe {
                reshape_and_cache(
                    key_ptr,
                    value_ptr,
                    key_cache_ptr,
                    value_cache_ptr,
                    slot_mapping_ptr,
                )
            };
        }

        let output = if input_metadata.is_prompt {
            self._normal_attention(
                query,
                key,
                value,
                input_metadata,
                seq_len,
                batch_size,
                &device,
                dtype,
            )?
        } else {
            self._paged_attention(
                query,
                key_cache.unwrap(),
                value_cache.unwrap(),
                input_metadata,
                None,
            )?
        };

        output
            .reshape((batch_size, seq_len, hidden_size))
            .map_err(APIError::from)
    }
}
