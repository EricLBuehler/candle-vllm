use candle_core::{DType, Device, Tensor};

use crate::{
    backend::{paged_attention_v1, paged_attention_v2, reshape_and_cache},
    openai::responses::APIError,
    try_api,
};

use self::input_metadata::InputMetadata;
mod attn_bias;
pub(crate) mod input_metadata;
mod memory_efficient_attention;
use memory_efficient_attention::_memory_efficient_attention;
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
    alibi_slopes: Option<Tensor>,
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
            Some(try_api!(Tensor::new(alibi_slopes, &device)))
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
        query: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        input_metadata: &mut InputMetadata,
        alibi_slopes: Option<Tensor>,
    ) -> Result<Tensor, APIError> {
        let block_size = *value_cache.shape().dims().get(3).unwrap();
        let (num_seqs, num_heads, _head_size) = try_api!(query.shape().dims3());
        let max_num_partitions =
            (input_metadata.max_context_len.unwrap() + _PARTITION_SIZE - 1) / _PARTITION_SIZE;

        let use_v1 = input_metadata.max_context_len.unwrap() <= 8192
            && (max_num_partitions == 1 || num_seqs * num_heads > 512);
        let output = if use_v1 {
            //Run PagedAttention V1
            paged_attention_v1(
                query,
                key_cache,
                value_cache,
                self.num_key_value_heads.try_into().unwrap(),
                self.scale,
                input_metadata.block_tables.as_ref().unwrap().clone(),
                input_metadata.context_lens.as_ref().unwrap().clone(),
                block_size,
                input_metadata.max_context_len.unwrap(),
                alibi_slopes,
                &input_metadata.kv_cache_dtype,
            )?
        } else {
            //Run PagedAttention V2
            assert_eq!(_PARTITION_SIZE % block_size, 0);

            let exp_sums = try_api!(Tensor::zeros(
                (num_seqs, num_heads, max_num_partitions),
                DType::F32,
                query.device(),
            ));
            let max_logits = try_api!(exp_sums.zeros_like());

            paged_attention_v2(
                exp_sums,
                max_logits,
                query,
                key_cache,
                value_cache,
                self.num_key_value_heads.try_into().unwrap(),
                self.scale,
                input_metadata.block_tables.as_ref().unwrap().clone(),
                input_metadata.context_lens.as_ref().unwrap().clone(),
                block_size,
                input_metadata.max_context_len.unwrap(),
                alibi_slopes,
            )
        };
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
        device: &Device,
    ) -> Result<Tensor, APIError> {
        let (batch_size, seq_len, attention_heads, hidden_size) = try_api!(query.shape().dims4());
        let (_, _, key_value_heads, _) = try_api!(key.shape().dims4());

        let key = try_api!(key.reshape(((), key_value_heads, hidden_size)));
        let value = try_api!(value.reshape(((), key_value_heads, hidden_size)));
        let dims = input_metadata.slot_mapping.dims();
        let slot_mapping = if dims.len() > 1 {
            try_api!(input_metadata
            .slot_mapping
            .flatten(0, input_metadata.slot_mapping.dims().len()))
        } else {
            input_metadata.slot_mapping.clone()
        };

        // key: Tensor,              // [num_tokens, num_heads, head_size]
        // value: Tensor,            // [num_tokens, num_heads, head_size]
        // key_cache: &mut Tensor,   // [num_blocks, num_heads, head_size/x, block_size, x] 48,32,16,16,8
        // value_cache: &mut Tensor, // [num_blocks, num_heads, head_size, block_size] 48,32,128,16
        // slot_mapping: Tensor,     // [num_tokens]

        if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            try_api!(unsafe {
                reshape_and_cache(
                    key.clone(),
                    value.clone(),
                    key_cache.as_mut().unwrap(),
                    value_cache.as_mut().unwrap(),
                    slot_mapping,
                )
            });
        }

        let query = try_api!(query.reshape(((), attention_heads, hidden_size)));

    //  Args:
    //  output: shape = [num_generation_tokens, num_heads, head_size]
    // 
    //  query: shape = [num_generation_tokens, num_heads, head_size]
    // 
    //  key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
    //      block_size, x]
    // 
    //  value_cache: shape = [num_blocks, num_kv_heads, head_size,
    //      block_size]
    // 
    //  input_metadata: metadata for paged attention.
    // 
    //  alibi_slopes: shape = [num_heads]
    //  
        let output = if input_metadata.is_prompt {
            self._normal_attention(
                query,
                key,
                value,
                input_metadata,
                seq_len,
                batch_size,
                device,
                dtype,
            )?
        } else {
            self._paged_attention( //TODO
                query,
                key_cache.as_ref().unwrap().clone(),
                value_cache.as_ref().unwrap().clone(),
                input_metadata,
                None,
            )?
        };

        output
            .reshape(((), seq_len, attention_heads * hidden_size))
            .map_err(APIError::from)
    }
}
