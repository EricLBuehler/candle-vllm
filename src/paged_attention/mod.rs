use candle_core::{DType, Device, Tensor};

use crate::openai::responses::APIError;

use self::input_metadata::InputMetadata;
mod input_metadata;

const _PARTION_SIZE: usize = 512;

pub struct PagedAttention {
    num_attention_heads: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
    num_queries_per_kv: usize,
    head_mapping: Tensor,
}

impl PagedAttention {
    pub fn new(
        num_attention_heads: usize,
        head_dim: usize,
        scale: f32,
        num_key_value_heads: Option<usize>,
        sliding_window: Option<usize>,
        device: Device,
    ) -> Result<Self, APIError> {
        let num_key_value_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let num_queries_per_kv = num_attention_heads / num_key_value_heads;
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
    fn single_query_kv_attention(
        &self,
        output: Tensor,
        query: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        input_metadata: InputMetadata,
        alibi_slopes: Option<Tensor>,
    ) -> Result<(), APIError> {
        let block_size = value_cache.shape().dims().get(3).unwrap();
        let (num_seqs, num_heads, head_size) = query.shape().dims3().map_err(APIError::from)?;
        let max_num_partions = (input_metadata.max_context_len + _PARTION_SIZE - 1) / _PARTION_SIZE;

        let use_v1 = input_metadata.max_context_len <= 8192
            && (max_num_partions == 1 || num_seqs * num_heads > 512);
        if use_v1 {
            //Run PagedAttention V1
            todo!()
        } else {
            //Run PagedAttention V2
            assert_eq!(_PARTION_SIZE % block_size, 0);

            let tmp_output = Tensor::zeros(
                (num_seqs, num_heads, max_num_partions, head_size),
                output.dtype(),
                output.device(),
            )
            .map_err(APIError::from)?;
            let exp_sums = Tensor::zeros(
                (num_seqs, num_heads, max_num_partions),
                DType::F32,
                output.device(),
            )
            .map_err(APIError::from)?;
            let max_logits = exp_sums.zeros_like().map_err(APIError::from)?;

            todo!()
        }
        Ok(())
    }
}
