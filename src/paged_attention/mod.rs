use candle_core::{DType, Device, Result, Tensor, D};

use crate::{
    backend::{paged_attention, reshape_and_cache},
    openai::responses::APIError,
    try_api,
};

use self::input_metadata::InputMetadata;
mod attn_bias;
pub(crate) mod input_metadata;
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
    ) -> Result<Self> {
        let num_key_value_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let num_queries_per_kv = num_attention_heads / num_key_value_heads;
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            Some(Tensor::new(alibi_slopes, &device)?)
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
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        mut key_cache: Option<Tensor>,
        mut value_cache: Option<Tensor>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let dims = input_metadata.slot_mapping.dims();
        let slot_mapping = if dims.len() > 1 {
            input_metadata
                .slot_mapping
                .flatten(0, input_metadata.slot_mapping.dims().len())?
        } else {
            input_metadata.slot_mapping.clone()
        };

        let att = match attention_mask {
            None => None,
            Some(mask) => {
                let att = (query.matmul(&key.t()?)? * self.scale as f64)?;
                let att = att.broadcast_add(mask)?;
                let att = candle_nn::ops::softmax(&att, D::Minus1)?;
                Some(att.matmul(&value)?)
            }
        };

        // // paged-attn expects (b_sz, seq_len, nheads, head_dim)
        let query = query.transpose(1, 2)?;
        let key = key.transpose(1, 2)?;
        let value = value.transpose(1, 2)?;

        //format [batch_size, num_tokens, num_heads, head_size]
        let (batch_size, seq_len, attention_heads, head_size) = query.shape().dims4()?;
        let (_, _, key_value_heads, _) = key.shape().dims4()?;
        let query = query.reshape(((), attention_heads, head_size))?;
        let key = key.reshape(((), key_value_heads, head_size))?;
        let value = value.reshape(((), key_value_heads, head_size))?;

        // key: Tensor,              // [num_tokens, num_heads, head_size]
        // value: Tensor,            // [num_tokens, num_heads, head_size]
        // key_cache: &mut Tensor,   // [num_blocks, num_heads, head_size/x, block_size, x] 48,32,16,16,8
        // value_cache: &mut Tensor, // [num_blocks, num_heads, head_size, block_size] 48,32,128,16
        // slot_mapping: Tensor,     // [num_tokens]
        if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            let _ = reshape_and_cache(
                &key,
                &value,
                &key_cache.as_mut().unwrap(),
                &value_cache.as_mut().unwrap(),
                &slot_mapping,
            )?;
        }

        if att.is_some() {
            //prefill result
            return Ok(att.unwrap());
        }
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
        paged_attention(
            &query,
            &key_cache.as_ref().unwrap(),
            &value_cache.as_ref().unwrap(),
            &input_metadata.block_tables.as_ref().unwrap(),
            &input_metadata.context_lens.as_ref().unwrap(),
            input_metadata.max_context_len.unwrap(),
            self.scale,
        )
    }
}
