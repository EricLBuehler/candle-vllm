use candle_core::{DType, Device, IndexOp, Tensor};

use crate::{
    openai::responses::APIError,
    paged_attention::attn_bias::{BlockDiagonalCausalMask, LowerTriangularMaskWithTensorBias},
};

use super::{input_metadata::InputMetadata, PagedAttention};

pub fn _memory_efficient_attention(
    this: &PagedAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    input_metadata: &mut InputMetadata,
    seq_len: usize,
    batch_size: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor, APIError> {
    let (query, key, value) = if this.num_key_value_heads != this.num_attention_heads {
        let query = query
            .reshape((
                *query.shape().dims().first().unwrap(),
                this.num_key_value_heads,
                this.num_queries_per_kv,
                *query.shape().dims().last().unwrap(),
            ))
            .map_err(APIError::from)?;

        let key = key
            .i((.., .., .., ..))
            .map_err(APIError::from)?
            .unsqueeze(2)
            .map_err(APIError::from)?
            .expand((
                *key.shape().dims().first().unwrap(),
                this.num_key_value_heads,
                this.num_queries_per_kv,
                *key.shape().dims().last().unwrap(),
            ))
            .map_err(APIError::from)?;

        let value = value
            .i((.., .., .., ..))
            .map_err(APIError::from)?
            .unsqueeze(2)
            .map_err(APIError::from)?
            .expand((
                *value.shape().dims().first().unwrap(),
                this.num_key_value_heads,
                this.num_queries_per_kv,
                *value.shape().dims().last().unwrap(),
            ))
            .map_err(APIError::from)?;

        (query, key, value)
    } else {
        (query, key, value)
    };

    if input_metadata.attn_bias.is_none() {
        if let Some(alibi_slopes) = &this.alibi_slopes {
            //make alibi bias
            let bias = Tensor::arange(
                0f64,
                TryInto::<i32>::try_into(seq_len)
                    .unwrap()
                    .try_into()
                    .unwrap(),
                &device,
            )
            .map_err(APIError::from)?
            .to_dtype(dtype)
            .map_err(APIError::from)?;
            let bias = (bias.unsqueeze(0).map_err(APIError::from)?
                - bias.unsqueeze(1).map_err(APIError::from)?)
            .map_err(APIError::from)?
            .to_device(alibi_slopes.device())
            .map_err(APIError::from)?;

            let padded_len = ((seq_len + 7) / 8) * 8;
            //use Tensor::empty, huggingface/candle#1374
            let mut bias_new = Tensor::zeros(
                (
                    batch_size,
                    alibi_slopes.shape().dims()[0],
                    seq_len,
                    padded_len,
                ),
                dtype,
                &device,
            )
            .map_err(APIError::from)?
            .i((.., .., .., ..seq_len))
            .map_err(APIError::from)?;

            bias_new = bias_new
                .slice_assign(&[.., .., .., ..], &bias)
                .map_err(APIError::from)?;

            bias_new = bias_new
                .mul(
                    &alibi_slopes
                        .i(..)
                        .map_err(APIError::from)?
                        .unsqueeze(1)
                        .map_err(APIError::from)?
                        .unsqueeze(2)
                        .map_err(APIError::from)?,
                )
                .map_err(APIError::from)?;
            let attn_bias = LowerTriangularMaskWithTensorBias::new(bias_new);
            input_metadata.attn_bias = Some(Box::new(attn_bias));
        } else {
            let mut attn_bias = BlockDiagonalCausalMask::from_seqlens(
                [seq_len.try_into().unwrap()].repeat(batch_size),
                None,
            )
            .map_err(APIError::from)?;
            if let Some(sliding_window) = this.sliding_window {
                attn_bias = attn_bias
                    .make_local_attention(sliding_window)
                    .map_err(APIError::from)?;
            }
            input_metadata.attn_bias = Some(attn_bias);
        }
    }

    let (query, key, value) = if this.alibi_slopes.is_none() {
        (
            query.unsqueeze(0).map_err(APIError::from)?,
            key.unsqueeze(0).map_err(APIError::from)?,
            value.unsqueeze(0).map_err(APIError::from)?,
        )
    } else {
        assert_eq!(query.shape().dims().len(), key.shape().dims().len());
        assert_eq!(value.shape().dims().len(), key.shape().dims().len());
        assert!(query.shape().dims().len() == 3 || query.shape().dims().len() == 4);
        if query.shape().dims().len() == 3 {
            (
                query
                    .reshape((
                        batch_size,
                        seq_len,
                        query.shape().dims()[1],
                        query.shape().dims()[2],
                    ))
                    .map_err(APIError::from)?,
                key.reshape((
                    batch_size,
                    seq_len,
                    key.shape().dims()[1],
                    key.shape().dims()[2],
                ))
                .map_err(APIError::from)?,
                value
                    .reshape((
                        batch_size,
                        seq_len,
                        value.shape().dims()[1],
                        value.shape().dims()[2],
                    ))
                    .map_err(APIError::from)?,
            )
        } else {
            (
                query
                    .reshape((
                        batch_size,
                        seq_len,
                        query.shape().dims()[1],
                        query.shape().dims()[2],
                        query.shape().dims()[3],
                    ))
                    .map_err(APIError::from)?,
                key.reshape((
                    batch_size,
                    seq_len,
                    key.shape().dims()[1],
                    key.shape().dims()[2],
                    key.shape().dims()[3],
                ))
                .map_err(APIError::from)?,
                value
                    .reshape((
                        batch_size,
                        seq_len,
                        value.shape().dims()[1],
                        value.shape().dims()[2],
                        value.shape().dims()[3],
                    ))
                    .map_err(APIError::from)?,
            )
        }
    };

    todo!("memory_efficient_attention_forward");
}
