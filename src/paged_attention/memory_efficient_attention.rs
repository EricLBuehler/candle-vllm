use candle_core::{DType, Device, IndexOp, Shape, Tensor, D};

use crate::{
    openai::responses::APIError,
    paged_attention::attn_bias::{BlockDiagonalCausalMask, LowerTriangularMaskWithTensorBias},
};

use super::{input_metadata::InputMetadata, PagedAttention};

#[allow(clippy::too_many_arguments)]
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
                device,
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
                device,
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
    //candle_flash_attn::flash_attn;
    let l = query.dim(D::Minus2).map_err(APIError::from)?;
    let s = key.dim(D::Minus2).map_err(APIError::from)?;

    scaled_dot_product_attention(
        &query,
        &key,
        &value,
        &input_metadata
            .attn_bias
            .as_ref()
            .unwrap()
            .materialize(&Shape::from_dims(&[l, s]), query.dtype(), device)
            .map_err(APIError::from)?,
        None,
        Some(this.scale as f64),
    )
}

// https://github.com/mokeyish/candle-ext/blob/main/src/scaled_dot_product_attention.rs

/// Computes scaled dot product attention on query, key and value tensors,
/// using an optional attention mask if passed, and applying dropout
/// if a probability greater than 0.0 is specified.
///
/// # Arguments
/// - query   - Query tensor; shape (N, ..., L, E)
/// - key     - Key tensor; shape (N, ..., S, E)
/// - value   - Value tensor; shape (N, ..., S, E)
///
/// https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
/// # Errors
///
/// This function will return an error if .
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_bias: &Tensor,
    dropout_p: Option<f32>,
    scale: Option<f64>,
) -> Result<Tensor, APIError> {
    let dim = query.dim(D::Minus1).map_err(APIError::from)?;

    let scale_factor = if let Some(scale) = scale {
        scale
    } else {
        1.0 / (dim as f64).sqrt()
    };

    let mut attn_weights = (query
        .matmul(
            &key.transpose(D::Minus2, D::Minus1)
                .map_err(APIError::from)?
                .contiguous()
                .map_err(APIError::from)?,
        )
        .map_err(APIError::from)?
        * scale_factor)
        .map_err(APIError::from)?;

    attn_weights = (&attn_weights
        + attn_bias
            .broadcast_as(attn_weights.shape())
            .map_err(APIError::from)?)
    .map_err(APIError::from)?;
    attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights).map_err(APIError::from)?;

    if let Some(drop_p) = dropout_p {
        attn_weights = candle_nn::ops::dropout(&attn_weights, drop_p).map_err(APIError::from)?;
    }
    attn_weights.matmul(value).map_err(APIError::from)
}
