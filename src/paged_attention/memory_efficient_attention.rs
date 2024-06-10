use candle_core::{DType, Device, IndexOp, Shape, Tensor, D};

use crate::{
    openai::responses::APIError,
    paged_attention::attn_bias::{BlockDiagonalCausalMask, LowerTriangularMaskWithTensorBias},
    try_api,
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
    if true {
        scaled_dot_product_attention(&query, &key, &value, None, None, this.scale)
    } else {
        //problems to be fixed
        let (query, key, value) = if this.num_key_value_heads != this.num_attention_heads {
            let query = try_api!(query.reshape((
                *query.shape().dims().first().unwrap(),
                this.num_key_value_heads,
                this.num_queries_per_kv,
                *query.shape().dims().last().unwrap(),
            )));

            let key = try_api!(
                try_api!(try_api!(key.i((.., .., .., ..))).unsqueeze(2)).expand((
                    *key.shape().dims().first().unwrap(),
                    this.num_key_value_heads,
                    this.num_queries_per_kv,
                    *key.shape().dims().last().unwrap(),
                ))
            );

            let value = try_api!(try_api!(try_api!(value.i((.., .., .., ..))).unsqueeze(2))
                .expand((
                    *value.shape().dims().first().unwrap(),
                    this.num_key_value_heads,
                    this.num_queries_per_kv,
                    *value.shape().dims().last().unwrap(),
                )));

            (query, key, value)
        } else {
            (query, key, value)
        };

        if input_metadata.attn_bias.is_none() {
            if let Some(alibi_slopes) = &this.alibi_slopes {
                //make alibi bias
                let bias = try_api!(try_api!(Tensor::arange(
                    0f64,
                    TryInto::<i32>::try_into(seq_len).unwrap().into(),
                    device,
                ))
                .to_dtype(dtype));
                let bias = try_api!(try_api!(
                    (try_api!(bias.unsqueeze(0)) - try_api!(bias.unsqueeze(1)))
                )
                .to_device(alibi_slopes.device()));

                let padded_len = ((seq_len + 7) / 8) * 8;
                let mut bias_new = try_api!(try_api!(Tensor::zeros(
                    (
                        batch_size,
                        alibi_slopes.shape().dims()[0],
                        seq_len,
                        padded_len,
                    ),
                    dtype,
                    device,
                ))
                .i((.., .., .., ..seq_len)));

                bias_new = try_api!(bias_new.slice_assign(&[.., .., .., ..], &bias));

                bias_new = try_api!(bias_new.mul(&try_api!(try_api!(
                    try_api!(alibi_slopes.i(..)).unsqueeze(1)
                )
                .unsqueeze(2)),));
                let attn_bias = LowerTriangularMaskWithTensorBias::new(bias_new);
                input_metadata.attn_bias = Some(Box::new(attn_bias));
            } else {
                let mut attn_bias = try_api!(BlockDiagonalCausalMask::from_seqlens(
                    [seq_len.try_into().unwrap()].repeat(batch_size),
                    None,
                ));
                if let Some(sliding_window) = this.sliding_window {
                    attn_bias = try_api!(attn_bias.make_local_attention(sliding_window));
                }
                input_metadata.attn_bias = Some(attn_bias);
            }
        }

        let (query, key, value) = if this.alibi_slopes.is_none() {
            (
                try_api!(query.unsqueeze(0)),
                try_api!(key.unsqueeze(0)),
                try_api!(value.unsqueeze(0)),
            )
        } else {
            assert_eq!(query.shape().dims().len(), key.shape().dims().len());
            assert_eq!(value.shape().dims().len(), key.shape().dims().len());
            assert!(query.shape().dims().len() == 3 || query.shape().dims().len() == 4);
            if query.shape().dims().len() == 3 {
                (
                    try_api!(query.reshape((
                        batch_size,
                        seq_len,
                        query.shape().dims()[1],
                        query.shape().dims()[2],
                    ))),
                    try_api!(key.reshape((
                        batch_size,
                        seq_len,
                        key.shape().dims()[1],
                        key.shape().dims()[2],
                    ))),
                    try_api!(value.reshape((
                        batch_size,
                        seq_len,
                        value.shape().dims()[1],
                        value.shape().dims()[2],
                    ))),
                )
            } else {
                (
                    try_api!(query.reshape((
                        batch_size,
                        seq_len,
                        query.shape().dims()[1],
                        query.shape().dims()[2],
                        query.shape().dims()[3],
                    ))),
                    try_api!(key.reshape((
                        batch_size,
                        seq_len,
                        key.shape().dims()[1],
                        key.shape().dims()[2],
                        key.shape().dims()[3],
                    ))),
                    try_api!(value.reshape((
                        batch_size,
                        seq_len,
                        value.shape().dims()[1],
                        value.shape().dims()[2],
                        value.shape().dims()[3],
                    ))),
                )
            }
        };

        let l = try_api!(query.dim(D::Minus2));
        let s = try_api!(key.dim(D::Minus2));
        scaled_dot_product_attention(
            &query,
            &key,
            &value,
            Some(&try_api!(input_metadata
                .attn_bias
                .as_ref()
                .unwrap()
                .materialize(
                    &Shape::from_dims(&[l, s]),
                    query.dtype(),
                    device
                ))),
            None,
            this.scale,
        )
    }
}

#[cfg(feature = "cuda")]
/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
#[cfg(feature = "flash-attn")]
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    _attn_bias: &Tensor,
    _dropout_p: Option<f32>,
    scale_factor: f32,
) -> Result<Tensor, APIError> {
    candle_flash_attn::flash_attn(query, key, value, scale_factor, false).map_err(APIError::from)
}

#[cfg(not(feature = "flash-attn"))]
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
    attn_bias: Option<&Tensor>,
    dropout_p: Option<f32>,
    scale_factor: f32,
) -> Result<Tensor, APIError> {
    let mut attn_weights = try_api!(
        try_api!(query.matmul(&try_api!(
            try_api!(key.transpose(D::Minus2, D::Minus1)).contiguous()
        ),)) * f64::from(scale_factor)
    );
    if attn_bias.is_some() {
        attn_weights = try_api!(
            &attn_weights + try_api!(attn_bias.unwrap().broadcast_as(attn_weights.shape()))
        );
    }
    attn_weights = try_api!(candle_nn::ops::softmax_last_dim(&attn_weights));

    if let Some(drop_p) = dropout_p {
        attn_weights = try_api!(candle_nn::ops::dropout(&attn_weights, drop_p));
    }
    attn_weights.matmul(value).map_err(APIError::from)
}
