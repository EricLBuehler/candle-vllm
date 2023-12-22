use candle_core::Tensor;

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v1(
    _query: Tensor,
    _key_cache: Tensor,
    _value_cache: Tensor,
    _head_mapping: Tensor,
    _scale: f32,
    _block_tables: Tensor,
    _context_lens: Tensor,
    _block_size: usize,
    _max_context_len: usize,
    _alibi_slopes: Option<Tensor>,
) -> Tensor {
    todo!()
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v2(
    _exp_sums: Tensor,
    _max_logits: Tensor,
    _query: Tensor,
    _key_cache: Tensor,
    _value_cache: Tensor,
    _head_mapping: Tensor,
    _scale: f32,
    _block_tables: Tensor,
    _context_lens: Tensor,
    _block_size: usize,
    _max_context_len: usize,
    _alibi_slopes: Option<Tensor>,
) -> Tensor {
    todo!()
}
