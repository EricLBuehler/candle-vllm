use candle_core::Tensor;

pub fn paged_attention_v1(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    head_mapping: Tensor,
    scale: f32,
    block_tables: Tensor,
    context_lens: Tensor,
    block_size: usize,
    max_context_len: usize,
    alibi_slopes: Option<Tensor>,
) -> Tensor {
    todo!()
}

pub fn paged_attention_v2(
    exp_sums: Tensor,
    max_logits: Tensor,
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    head_mapping: Tensor,
    scale: f32,
    block_tables: Tensor,
    context_lens: Tensor,
    block_size: usize,
    max_context_len: usize,
    alibi_slopes: Option<Tensor>,
) -> Tensor {
    todo!()
}
