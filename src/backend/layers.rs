use candle_core::Tensor;

pub fn rotary_embedding(
    positions: Tensor,
    query: &mut Tensor,
    key: &mut Tensor,
    head_size: usize,
    cos_sin_cache: Tensor,
    is_neox: bool,
) {
    todo!()
}
