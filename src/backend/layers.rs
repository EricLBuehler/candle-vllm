use candle_core::Tensor;

pub fn rotary_embedding(
    _positions: Tensor,
    _query: &mut Tensor,
    _key: &mut Tensor,
    _head_size: usize,
    _cos_sin_cache: Tensor,
    _is_neox: bool,
) {
    todo!()
}
