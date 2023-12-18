use std::collections::HashMap;

use candle_core::Tensor;

pub fn reshape_and_cache(
    key: Tensor,
    value: Tensor,
    key_cache: &mut Tensor,
    value_cache: &mut Tensor,
    slot_mapping: Tensor,
) {
    todo!()
}

pub fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: HashMap<usize, Vec<usize>>,
) {
    todo!()
}

pub fn swap_blocks(src: Tensor, dst: &mut Tensor, block_mapping: HashMap<usize, usize>) {
    todo!()
}
