use candle_core::Tensor;

#[derive(Debug)]
pub struct InputMetadata {
    pub max_context_len: usize,
    pub block_tables: Tensor,
    pub context_lens: Tensor,
}

impl InputMetadata {
    pub fn new(max_context_len: usize, block_tables: Tensor, context_lens: Tensor) -> Self {
        Self {
            max_context_len,
            block_tables,
            context_lens,
        }
    }
}
