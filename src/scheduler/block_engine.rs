pub struct LogicalTokenBlock {
    tokens: Vec<usize>,
    block_id: usize,
    block_size: usize,
    num_tokens: usize,
}

impl LogicalTokenBlock {
    pub fn new(block_id: usize, block_size: usize) -> Self {
        Self {
            tokens: vec![0].repeat(block_size),
            block_id,
            block_size,
            num_tokens: 0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.num_tokens == self.block_size
    }

    pub fn append_token_id(&mut self, token: usize) {
        assert!(!self.is_full());
        self.tokens[self.num_tokens] = token;
        self.num_tokens += 1;
    }

    pub fn append_tokens(&mut self, tokens: &[usize]) {
        for token in tokens {
            self.append_token_id(*token);
        }
    }
}
