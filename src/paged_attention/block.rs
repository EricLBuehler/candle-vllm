use std::{
    cell::{Ref, RefCell, RefMut},
    hash::Hash,
    ops::Deref,
};

use candle_core::Device;

const BLANK_TOKEN_ID: usize = 0;

/// Block that stores a contiguous chunk of tokens from left to right
///
/// Used to represent the states of the corresponding physical blocks in the KV cache.
pub struct LogicalTokenBlock {
    pub block_number: usize,
    block_size: usize,
    token_ids: Vec<usize>,
    num_tokens: usize,
}

impl LogicalTokenBlock {
    pub fn new(block_number: usize, block_size: usize) -> Self {
        Self {
            block_number,
            block_size,
            token_ids: vec![BLANK_TOKEN_ID].repeat(block_size),
            num_tokens: 0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.block_size == self.num_tokens
    }

    pub fn get_num_empty_slots(&self) -> usize {
        self.block_size - self.num_tokens
    }

    pub fn append_tokens(&mut self, token_ids: Vec<usize>) {
        assert!(token_ids.len() <= self.get_num_empty_slots());
        let curr_idx = self.num_tokens;

        let ids = &mut self.token_ids[curr_idx..curr_idx + token_ids.len()];
        // TODO(EricLBuehler): Have a panic check here in Rust impl, above assert is unnecessary?
        ids.copy_from_slice(&token_ids[..]);
        self.num_tokens += token_ids.len();
    }
}

/// State of a block in the KV cache
#[derive(Debug)]
pub struct PhysicalTokenBlockInner {
    pub device: Device,
    pub block_number: usize,
    block_size: usize,
    pub ref_count: usize,
}

impl PartialEq for PhysicalTokenBlockInner {
    fn eq(&self, other: &Self) -> bool {
        self.block_number == other.block_number
    }
}

impl Eq for PhysicalTokenBlockInner {}

impl Hash for PhysicalTokenBlockInner {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.block_number)
    }
}

impl PhysicalTokenBlockInner {
    pub fn new(device: Device, block_number: usize, block_size: usize) -> Self {
        Self {
            device,
            block_number,
            block_size,
            ref_count: 0,
        }
    }
}

#[derive(Debug)]
pub struct PhysicalTokenBlock {
    inner: RefCell<PhysicalTokenBlockInner>,
}

impl PhysicalTokenBlock {
    pub fn new(device: Device, block_number: usize, block_size: usize) -> Self {
        Self {
            inner: RefCell::new(PhysicalTokenBlockInner::new(
                device,
                block_number,
                block_size,
            )),
        }
    }

    pub fn deref_mut(&self) -> RefMut<'_, PhysicalTokenBlockInner> {
        loop {
            if let Ok(reference) = self.inner.try_borrow_mut() {
                return reference;
            }
        }
    }

    pub fn deref(&self) -> Ref<'_, PhysicalTokenBlockInner> {
        loop {
            if let Ok(reference) = self.inner.try_borrow() {
                return reference;
            }
        }
    }
}

impl PartialEq for PhysicalTokenBlock {
    fn eq(&self, other: &Self) -> bool {
        self.eq(other)
    }
}

impl Eq for PhysicalTokenBlock {}

impl Hash for PhysicalTokenBlock {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.deref().block_number)
    }
}
