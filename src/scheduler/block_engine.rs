use std::{collections::HashMap, marker::PhantomData, ops::Deref, rc::Rc};

use super::sequence::Sequence;

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

struct PhysicalTokenBlock {
    block_id: usize,
    block_size: usize,
    refcount: usize,
}

type BlockTable = Vec<PhysicalTokenBlock>;
struct GPUAllocator;
struct CPUAllocator;

struct GPUAllocatorWrapper(usize);
struct CPUAllocatorWrapper(usize);

impl Deref for GPUAllocatorWrapper {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for CPUAllocatorWrapper {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct Allocator<T> {
    block_size: usize,
    free_blocks: BlockTable,
    _ghost: PhantomData<T>,
}

impl<T> Allocator<T> {
    fn new(block_size: usize, num_blocks: usize) -> Self {
        let mut free_blocks = Vec::new();
        for id in 0..num_blocks {
            free_blocks.push(PhysicalTokenBlock {
                block_id: id,
                block_size,
                refcount: 0,
            })
        }
        Allocator {
            block_size,
            free_blocks,
            _ghost: PhantomData,
        }
    }

    fn allocate(&self) -> PhysicalTokenBlock {
        let mut block = self.free_blocks.pop().unwrap();
        block.refcount = 1;
        block
    }
}

impl Allocator<GPUAllocator> {
    fn get_num_free_blocks(&self) -> GPUAllocatorWrapper {
        GPUAllocatorWrapper(self.free_blocks.len())
    }
}

impl Allocator<CPUAllocator> {
    fn get_num_free_blocks(&self) -> CPUAllocatorWrapper {
        CPUAllocatorWrapper(self.free_blocks.len())
    }
}

pub enum AllocStatus {
    Ok,
    Later,
    Impossible,
}

type SeqID = usize;

pub struct BlockEngine {
    block_size: usize,
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,
    gpu_allocator: Allocator<GPUAllocator>,
    cpu_allocator: Allocator<CPUAllocator>,
    block_tables: HashMap<SeqID, BlockTable>,
}

impl BlockEngine {
    pub fn new(block_size: usize, num_gpu_blocks: usize, num_cpu_blocks: usize) -> Self {
        Self {
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            gpu_allocator: Allocator::new(block_size, num_gpu_blocks),
            cpu_allocator: Allocator::new(block_size, num_cpu_blocks),
            block_tables: HashMap::new(),
        }
    }

    pub fn can_allocate(&self, sequence: &Sequence) -> AllocStatus {
        let num_required_blocks = sequence.get_logical_token_blocks();
        let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();

        if self.num_gpu_blocks > *num_free_gpu_blocks + num_required_blocks {
            AllocStatus::Later
        } else if self.num_gpu_blocks < num_required_blocks {
            AllocStatus::Impossible
        } else {
            AllocStatus::Ok
        }
    }

    pub fn allocate(&mut self, sequence: &Sequence) {
        let mut block_table = Vec::new();
        for logcical_idx in 0..sequence.get_logical_token_blocks() {
            block_table.push(self.gpu_allocator.allocate());
        }
        self.block_tables.insert(sequence.get_id(), block_table);
    }

    pub fn can_append_token_to_seq(&self, sequence: &Sequence) -> bool {
        let free_blocks = self.gpu_allocator.get_num_free_blocks();
        // Physical blocks = logical blocks
        sequence.blocks_to_add_new_tok() <= *free_blocks
    }
}
