use std::{collections::HashMap, marker::PhantomData, ops::Deref, rc::Rc};

use super::sequence::{Sequence, SequenceGroup};

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

#[derive(Hash, PartialEq, Eq)]
struct PhysicalTokenBlock {
    block_id: usize,
    block_size: usize,
    refcount: usize,
    is_gpu: bool,
}

type BlockTable = Vec<Rc<PhysicalTokenBlock>>;
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
    fn allocate(&self) -> Rc<PhysicalTokenBlock> {
        let mut block = self.free_blocks.pop().unwrap();
        block.refcount = 1;
        block
    }

    fn free_block(&self, mut block: Rc<PhysicalTokenBlock>) {
        if block.refcount == 0 {
            panic!(
                "PhysicalTokenBlock with id {} experienced a double free!",
                block.block_id
            );
        }
        block.refcount -= 1;
        if block.refcount == 0 {
            self.free_blocks.push(block);
        }
    }
}

impl Allocator<GPUAllocator> {
    fn new(block_size: usize, num_blocks: usize) -> Self {
        let mut free_blocks = Vec::new();
        for id in 0..num_blocks {
            free_blocks.push(Rc::new(PhysicalTokenBlock {
                block_id: id,
                block_size,
                refcount: 0,
                is_gpu: true,
            }))
        }
        Allocator {
            block_size,
            free_blocks,
            _ghost: PhantomData,
        }
    }

    const fn get_num_free_blocks(&self) -> GPUAllocatorWrapper {
        GPUAllocatorWrapper(self.free_blocks.len())
    }

    #[inline(always)]
    const fn is_gpu(&self) -> bool {
        true
    }
}

impl Allocator<CPUAllocator> {
    fn new(block_size: usize, num_blocks: usize) -> Self {
        let mut free_blocks = Vec::new();
        for id in 0..num_blocks {
            free_blocks.push(Rc::new(PhysicalTokenBlock {
                block_id: id,
                block_size,
                refcount: 0,
                is_gpu: true,
            }))
        }
        Allocator {
            block_size,
            free_blocks,
            _ghost: PhantomData,
        }
    }

    const fn get_num_free_blocks(&self) -> CPUAllocatorWrapper {
        CPUAllocatorWrapper(self.free_blocks.len())
    }

    #[inline(always)]
    const fn is_gpu(&self) -> bool {
        false
    }
}

pub enum AllocStatus {
    Ok,
    Later,
    Impossible,
}

type SeqID = usize;

/// A BlockEngine maps eachs Sequence (identified by its SeqID), to physical token blocks.
/// The physical token blocks may not match the logical token blocks because during
/// scheduling, physical blocks are allocated to accomodate the new tokens generated.
/// These new tokens will be added to the logical token block for each sequence.
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
            gpu_allocator: Allocator::<GPUAllocator>::new(block_size, num_gpu_blocks),
            cpu_allocator: Allocator::<CPUAllocator>::new(block_size, num_cpu_blocks),
            block_tables: HashMap::new(),
        }
    }

    pub fn can_allocate(&self, seq_group: &SequenceGroup) -> AllocStatus {
        let num_required_blocks = seq_group.get_total_logical_token_blocks();
        let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();

        if self.num_gpu_blocks > *num_free_gpu_blocks + num_required_blocks {
            AllocStatus::Later
        } else if self.num_gpu_blocks < num_required_blocks {
            AllocStatus::Impossible
        } else {
            AllocStatus::Ok
        }
    }

    pub fn allocate(&mut self, seq_group: &SequenceGroup) {
        let mut block_table = Vec::new();
        for logcical_idx in 0..seq_group.get_total_logical_token_blocks() {
            block_table.push(self.gpu_allocator.allocate());
        }
        for (seq_id, _) in seq_group.get_seqs() {
            self.block_tables.insert(*seq_id, block_table);
        }
    }

    pub fn can_append_token_to_seq(&self, seq_group: &SequenceGroup) -> bool {
        let free_blocks = self.gpu_allocator.get_num_free_blocks();
        // Physical blocks = logical blocks
        seq_group.total_blocks_to_add_new_tok() <= *free_blocks
    }

    pub fn free_sequence(&mut self, sequence: &Sequence) {
        let block_table = self.block_tables.get(&sequence.get_id()).unwrap();

        // Free from block table
        for block in block_table {
            if block.is_gpu {
                self.gpu_allocator.free_block(block.clone())
            } else {
                self.cpu_allocator.free_block(block.clone())
            }
        }

        self.block_tables.remove(&sequence.get_id());
    }

    pub fn can_swap_out_seq_group(&self, seq_group: &SequenceGroup) -> bool {
        let blocks_required: usize = self
            .block_tables
            .iter()
            .filter(|(id, _)| seq_group.get_seqs().contains_key(id))
            .map(|(_, table)| table.len())
            .sum();
        blocks_required <= self.cpu_allocator.free_blocks.len()
    }

    /// Update the block table so that the sequence does no longer reserve any GPU
    /// physical blocks, and only has CPU physical blocks.
    pub fn swap_out(&self, seq_group: &SequenceGroup) -> HashMap<usize, usize> {
        // GPU block to a CPU block
        let mut new_mapping = HashMap::new();
        for (seq_id, seq) in seq_group.get_seqs() {
            let mut new_block_table = Vec::new();
            let block_table = self.block_tables.get(seq_id).unwrap();

            for gpu_block in block_table {
                let cpu_block = if new_mapping.contains_key(gpu_block) {
                    // Reuse a block
                    let mut cpu_block: &Rc<PhysicalTokenBlock> =
                        new_mapping.get(gpu_block).unwrap();
                    cpu_block.refcount += 1;
                    *cpu_block
                } else {
                    // Create a new block
                    let cpu_block = self.cpu_allocator.allocate();
                    new_mapping.insert(gpu_block.clone(), cpu_block);
                    cpu_block
                };
                new_block_table.push(cpu_block);
                self.gpu_allocator.free_block(gpu_block.clone());
            }
            self.block_tables.insert(*seq_id, new_block_table);
        }

        new_mapping
            .iter()
            .map(|(k, v)| (k.block_id, v.block_id))
            .collect::<HashMap<_, _>>()
    }

    // Returns the COW mapping (src, dst).
    // COW is performed if there are multiple references to the last phyiscal block.
    pub fn append_token_slot_to_seq(&mut self, sequence: &Sequence) -> Option<(usize, usize)> {
        let table = self.block_tables.get_mut(&sequence.get_id()).unwrap();

        match sequence.blocks_to_add_new_tok() {
            1 => {
                table.push(self.gpu_allocator.allocate());
                None
            }
            0 => {
                let last_block = table.last_mut().unwrap();
                assert!(last_block.is_gpu);
                if last_block.refcount == 1 {
                    None
                } else {
                    // We would be writing into shared, so COW.
                    let new_block = self.gpu_allocator.allocate();
                    self.gpu_allocator.free_block(last_block.clone());
                    let old_number = last_block.block_id;
                    *last_block = new_block;
                    Some((old_number, new_block.block_id))
                }
            }
        }
    }
}
