use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use candle_core::Device;
use range_checked::F64Bounded;

use crate::openai::responses::APIError;

use super::{
    bindings,
    block::PhysicalTokenBlock,
    sequence::{Sequence, SequenceGroup, SequenceStatus},
};

pub struct BlockAllocator {
    device: Device,
    block_size: usize,
    num_blocks: usize,
    free_blocks: Vec<Arc<PhysicalTokenBlock>>,
}

impl BlockAllocator {
    pub fn new(device: Device, block_size: usize, num_blocks: usize) -> Self {
        Self {
            device: device.clone(),
            block_size,
            num_blocks,
            free_blocks: {
                let mut blocks = Vec::new();
                for i in 0..num_blocks {
                    blocks.push(Arc::new(PhysicalTokenBlock::new(
                        device.clone(),
                        i,
                        block_size,
                    )));
                }
                blocks
            },
        }
    }

    fn get_num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    fn allocate(&mut self) -> Result<Arc<PhysicalTokenBlock>, APIError> {
        if self.free_blocks.is_empty() {
            return Err(APIError::new(format!(
                "**Out of memory!** No free blocks are available."
            )));
        }
        let binding = self.free_blocks.pop().unwrap();
        {
            let mut block = binding.deref_mut();

            block.ref_count = 1;
        }
        Ok(binding)
    }

    fn free(&mut self, block: Arc<PhysicalTokenBlock>) -> Result<(), APIError> {
        if block.deref().ref_count == 0 {
            return Err(APIError::new(format!(
                "**Double free!** {block:?} is already freed."
            )));
        }
        block.deref_mut().ref_count -= 1;
        if block.deref().ref_count == 0 {
            self.free_blocks.push(block.clone());
        }
        Ok(())
    }
}

pub enum AllocStatus {
    /// Now
    Ok,
    /// Later, not enough space yet
    Later,
    /// Cannot fit on GPU!
    Never,
}

pub struct NewBlockAllocated {
    pub src_block: usize,
    pub dst_block: usize,
}

pub struct BlockSpaceManager {
    block_size: usize,
    num_total_gpu_blocks: usize,
    num_total_cpu_blocks: usize,
    block_sliding_window: Option<usize>,
    watermark: f64,
    watermark_blocks: usize,
    gpu_allocator: BlockAllocator,
    cpu_allocator: BlockAllocator,
    block_tables: HashMap<usize, Vec<Arc<PhysicalTokenBlock>>>,
}

impl BlockSpaceManager {
    pub fn new(
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        watermark: F64Bounded<0, { i32::MAX }, true>,
        sliding_window: Option<usize>,
    ) -> Result<Self, APIError> {
        Ok(Self {
            block_size,
            num_total_cpu_blocks: num_cpu_blocks,
            num_total_gpu_blocks: num_gpu_blocks,
            block_sliding_window: if let Some(sliding_window) = sliding_window {
                assert_eq!(sliding_window % block_size, 0);
                Some(sliding_window / block_size)
            } else {
                None
            },
            watermark: *watermark,
            watermark_blocks: (*watermark * num_gpu_blocks as f64) as usize,
            gpu_allocator: BlockAllocator::new(
                Device::new_cuda(0).map_err(APIError::from)?,
                block_size,
                num_gpu_blocks,
            ),
            cpu_allocator: BlockAllocator::new(Device::Cpu, block_size, num_cpu_blocks),
            block_tables: HashMap::new(),
        })
    }

    pub fn can_allocate(&self, seq_group: &SequenceGroup) -> AllocStatus {
        // TODO(EricLBuehler): Assume all seqs in a group share the same prompt, may not be true for preempted seqs?
        let binding = seq_group.deref();
        let seqs = binding.get_seqs(None);
        let seq = seqs.get(0).unwrap();
        let mut num_required_blocks = seq.logical_token_blocks.len();
        if let Some(block_sliding_window) = self.block_sliding_window {
            num_required_blocks = num_required_blocks.min(block_sliding_window);
        }
        let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();

        // Use watermark to avoid frequent cache eviction
        if self.num_total_gpu_blocks - num_required_blocks < self.watermark_blocks {
            AllocStatus::Never
        } else if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks {
            AllocStatus::Ok
        } else {
            AllocStatus::Later
        }
    }

    pub fn can_append_slot(&self, seq_group: &SequenceGroup) -> bool {
        let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();
        let num_seqs = seq_group.deref().num_seqs(Some(SequenceStatus::Running));
        num_seqs <= num_free_gpu_blocks
    }

    pub fn free(&mut self, seq: &Sequence) {
        if !self.block_tables.contains_key(&seq.seq_id) {
            return;
        }
        let block_table = self.block_tables.get(&seq.seq_id).unwrap().clone();
        self._free_block_table(block_table);
        self.block_tables.remove(&seq.seq_id);
    }

    fn _free_block_table(&mut self, block_table: Vec<Arc<PhysicalTokenBlock>>) {
        for block in block_table {
            if block.deref().device.is_cuda() {
                self.gpu_allocator.free(block.clone());
            } else {
                self.cpu_allocator.free(block.clone());
            }
        }
    }

    fn _get_physical_blocks(&self, seq_group: Arc<SequenceGroup>) -> Vec<&Arc<PhysicalTokenBlock>> {
        let mut blocks = HashSet::new();
        for seq in seq_group.deref().get_seqs(None) {
            if seq.is_finished() {
                continue;
            }
            blocks.extend(self.block_tables.get(&seq.seq_id).unwrap());
        }
        Vec::from_iter(blocks)
    }

    pub fn can_swap_out(&self, seq_group: Arc<SequenceGroup>) -> bool {
        self._get_physical_blocks(seq_group).len() <= self.cpu_allocator.get_num_free_blocks()
    }

    pub fn swap_out(
        &mut self,
        seq_group: &SequenceGroup,
    ) -> Result<HashMap<usize, usize>, APIError> {
        // GPU to CPU block
        let mut mapping: HashMap<Arc<PhysicalTokenBlock>, Arc<PhysicalTokenBlock>> = HashMap::new();
        for seq in seq_group.deref().get_seqs(Some(SequenceStatus::Running)) {
            let mut new_block_table = Vec::new();
            let block_table = self.block_tables.get(&seq.seq_id).unwrap();

            for gpu_block in block_table {
                let cpu_block = if mapping.contains_key(gpu_block) {
                    let mut cpu_block = mapping.get(gpu_block).unwrap().deref_mut();
                    cpu_block.ref_count += 1;
                    mapping.get(gpu_block).unwrap().clone()
                } else {
                    let cpu_block = self.cpu_allocator.allocate()?;
                    mapping.insert(gpu_block.clone(), cpu_block.clone());
                    cpu_block
                };
                new_block_table.push(cpu_block);
                // Free the CPU block swapped into the GPU
                self.cpu_allocator.free(gpu_block.clone());
            }
            self.block_tables.insert(seq.seq_id, new_block_table);
        }

        let mut block_number_mapping = HashMap::new();
        for (cpu_block, gpu_block) in mapping {
            block_number_mapping.insert(
                gpu_block.deref().block_number,
                cpu_block.deref().block_number,
            );
        }
        Ok(block_number_mapping)
    }

    pub fn append_slot(&mut self, seq: &Sequence) -> Result<Option<NewBlockAllocated>, APIError> {
        // Allocate a physical slot for a new token
        let logical_blocks = &seq.logical_token_blocks;
        let block_table = self.block_tables.get_mut(&seq.seq_id).unwrap();

        if block_table.len() < logical_blocks.len() {
            if self.block_sliding_window.is_some()
                && block_table.len() >= self.block_sliding_window.unwrap()
            {
                // re-use a block
                block_table.push(
                    block_table
                        .get(block_table.len() % self.block_sliding_window.unwrap())
                        .unwrap()
                        .clone(),
                );
            } else {
                // Seq has a new logical block. Allocate a new physical block.
                let block = self.gpu_allocator.allocate()?;
                block_table.push(block);
            }
        }

        // Want to append token to the last physical block
        let last_block = block_table.last().unwrap().clone();
        assert!(last_block.deref().device.is_cuda());

        if last_block.deref().ref_count == 1 {
            // Not shared with other seqs, appendable
            return Ok(None);
        } else {
            // Shared with other seqs, do a COW.
            let new_block = self.gpu_allocator.allocate()?;
            let new_block_number = new_block.deref().block_number;
            *block_table.last_mut().unwrap() = new_block;
            self.gpu_allocator.free(last_block.clone());
            return Ok(Some(NewBlockAllocated {
                src_block: last_block.deref().block_number,
                dst_block: new_block_number,
            }));
        }
    }

    pub fn can_swap_in(&self, seq_group: &Arc<SequenceGroup>) -> bool {
        let blocks = self._get_physical_blocks(seq_group.clone());
        let num_swapped_seqs = seq_group.deref().num_seqs(Some(SequenceStatus::Swapped));
        let num_free_blocks = self.gpu_allocator.get_num_free_blocks();
        // Note: conservatively, assume every sequence will icnlude at least one free block after the swap-in.
        // Should match the logic in can_append_slot
        let num_required_blocks = blocks.len() + num_swapped_seqs;
        num_free_blocks - num_required_blocks >= self.watermark_blocks
    }

    pub fn swap_in(
        &mut self,
        seq_group: &SequenceGroup,
    ) -> Result<HashMap<usize, usize>, APIError> {
        // GPU to CPU block
        let mut mapping: HashMap<Arc<PhysicalTokenBlock>, Arc<PhysicalTokenBlock>> = HashMap::new();
        for seq in seq_group.deref().get_seqs(Some(SequenceStatus::Running)) {
            let mut new_block_table = Vec::new();
            let block_table = self.block_tables.get(&seq.seq_id).unwrap();

            for gpu_block in block_table {
                let cpu_block = if mapping.contains_key(gpu_block) {
                    let mut cpu_block = mapping.get(gpu_block).unwrap().deref_mut();
                    cpu_block.ref_count += 1;
                    mapping.get(gpu_block).unwrap().clone()
                } else {
                    let cpu_block = self.cpu_allocator.allocate()?;
                    mapping.insert(gpu_block.clone(), cpu_block.clone());
                    cpu_block
                };
                new_block_table.push(cpu_block);
                // Free the CPU block swapped into the GPU
                self.cpu_allocator.free(gpu_block.clone());
            }
            self.block_tables.insert(seq.seq_id, new_block_table);
        }

        let mut block_number_mapping = HashMap::new();
        for (cpu_block, gpu_block) in mapping {
            block_number_mapping.insert(
                gpu_block.deref().block_number,
                cpu_block.deref().block_number,
            );
        }
        Ok(block_number_mapping)
    }

    pub fn allocate(&mut self, seq_group: &SequenceGroup) -> Result<(), APIError> {
        // TODO(EricLBuehler): Assume all seqs in a group have the same prompt.
        let binding = seq_group.deref();
        let binding = binding.get_seqs(None);
        let seq = binding.get(0).unwrap();

        // Alloc new physical token blocks that will store the prompt tokens.
        let mut block_table: Vec<Arc<PhysicalTokenBlock>> = Vec::new();
        for logical_idx in 0..seq.logical_token_blocks.len() {
            let block = if self
                .block_sliding_window
                .is_some_and(|blk_sliding_window| logical_idx >= blk_sliding_window)
            {
                block_table
                    .get(logical_idx % self.block_sliding_window.unwrap())
                    .unwrap()
                    .clone()
            } else {
                self.gpu_allocator.allocate()?
            };
            block.deref_mut().ref_count = seq_group.deref().num_seqs(None);
            block_table.push(block);
        }

        for seq in seq_group.deref_mut().get_seqs(None) {
            self.block_tables.insert(seq.seq_id, block_table.clone());
        }
        Ok(())
    }

    pub fn get_block_table(&mut self, seq: &Sequence) -> Vec<usize> {
        self.block_tables
            .get(&seq.seq_id)
            .unwrap()
            .iter()
            .map(|block| block.deref().block_number)
            .collect::<Vec<_>>()
    }
}
