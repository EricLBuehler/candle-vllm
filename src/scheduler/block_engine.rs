use std::{
    collections::{hash_map::Entry, HashMap, VecDeque},
    hash::Hash,
    marker::PhantomData,
    ops::Deref,
    sync::{Arc, Mutex, MutexGuard},
};

use super::prefix_cache::{PrefixCache, PrefixCacheConfig, PrefixMatch};
use super::sequence::{Sequence, SequenceGroup};

pub struct LogicalTokenBlock {
    tokens: Vec<u32>,
    block_size: usize,
    num_tokens: usize,
}

impl LogicalTokenBlock {
    pub fn new(block_size: usize) -> Self {
        Self {
            tokens: [0].repeat(block_size),
            block_size,
            num_tokens: 0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.num_tokens == self.block_size
    }

    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    pub fn append_token_id(&mut self, token: u32) {
        assert!(!self.is_full());
        self.tokens[self.num_tokens] = token;
        self.num_tokens += 1;
    }

    pub fn append_tokens(&mut self, tokens: &[u32]) {
        for token in tokens {
            self.append_token_id(*token);
        }
    }
}

#[derive(Hash, PartialEq, Eq)]
pub struct _PhysicalTokenBlock {
    pub block_id: usize,
    block_size: usize,
    pub refcount: usize,
    is_gpu: bool,
}

pub struct PhysicalTokenBlock(pub Mutex<_PhysicalTokenBlock>);

impl PhysicalTokenBlock {
    pub fn deref_mut(&self) -> MutexGuard<'_, _PhysicalTokenBlock> {
        loop {
            if let Ok(v) = self.0.try_lock() {
                return v;
            }
        }
    }
}

impl PartialEq for PhysicalTokenBlock {
    fn eq(&self, other: &Self) -> bool {
        *self.deref_mut() == *other.deref_mut()
    }
}

impl Hash for PhysicalTokenBlock {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.deref_mut().hash(state)
    }
}

impl Eq for PhysicalTokenBlock {}

type BlockTable = VecDeque<Arc<PhysicalTokenBlock>>;
struct GPUAllocator;
struct CPUAllocator;

struct GPUAllocatorWrapper(usize);
// struct CPUAllocatorWrapper(usize);

impl Deref for GPUAllocatorWrapper {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// impl Deref for CPUAllocatorWrapper {
//     type Target = usize;

//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }

struct Allocator<T> {
    free_blocks: BlockTable,
    _ghost: PhantomData<T>,
    num_blocks: usize,
}

impl<T> Allocator<T> {
    fn allocate(&mut self) -> Arc<PhysicalTokenBlock> {
        let block = self.free_blocks.pop_front().unwrap();
        block.deref_mut().refcount = 1;
        block
    }

    fn free_block(&mut self, block: Arc<PhysicalTokenBlock>) {
        if block.deref_mut().refcount == 0 {
            panic!(
                "PhysicalTokenBlock with id {} experienced a double free!",
                block.deref_mut().block_id
            );
        }
        block.deref_mut().refcount -= 1;
        if block.deref_mut().refcount == 0 {
            self.free_blocks.push_back(block);
        }
    }
}

impl Allocator<GPUAllocator> {
    fn new(block_size: usize, num_blocks: usize) -> Self {
        let mut free_blocks = VecDeque::new();
        for id in 0..num_blocks {
            free_blocks.push_back(Arc::new(PhysicalTokenBlock(Mutex::new(
                _PhysicalTokenBlock {
                    block_id: id,
                    block_size,
                    refcount: 0,
                    is_gpu: true,
                },
            ))))
        }
        Allocator {
            free_blocks,
            num_blocks,
            _ghost: PhantomData,
        }
    }

    fn get_num_free_blocks(&self) -> GPUAllocatorWrapper {
        GPUAllocatorWrapper(self.free_blocks.len())
    }

    fn get_num_blocks(&self) -> GPUAllocatorWrapper {
        GPUAllocatorWrapper(self.num_blocks)
    }
}

impl Allocator<CPUAllocator> {
    fn new(block_size: usize, num_blocks: usize) -> Self {
        let mut free_blocks = VecDeque::new();
        for id in 0..num_blocks {
            free_blocks.push_back(Arc::new(PhysicalTokenBlock(Mutex::new(
                _PhysicalTokenBlock {
                    block_id: id,
                    block_size,
                    refcount: 0,
                    is_gpu: true,
                },
            ))))
        }
        Allocator {
            free_blocks,
            _ghost: PhantomData,
            num_blocks,
        }
    }
}

pub enum AllocStatus {
    Ok,
    Later,
    Impossible,
}

type SeqID = usize;

/// A BlockEngine maps each Sequence (identified by its SeqID), to physical token blocks.
/// The physical token blocks may not match the logical token blocks because during
/// scheduling, physical blocks are allocated to accommodate the new tokens generated.
/// These new tokens will be added to the logical token block for each sequence.
pub struct BlockEngine {
    num_gpu_blocks: usize,
    gpu_allocator: Allocator<GPUAllocator>,
    cpu_allocator: Allocator<CPUAllocator>,
    pub block_tables: HashMap<SeqID, BlockTable>,
    block_size: usize,
    kvcache_mem_gpu: usize,
    prefix_cache: Option<PrefixCache>,
}

impl BlockEngine {
    #[must_use]
    pub fn new(
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        kvcache_mem_gpu: usize,
        prefix_cache: PrefixCacheConfig,
    ) -> Self {
        let prefix_cache = if prefix_cache.enabled && prefix_cache.max_cached_blocks > 0 {
            Some(PrefixCache::new(block_size, prefix_cache))
        } else {
            None
        };
        Self {
            num_gpu_blocks,
            gpu_allocator: Allocator::<GPUAllocator>::new(block_size, num_gpu_blocks),
            cpu_allocator: Allocator::<CPUAllocator>::new(block_size, num_cpu_blocks),
            block_tables: HashMap::new(),
            block_size,
            kvcache_mem_gpu,
            prefix_cache,
        }
    }

    pub fn get_block_size(&self) -> usize {
        self.block_size
    }

    pub fn get_num_free_blocks(&self) -> usize {
        *self.gpu_allocator.get_num_free_blocks()
    }

    pub fn get_num_blocks(&self) -> usize {
        *self.gpu_allocator.get_num_blocks()
    }

    pub fn get_kvcache_mem_size(&self) -> usize {
        self.kvcache_mem_gpu
    }

    pub fn can_allocate(&mut self, seq_group: &SequenceGroup) -> AllocStatus {
        let block_size = self.block_size;
        let num_required_blocks = if let Some(prefix_cache) = self.prefix_cache.as_mut() {
            let seq = seq_group.get_seqs().values().nth(0).unwrap();
            let tokens = seq.deref().deref().get_token_ids();
            let PrefixMatch { matched_blocks, .. } = prefix_cache.match_prefix(&tokens);
            let full_blocks = tokens.len() / block_size;
            let matched_blocks = if matched_blocks == full_blocks
                && tokens.len() % block_size == 0
                && matched_blocks > 0
            {
                matched_blocks - 1
            } else {
                matched_blocks
            };
            let logical_blocks = seq_group.get_total_logical_token_blocks();
            logical_blocks.saturating_sub(matched_blocks)
        } else {
            seq_group.get_total_logical_token_blocks()
        };
        let num_free_gpu_blocks = *self.gpu_allocator.get_num_free_blocks();

        if self.num_gpu_blocks < num_required_blocks {
            AllocStatus::Impossible
        } else if num_free_gpu_blocks > num_required_blocks {
            AllocStatus::Ok
        } else {
            AllocStatus::Later
        }
    }

    pub fn allocate(
        &mut self,
        seq_group: &SequenceGroup,
        _blocks_to_copy: &mut HashMap<usize, Vec<usize>>,
    ) {
        if self.prefix_cache.is_some() {
            self.allocate_with_prefix(seq_group);
        } else {
            let mut block_table = VecDeque::new();
            for _logcical_idx in 0..seq_group.get_total_logical_token_blocks() {
                block_table.push_back(self.gpu_allocator.allocate());
            }
            for (idx, seq_id) in seq_group.get_seqs().keys().enumerate() {
                let table = block_table.clone();
                if idx > 0 {
                    for block in &table {
                        block.deref_mut().refcount += 1;
                    }
                }
                self.block_tables.insert(*seq_id, table);
            }
        }
    }

    pub fn can_append_token_to_seq(&self, seq_group: &SequenceGroup) -> bool {
        let free_blocks = self.gpu_allocator.get_num_free_blocks();
        // Physical blocks = logical blocks
        seq_group.total_blocks_to_add_new_tok() <= *free_blocks
    }

    pub fn free_sequence(&mut self, sequence: &Sequence) {
        let block_table = self
            .block_tables
            .get(&sequence.deref_mut().get_id())
            .unwrap();

        // Free from block table
        for block in block_table {
            if block.deref_mut().is_gpu {
                self.gpu_allocator.free_block(block.clone())
            } else {
                self.cpu_allocator.free_block(block.clone())
            }
        }

        self.block_tables.remove(&sequence.deref_mut().get_id());
    }

    pub fn cache_sequence(&mut self, sequence: &Sequence) {
        let Some(prefix_cache) = self.prefix_cache.as_mut() else {
            return;
        };
        if !prefix_cache.enabled() {
            return;
        }

        let tokens = sequence.deref().get_token_ids();
        let full_blocks = tokens.len() / self.block_size;
        if full_blocks == 0 {
            return;
        }

        let table = match self.block_tables.get(&sequence.deref().get_id()) {
            Some(table) => table,
            None => return,
        };
        if table.len() < full_blocks {
            return;
        }

        let blocks: Vec<Arc<PhysicalTokenBlock>> =
            table.iter().take(full_blocks).cloned().collect();
        if blocks.iter().any(|block| !block.deref_mut().is_gpu) {
            return;
        }

        tracing::info!(
            "Prefix cache insert seq {} ({} tokens, {} blocks)",
            sequence.deref().get_id(),
            tokens.len(),
            full_blocks
        );
        let evicted = prefix_cache.insert_prefix(&tokens, &blocks);
        if !evicted.is_empty() {
            tracing::info!("Prefix cache evicted {} blocks after insert", evicted.len());
        }
        for block in evicted {
            self.release_block(block);
        }
    }

    pub fn prefix_cache_enabled(&self) -> bool {
        self.prefix_cache.is_some()
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
    pub fn swap_out(&mut self, seq_group: &SequenceGroup) -> HashMap<usize, usize> {
        // GPU block to a CPU block
        let mut new_mapping = HashMap::new();
        for seq_id in seq_group.get_seqs().keys() {
            let mut new_block_table = VecDeque::new();
            let block_table = self.block_tables.get(seq_id).unwrap();

            for gpu_block in block_table {
                let cpu_block =
                    if let Entry::Vacant(e) = new_mapping.entry(gpu_block.deref_mut().block_id) {
                        // Create a new block
                        let cpu_block = self.cpu_allocator.allocate();
                        e.insert(cpu_block.clone());
                        cpu_block
                    } else {
                        // Reuse a block
                        let cpu_block = new_mapping
                            .get(&gpu_block.deref_mut().block_id)
                            .unwrap()
                            .clone();
                        cpu_block.deref_mut().refcount += 1;
                        cpu_block
                    };
                new_block_table.push_back(cpu_block);
                self.gpu_allocator.free_block(gpu_block.clone());
            }
            self.block_tables.insert(*seq_id, new_block_table);
        }

        new_mapping
            .iter()
            .map(|(k, v)| (*k, v.deref_mut().block_id))
            .collect::<HashMap<_, _>>()
    }

    // Returns the COW mapping (src, dst).
    // COW is performed if there are multiple references to the last physical block.
    pub fn append_token_slot_to_seq(&mut self, sequence: &Sequence) -> Option<(usize, usize)> {
        let table = self
            .block_tables
            .get_mut(&sequence.deref_mut().get_id())
            .unwrap();

        match sequence.deref_mut().blocks_to_add_new_tok() {
            1 => {
                table.push_back(self.gpu_allocator.allocate());
                None
            }
            0 => {
                let last_block = table.back_mut().unwrap();
                assert!(last_block.deref_mut().is_gpu);
                if last_block.deref_mut().refcount == 1 {
                    None
                } else {
                    // We would be writing into shared, so COW.
                    let new_block = self.gpu_allocator.allocate();
                    self.gpu_allocator.free_block(last_block.clone());
                    let old_number = last_block.deref_mut().block_id;
                    let new_number = new_block.deref_mut().block_id;
                    *last_block = new_block;
                    Some((old_number, new_number))
                }
            }
            _ => {
                unreachable!()
            }
        }
    }

    pub fn can_swap_in_seq_group(&self, seq_group: &SequenceGroup) -> bool {
        let blocks_required: usize = self
            .block_tables
            .iter()
            .filter(|(id, _)| seq_group.get_seqs().contains_key(id))
            .map(|(_, table)| table.len())
            .sum();
        blocks_required <= self.gpu_allocator.free_blocks.len()
    }

    /// Update the block table so that the sequence does no longer reserve any CPU
    /// physical blocks, and only has GPU physical blocks.
    pub fn swap_in(&mut self, seq_group: &SequenceGroup) -> HashMap<usize, usize> {
        // CPU block to a GPU block
        let mut new_mapping = HashMap::new();
        for seq_id in seq_group.get_seqs().keys() {
            let mut new_block_table = VecDeque::new();
            let block_table = self.block_tables.get(seq_id).unwrap();

            for cpu_block in block_table {
                let gpu_block =
                    if let Entry::Vacant(e) = new_mapping.entry(cpu_block.deref_mut().block_id) {
                        // Create a new block
                        let gpu_block = self.cpu_allocator.allocate();
                        e.insert(gpu_block.clone());
                        gpu_block
                    } else {
                        // Reuse a block
                        let gpu_block = new_mapping
                            .get(&cpu_block.deref_mut().block_id)
                            .unwrap()
                            .clone();
                        gpu_block.deref_mut().refcount += 1;
                        gpu_block
                    };
                new_block_table.push_back(gpu_block);
                self.gpu_allocator.free_block(cpu_block.clone());
            }
            self.block_tables.insert(*seq_id, new_block_table);
        }

        new_mapping
            .iter()
            .map(|(k, v)| (*k, v.deref_mut().block_id))
            .collect::<HashMap<_, _>>()
    }

    fn allocate_with_prefix(&mut self, seq_group: &SequenceGroup) {
        let block_size = self.block_size;
        let seqs: Vec<_> = seq_group.get_seqs().values().cloned().collect();
        let mut cached_tokens = 0usize;
        let mut block_table = VecDeque::new();

        if let Some(seq) = seqs.first() {
            let tokens = seq.deref().deref().get_token_ids();
            if let Some(prefix_cache) = self.prefix_cache.as_mut() {
                let PrefixMatch {
                    matched_blocks,
                    last_hash,
                } = prefix_cache.match_prefix(&tokens);
                let full_blocks = tokens.len() / block_size;
                let matched_blocks = if matched_blocks == full_blocks
                    && tokens.len() % block_size == 0
                    && matched_blocks > 0
                {
                    matched_blocks - 1
                } else {
                    matched_blocks
                };
                cached_tokens = matched_blocks * block_size;
                if matched_blocks > 0 {
                    tracing::info!(
                        "Prefix cache hit seq {} ({} cached tokens, {} blocks)",
                        seq.deref().deref().get_id(),
                        cached_tokens,
                        matched_blocks
                    );
                } else {
                    tracing::debug!("Prefix cache miss seq {}", seq.deref().deref().get_id());
                }
                if matched_blocks > 0 {
                    let mut blocks = prefix_cache.blocks_for_match(last_hash.unwrap());
                    blocks.truncate(matched_blocks);
                    for block in blocks {
                        block.deref_mut().refcount += 1;
                        block_table.push_back(block);
                    }
                }
            }
            seq.deref_mut().set_num_cached_tokens(cached_tokens);
            let logical_blocks = seq.deref().deref().get_logical_token_blocks();
            for _ in block_table.len()..logical_blocks {
                block_table.push_back(self.gpu_allocator.allocate());
            }
        }

        for (idx, seq) in seqs.iter().enumerate() {
            seq.deref_mut().set_num_cached_tokens(cached_tokens);
            let table = block_table.clone();
            if idx > 0 {
                for block in &table {
                    block.deref_mut().refcount += 1;
                }
            }
            self.block_tables
                .insert(seq.deref().deref().get_id(), table);
        }
    }

    fn release_block(&mut self, block: Arc<PhysicalTokenBlock>) {
        if block.deref_mut().is_gpu {
            self.gpu_allocator.free_block(block);
        } else {
            self.cpu_allocator.free_block(block);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockEngine, PrefixCacheConfig};
    use crate::openai::requests::{EmbeddingType, EncodingFormat};
    use crate::openai::sampling_params::{EarlyStoppingCondition, SamplingParams};
    use crate::scheduler::sequence::{Sequence, SequenceGroup, _Sequence};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::SystemTime;

    fn make_group(
        seq_id: usize,
        group_id: usize,
        block_size: usize,
        tokens: Vec<u32>,
    ) -> (SequenceGroup, Arc<Sequence>) {
        let seq = Arc::new(Sequence(std::sync::RwLock::new(_Sequence::new(
            &tokens, seq_id, block_size,
        ))));
        let sampling_params = SamplingParams::new(
            1,
            None,
            0.0,
            0.0,
            None,
            None,
            None,
            None,
            None,
            false,
            1.0,
            EarlyStoppingCondition::UnlikelyBetterCandidates,
            None,
            vec![],
            false,
            16,
            None,
            None,
            true,
            None,
        )
        .expect("sampling params");
        let group = SequenceGroup::new(
            &[seq.clone()],
            0,
            group_id,
            "req".to_string(),
            SystemTime::now(),
            sampling_params,
            false,
            false,
            EncodingFormat::Float,
            EmbeddingType::Last,
            None,
        );
        (group, seq)
    }

    #[test]
    fn allocate_with_prefix_cache_reuses_blocks() {
        let block_size = 4;
        let mut engine = BlockEngine::new(
            block_size,
            8,
            8,
            0,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 4,
            },
        );

        let (group1, seq1) = make_group(1, 1, block_size, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let mut blocks_to_copy = HashMap::new();
        let free_before = engine.get_num_free_blocks();
        engine.allocate(&group1, &mut blocks_to_copy);
        let free_after_alloc = engine.get_num_free_blocks();
        assert!(free_after_alloc < free_before);

        let cached_block_ids: Vec<usize> = engine
            .block_tables
            .get(&seq1.deref().get_id())
            .unwrap()
            .iter()
            .take(2)
            .map(|block| block.deref_mut().block_id)
            .collect();

        engine.cache_sequence(&seq1);
        engine.free_sequence(&seq1);
        let free_after_free = engine.get_num_free_blocks();
        assert_eq!(free_after_free, free_after_alloc + 1);

        let (group2, seq2) = make_group(
            2,
            2,
            block_size,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        );
        engine.allocate(&group2, &mut blocks_to_copy);
        assert_eq!(seq2.deref().get_num_cached_tokens(), 8);
        let table = engine.block_tables.get(&seq2.deref().get_id()).unwrap();
        assert_eq!(table[0].deref_mut().block_id, cached_block_ids[0]);
        assert_eq!(table[1].deref_mut().block_id, cached_block_ids[1]);
    }
}
