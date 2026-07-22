//! The Scheduler uses a BlockEngine to schedule and automatically batch sequences. The
//! primary method `schedule` returns the batched sequences as inputs, as well as the
//! operations to be executed on the cache by the CacheEngine.

/// The higher-level manager of the blocks allocated. Operations performed by the block engine do
/// not directly change memory.
pub mod block_engine;
/// This is the lower-level manager of the cache. It manages swapping and copying the blocks and
/// actually allocates the KV cache for the CPU and GPU. It is used by the LLMEngine to execute
/// operations issued by the scheduler.
pub mod cache_engine;
pub mod mamba;
pub mod prefix_cache;
pub mod sequence;
use tracing::warn;
type CPUBlockFrom = usize;
type GPUBlockFrom = usize;
type CPUBlockTo = usize;
type GPUBlockTo = usize;
type SrcBlockFrom = usize;
type DstBlocksTo = Vec<usize>;

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::{Duration, SystemTime},
};

use crate::scheduler::{block_engine::AllocStatus, sequence::SequenceStatus};

use self::mamba::MambaState;
use self::{
    block_engine::BlockEngine, cache_engine::CacheConfig, prefix_cache::PrefixCacheConfig,
    sequence::SequenceGroup,
};

const PREFIX_CACHE_PRESSURE_EVICT_PERCENT: f32 = 0.1; // evict 10% of prefix cache when under pressure
const FINISHED_CACHED_TOKENS_MAX: usize = 16_384;
const SWAP_COOLING_PERIOD: Duration = Duration::from_millis(300);

fn active_sequence_limit(max_num_seqs: usize, mamba_cache_capacity: Option<usize>) -> usize {
    match mamba_cache_capacity {
        Some(mamba_capacity) if mamba_capacity > 0 => max_num_seqs.min(mamba_capacity),
        _ => max_num_seqs,
    }
    .max(1)
}

#[cfg(test)]
mod tests {
    use super::active_sequence_limit;

    #[test]
    fn mamba_capacity_cannot_raise_user_sequence_limit() {
        assert_eq!(active_sequence_limit(4, Some(8)), 4);
        assert_eq!(active_sequence_limit(8, Some(4)), 4);
    }

    #[test]
    fn active_sequence_limit_is_at_least_one() {
        assert_eq!(active_sequence_limit(0, None), 1);
    }
}

pub struct SchedulerOutput {
    pub scheduled: Arc<VecDeque<Arc<SequenceGroup>>>,
    pub blocks_to_swap_in: HashMap<CPUBlockFrom, GPUBlockTo>,
    pub blocks_to_swap_out: HashMap<GPUBlockFrom, CPUBlockTo>,
    pub blocks_to_copy: HashMap<SrcBlockFrom, DstBlocksTo>,
    pub swap_in_groups: Vec<usize>,
    pub swap_out_groups: Vec<usize>,
    pub ignored_seq_groups: Arc<VecDeque<Arc<SequenceGroup>>>,
}

pub struct SchedulerConfig {
    /// User request/preallocation limit used while sizing the KV cache.
    pub max_num_seqs: usize,
    /// Resource-derived active request capacity, bounded independently from
    /// the user request/preallocation limit.
    pub max_num_parallel_reqs: usize,
    /// Per-step prefill token budget, distinct from the total KV-cache pool.
    pub max_num_batched_tokens: usize,
    pub prefix_cache: PrefixCacheConfig,
    pub mamba_cache_capacity: Option<usize>,
}

pub struct Scheduler {
    waiting: VecDeque<Arc<SequenceGroup>>,
    running: VecDeque<Arc<SequenceGroup>>,
    swapped_out: VecDeque<Arc<SequenceGroup>>,
    config: SchedulerConfig,
    pub block_engine: BlockEngine,
    mamba_state: MambaState,
    is_last_prefill: bool,
    prefill_chunk_size: usize,
    finished_cached_tokens: HashMap<usize, usize>,
    pending_runner_releases: Vec<usize>,
}

impl Scheduler {
    pub fn new(
        config: SchedulerConfig,
        cache_config: &CacheConfig,
        require_mamba_prefix_snapshots: bool,
        prefill_chunk_size: usize,
    ) -> Self {
        assert!(cache_config.fully_init);
        let prefix_cache_cfg = config.prefix_cache.clone();
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            swapped_out: VecDeque::new(),
            config,
            block_engine: BlockEngine::new(
                cache_config.block_size,
                cache_config.num_gpu_blocks.unwrap(),
                cache_config.num_cpu_blocks.unwrap(),
                cache_config.kvcache_mem_gpu,
                prefix_cache_cfg,
                require_mamba_prefix_snapshots,
            ),
            mamba_state: MambaState::default(),
            is_last_prefill: false,
            prefill_chunk_size,
            finished_cached_tokens: HashMap::new(),
            pending_runner_releases: Vec::new(),
        }
    }

    pub fn add_sequence(&mut self, seq_group: SequenceGroup) {
        self.waiting.push_back(Arc::new(seq_group));
    }

    pub fn take_pending_runner_releases(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.pending_runner_releases)
    }

    fn request_runner_release_for_group(&mut self, seq_group: &SequenceGroup) {
        for seq in seq_group.get_seqs().values() {
            let seq_id = seq.deref().get_id();
            self.forget_mamba_sequence_state(seq_id);
            if !self.pending_runner_releases.contains(&seq_id) {
                self.pending_runner_releases.push(seq_id);
            }
        }
    }

    pub fn rollback_swap_in_groups(&mut self, group_ids: &[usize]) {
        for group_id in group_ids {
            let Some(index) = self
                .running
                .iter()
                .position(|group| group.get_id() == group_id)
            else {
                continue;
            };
            let group = self.running.remove(index).unwrap();
            group.set_status(SequenceStatus::Swapped);
            self.swapped_out.push_front(group);
        }
    }

    pub fn rollback_swap_out_groups(&mut self, group_ids: &[usize]) {
        for group_id in group_ids {
            let Some(index) = self
                .swapped_out
                .iter()
                .position(|group| group.get_id() == group_id)
            else {
                continue;
            };
            let group = self.swapped_out.remove(index).unwrap();
            group.set_status(SequenceStatus::Running);
            for seq in group.get_seqs().values() {
                seq.deref_mut().set_swapped_time(None);
            }
            self.running.push_front(group);
        }
    }

    pub fn schedule(&mut self) -> SchedulerOutput {
        // If there are no swapped seqs (they have higher priority), add seqs that are in the
        // waiting queue to the running queue.
        if self.swapped_out.is_empty() {
            let mut scheduled = VecDeque::new();
            let mut ignored_seq_groups = VecDeque::new();
            let mut blocks_to_copy = HashMap::new();
            let pre_existing_running = self.running.len();
            let mut num_scheduled_tokens = 0usize;

            let max_seqs_limit = active_sequence_limit(
                self.config.max_num_parallel_reqs.max(1),
                self.config.mamba_cache_capacity,
            );

            while !self.waiting.is_empty() {
                if self.is_last_prefill && pre_existing_running > 0 {
                    break; // interleaved scheduling
                }
                let seq_group = self.waiting.front().unwrap().clone();

                let group_tokens = seq_group
                    .get_seqs()
                    .values()
                    .map(|seq| seq.deref().prefill_chunk_tokens(self.prefill_chunk_size))
                    .sum::<usize>();
                if group_tokens > 0
                    && num_scheduled_tokens.saturating_add(group_tokens)
                        > self.config.max_num_batched_tokens.max(1)
                {
                    break;
                }

                if self.running.len() >= max_seqs_limit {
                    break;
                }
                let total_individual_seqs: usize = self
                    .running
                    .iter()
                    .map(|group| group.get_seqs().len())
                    .sum();
                if total_individual_seqs + 1 > self.config.max_num_parallel_reqs.max(1) {
                    break;
                }

                let has_block_table = self.block_engine.has_block_table(&seq_group);
                if !has_block_table {
                    // If we cannot allocate either now or in the future, either do not continue or remove the sequence.
                    let can_allocate = self
                        .block_engine
                        .can_allocate_for_prefill(&seq_group, self.prefill_chunk_size);
                    match can_allocate {
                        AllocStatus::Later => break, //If we can only allocate later, do not bother iterating over the rest.
                        AllocStatus::Impossible => {
                            warn!(
                                "Input prompt with length of {} tokens is too long and exceeds capacity of block engine.",
                                seq_group.get_prompt_len()
                            );
                            seq_group.set_status(SequenceStatus::FinishedIgnored);
                            ignored_seq_groups.push_back(self.waiting.pop_front().unwrap());
                            continue;
                        }
                        AllocStatus::Ok => {
                            self._allocate(&seq_group, &mut blocks_to_copy);
                        }
                    }
                } else if !self.ensure_prefill_chunk_slots(&seq_group) {
                    break;
                }

                seq_group.set_status(SequenceStatus::Running);

                let seq_group = self.waiting.pop_front().unwrap();
                self.running.push_back(seq_group.clone());
                scheduled.push_back(seq_group);
                num_scheduled_tokens = num_scheduled_tokens.saturating_add(group_tokens);
            }

            // If we did schedule, or we ignored sequences.
            if !scheduled.is_empty() || !ignored_seq_groups.is_empty() {
                self.is_last_prefill = true;
                return SchedulerOutput {
                    scheduled: Arc::new(scheduled),
                    blocks_to_swap_in: HashMap::new(),
                    blocks_to_copy,
                    blocks_to_swap_out: HashMap::new(),
                    swap_in_groups: Vec::new(),
                    swap_out_groups: Vec::new(),
                    ignored_seq_groups: Arc::new(ignored_seq_groups),
                };
            }
        }

        let mut blocks_to_swap_out = HashMap::new();
        let mut blocks_to_swap_in = HashMap::new();
        let mut blocks_to_copy = HashMap::new();
        let mut swap_in_groups = Vec::new();
        let mut swap_out_groups = Vec::new();

        // Reserve token slots for the running sequence groups, preempting the lowest (earliest) first.
        // Preempt lowest priority sequences that are in the running queue, forming a
        // new running queue that has the actually running sequences. Remember the preempted
        // sequences, which will be put into the waiting or swapped out state depending on
        // the preemption method (recompute or swap, respectively).

        // Sorts by creation time, in descending order so that earliest are latest (first come first serve).
        self.sort_running_by_priority_fcfs();

        let decode_max_seqs = if let Some(mamba_cap) = self.config.mamba_cache_capacity {
            if mamba_cap > 0 {
                // A swapped hybrid sequence keeps its active GDN/Mamba slot
                // resident while only its KV suffix is offloaded. Do not
                // admit more decode groups than the remaining slots can hold.
                let retained_mamba_slots = self
                    .swapped_out
                    .iter()
                    .map(|group| group.get_seqs().len())
                    .sum::<usize>();
                std::cmp::min(
                    mamba_cap.saturating_sub(retained_mamba_slots),
                    self.config.max_num_parallel_reqs.max(1),
                )
            } else {
                self.config.max_num_parallel_reqs.max(1)
            }
        } else {
            self.config.max_num_parallel_reqs.max(1)
        };

        let mut running = VecDeque::new();
        let mut preempted = VecDeque::new();
        while !self.running.is_empty() {
            if running.len() >= decode_max_seqs {
                while let Some(excess) = self.running.pop_front() {
                    self._preempt(
                        excess.clone(),
                        &mut blocks_to_swap_out,
                        &mut swap_out_groups,
                    );
                    preempted.push_back(excess);
                }
                break;
            }
            let seq_group = self.running.pop_front().unwrap();
            let mut finished_with_break = false;
            while !self.block_engine.can_append_token_to_seq(&seq_group) {
                let evicted = self.evict_prefix_cache_under_pressure();
                if evicted > 0 {
                    warn!("Evicted {} prefix cache block(s) under pressure.", evicted);
                    continue;
                }
                // If we cannot, now we need to preempt some seqs
                if !self.running.is_empty() {
                    // There is something to preempt.
                    let seq_to_preempt = self.running.pop_back().unwrap();
                    self._preempt(
                        seq_to_preempt.clone(),
                        &mut blocks_to_swap_out,
                        &mut swap_out_groups,
                    );
                    preempted.push_back(seq_to_preempt);
                } else {
                    // Nothing to preempt, preempt ourselves. Also, do not bother looking at anything else.
                    self._preempt(
                        seq_group.clone(),
                        &mut blocks_to_swap_out,
                        &mut swap_out_groups,
                    );
                    preempted.push_back(seq_group.clone());
                    finished_with_break = true;
                    break;
                }
            }
            if !finished_with_break {
                // If we need to, append physical blocks for a new token. We do not need to if there is enough space.
                // If we just got preempted, there is no reason to allocate
                self._append_token_slot_to_seq_group(&seq_group, &mut blocks_to_copy);
                running.push_back(seq_group);
            }
        }
        self.running = running;

        // Try to swap in the swapped out sequences and add these to the
        // running state if possible.

        // Sorts by creation time, in descending order so that earliest are latest (first come first serve).
        self.sort_swapped_out_by_priority_fcfs();

        if preempted.is_empty() {
            while !self.swapped_out.is_empty() {
                let seq_group = self.swapped_out.front().unwrap();
                let primary = seq_group
                    .get_seqs()
                    .values()
                    .min_by_key(|seq| seq.deref().get_id())
                    .unwrap();
                if let Some(swapped_time) = primary.deref().swapped_time() {
                    if SystemTime::now()
                        .duration_since(swapped_time)
                        .unwrap_or_default()
                        < SWAP_COOLING_PERIOD
                    {
                        break;
                    }
                }

                // If the GPU cannot handle the group being swapped in, stop
                if !self.block_engine.can_swap_in_seq_group(seq_group) {
                    let required_blocks = self.block_engine.seq_group_block_count(seq_group);
                    let evicted = self
                        .block_engine
                        .evict_prefix_cache_until_free(required_blocks);
                    if evicted > 0 {
                        warn!("Evicted {} prefix cache block(s) for swap-in.", evicted);
                    }
                    if !self.block_engine.can_swap_in_seq_group(seq_group) {
                        break;
                    }
                }

                let seq_group = self.swapped_out.pop_front().unwrap();
                // Swap in the blocks
                let to_swap_in = self.block_engine.swap_in(&seq_group);
                blocks_to_swap_in.extend(to_swap_in);
                swap_in_groups.push(*seq_group.get_id());
                for seq in seq_group.get_seqs().values() {
                    seq.deref_mut().set_swapped_time(None);
                }
                // Reserve a new slot
                self._append_token_slot_to_seq_group(&seq_group, &mut blocks_to_copy);
                self.running.push_back(seq_group);
            }
        }

        self.is_last_prefill = false;
        SchedulerOutput {
            scheduled: self.running.clone().into(),
            blocks_to_swap_in,
            blocks_to_copy,
            blocks_to_swap_out,
            swap_in_groups,
            swap_out_groups,
            ignored_seq_groups: Arc::new(VecDeque::new()),
        }
    }

    pub fn has_unfinished_sequences(&self) -> bool {
        !self.running.is_empty() || !self.waiting.is_empty()
    }

    pub fn has_waiting_sequences(&self) -> bool {
        !self.waiting.is_empty()
    }

    pub fn free_finished_sequence_groups(&mut self) -> Vec<usize> {
        self.free_finished_sequence_groups_with(|_, _, _| {})
    }

    pub fn free_finished_sequence_groups_with<F>(&mut self, mut on_finished: F) -> Vec<usize>
    where
        F: FnMut(usize, Option<u64>, Option<usize>),
    {
        let mut to_free = Vec::new();
        let mut released_ids = Vec::new();
        let clone = self.running.clone();
        self.running = clone
            .iter()
            .filter(|group| {
                if group.is_finished() {
                    to_free.push((*group).clone());
                    false
                } else {
                    true
                }
            })
            .cloned()
            .collect::<VecDeque<_>>();
        for group in to_free {
            for seq in group.get_seqs().values() {
                let seq_id = seq.deref().get_id();
                self.remember_finished_cached_tokens(seq_id, seq.deref().get_num_cached_tokens());
                let full_blocks = seq.deref().get_len() / self.block_engine.get_block_size();
                let block_id = self
                    .block_engine
                    .prefix_block_id_for_sequence(seq, full_blocks);
                let hash = self
                    .block_engine
                    .prefix_hash_for_sequence(seq, seq.deref().get_len());
                on_finished(seq_id, hash, block_id);
                released_ids.push(seq_id);
            }
            self._free(&group, true);
        }
        released_ids
    }

    pub fn get_num_cached_tokens_for_seq(&self, seq_id: usize) -> Option<usize> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.swapped_out.iter())
            .find_map(|group| {
                group
                    .get_seqs()
                    .values()
                    .find(|seq| seq.deref().get_id() == seq_id)
                    .map(|seq| seq.deref().get_num_cached_tokens())
            })
            .or_else(|| self.finished_cached_tokens.get(&seq_id).copied())
    }

    fn remember_finished_cached_tokens(&mut self, seq_id: usize, num_cached_tokens: usize) {
        self.finished_cached_tokens
            .insert(seq_id, num_cached_tokens);
        while self.finished_cached_tokens.len() > FINISHED_CACHED_TOKENS_MAX {
            let Some(oldest_seq_id) = self.finished_cached_tokens.keys().min().copied() else {
                break;
            };
            self.finished_cached_tokens.remove(&oldest_seq_id);
        }
    }

    pub fn prefix_cache_enabled(&self) -> bool {
        self.block_engine.prefix_cache_enabled()
    }

    pub fn query_prefix_cache_match_tokens(&mut self, tokens: &[u32]) -> usize {
        self.block_engine.query_prefix_cache_match_tokens(tokens)
    }

    pub fn print_free_blocks(&self) {
        let free_blocks = self.block_engine.get_num_free_blocks();
        let num_blocks = self.block_engine.get_num_blocks();
        let kvcache_mem_size = self.block_engine.get_kvcache_mem_size() as f32 / 1024f32;
        let used_percent = (num_blocks - free_blocks) as f32 * 100f32 / num_blocks as f32;
        tracing::info!(
            "GPU KvCache used {:.02}% ({:.02}/{:.02}GB, available {} KvCache tokens)",
            used_percent,
            used_percent / 100f32 * kvcache_mem_size,
            kvcache_mem_size,
            free_blocks * self.block_engine.get_block_size(),
        );
    }

    pub fn get_available_kv_tokens(&self) -> usize {
        let free_blocks = self.block_engine.get_num_free_blocks();
        free_blocks * self.block_engine.get_block_size()
    }

    pub fn ensure_available_kv_tokens(&mut self, required_tokens: usize) -> (usize, usize) {
        if required_tokens == 0 {
            return (self.get_available_kv_tokens(), 0);
        }

        let required_blocks = required_tokens.div_ceil(self.block_engine.get_block_size());
        let evicted = self
            .block_engine
            .evict_prefix_cache_until_free(required_blocks);
        (self.get_available_kv_tokens(), evicted)
    }

    pub fn filter_prefill_finished(
        &mut self,
        scheduled: &VecDeque<Arc<SequenceGroup>>,
        chunk_size: usize,
    ) -> (Vec<u32>, VecDeque<Arc<SequenceGroup>>) {
        let mut finished_indices = Vec::new();
        let mut remove_ids = Vec::new();
        let mut chunked_info = Vec::new();
        let mut chunk_finished_info = Vec::new();
        assert!(chunk_size > 0, "Invalid prefill chunk size!");
        for (i, group) in scheduled.iter().enumerate() {
            let seq = group.get_seqs().values().nth(0).unwrap();
            let prompt_len = seq.deref().get_prompt_len();
            let num_cached_tokens = seq.deref().get_num_cached_tokens();
            let chunk_tokens = seq.deref().prefill_chunk_tokens(chunk_size);
            if chunk_tokens == 0 || num_cached_tokens + chunk_tokens >= prompt_len {
                if num_cached_tokens > 0 {
                    chunk_finished_info.push((seq.deref().get_id(), prompt_len));
                }
                finished_indices.push(i as u32);
            } else {
                remove_ids.push(seq.deref().get_id());
                //unfinished due to chunked_prefill, push back to waiting list
                let group = group.clone();
                let seq = group.get_seqs().values().nth(0).unwrap();
                seq.deref_mut()
                    .set_num_cached_tokens(num_cached_tokens + chunk_tokens);
                if seq.deref().active_mamba_prefix_warmup_target().is_none() {
                    seq.deref_mut().clear_mamba_prefix_warmup();
                }
                group.set_status(SequenceStatus::Pending);
                chunked_info.push((
                    seq.deref().get_id(),
                    seq.deref().get_num_cached_tokens(),
                    prompt_len,
                ));
                self.waiting.push_back(group);
            }
        }
        if !chunked_info.is_empty() {
            let total_chunked: usize = chunked_info.iter().map(|(_, cached, _)| *cached).sum();
            let seq_details = chunked_info
                .iter()
                .map(|(id, cached, total)| format!("{}:{}/{}", id, cached, total))
                .collect::<Vec<_>>();
            tracing::info!(
                "Chunk prefilled {} seq(s) [{}] ({} total tokens processed)",
                chunked_info.len(),
                seq_details.join(", "),
                total_chunked
            );
        }
        if !chunk_finished_info.is_empty() {
            let seq_ids = chunk_finished_info
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>();
            let total: usize = chunk_finished_info.iter().map(|(_, len)| *len).sum();
            tracing::info!(
                "Chunk prefill finished for {} seq(s) {:?} ({} total tokens)",
                chunk_finished_info.len(),
                seq_ids,
                total
            );
        }
        self.running.retain(|s| {
            !remove_ids.contains(&s.get_seqs().values().nth(0).unwrap().deref().get_id())
        });

        let finished_groups: VecDeque<Arc<SequenceGroup>> = finished_indices
            .iter()
            .map(|&i| Arc::clone(&scheduled[i as usize]))
            .collect();
        (finished_indices, finished_groups)
    }

    pub fn abort_sequences(&mut self, seq_ids: &[usize]) -> Vec<usize> {
        if seq_ids.is_empty() {
            return Vec::new();
        }

        let seq_id_set = seq_ids.iter().copied().collect::<HashSet<_>>();
        let groups_to_abort = self
            .waiting
            .iter()
            .chain(self.running.iter())
            .chain(self.swapped_out.iter())
            .filter(|group| {
                group
                    .get_seqs()
                    .values()
                    .any(|seq| seq_id_set.contains(&seq.deref().get_id()))
            })
            .cloned()
            .collect::<Vec<_>>();

        let mut aborted_seq_ids = Vec::new();
        let mut aborted_group_ids = HashSet::new();
        for group in groups_to_abort {
            if !aborted_group_ids.insert(*group.get_id()) {
                continue;
            }

            for seq in group.get_seqs().values() {
                let seq_id = seq.deref().get_id();
                if seq_id_set.contains(&seq_id) {
                    aborted_seq_ids.push(seq_id);
                }
            }
            self._abort_seq_group(&group);
        }

        aborted_seq_ids.sort_unstable();
        aborted_seq_ids.dedup();
        aborted_seq_ids
    }
}

impl Scheduler {
    fn remove_seq_group(&mut self, seq_group: &SequenceGroup) {
        // Remove it if it is in waiting
        if let Some(idx) = self
            .waiting
            .iter()
            .position(|grp| grp.get_id() == seq_group.get_id())
        {
            self.waiting.remove(idx);
        };
        // Remove it if it is in running
        if let Some(idx) = self
            .running
            .iter()
            .position(|grp| grp.get_id() == seq_group.get_id())
        {
            self.running.remove(idx);
        };
        // Remove it if it is in swapped out
        if let Some(idx) = self
            .swapped_out
            .iter()
            .position(|grp| grp.get_id() == seq_group.get_id())
        {
            self.swapped_out.remove(idx);
        };
    }
    fn _append_token_slot_to_seq_group(
        &mut self,
        seq_group: &SequenceGroup,
        blocks_to_copy: &mut HashMap<usize, Vec<usize>>,
    ) {
        for seq in seq_group.get_seqs().values() {
            let op = self.block_engine.append_token_slot_to_seq(seq);
            if let Some((src_block, dst_block)) = op {
                if let std::collections::hash_map::Entry::Vacant(e) =
                    blocks_to_copy.entry(src_block)
                {
                    e.insert(vec![dst_block]);
                } else {
                    blocks_to_copy.get_mut(&src_block).unwrap().push(dst_block);
                }
            }
        }
    }

    fn _abort_seq_group(&mut self, seq_group: &SequenceGroup) {
        self.remove_seq_group(seq_group);
        for seq in seq_group.get_seqs().values() {
            self.forget_mamba_sequence_state(seq.deref().get_id());
        }
        seq_group.set_status(SequenceStatus::FinishedAborted);
        self._free(seq_group, false);
    }

    /// Preempt by recomputation when prefix caching is unavailable for a single
    /// sequence; otherwise preserve the sequence state and swap its KV blocks.
    fn _preempt(
        &mut self,
        seq_group: Arc<SequenceGroup>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
        swap_out_groups: &mut Vec<usize>,
    ) {
        if !self.block_engine.cpu_swap_enabled()
            || (seq_group.get_seqs().len() == 1 && !self.block_engine.prefix_cache_enabled())
        {
            self._preempt_by_recompute(seq_group);
        } else {
            self._preempt_by_swap(seq_group, blocks_to_swap_out, swap_out_groups);
        }
    }

    fn _preempt_by_recompute(&mut self, seq_group: Arc<SequenceGroup>) {
        self.request_runner_release_for_group(&seq_group);
        seq_group.set_status(SequenceStatus::Waiting);
        self._free(&seq_group, false);
        self.waiting.push_front(seq_group);
    }

    fn _preempt_by_swap(
        &mut self,
        seq_group: Arc<SequenceGroup>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
        swap_out_groups: &mut Vec<usize>,
    ) {
        if !self.block_engine.can_swap_out_seq_group(&seq_group) {
            if seq_group.get_seqs().len() == 1 {
                // Prefix-cache offload is opportunistic. If the suffix cannot
                // be copied (for example, CPU swap is exhausted), recompute
                // this single sequence instead of aborting the request.
                self._preempt_by_recompute(seq_group);
                return;
            }
            // If we cannot swap it out, abort the sequence group.
            self.request_runner_release_for_group(&seq_group);
            self._abort_seq_group(&seq_group);
            return;
        }
        let new_to_swap = self.block_engine.swap_out(&seq_group);
        blocks_to_swap_out.extend(new_to_swap);
        swap_out_groups.push(*seq_group.get_id());
        let swapped_time = Some(SystemTime::now());
        for seq in seq_group.get_seqs().values() {
            seq.deref_mut().set_swapped_time(swapped_time);
        }
        seq_group.set_status(SequenceStatus::Swapped);

        self.swapped_out.push_back(seq_group);
    }

    fn _allocate(
        &mut self,
        seq_group: &SequenceGroup,
        blocks_to_copy: &mut HashMap<usize, Vec<usize>>,
    ) {
        self.block_engine
            .allocate_for_prefill(seq_group, blocks_to_copy, self.prefill_chunk_size)
    }

    fn _free(&mut self, seq_group: &SequenceGroup, cache_prefix: bool) {
        for seq in seq_group.get_seqs().values() {
            if cache_prefix {
                if matches!(seq.deref().get_status(), SequenceStatus::Finished(_)) {
                    self.block_engine.cache_sequence(seq);
                }
            }
            self.block_engine.free_sequence(seq);
        }
    }

    fn sort_running_by_priority_fcfs(&mut self) {
        self.running
            .make_contiguous()
            .sort_by_key(|seq_group| seq_group.arrival_time());
        self.running.make_contiguous().reverse();
    }

    fn sort_swapped_out_by_priority_fcfs(&mut self) {
        self.swapped_out
            .make_contiguous()
            .sort_by_key(|seq_group| seq_group.arrival_time());
        self.swapped_out.make_contiguous().reverse();
    }

    fn evict_prefix_cache_under_pressure(&mut self) -> usize {
        let cached_blocks = self.block_engine.prefix_cache_blocks();
        if cached_blocks == 0 {
            return 0;
        }
        let blocks = ((cached_blocks as f32) * PREFIX_CACHE_PRESSURE_EVICT_PERCENT).ceil() as usize;
        let blocks = blocks.max(1);
        self.block_engine.evict_prefix_cache_blocks(blocks)
    }

    fn ensure_prefill_chunk_slots(&mut self, seq_group: &SequenceGroup) -> bool {
        let required_blocks = self
            .block_engine
            .prefill_chunk_blocks_required(seq_group, self.prefill_chunk_size);
        if required_blocks > self.block_engine.get_num_free_blocks() {
            let evicted = self
                .block_engine
                .evict_prefix_cache_until_free(required_blocks);
            if evicted > 0 {
                warn!(
                    "Evicted {} prefix cache block(s) for prefill chunk.",
                    evicted
                );
            }
        }
        if !self
            .block_engine
            .can_append_prefill_chunk_to_seq_group(seq_group, self.prefill_chunk_size)
        {
            return false;
        }
        self.block_engine
            .append_prefill_chunk_slots_to_seq_group(seq_group, self.prefill_chunk_size);
        true
    }
}
