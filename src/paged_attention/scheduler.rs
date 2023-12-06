use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use crate::{
    openai::{responses::APIError, utils::get_created_time_secs},
    paged_attention::{
        block_manager::{AllocStatus, NewBlockAllocated},
        sequence::SequenceStatus,
    },
};

use super::{
    block_manager::BlockSpaceManager,
    cache_engine::CacheConfig,
    policy::{Policy, PolicyFactory, PolicyType},
    sequence::{SequenceData, SequenceGroup, SequenceGroupMetadata},
};

pub struct SchedulerOutputs {
    prompt_run: bool,
    num_batched_tokens: usize,
    blocks_to_swap_in: HashMap<usize, usize>,
    blocks_to_swap_out: HashMap<usize, usize>,
    blocks_to_copy: HashMap<usize, Vec<usize>>,
    scheduled_seq_groups: Vec<Arc<SequenceGroup>>,
    ignored_seq_groups: Vec<SequenceGroup>,
}

impl SchedulerOutputs {
    pub fn new(
        prompt_run: bool,
        num_batched_tokens: usize,
        blocks_to_swap_in: HashMap<usize, usize>,
        blocks_to_swap_out: HashMap<usize, usize>,
        blocks_to_copy: HashMap<usize, Vec<usize>>,
        scheduled_seq_groups: Vec<Arc<SequenceGroup>>,
        ignored_seq_groups: Vec<SequenceGroup>,
    ) -> Self {
        Self {
            prompt_run,
            num_batched_tokens,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            scheduled_seq_groups,
            ignored_seq_groups,
        }
    }

    pub fn is_empty(&self) -> bool {
        // NOTE: ignored sequence groups are not considered
        self.scheduled_seq_groups.is_empty()
            && self.blocks_to_swap_in.is_empty()
            && self.blocks_to_swap_out.is_empty()
            && self.blocks_to_copy.is_empty()
    }
}

pub struct SchedulerConfig {
    max_num_batched_tokens: usize,
    max_num_seqs: usize,
    max_model_len: usize,
    max_paddings: usize,
}

impl SchedulerConfig {
    pub fn new(
        max_num_batched_tokens: Option<usize>,
        max_num_seqs: usize,
        max_model_len: usize,
        max_paddings: usize,
    ) -> Result<Self, APIError> {
        let max_num_batched_tokens = if let Some(max_num_batched_tokens) = max_num_batched_tokens {
            max_num_batched_tokens
        } else {
            max_model_len.max(2048)
        };
        if max_num_batched_tokens < max_model_len {
            return Err(APIError::new(format!("max_num_batched_tokens ({max_num_batched_tokens}) is smaller than max_model_len ({max_model_len}).\
            This effectively limits the maximum sequence length to \
            max_num_batched_tokens and makes candle vLLM reject longer \
            sequences. Please increase max_num_batched_tokens or \
            decrease max_model_len.")));
        }
        if max_num_batched_tokens < max_num_seqs {
            return Err(APIError::new(format!("max_num_batched_tokens ({max_num_batched_tokens}) must be greater than or equal to \
            max_num_seqs ({max_num_seqs})")));
        }

        Ok(Self {
            max_num_batched_tokens,
            max_num_seqs,
            max_model_len,
            max_paddings,
        })
    }
}

#[derive(PartialEq)]
pub enum PreemptionMode {
    Recompute,
    Swap,
}

pub struct Scheduler {
    waiting: VecDeque<SequenceGroup>,
    running: VecDeque<Arc<SequenceGroup>>,
    swapped: VecDeque<Arc<SequenceGroup>>,
    prompt_limit: usize,
    block_manager: BlockSpaceManager,
    scheduler_config: SchedulerConfig,
    policy: Box<dyn Policy>,
}

impl Scheduler {
    pub fn new(
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> Result<Self, APIError> {
        let prompt_limit = scheduler_config
            .max_model_len
            .min(scheduler_config.max_num_batched_tokens);
        let block_manager = BlockSpaceManager::new(
            cache_config.block_size,
            cache_config.num_gpu_blocks.unwrap(),
            cache_config.num_cpu_blocks.unwrap(),
            0.01.try_into().unwrap(),
            cache_config.sliding_window,
        )?;

        Ok(Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            swapped: VecDeque::new(),
            prompt_limit,
            block_manager,
            scheduler_config,
            policy: PolicyFactory::get_policy(PolicyType::FirstComeFirstServe),
        })
    }

    pub fn schedule(&mut self) -> (Vec<SequenceGroupMetadata>, SchedulerOutputs) {
        let scheduler_outputs = self._schedule();

        // Create input data structures
        let mut seq_group_metadata_list = Vec::new();
        for seq_group in &scheduler_outputs.scheduled_seq_groups {
            let mut seq_data = HashMap::new();
            let mut block_tables = HashMap::new();
            for seq in seq_group.deref().get_seqs(Some(SequenceStatus::Running)) {
                seq_data.insert(seq.seq_id, seq.data.clone());
                block_tables.insert(seq.seq_id, self.block_manager.get_block_table(seq));
            }

            seq_group_metadata_list.push(SequenceGroupMetadata {
                request_id: seq_group.deref().request_id.clone(),
                is_prompt: scheduler_outputs.prompt_run,
                seq_data,
                sampling_params: seq_group.deref().sampling_params.clone(),
                block_tables,
            });
        }
        (seq_group_metadata_list, scheduler_outputs)
    }

    fn _schedule(&mut self) -> SchedulerOutputs {
        let mut blocks_to_swap_in = HashMap::new();
        let mut blocks_to_swap_out = HashMap::new();
        let mut blocks_to_copy = HashMap::new();

        let now = get_created_time_secs();

        if self.swapped.is_empty() {
            let mut ignored_seq_groups = Vec::new();
            let mut scheduled = Vec::new();

            // Total number of sequences on the fly, including in the generation phase
            let mut num_curr_seqs: usize = self
                .running
                .iter()
                .map(|group| group.deref().get_max_num_running_seqs())
                .sum();
            let mut seq_lens = Vec::new();

            // Optimization: do not sort the waiting queue since the preempted sequence groups are added to the front and the new
            // sequence groups are added to the back.
            while !self.waiting.is_empty() {
                let seq_group = self.waiting.front_mut().unwrap();

                assert_eq!(
                    seq_group.deref().num_seqs(None),
                    1,
                    "Waiting sequence group should have only one prompt sequence"
                );
                let num_prompt_tokens = seq_group.deref().get_seqs(None).get(0).unwrap().len();
                if num_prompt_tokens > self.prompt_limit {
                    eprintln!("Warning: Input prompt ({num_prompt_tokens} tokens) is too long and exceeds the limit of {}", self.prompt_limit);
                    for seq in seq_group.deref_mut().get_mut_seqs(None) {
                        seq.set_status(SequenceStatus::FinishedIgnored);
                    }
                    let seq_group = self.waiting.pop_front().unwrap();
                    ignored_seq_groups.push(seq_group);
                    continue;
                }

                // If the sequence group cannot be allocated, stop.
                let can_allocate = self.block_manager.can_allocate(&seq_group);
                if matches!(can_allocate, AllocStatus::Later) {
                    break;
                } else if matches!(can_allocate, AllocStatus::Never) {
                    eprintln!("Warning: Input prompt ({num_prompt_tokens} tokens) is too long and exceeds the limit of {}", self.prompt_limit);
                    for seq in seq_group.deref_mut().get_mut_seqs(None) {
                        seq.set_status(SequenceStatus::FinishedIgnored);
                    }
                    let seq_group = self.waiting.pop_front().unwrap();
                    ignored_seq_groups.push(seq_group);
                    continue;
                }

                // If the number of batched tokens exceeds the limit, stop
                let mut new_seq_lens = seq_lens.clone();
                new_seq_lens.extend(vec![num_prompt_tokens]);
                let num_batched_tokens = new_seq_lens.len() * new_seq_lens.iter().max().unwrap();
                if num_batched_tokens > self.scheduler_config.max_num_batched_tokens {
                    break;
                }

                // The total number of sequences in the RUNNING state should not exceed the maximum number of sequences.
                let num_new_seqs = seq_group.deref().get_max_num_running_seqs();
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs {
                    break;
                }
                seq_lens = new_seq_lens;

                let seq_group = Arc::new(self.waiting.pop_front().unwrap());
                self._allocate(&seq_group);
                self.running.push_back(seq_group);
                num_curr_seqs += num_new_seqs;
                scheduled.push(self.running.back().unwrap().clone());
            }

            if !scheduled.is_empty() || !ignored_seq_groups.is_empty() {
                return SchedulerOutputs::new(
                    true,
                    seq_lens.len()
                        * if !seq_lens.is_empty() {
                            *seq_lens.iter().max().unwrap()
                        } else {
                            0
                        },
                    blocks_to_swap_in,
                    blocks_to_swap_out,
                    blocks_to_copy,
                    scheduled,
                    ignored_seq_groups,
                );
            }
        }

        // Policy is responsible for deciding which sequence groups to preempt.
        self.policy.sort_by_priority(now, &mut self.running);

        let mut running = VecDeque::new();
        let mut preempted = VecDeque::new();
        while !self.running.is_empty() {
            let seq_group = self.running.pop_front().unwrap();
            let mut did_not_break = true;
            while !self.block_manager.can_append_slot(&seq_group) {
                if !self.running.is_empty() {
                    // Preempt the lowest priority sequence groups
                    let victim_seq_group = self.running.pop_back().unwrap();
                    self._preempt(victim_seq_group.clone(), &mut blocks_to_swap_out, None);
                    preempted.push_back(victim_seq_group.clone());
                } else {
                    //No other sequence groups can be preempted; Preempt the current sequence group
                    self._preempt(seq_group.clone(), &mut blocks_to_swap_out, None);
                    preempted.push_back(seq_group.clone());
                    did_not_break = false;
                    break;
                }
            }
            if did_not_break {
                self._append_slot(seq_group.clone(), &mut blocks_to_copy);
                running.push_back(seq_group);
            }
        }
        self.running = running;

        // Swap in the sequence groups in the SWAPPED state if possible
        self.policy.sort_by_priority(now, &mut self.swapped);
        if preempted.is_empty() {
            let mut num_curr_seqs: usize = self
                .running
                .iter()
                .map(|grp| grp.deref().get_max_num_running_seqs())
                .sum();

            while !self.swapped.is_empty() {
                let seq_group = self.swapped.front().unwrap();
                //If the sequence group cannot be swapped in, stop
                if !self.block_manager.can_swap_in(&seq_group) {
                    break;
                }

                // The total number of seq in the RUNNING should not exceed the max num of seqs
                let num_new_seqs = seq_group.deref().get_max_num_running_seqs();
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs {
                    break;
                }

                let seq_group = self.swapped.pop_front().unwrap();
                self._swap_in(seq_group.clone(), &mut blocks_to_swap_in);
                self._append_slot(seq_group.clone(), &mut blocks_to_copy);
                num_curr_seqs += num_new_seqs;
                self.running.push_back(seq_group);
            }
        }

        // Each seq in the generation phase only takes one token slot. Therefore,
        // the num of batched toks is equal to the number of seqs in the RUNNING state.
        let num_batched_tokens: usize = self
            .running
            .iter()
            .map(|group| group.deref().num_seqs(Some(SequenceStatus::Running)))
            .sum();

        SchedulerOutputs::new(
            false,
            num_batched_tokens,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            self.running.clone().into(),
            vec![],
        )
    }

    fn _append_slot(
        &mut self,
        mut seq_group: Arc<SequenceGroup>,
        blocks_to_copy: &mut HashMap<usize, Vec<usize>>,
    ) -> Result<(), APIError> {
        for seq in seq_group
            .deref_mut()
            .get_mut_seqs(Some(SequenceStatus::Running))
        {
            let ret = self.block_manager.append_slot(seq)?;
            if let Some(NewBlockAllocated {
                src_block,
                dst_block,
            }) = ret
            {
                if blocks_to_copy.contains_key(&src_block) {
                    blocks_to_copy.get_mut(&src_block).unwrap().push(dst_block);
                } else {
                    *blocks_to_copy.get_mut(&src_block).unwrap() = vec![dst_block];
                }
            }
        }
        Ok(())
    }

    fn _preempt(
        &mut self,
        mut victim_group: Arc<SequenceGroup>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
        mut preemption_mode: Option<PreemptionMode>,
    ) {
        let preemption_mode = if let None = preemption_mode {
            if victim_group.deref().get_max_num_running_seqs() == 1 {
                PreemptionMode::Recompute
            } else {
                PreemptionMode::Swap
            }
        } else {
            preemption_mode.unwrap()
        };

        match preemption_mode {
            PreemptionMode::Recompute => {
                self._preempt_by_recompute(victim_group);
            }
            PreemptionMode::Swap => self._preempt_by_swap(victim_group, blocks_to_swap_out),
        }
    }

    fn _preempt_by_recompute(&mut self, mut victim_group: Arc<SequenceGroup>) {
        let mut binding = victim_group.deref_mut();
        let seqs = binding.get_mut_seqs(Some(SequenceStatus::Running));
        assert_eq!(seqs.len(), 1);
        for seq in seqs {
            seq.set_status(SequenceStatus::Waiting);
            self.block_manager.free(&seq);
        }
    }

    fn _preempt_by_swap(
        &mut self,
        mut victim_group: Arc<SequenceGroup>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
    ) {
        self._swap_out(victim_group.clone(), blocks_to_swap_out);
        self.swapped.push_back(victim_group.clone())
    }

    fn _swap_out(
        &mut self,
        mut victim_group: Arc<SequenceGroup>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
    ) -> Result<(), APIError> {
        if !self.block_manager.can_swap_out(victim_group.clone()) {
            panic!("Aborted due to the lack of CPU swap space. Please increase the swap space to avoid this error.");
        }
        let mapping = self.block_manager.swap_out(&victim_group)?;
        blocks_to_swap_out.extend(mapping);
        for seq in victim_group
            .deref_mut()
            .get_mut_seqs(Some(SequenceStatus::Running))
        {
            seq.set_status(SequenceStatus::Swapped);
        }
        Ok(())
    }

    fn _swap_in(
        &mut self,
        seq_group: Arc<SequenceGroup>,
        blocks_to_swap_in: &mut HashMap<usize, usize>,
    ) -> Result<(), APIError> {
        let mapping = self.block_manager.swap_in(&seq_group)?;
        blocks_to_swap_in.extend(mapping.iter());
        for seq in seq_group
            .deref_mut()
            .get_mut_seqs(Some(SequenceStatus::Swapped))
        {
            seq.set_status(SequenceStatus::Running);
        }
        Ok(())
    }

    fn _allocate(&mut self, seq_group: &SequenceGroup) {
        self.block_manager.allocate(seq_group);
        for seq in seq_group.deref_mut().get_mut_seqs(None) {
            seq.set_status(SequenceStatus::Running);
        }
    }
}
