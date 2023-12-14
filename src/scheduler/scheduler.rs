use std::{
    collections::{HashMap, VecDeque},
    rc::Rc,
};

use crate::{
    log_warning,
    scheduler::{block_engine::AllocStatus, sequence::SequenceStatus},
};

use super::{block_engine::BlockEngine, cache_engine::CacheConfig, sequence::SequenceGroup};

type CPUBlockFrom = usize;
type GPUBlockFrom = usize;
type CPUBlockTo = usize;
type GPUBlockTo = usize;
type SrcBlockFrom = usize;
type DstBlocksTo = Vec<usize>;

pub struct SchedulerOutput {
    scheduled: Rc<Vec<Rc<SequenceGroup>>>,
    blocks_to_swap_in: HashMap<CPUBlockFrom, GPUBlockTo>,
    blocks_to_swap_out: HashMap<GPUBlockFrom, CPUBlockTo>,
    blocks_to_copy: HashMap<SrcBlockFrom, DstBlocksTo>,
}

pub struct SchedulerConfig {
    pub max_num_seqs: usize,
}

pub struct Scheduler {
    waiting: VecDeque<Rc<SequenceGroup>>,
    running: Rc<VecDeque<Rc<SequenceGroup>>>,
    swapped_out: VecDeque<Rc<SequenceGroup>>,
    config: SchedulerConfig,
    block_engine: BlockEngine,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, cache_config: &CacheConfig) -> Self {
        assert!(cache_config.fully_init);
        Self {
            waiting: VecDeque::new(),
            running: Rc::new(VecDeque::new()),
            swapped_out: VecDeque::new(),
            config,
            block_engine: BlockEngine::new(
                cache_config.block_size,
                cache_config.num_gpu_blocks.unwrap(),
                cache_config.num_cpu_blocks.unwrap(),
            ),
        }
    }

    pub fn add_sequence(&mut self, seq: SequenceGroup) {
        self.waiting.push_back(Rc::new(seq));
    }

    pub fn schedule(&mut self) -> SchedulerOutput {
        // If there are no swapped seqs (they have higher priority), add seqs that are in the
        // waiting queue to the running queue.
        if self.swapped_out.is_empty() {
            let mut scheduled = Vec::new();
            let mut ignored_seq_groups = Vec::new();
            while !self.waiting.is_empty() {
                let seq_group = self.waiting.get(0).unwrap();

                // If adding this seq means we will have too many, stop as no more could be added.
                if self.config.max_num_seqs == self.running.len() + 1 {
                    break;
                }

                // If we cannot allocate either now or in the future, either do not continue or remove the sequence.
                let can_allocate = self.block_engine.can_allocate(seq_group);
                match can_allocate {
                    AllocStatus::Later => break, //If we can only allocate later, do not bother iterating over the rest.
                    AllocStatus::Impossible => {
                        log_warning(
                            &format!("Input prompt with length of {} tokens is too long and exceeds capacity of block engine.",
                            seq_group.get_prompt_len())
                        );
                        seq_group.set_status(SequenceStatus::FinishedIgnored);
                        ignored_seq_groups.push(self.waiting.pop_front());
                    }
                    _ => {}
                }

                seq_group.set_status(SequenceStatus::Running);
                self._allocate(seq_group);

                let seq_group = self.waiting.pop_front().unwrap();
                self.running.push_back(seq_group.clone());
                scheduled.push(seq_group);
            }

            // If we did schedule, or we ignored sequences.
            if !scheduled.is_empty() || !ignored_seq_groups.is_empty() {
                return SchedulerOutput {
                    scheduled: Rc::new(scheduled),
                    blocks_to_swap_in: HashMap::new(),
                    blocks_to_copy: HashMap::new(),
                    blocks_to_swap_out: HashMap::new(),
                };
            }
        }

        let mut blocks_to_swap_out = HashMap::new();

        // Reserve token slots for the running sequence groups, preempting the lowest (earliest) first.
        // Preempt lowest priority sequences that are in the running queue, forming a
        // new running queue that has the actually running sequences. Remember the preempted
        // sequences, which will be put into the waiting or swapped out state depending on
        // the preemption method (recompute or swap, repectively).

        // Sorts by creation time, in descending order so that earliest are latest (first come first serve).
        self.sort_running_by_priority_fcfs();

        let mut running = VecDeque::new();
        let mut preempted = VecDeque::new();
        while !self.running.is_empty() {
            let mut seq_group = self.running.pop_front().unwrap();
            let mut finished_with_break = false;
            while !self.block_engine.can_append_token_to_seq(&seq_group) {
                // If we cannot, now we need to preempt some seqs
                if !self.running.is_empty() {
                    // There is something to preempt.
                    let mut seq_to_preempt = self.running.pop_back().unwrap();
                    self._preempt(seq_to_preempt.clone(), &mut blocks_to_swap_out);
                    preempted.push_back(seq_to_preempt);
                } else {
                    // Nothing to preempt, preempt ourselves. Also, do not bother looking at anything else.
                    self._preempt(seq_group.clone(), &mut blocks_to_swap_out);
                    preempted.push_back(seq_group);
                    finished_with_break = true;
                    break;
                }
            }
            if !finished_with_break {
                // If we need to, append physical blocks for a new token. We do not need to if there is enough space.
                // TODO(EricLBuehler): (possibly) Append the slot
                running.push_back(seq_group);
            }
        }
        self.running = Rc::new(running);

        // Try to swap in the swapped out sequences and add these to the
        // running state if possible.

        todo!()
    }
}

impl Scheduler {
    fn _abort_seq_group(&self, seq_group: &SequenceGroup) {
        // Remove it if it is in waiting
        match self
            .waiting
            .iter()
            .position(|grp| grp.get_id() == seq_group.get_id())
        {
            Some(idx) => {
                self.waiting.remove(idx);
            }
            None => {}
        };
        // Remove it if it is in running
        match self
            .running
            .iter()
            .position(|grp| grp.get_id() == seq_group.get_id())
        {
            Some(idx) => {
                self.running.remove(idx);
            }
            None => {}
        };
        // Remove it if it is in swapped out
        match self
            .swapped_out
            .iter()
            .position(|grp| grp.get_id() == seq_group.get_id())
        {
            Some(idx) => {
                self.swapped_out.remove(idx);
            }
            None => {}
        };
        seq_group.set_status(SequenceStatus::FinishedAborted);
        self._free(&seq_group);
    }

    fn _preempt(
        &self,
        seq_group: Rc<SequenceGroup>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
    ) {
        match seq_group.get_seqs().len() {
            1 => self._preempt_by_recompute(seq_group),
            _ => self._preempt_by_swap(seq_group, blocks_to_swap_out),
        }
    }

    fn _preempt_by_recompute(&self, mut seq_group: Rc<SequenceGroup>) {
        seq_group.set_status(SequenceStatus::Waiting);
        self._free(&seq_group);
        self.waiting.push_front(seq_group);
    }

    fn _preempt_by_swap(
        &self,
        mut seq_group: Rc<SequenceGroup>,
        blocks_to_swap_out: &mut HashMap<usize, usize>,
    ) {
        if !self.block_engine.can_swap_out_seq_group(&seq_group) {
            self._abort_seq_group(&seq_group);
        }
        let new_to_swap = self.block_engine.swap_out(&seq_group);
        blocks_to_swap_out.extend(new_to_swap);
        seq_group.set_status(SequenceStatus::Swapped);

        self.swapped_out.push_back(seq_group);
    }

    fn _allocate(&self, seq_group: &SequenceGroup) {
        self.block_engine.allocate(seq_group)
    }

    fn _free(&self, seq_group: &SequenceGroup) {
        for (_, seq) in seq_group.get_seqs() {
            self.block_engine.free_sequence(&seq);
        }
    }

    fn sort_running_by_priority_fcfs(&self) {
        self.running
            .make_contiguous()
            .sort_by_key(|seq_group| seq_group.arrival_time());
        self.running.make_contiguous().reverse();
    }
}
