use std::{
    collections::{HashMap, VecDeque},
    rc::Rc,
};

use crate::{
    log_warning,
    scheduler::{block_engine::AllocStatus, sequence::SequenceStatus},
};

use super::{block_engine::BlockEngine, cache_engine::CacheConfig, sequence::Sequence};

type CPUBlockFrom = usize;
type GPUBlockFrom = usize;
type CPUBlockTo = usize;
type GPUBlockTo = usize;
type SrcBlockFrom = usize;
type DstBlocksTo = Vec<usize>;

pub struct SchedulerOutput {
    scheduled: Rc<Vec<Rc<Sequence>>>,
    blocks_to_swap_in: HashMap<CPUBlockFrom, GPUBlockTo>,
    blocks_to_swap_out: HashMap<GPUBlockFrom, CPUBlockTo>,
    blocks_to_copy: HashMap<SrcBlockFrom, DstBlocksTo>,
}

pub struct SchedulerConfig {
    pub max_num_seqs: usize,
}

pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Rc<VecDeque<Rc<Sequence>>>,
    swapped_out: VecDeque<Sequence>,
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

    pub fn add_sequence(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    pub fn schedule(&mut self) -> SchedulerOutput {
        // If there are no swapped seqs (they have higher priority), add seqs that are in the
        // waiting queue to the running queue.
        if self.swapped_out.is_empty() {
            let mut scheduled = Vec::new();
            let mut ignored_seq_groups = Vec::new();
            while !self.waiting.is_empty() {
                let seq = self.waiting.get(0).unwrap();

                // If adding this seq means we will have too many, stop as no more could be added.
                if self.config.max_num_seqs == self.running.len() + 1 {
                    break;
                }

                // If we cannot allocate either now or in the future, either do not continue or remove the sequence.
                let can_allocate = self.block_engine.can_allocate(seq);
                match can_allocate {
                    AllocStatus::Later => break, //If we can only allocate later, do not bother iterating over the rest.
                    AllocStatus::Impossible => {
                        log_warning(
                            &format!("Input prompt with length of {} tokens is too long and exceeds capacity of block engine.",
                            seq.get_len())
                        );
                        seq.deref_mut().set_status(SequenceStatus::FinishedIgnored);
                        ignored_seq_groups.push(self.waiting.pop_front());
                    }
                    _ => {}
                }

                seq.deref_mut().set_status(SequenceStatus::Running);
                self._allocate(seq);

                let seq = Rc::new(self.waiting.pop_front().unwrap());
                self.running.push_back(seq.clone());
                scheduled.push(seq);
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
            let seq = self.running.pop_front().unwrap();
            let mut finished_with_break = false;
            while !self.block_engine.can_append_token_to_seq(&seq) {
                // If we cannot, now we need to preempt some seqs
                if !self.running.is_empty() {
                    // There is something to preempt.
                    let seq_to_preempt = self.running.pop_back().unwrap();
                    // TODO(EricLBuehler): Actually preempt here.
                    preempted.push_back(seq_to_preempt);
                } else {
                    // Nothing to preempt, preempt ourselves. Also, do not bother looking at anything else.
                    // TODO(EricLBuehler): Actually preempt here.
                    preempted.push_back(seq);
                    finished_with_break = true;
                    break;
                }
            }
            if !finished_with_break {
                // If we need to, append a physical block for a new token. We do not need to if there is enough space.
                // TODO(EricLBuehler): (possibly) Append the slot
                running.push_back(seq);
            }
        }
        self.running = Rc::new(running);

        // Try to swap in the swapped out sequences and add these to the
        // running state if possible.

        todo!()
    }
}

impl Scheduler {
    fn _allocate(&self, seq: &Sequence) {
        self.block_engine.allocate(seq)
    }

    fn sort_running_by_priority_fcfs(&self) {
        self.running
            .make_contiguous()
            .sort_by_key(|seq| seq.get_id());
        self.running.make_contiguous().reverse();
    }
}
