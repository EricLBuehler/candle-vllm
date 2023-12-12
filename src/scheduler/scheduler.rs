use std::collections::{HashMap, VecDeque};

use super::sequence::Sequence;

type CPUBlockFrom = usize;
type GPUBlockFrom = usize;
type CPUBlockTo = usize;
type GPUBlockTo = usize;
type SrcBlockFrom = usize;
type DstBlocksTo = Vec<usize>;

pub struct SchedulerOutput<'a> {
    running: &'a [Sequence],
    blocks_to_swap_in: HashMap<CPUBlockFrom, GPUBlockTo>,
    blocks_to_swap_out: HashMap<GPUBlockFrom, CPUBlockTo>,
    blocks_to_copy: HashMap<SrcBlockFrom, DstBlocksTo>,
}

pub struct SchedulerConfig {
    pub max_num_seqs: usize,
}

pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: VecDeque<Sequence>,
    swapped_out: VecDeque<Sequence>,
    config: SchedulerConfig,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            swapped_out: VecDeque::new(),
            config,
        }
    }

    pub fn add_sequence(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    pub fn schedule(&mut self) -> SchedulerOutput<'_> {
        // If there are no swapped seqs (they have higher priority), add seqs that are in the
        // waiting queue to the running queue.
        if self.swapped_out.is_empty() {
            while !self.waiting.is_empty() {
                let seq = self.waiting.get(0).unwrap();

                // If adding this seq means we will have too many, stop as no more could be added.
                if self.config.max_num_seqs == self.running.len() + 1 {
                    break;
                }

                // TODO(EricLBuehler): More conditions must be added here, specifically whether it can be allocated!

                self.running.push_back(self.waiting.pop_front().unwrap());
            }
        }

        // Preempt lowest priority sequences that are in the running queue, forming a
        // new running queue that has the actually running sequences. Remember the preempted
        // sequences, which will be put into the waiting or swapped out state depending on
        // the preemption method (recompute or swap, repectively).

        // Try to swap in the swapped out sequences and add these to the
        // running state if possible.

        todo!()
    }
}

impl Scheduler {
    fn get_running_slice(&self) -> &[Sequence] {
        &*self.running.make_contiguous()
    }
}
