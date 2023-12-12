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

pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: VecDeque<Sequence>,
    swapped_out: VecDeque<Sequence>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            swapped_out: VecDeque::new(),
        }
    }

    pub fn add_sequence(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    pub fn schedule(&mut self) -> SchedulerOutput<'_> {
        // If there are no swapped seqs (they have higher priority), add seqs that are in the
        // waiting queue to the running queue.

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
