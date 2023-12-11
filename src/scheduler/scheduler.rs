use std::collections::{HashMap, VecDeque};

use super::sequence::Sequence;

pub struct SchedulerOutput<'a> {
    running: &'a [Sequence],
    blocks_to_swap_in: HashMap<usize, usize>,
    blocks_to_swap_out: HashMap<usize, usize>,
    blocks_to_copy: HashMap<usize, Vec<usize>>,
}

pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: VecDeque<Sequence>,
    swapped: VecDeque<Sequence>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            swapped: VecDeque::new(),
        }
    }

    pub fn add_sequence(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    pub fn schedule(&mut self) -> SchedulerOutput<'_> {
        todo!()
    }
}

impl Scheduler {
    fn get_running_slice(&self) -> &[Sequence] {
        &*self.running.make_contiguous()
    }
}
