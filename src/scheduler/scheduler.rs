use std::collections::VecDeque;

use super::sequence::Sequence;

pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    runing: VecDeque<Sequence>,
    swapped: VecDeque<Sequence>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            waiting: VecDeque::new(),
            runing: VecDeque::new(),
            swapped: VecDeque::new(),
        }
    }

    pub fn add_sequence(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }
}
