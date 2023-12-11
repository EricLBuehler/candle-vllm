use std::collections::VecDeque;

use super::sequence::SequenceGroup;

pub struct Scheduler {
    waiting: VecDeque<SequenceGroup>,
    runing: VecDeque<SequenceGroup>,
    swapped: VecDeque<SequenceGroup>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            waiting: VecDeque::new(),
            runing: VecDeque::new(),
            swapped: VecDeque::new(),
        }
    }

    pub fn add_sequence(&mut self, seq: SequenceGroup) {
        self.waiting.push_back(seq);
    }
}
