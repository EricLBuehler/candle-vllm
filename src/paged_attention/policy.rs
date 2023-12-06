use std::{collections::VecDeque, sync::Arc};

use super::sequence::SequenceGroup;

pub trait Policy {
    fn get_priority(&self, now: u64, seq_group: &Arc<SequenceGroup>) -> u64;
    fn sort_by_priority(&self, now: u64, seq_groups: &mut VecDeque<Arc<SequenceGroup>>) {
        seq_groups
            .make_contiguous()
            .sort_by_key(|seq_group| self.get_priority(now, seq_group));
        seq_groups.make_contiguous().reverse();
    }
}

/// First come first serve
struct FCFS;

impl Policy for FCFS {
    fn get_priority(&self, now: u64, seq_group: &Arc<SequenceGroup>) -> u64 {
        now - seq_group.deref().arrival_time
    }
}

pub struct PolicyFactory;

pub enum PolicyType {
    FirstComeFirstServe,
}

impl PolicyFactory {
    pub fn get_policy(policy_name: PolicyType) -> Box<dyn Policy> {
        match policy_name {
            PolicyType::FirstComeFirstServe => Box::new(FCFS),
        }
    }
}
