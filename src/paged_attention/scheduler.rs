use std::collections::HashMap;

pub struct SchedulerOutputs {
    prompt_run: bool,
    num_batched_tokens: usize,
    blocks_to_swap_in: HashMap<usize, usize>,
    blocks_to_swap_out: HashMap<usize, usize>,
    blocks_to_copy: HashMap<usize, Vec<usize>>,
    n_scheduled_seq_groups: usize,
    n_ignored_seq_groups: usize,
}

impl SchedulerOutputs {
    pub fn new(
        prompt_run: bool,
        num_batched_tokens: usize,
        blocks_to_swap_in: HashMap<usize, usize>,
        blocks_to_swap_out: HashMap<usize, usize>,
        blocks_to_copy: HashMap<usize, Vec<usize>>,
        n_scheduled_seq_groups: usize,
        n_ignored_seq_groups: usize,
    ) -> Self {
        Self {
            prompt_run,
            num_batched_tokens,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            n_scheduled_seq_groups,
            n_ignored_seq_groups,
        }
    }

    pub fn is_empty(&self) -> bool {
        // NOTE: ignored sequence groups are not considered
        self.n_scheduled_seq_groups == 0
            && self.blocks_to_swap_in.is_empty()
            && self.blocks_to_swap_out.is_empty()
            && self.blocks_to_copy.is_empty()
    }
}

pub struct Scheduler {}
