use std::collections::{HashMap, VecDeque};

use crate::openai::responses::APIError;

use super::{block_manager::BlockSpaceManager, cache_engine::CacheConfig, sequence::SequenceGroup};

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

pub struct Scheduler {
    waiting: VecDeque<SequenceGroup>,
    running: VecDeque<SequenceGroup>,
    swapped: VecDeque<SequenceGroup>,
    prompt_limit: usize,
    block_manager: BlockSpaceManager,
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
        })
    }
}
