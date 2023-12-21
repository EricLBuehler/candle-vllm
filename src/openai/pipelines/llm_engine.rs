use std::{
    collections::{HashMap, VecDeque},
    iter::zip,
    sync::{Arc, Mutex},
};

use either::Either;
use tokenizers::Encoding;

use crate::{
    openai::{
        responses::{
            APIError, ChatChoice, ChatChoiceData, ChatCompletionUsageResponse, WrapperLogprobs,
        },
        sampling_params::SamplingParams,
        utils::get_created_time_secs,
    },
    paged_attention::input_metadata::InputMetadata,
    scheduler::{
        cache_engine::{CacheConfig, CacheEngine},
        sequence::{Sequence, SequenceGroup, _Sequence},
        SchedulerConfig, SchedulerOutput,
    },
    try_api,
};

use crate::scheduler::Scheduler;

use super::{ModulePipeline, _make_tensor_with_pad};

use candle_core::{Device, Tensor};

#[allow(dead_code)]
struct PreparedInputs {
    tokens: Tensor,
    positions: Tensor,
    metadata: InputMetadata,
}

const _PAD_SLOT_ID: i64 = -1;

pub struct LLMEngine<'a> {
    pipeline: Box<dyn ModulePipeline<'a>>,
    scheduler: Scheduler,
    seq_id: usize,
    cache_config: CacheConfig,
    group_id: usize,
    cache_engine: CacheEngine,
    sliding_window: Option<usize>,
}

impl<'a> LLMEngine<'a> {
    pub fn new(
        pipeline: Box<dyn ModulePipeline<'a>>,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> Result<Self, APIError> {
        let cache_engine = CacheEngine::new(
            pipeline.get_model_config(),
            cache_config.clone(),
            pipeline.get_dtype(),
        )?;
        let sliding_window = pipeline.get_model_config().get_sliding_window();
        Ok(Self {
            pipeline,
            scheduler: Scheduler::new(scheduler_config, &cache_config),
            seq_id: 0,
            cache_config,
            group_id: 0,
            cache_engine,
            sliding_window,
        })
    }

    pub fn get_pipeline(&self) -> &dyn ModulePipeline<'a> {
        &*self.pipeline
    }

    pub fn get_mut_pipeline(&mut self) -> &mut dyn ModulePipeline<'a> {
        &mut *self.pipeline
    }

    pub fn generate(
        &mut self,
        prompt: Encoding,
        request_id: String,
        created: u64,
        sampling_params: SamplingParams,
    ) -> Result<Vec<(Vec<ChatChoice>, ChatCompletionUsageResponse)>, APIError> {
        self.add_request(prompt, request_id, created);

        let mut responses = HashMap::new();
        while self.scheduler.has_unfinished_sequences() {
            let scheduler_outputs = self.scheduler.schedule();
            if !scheduler_outputs.ignored_seq_groups.is_empty() {
                todo!();
            }

            self.execute_scheduler_ops(&scheduler_outputs);

            let scheduled = &*scheduler_outputs.scheduled;

            let seqs = scheduled
                .iter()
                .flat_map(|group| group.get_seqs())
                .collect::<Vec<_>>();

            let PreparedInputs {
                tokens,
                positions,
                metadata,
            } = if scheduled
                .front()
                .unwrap()
                .get_seqs()
                .values()
                .nth(0)
                .unwrap()
                .deref_mut()
                .is_prompt()
            {
                self.prepare_prompt(scheduled)
            } else {
                // Because of the KV cache, we only need to take
                // the last token.
                self.prepare_decode(scheduled)
            }?;

            let logits = self.pipeline.forward(
                tokens,
                positions,
                Some(self.cache_engine.get_kv_cache()),
                metadata,
            )?;
            let result = self.pipeline.sample(logits, &sampling_params, &seqs)?;

            for (result, (_, seq)) in zip(result, seqs) {
                match result {
                    Either::Left(logprobs) => {
                        seq.deref_mut().add_token(logprobs);
                    }
                    Either::Right(finish_reason) => {
                        seq.deref_mut().set_finish_reason(finish_reason)
                    }
                }
            }

            self.scheduler.free_finished_sequence_groups();

            for group in scheduler_outputs.scheduled.iter() {
                if group.is_finished() && !responses.contains_key(group.get_id()) {
                    // Create choices from the group
                    let mut seqs = group.get_seqs().values().collect::<Vec<_>>();
                    seqs.sort_by(|seq_a, seq_b| {
                        seq_b
                            .deref_mut()
                            .get_cumulative_logprob()
                            .partial_cmp(&seq_a.deref_mut().get_cumulative_logprob())
                            .unwrap()
                    });
                    let top_n = seqs.get(0..sampling_params.n).unwrap();

                    let mut choices = Vec::new();
                    for (index, seq) in top_n.iter().enumerate() {
                        let outputs = seq.deref_mut().get_output_tokens();
                        let data = outputs
                            .iter()
                            .map(|x| x.token.try_into().unwrap())
                            .collect::<Vec<_>>();
                        let data = self.pipeline.tokenizer().detokenize(&data)?;
                        let choice = ChatChoice {
                            message: ChatChoiceData {
                                role: self.pipeline.get_conversation().get_roles().0.clone(),
                                content: Some(data),
                            },
                            finish_reason: Some(seq.deref_mut().get_finish_reason().clone()),
                            index,
                            logprobs: Some(WrapperLogprobs { content: outputs }),
                        };
                        choices.push(choice);
                    }

                    let usage = ChatCompletionUsageResponse {
                        completion_tokens: top_n
                            .iter()
                            .map(|seq| seq.deref_mut().get_len() - seq.deref_mut().get_prompt_len())
                            .sum(),
                        prompt_tokens: top_n.get(0).unwrap().deref_mut().get_prompt_len(),
                        total_tokens: top_n
                            .iter()
                            .map(|seq| seq.deref_mut().get_len() - seq.deref_mut().get_prompt_len())
                            .sum::<usize>()
                            + top_n.get(0).unwrap().deref_mut().get_prompt_len(),
                    };

                    responses.insert(*group.get_id(), (choices, usage));
                }
            }
        }

        Ok(responses.into_values().collect::<Vec<_>>())
    }
}

impl<'a> LLMEngine<'a> {
    fn execute_scheduler_ops(&self, scheduler_output: &SchedulerOutput) {
        self.cache_engine
            .swap_in(scheduler_output.blocks_to_swap_in.clone());
        self.cache_engine
            .swap_out(scheduler_output.blocks_to_swap_out.clone());
        self.cache_engine
            .copy(scheduler_output.blocks_to_copy.clone());
    }

    fn prepare_prompt(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
    ) -> Result<PreparedInputs, APIError> {
        let mut prompt_lens = Vec::new();
        let mut input_tokens = Vec::new();
        let mut input_positions = Vec::new();
        let mut slot_mappings = Vec::new();
        for group in groups {
            for seq in group.get_seqs().values() {
                let prompt_ids = seq.deref_mut().get_token_ids();

                let prompt_len = prompt_ids.len();
                prompt_lens.push(prompt_len);

                input_tokens.push(prompt_ids);
                input_positions.push((0..prompt_len).collect::<Vec<_>>());
                let table = self
                    .scheduler
                    .block_engine
                    .block_tables
                    .get(&seq.deref_mut().get_id());
                if table.is_none() {
                    // Will be None during profiling.
                    slot_mappings.push([_PAD_SLOT_ID].repeat(prompt_len));
                    continue;
                }
                let table = table
                    .unwrap()
                    .iter()
                    .map(|block| block.deref_mut().block_id)
                    .collect::<Vec<_>>();

                let start_idx = if let Some(sliding_window) = self.sliding_window {
                    0.min(prompt_len - sliding_window)
                } else {
                    0
                };

                let mut slot_mapping = Vec::new();
                for i in 0..prompt_len {
                    if i < start_idx {
                        // Pad [0,start_idx) with _PAD_TOKEN_ID
                        slot_mapping.push(_PAD_SLOT_ID);
                    }

                    let block_number = table.get(i / self.cache_config.block_size).unwrap();
                    let block_offset = i % self.cache_config.block_size;
                    let slot = block_number * self.cache_config.block_size + block_offset;
                    slot_mapping.push(slot.try_into().unwrap());
                }
                slot_mappings.push(slot_mapping);
            }
        }

        let max_prompt_len = prompt_lens.iter().max().unwrap();
        let input_tokens = _make_tensor_with_pad(
            input_tokens
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            *max_prompt_len,
            0,
        )?;
        let input_positions = _make_tensor_with_pad(
            input_positions
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            *max_prompt_len,
            0,
        )?;
        let slot_mapping = _make_tensor_with_pad(slot_mappings, *max_prompt_len, _PAD_SLOT_ID)?;

        Ok(PreparedInputs {
            tokens: input_tokens,
            positions: input_positions,
            metadata: InputMetadata {
                prompt_lens,
                slot_mapping,
                max_context_len: None,
                context_lens: None,
                block_tables: None,
                attn_bias: None,
                is_prompt: true,
            },
        })
    }

    fn prepare_decode(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
    ) -> Result<PreparedInputs, APIError> {
        let mut input_tokens = Vec::new();
        let mut input_positions = Vec::new();
        let mut context_lens = Vec::new();
        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        for group in groups {
            for seq in group.get_seqs().values() {
                let last_token_id = seq.deref_mut().get_last_token_id();
                input_tokens.push(vec![last_token_id]);

                let position = seq.deref_mut().get_len() - 1;
                input_positions.push(vec![position]);

                let context_len = if let Some(sliding_window) = self.sliding_window {
                    seq.deref_mut().get_len().min(sliding_window)
                } else {
                    seq.deref_mut().get_len()
                };
                context_lens.push(context_len);

                let table = self
                    .scheduler
                    .block_engine
                    .block_tables
                    .get(&seq.deref_mut().get_id())
                    .unwrap();
                let table = table
                    .iter()
                    .map(|block| block.deref_mut().block_id)
                    .collect::<Vec<_>>();

                let block_number = table.get(position / self.cache_config.block_size).unwrap();
                let block_offset = position % self.cache_config.block_size;
                let slot = block_number * self.cache_config.block_size + block_offset;
                let slot = slot.try_into().unwrap();
                slot_mappings.push(vec![slot]);

                if let Some(sliding_window) = self.sliding_window {
                    let sliding_window_blocks = sliding_window / self.cache_config.block_size;
                    block_tables.push(
                        table
                            .get(table.len() - sliding_window_blocks..)
                            .unwrap()
                            .to_vec(),
                    );
                } else {
                    block_tables.push(table);
                }
            }
        }

        let input_tokens = _make_tensor_with_pad(
            input_tokens
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            1,
            0,
        )?;
        let input_positions = _make_tensor_with_pad(
            input_positions
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            1,
            0,
        )?;
        let slot_mapping = _make_tensor_with_pad(slot_mappings, 1, _PAD_SLOT_ID)?;

        let max_context_len = context_lens.iter().max().unwrap();
        let context_lens = try_api!(Tensor::from_vec(
            context_lens.iter().map(|x| *x as i64).collect::<Vec<_>>(),
            (context_lens.len(),),
            &try_api!(Device::new_cuda(0)),
        ));

        let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
        let block_tables = _make_tensor_with_pad(
            block_tables
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            max_block_table_len,
            0,
        )?;

        Ok(PreparedInputs {
            tokens: input_tokens,
            positions: input_positions,
            metadata: InputMetadata {
                prompt_lens: vec![],
                slot_mapping,
                max_context_len: Some(*max_context_len),
                context_lens: Some(context_lens),
                block_tables: Some(block_tables),
                attn_bias: None,
                is_prompt: false,
            },
        })
    }

    fn add_request(&mut self, prompt: Encoding, request_id: String, created: u64) {
        let seq = Arc::new(Sequence(Mutex::new(_Sequence::new(
            prompt
                .get_ids()
                .to_vec()
                .iter()
                .map(|x| *x as usize)
                .collect::<Vec<_>>(),
            self.seq_id,
            self.cache_config.block_size,
        ))));
        self.seq_id += 1;
        let seq_group = SequenceGroup::new(
            &[seq],
            get_created_time_secs(),
            self.group_id,
            request_id,
            created,
        );
        self.group_id += 1;

        self.scheduler.add_sequence(seq_group);
    }
}
