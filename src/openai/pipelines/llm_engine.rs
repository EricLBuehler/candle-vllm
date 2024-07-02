use std::{
    collections::{HashMap, VecDeque},
    iter::zip,
    sync::Arc,
};

use crate::openai::streaming::SenderError;
use actix_web::web::Bytes;
use either::Either;
use std::time::Instant;
use tokenizers::Encoding;
use tokio::sync::mpsc::Sender;

use crate::{
    openai::{
        responses::{
            APIError, ChatChoice, ChatChoiceData, ChatCompletionChunk, ChatCompletionUsageResponse,
            Choice, ChoiceData, WrapperLogprobs,
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

use super::{ModulePipeline, _make_tensor_with_pad};
use crate::scheduler::Scheduler;
use candle_core::Tensor;

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
            &pipeline.device(),
        )?;
        let sliding_window = pipeline.get_model_config().sliding_window;
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

    fn get_stream_response(
        &mut self,
        request_id: String,
        created: u64,
        content: Option<String>,
        finish_reason: Option<String>,
    ) -> String {
        let mut choices = Vec::new();
        let choice = Choice {
            delta: ChoiceData {
                role: self.pipeline.get_conversation(true).get_roles().0.clone(),
                content: content,
            },
            finish_reason: finish_reason,
            index: 0,
        };
        choices.push(choice);

        let response = ChatCompletionChunk {
            id: request_id,
            choices: choices,
            created: created,
            model: self.pipeline.name().to_string(),
            object: "chat.completion.chunk",
            system_fingerprint: None,
        };

        format!("data: {}\n\n", serde_json::to_string(&response).unwrap())
    }

    pub async fn generate(
        &mut self,
        prompt: Encoding,
        request_id: String,
        created: u64,
        sampling_params: SamplingParams,
        use_logprobs: bool,
        stream: Option<&Sender<Result<Bytes, SenderError>>>,
    ) -> Result<Vec<(Vec<ChatChoice>, ChatCompletionUsageResponse)>, APIError> {
        self.add_request(prompt, request_id.clone(), created);
        let mut responses = HashMap::new();
        let mut start_time = Instant::now();
        let mut prompt_time_costs = 1;
        let mut completion_time_costs = 1;
        while self.scheduler.has_unfinished_sequences() {
            let scheduler_outputs = self.scheduler.schedule();
            if !scheduler_outputs.ignored_seq_groups.is_empty() {
                todo!();
            }

            try_api!(self.execute_scheduler_ops(&scheduler_outputs));

            let scheduled = &*scheduler_outputs.scheduled;

            let seqs = scheduled
                .iter()
                .flat_map(|group| group.get_seqs())
                .collect::<Vec<_>>();

            let PreparedInputs {
                tokens,
                positions,
                metadata,
            } = if scheduled.front().is_some()
                && scheduled
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
                if scheduled.front().is_some() {
                    self.prepare_decode(scheduled)
                } else {
                    println!("Response should be finished!");
                    unreachable!();
                }
            }?;
            let token_len = tokens.shape().dims()[0];

            let logits = self.pipeline.forward(
                tokens,
                positions,
                Some(&*self.cache_engine.get_kv_cache()),
                metadata,
            )?;
            let result = self.pipeline.sample(logits, &sampling_params, &seqs)?;

            if token_len > 1 {
                let end_time = Instant::now();
                prompt_time_costs = end_time.duration_since(start_time).as_millis();
                start_time = Instant::now();
            }

            for (result_, (_, seq)) in zip(result, seqs) {
                match result_ {
                    Either::Left(logprobs) => {
                        if let Some(sender) = stream {
                            let str_response = self.get_stream_response(
                                request_id.clone(),
                                created,
                                Some(logprobs.bytes.clone()),
                                None,
                            );
                            let _ = sender.send(Ok(Bytes::from(str_response))).await;
                        };
                        print!("{}", logprobs.bytes.clone());
                        seq.deref_mut().add_token(logprobs);
                    }
                    Either::Right(finish_reason) => {
                        if let Some(sender) = stream {
                            let str_response = self.get_stream_response(
                                request_id.clone(),
                                created,
                                None,
                                Some(finish_reason.clone()),
                            );
                            let _ = sender.send(Ok(Bytes::from(str_response))).await;
                        };
                        seq.deref_mut().set_finish_reason(finish_reason)
                    }
                }
            }

            self.scheduler.free_finished_sequence_groups();

            for group in scheduler_outputs.scheduled.iter() {
                if group.is_finished() && !responses.contains_key(group.get_id()) {
                    let end_time = Instant::now();
                    completion_time_costs = end_time.duration_since(start_time).as_millis();
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
                        let data = self
                            .pipeline
                            .tokenizer()
                            .tokenizer()
                            .decode(&data, false)
                            .map_err(APIError::from)?;
                        let choice = ChatChoice {
                            message: ChatChoiceData {
                                role: self.pipeline.get_conversation(true).get_roles().0.clone(),
                                content: Some(data),
                            },
                            finish_reason: Some(seq.deref_mut().get_finish_reason().clone()),
                            index,
                            logprobs: if use_logprobs {
                                Some(WrapperLogprobs { content: outputs })
                            } else {
                                None
                            },
                        };
                        choices.push(choice);
                    }

                    let completion_tokens = top_n
                        .iter()
                        .map(|seq| seq.deref().get_len() - seq.deref().get_prompt_len())
                        .sum();
                    let prompt_tokens = top_n.first().unwrap().deref().get_prompt_len();

                    let usage = ChatCompletionUsageResponse {
                        completion_tokens: completion_tokens,
                        prompt_tokens: prompt_tokens,
                        total_tokens: completion_tokens + prompt_tokens,
                        prompt_time_costs: prompt_time_costs as usize,
                        completion_time_costs: completion_time_costs as usize,
                    };

                    responses.insert(*group.get_id(), (choices, usage));
                }
            }
        }
        if let Some(sender) = stream {
            let str_response = "data: [DONE]\n\n"; //stream finish flag
            let _ = sender.send(Ok(Bytes::from(str_response))).await;
        };
        Ok(responses.into_values().collect::<Vec<_>>())
    }
}

impl<'a> LLMEngine<'a> {
    fn execute_scheduler_ops(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> Result<(), APIError> {
        if scheduler_output.blocks_to_swap_in.len() > 0 {
            try_api!(self
                .cache_engine
                .swap_in(scheduler_output.blocks_to_swap_in.clone()));
        }
        if scheduler_output.blocks_to_swap_out.len() > 0 {
            try_api!(self
                .cache_engine
                .swap_out(scheduler_output.blocks_to_swap_out.clone()));
        }
        if scheduler_output.blocks_to_copy.len() > 0 {
            try_api!(self
                .cache_engine
                .copy(scheduler_output.blocks_to_copy.clone()));
        }
        Ok(())
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
                    if prompt_len > sliding_window {
                        0.min(prompt_len - sliding_window)
                    } else {
                        0
                    }
                } else {
                    0
                };

                let mut slot_mapping = Vec::new();
                for i in 0..prompt_len {
                    if i < start_idx {
                        // Pad [0,start_idx) with _PAD_TOKEN_ID
                        slot_mapping.push(_PAD_SLOT_ID);
                    }

                    let block_number = if i / self.cache_config.block_size >= table.len() {
                        table.get(table.len() - 1).unwrap() //position exceed! use last position
                    } else {
                        table.get(i / self.cache_config.block_size).unwrap()
                    };
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
            &self.pipeline.device(),
        )?;
        let input_positions = _make_tensor_with_pad(
            input_positions
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            *max_prompt_len,
            0,
            &self.pipeline.device(),
        )?;
        let slot_mapping = _make_tensor_with_pad(
            slot_mappings,
            *max_prompt_len,
            _PAD_SLOT_ID,
            &self.pipeline.device(),
        )?;

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
                kv_cache_dtype: "auto".to_string(), // TODO(EricLBuehler): specialize for models
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

                let block_number = if position / self.cache_config.block_size >= table.len() {
                    table.get(table.len() - 1).unwrap() //position exceed! use last position; TODO (bug fix)
                } else {
                    table.get(position / self.cache_config.block_size).unwrap()
                };
                let block_offset = position % self.cache_config.block_size;
                let slot = block_number * self.cache_config.block_size + block_offset;
                let slot = slot.try_into().unwrap();
                slot_mappings.push(vec![slot]);

                if let Some(sliding_window) = self.sliding_window {
                    let sliding_window_blocks = sliding_window / self.cache_config.block_size;
                    let slide_idx = if table.len() > sliding_window_blocks {
                        table.len() - sliding_window_blocks
                    } else {
                        0
                    };
                    block_tables.push(table.get(slide_idx..).unwrap().to_vec());
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
            &self.pipeline.device(),
        )?;
        let input_positions = _make_tensor_with_pad(
            input_positions
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            1,
            0,
            &self.pipeline.device(),
        )?;
        let slot_mapping =
            _make_tensor_with_pad(slot_mappings, 1, _PAD_SLOT_ID, &self.pipeline.device())?;

        let max_context_len = context_lens.iter().max().unwrap();
        let context_lens = try_api!(Tensor::from_vec(
            context_lens.iter().map(|x| *x as u32).collect::<Vec<_>>(),
            (context_lens.len(),),
            &self.pipeline.device(),
        ));

        let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
        let block_tables = _make_tensor_with_pad(
            block_tables
                .iter()
                .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            max_block_table_len,
            0,
            &self.pipeline.device(),
        )?;
        let block_tables = try_api!(block_tables.reshape(((), max_block_table_len)));
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
                kv_cache_dtype: "auto".to_string(), // TODO(EricLBuehler): specialize for models
            },
        })
    }

    fn add_request(&mut self, prompt: Encoding, request_id: String, created: u64) {
        let seq = Arc::new(Sequence(std::sync::RwLock::new(_Sequence::new(
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
