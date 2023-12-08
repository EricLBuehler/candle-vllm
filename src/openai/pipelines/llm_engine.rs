use std::{collections::HashMap, iter::zip, sync::Arc};

use actix_web::body::None;
use candle_core::{Device, Tensor};
use futures::future::Either;

use crate::{
    openai::{
        responses::APIError,
        sampling_params::{self, SamplingParams},
        utils::get_created_time_secs,
    },
    paged_attention::{
        cache_engine::{CacheConfig, CacheEngine, ModelConfig, ParallelConfig},
        scheduler::{Scheduler, SchedulerConfig, SchedulerOutputs},
        sequence::{Sequence, SequenceGroup, SequenceGroupMetadata, SequenceGroupOutput},
    },
};

use super::{outputs::RequestOutput, ModulePipeline};

pub struct LlmEngine<'a> {
    pipeline: Box<dyn ModulePipeline<'a>>,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    parallel_config: ParallelConfig,
    cache_engine: Option<CacheEngine>,
    scheduler: Scheduler,
    seq_counter: usize,
    request_counter: usize,
}

pub struct ProfiledBlocks {
    num_gpu_blocks: usize,
    num_cpu_blocks: usize,
}

impl<'a> LlmEngine<'a> {
    pub fn new(
        pipeline: Box<dyn ModulePipeline<'a>>,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, APIError> {
        let mut this = Self {
            pipeline,
            model_config,
            cache_config,
            parallel_config,
            cache_engine: None,
            scheduler: Scheduler::new(scheduler_config, cache_config)?,
            seq_counter: 0,
            request_counter: 0,
        };
        this._init_cache();
        Ok(this)
    }

    // The following 2 are functions with todo!() internals
    // TODO_URGENT(EricLBuehler)
    fn get_peak_memory() -> usize {
        // Pending on https://github.com/huggingface/candle/pull/1412
        todo!()
    }
    fn total_gpu_memory() -> usize {
        todo!()
    }

    fn profile_num_available_blocks(
        &mut self,
        block_size: usize,
        gpu_memory_utilization: f64,
        cpu_swap_space: usize,
    ) -> ProfiledBlocks {
        // Pending on https://github.com/huggingface/candle/pull/1412
        // reset_peak_memory_stats

        self.pipeline.profile_run();

        let peak_memory: usize = Self::get_peak_memory();

        let total_gpu_memory = Self::total_gpu_memory();
        let cache_block_size = CacheEngine::get_cache_block_size(
            block_size,
            &self.model_config,
            &self.parallel_config,
        );
        let num_gpu_blocks = (total_gpu_memory as f64 * gpu_memory_utilization - peak_memory as f64)
            as usize
            / cache_block_size;
        let num_cpu_blocks = cpu_swap_space / cache_block_size;
        let num_gpu_blocks = num_gpu_blocks.max(0);
        let num_cpu_blocks = num_cpu_blocks.max(0);

        ProfiledBlocks {
            num_gpu_blocks,
            num_cpu_blocks,
        }
    }

    pub fn _init_cache(&mut self) -> Result<(), APIError> {
        let ProfiledBlocks {
            num_gpu_blocks,
            num_cpu_blocks,
        } = self.profile_num_available_blocks(
            self.cache_config.block_size,
            self.cache_config.gpu_mem_utilization,
            self.cache_config.swap_space_bytes,
        );
        eprintln!("{num_gpu_blocks} GPU blocks.");
        eprintln!("{num_cpu_blocks} CPU blocks.");

        if num_gpu_blocks <= 0 {
            return Err(APIError::new_str("No available memory for the cache blocks. Try increasing `gpu_mem_utilization` when initializing the engine."));
        }

        self.cache_config.num_cpu_blocks = Some(num_cpu_blocks);
        self.cache_config.num_gpu_blocks = Some(num_gpu_blocks);

        todo!("init_cache_engine");

        Ok(())
    }

    fn _schedule(&self) -> (Vec<SequenceGroupMetadata>, SchedulerOutputs, ()) {
        let (seq_group_metadata_list, scheduler_outputs) = self.scheduler.schedule();
        (seq_group_metadata_list, scheduler_outputs, ())
    }

    fn _process_model_group_outputs(
        &self,
        seq_group: &Arc<SequenceGroup>,
        outputs: &SequenceGroupOutput,
    ) {
        // https://github.com/vllm-project/vllm/blob/60dc62dc9e53428912953276e0d12a034b353fb6/vllm/engine/llm_engine.py#L368
        todo!()
    }

    fn _process_model_outputs(
        &self,
        output: &Vec<SequenceGroupOutput>,
        scheduler_outputs: SchedulerOutputs,
    ) -> Vec<RequestOutput> {
        let sched_seq_groups = &scheduler_outputs.scheduled_seq_groups;
        for (seq_group, outputs) in zip(sched_seq_groups, output) {
            self._process_model_group_outputs(seq_group, outputs)
        }
        todo!()
    }

    pub fn generate(
        &mut self,
        prompts: Either<String, Vec<String>>,
        sampling_params: Option<SamplingParams>,
        prompt_token_ids: Option<Vec<Vec<usize>>>,
    ) -> Result<Vec<RequestOutput>, APIError> {
        let prompts = match prompts {
            Either::Left(single) => vec![single],
            Either::Right(multi) => multi,
        };
        let sampling_params = if let Some(params) = sampling_params {
            params
        } else {
            SamplingParams::new(
                1,
                Some(1),
                0.,
                0.,
                1.,
                1.,
                1.,
                -1,
                false,
                1.,
                sampling_params::EarlyStoppingCondition::UnlikelyBetterCandidates,
                None,
                Vec::new(),
                false,
                16,
                None,
                None,
                true,
            )?
        };
        let num_requests = prompts.len();
        for i in 0..num_requests {
            let prompt = prompts.get(i).unwrap();
            let token_ids = if let Some(prompt_token_ids) = &prompt_token_ids {
                prompt_token_ids.get(i)
            } else {
                None
            };
            self.add_request(
                self.request_counter.to_string(),
                prompt.clone(),
                token_ids.cloned(),
                sampling_params,
            );
            self.request_counter += 1;
        }
        Ok(self._run_engine())
    }

    fn add_request(
        &mut self,
        request_id: String,
        prompt: String,
        prompt_token_ids: Option<Vec<usize>>,
        sampling_params: SamplingParams,
    ) -> Result<(), APIError> {
        let prompt_token_ids = if let Some(prompt_token_ids) = prompt_token_ids {
            prompt_token_ids
        } else {
            self.pipeline
                .tokenizer()
                .tokenize(prompt)?
                .get_ids()
                .iter()
                .map(|x| *x as usize)
                .collect::<Vec<_>>()
        };

        let block_size = self.cache_config.block_size;

        let seq_id = self.seq_counter;
        self.seq_counter += 1;

        let seq = Sequence::new(seq_id, prompt, prompt_token_ids, block_size);
        let seq_group = SequenceGroup::new(
            request_id,
            vec![seq],
            sampling_params,
            get_created_time_secs(),
        );
        self.scheduler.add_seq_group(seq_group);
        Ok(())
    }

    fn _run_engine(&mut self) -> Vec<RequestOutput> {
        let mut outputs = Vec::new();
        while self.scheduler.has_unfinished_request() {
            let step_outputs = self.step();
            for output in step_outputs {
                if output.finished {
                    outputs.push(output);
                }
            }
        }
        outputs.sort_by_key(|x| x.request_id.parse::<usize>().unwrap());
        outputs
    }

    // Calls execute_model (schedules seqs), called by .generate
    fn step(&mut self) -> Vec<RequestOutput> {
        let (seq_group_metadata_list, scheduler_outputs, ignored) = self._schedule();

        let output = self.execute_model(
            seq_group_metadata_list,
            scheduler_outputs.blocks_to_swap_in,
            scheduler_outputs.blocks_to_swap_out,
            scheduler_outputs.blocks_to_copy,
        );

        self._process_model_outputs(&output, scheduler_outputs)
    }

    // Calls the module pipeline model executer, called by .step
    fn execute_model(
        &mut self,
        seq_group_metadatas: Vec<SequenceGroupMetadata>,
        blocks_to_swap_in: HashMap<usize, usize>,
        blocks_to_swap_out: HashMap<usize, usize>,
        blocks_to_copy: HashMap<usize, Vec<usize>>,
    ) -> Vec<SequenceGroupOutput> {
        if !blocks_to_swap_in.is_empty() {
            self.cache_engine.unwrap().swap_in(blocks_to_swap_in);
        }
        if !blocks_to_swap_out.is_empty() {
            self.cache_engine.unwrap().swap_out(blocks_to_swap_out);
        }
        if !blocks_to_copy.is_empty() {
            self.cache_engine.unwrap().copy(blocks_to_copy);
        }

        //https://github.com/vllm-project/vllm/blob/05ff90b692a6cdac4d8c06e7a4a4606d1b8fe1d6/vllm/worker/worker.py#L119

        self.pipeline.execute_model(
            seq_group_metadatas,
            Some(self.cache_engine.unwrap().gpu_cache.clone()),
        );

        todo!();
    }
}
