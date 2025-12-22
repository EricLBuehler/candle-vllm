use crate::openai::models::Config;
use crate::openai::pipelines::llm_engine::LLMEngine;
use crate::openai::pipelines::pipeline::DefaultLoader;
use crate::openai::sampling_params::{GenerationConfig, SamplingParams};
use crate::scheduler::cache_engine::{CacheConfig, CacheEngine};
use crate::scheduler::SchedulerConfig;

use candle_core::{DType, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Notify;

use crate::openai::conversation::Conversation;
#[derive(Clone, Debug)]
pub enum ModelRepo {
    /// (model_id, filename) -- when filename is None, treat as safetensor model id.
    /// When filename is Some, treat as GGUF model id + GGUF filename.
    ModelID((&'static str, Option<&'static str>)),
    /// Safetensor local path.
    ModelPath(&'static str),
    /// GGUF file(s). Only the first file is used today.
    ModelFile(Vec<&'static str>),
}

/// Builder for creating an `Engine` instance.
///
/// This builder allows configuring various parameters of the inference engine,
/// such as the model to load, quantization settings, and hardware utilization.
#[derive(Clone, Debug)]
pub struct EngineBuilder {
    repo: ModelRepo,
    isq: Option<String>,
    dtype: Option<DType>,
    flash_attn: Option<bool>,
    fp8_kvcache: Option<bool>,
    device_ids: Option<Vec<usize>>,
    max_num_seqs: usize,
    block_size: usize,
    kvcache_mem_gpu: usize,
    kvcache_mem_cpu: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    top_k: Option<isize>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    prefill_chunk_size: Option<usize>,
}

impl EngineBuilder {
    pub fn new(repo: ModelRepo) -> Self {
        Self {
            repo,
            isq: None,
            dtype: None,
            flash_attn: None,
            fp8_kvcache: None,
            device_ids: None,
            max_num_seqs: 16,
            block_size: 64,
            kvcache_mem_gpu: 4096,
            kvcache_mem_cpu: 128,
            temperature: None,
            top_p: None,
            min_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            prefill_chunk_size: None,
        }
    }

    pub fn with_isq(mut self, isq: impl Into<String>) -> Self {
        self.isq = Some(isq.into());
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    pub fn without_flash_attn(mut self) -> Self {
        self.flash_attn = Some(false);
        self
    }

    pub fn with_fp8_kvcache(mut self) -> Self {
        self.fp8_kvcache = Some(true);
        self
    }

    pub fn with_device_ids(mut self, device_ids: Vec<usize>) -> Self {
        self.device_ids = Some(device_ids);
        self
    }

    pub fn with_max_num_seqs(mut self, max_num_seqs: usize) -> Self {
        self.max_num_seqs = max_num_seqs;
        self
    }

    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    pub fn with_kvcache_mem_gpu(mut self, kvcache_mem_gpu: usize) -> Self {
        self.kvcache_mem_gpu = kvcache_mem_gpu;
        self
    }

    pub fn with_kvcache_mem_cpu(mut self, kvcache_mem_cpu: usize) -> Self {
        self.kvcache_mem_cpu = kvcache_mem_cpu;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = Some(min_p);
        self
    }
    pub fn with_top_k(mut self, top_k: isize) -> Self {
        self.top_k = Some(top_k);
        self
    }
    pub fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }
    pub fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }
    pub fn with_prefill_chunk_size(mut self, prefill_chunk_size: usize) -> Self {
        self.prefill_chunk_size = Some(prefill_chunk_size);
        self
    }

    pub async fn build_async(self) -> Result<Engine> {
        let (model_id, weight_path, weight_file) = match self.repo {
            ModelRepo::ModelID((model_id, filename)) => (
                Some(model_id.to_string()),
                None,
                filename.map(|f| f.to_string()),
            ),
            ModelRepo::ModelPath(path) => (None, Some(path.to_string()), None),
            ModelRepo::ModelFile(files) => {
                let weight_file = files.into_iter().next().map(|f| f.to_string());
                (None, None, weight_file)
            }
        };

        let loader = Box::new(DefaultLoader::new(
            model_id.clone(),
            weight_path.clone(),
            weight_file.clone(),
        ));

        // Use cached token if available, or try to load safely without prompt (api mode should not block on stdin)
        // For now, passing None will use default path or fail if not found/env var not set.
        let (paths, gguf) = loader.prepare_model_weights(None, None)?;

        let dtype = self.dtype.unwrap_or_else(|| crate::get_dtype(None));
        let kv_cache_dtype = if self.fp8_kvcache.unwrap_or(false) {
            DType::U8
        } else {
            dtype
        };

        let device_ids = self.device_ids.unwrap_or(vec![0]);
        // TODO: Handle multi-rank logic. For checking, stick to simpler single rank first or map to loader.

        // Load model
        let (pipelines, _) = loader
            .load_model(
                paths,
                dtype,
                kv_cache_dtype,
                gguf,
                self.isq,
                self.block_size,
                self.max_num_seqs,
                device_ids.clone(),
                #[cfg(feature = "nccl")]
                None, // Todo: Handle nccl
                None,
                None,
                #[cfg(feature = "nccl")]
                None,
                #[cfg(feature = "nccl")]
                None,
            )
            .await?;

        let mut config: Option<Config> = None;
        let mut cache_config: Option<CacheConfig> = None;

        let num_shards = device_ids.len();

        let pipelines: HashMap<
            usize,
            (
                Box<crate::openai::pipelines::pipeline::DefaultPipeline>,
                CacheEngine,
            ),
        > = pipelines
            .into_iter()
            .enumerate()
            .map(|(rank, pipeline)| {
                let cfg = pipeline.get_model_config();
                let cache_cfg = crate::get_cache_config(
                    self.kvcache_mem_gpu,
                    self.kvcache_mem_cpu,
                    self.block_size,
                    &cfg,
                    kv_cache_dtype,
                    num_shards,
                );

                let cache_engine = CacheEngine::new(
                    &cfg,
                    &cache_cfg,
                    cache_cfg.dtype,
                    pipeline.device(),
                    num_shards,
                )
                .unwrap();

                if config.is_none() {
                    config = Some(cfg.clone());
                }
                if cache_config.is_none() {
                    cache_config = Some(cache_cfg.clone());
                }

                (rank, (pipeline, cache_engine))
            })
            .collect();

        let cache_config = cache_config.unwrap();
        let config = config.unwrap();

        let scheduler_config = SchedulerConfig {
            max_num_seqs: self.max_num_seqs,
        };

        let notify = Arc::new(Notify::new());
        // holding_time logic
        let holding_time = 500; // Default

        let engine = LLMEngine::new(
            pipelines,
            scheduler_config,
            &cache_config,
            &config,
            notify.clone(),
            holding_time,
            num_shards,
            false, // multi_process: Assume false for simple API usage for now
            #[cfg(feature = "nccl")]
            None,
            self.prefill_chunk_size,
        )?;

        // Return the Engine wrapper
        Ok(Engine {
            engine,
            notify,
            pipeline_config: crate::openai::PipelineConfig {
                max_model_len: config.max_seq_len,
                default_max_tokens: config.max_seq_len / 5, // Approximate default
                generation_cfg: Some(GenerationConfig {
                    temperature: self.temperature,
                    top_k: self.top_k,
                    top_p: self.top_p,
                    min_p: self.min_p,
                    frequency_penalty: self.frequency_penalty,
                    presence_penalty: self.presence_penalty,
                }),
            },
            _runtime: None,
        })
    }

    pub fn build(self) -> Result<Engine> {
        let rt = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap(),
        );

        let mut engine = rt.block_on(self.build_async())?;
        engine._runtime = Some(rt);
        Ok(engine)
    }
}

/// The main inference engine context.
///
/// This struct holds the initialized `LLMEngine` and allows performing
/// text generation tasks. It is thread-safe and can be shared across threads.
#[derive(Clone)]
#[allow(dead_code)]
pub struct Engine {
    engine: Arc<RwLock<LLMEngine>>,
    notify: Arc<Notify>,
    pipeline_config: crate::openai::PipelineConfig,
    _runtime: Option<Arc<tokio::runtime::Runtime>>,
}

use crate::openai::requests::ChatCompletionRequest;
use crate::openai::responses::ChatCompletionResponse;

impl Engine {
    pub fn generate(&self, messages: Vec<ChatCompletionRequest>) -> Result<ChatCompletionResponse> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(self.generate_async(messages))
    }

    pub async fn generate_async(
        &self,
        messages: Vec<ChatCompletionRequest>,
    ) -> Result<ChatCompletionResponse> {
        self.generate_request(messages.into_iter().next().unwrap())
            .await
    }

    pub async fn generate_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let (prompt, tokenizer) = {
            let e = self.engine.read();
            let (pipeline, _) = e.get_pipeline(0).unwrap();
            // tokenizer is inside DefaultPipeline
            let conversation = pipeline.conversation.clone();

            // Logic to get prompt from messages
            // We need to access `messages` from request.
            let prompt = match &request.messages {
                crate::openai::requests::Messages::Literal(msg) => msg.clone(),
                crate::openai::requests::Messages::Map(messages) => {
                    let mut conv = conversation.clone();
                    for message in messages {
                        if let (Some(role), Some(content)) =
                            (message.get("role"), message.get("content"))
                        {
                            if role == "system" {
                                conv.set_system_message(Some(content.clone()));
                            }
                            conv.append_message(role.to_string(), content.clone());
                        }
                    }
                    conv.get_prompt(request.thinking.unwrap_or(false))
                }
            };

            (prompt, pipeline.tokenizer.clone())
        };

        let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());

        let token_ids = tokenizer
            .encode(prompt, false)
            .map_err(candle_core::Error::msg)?
            .get_ids()
            .to_vec();

        // Let's create a local notify for this request.
        let req_notify = std::sync::Arc::new(tokio::sync::Notify::new());

        // Re-add request with req_notify
        {
            let mut e = self.engine.write();
            e.add_request(
                token_ids,
                request_id.clone(),
                std::time::SystemTime::now(),
                 SamplingParams::new(
                    request.n.unwrap_or(1),
                    request.best_of,
                    request.presence_penalty.unwrap_or(0.0),
                    request.frequency_penalty.unwrap_or(0.0),
                    request.repeat_last_n,
                    request.temperature,
                    request.top_p,
                    request.min_p,
                    request.top_k,
                    request.use_beam_search.unwrap_or(false),
                    1.0,
                    crate::openai::sampling_params::EarlyStoppingCondition::UnlikelyBetterCandidates,
                    request.stop.clone(),
                    request.stop_token_ids.clone().unwrap_or_default(),
                    request.ignore_eos.unwrap_or(false),
                    request.max_tokens.unwrap_or(16),
                    None,
                    None,
                    request.skip_special_tokens.unwrap_or(true),
                    request.thinking,
                ).map_err(candle_core::Error::msg)?,
                request.logprobs.unwrap_or(false),
                None, // streamer
                Some(req_notify.clone()), // Use the local notify
            );
            self.notify.notify_one();
            // ...
        }

        // The loop below handles the wait.
        loop {
            let e = self.engine.read();
            if e.completion_records.contains_key(&request_id) {
                break;
            }
            drop(e);
            req_notify.notified().await;
        }

        let e = self.engine.read();
        if let Some(record) = e.completion_records.get(&request_id) {
            Ok(ChatCompletionResponse {
                id: request_id,
                choices: record.0.clone(),
                created: record.1.created,
                model: request.model.clone(),
                object: "chat.completion",
                usage: record.1.clone(),
            })
        } else {
            Err(candle_core::Error::msg("Failed to get response"))
        }
    }

    pub fn shutdown(&self) {
        let e = self.engine.read();
        e.exit_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.notify.notify_waiters();
    }
}
