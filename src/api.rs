use crate::openai::models::{Config, KvCacheDtype};
use crate::openai::multimodal::build_messages_and_images;
use crate::openai::pipelines::llm_engine::LLMEngine;
use crate::openai::pipelines::pipeline::DefaultLoader;
use crate::openai::requests::Messages;
use crate::openai::resolve_tools_for_request;
use crate::openai::sampling_params::{GenerationConfig, SamplingParams};
use crate::openai::PipelineConfig;
use crate::scheduler::cache_engine::{CacheConfig, CacheEngine};
use crate::scheduler::prefix_cache::PrefixCacheConfig;
use crate::scheduler::SchedulerConfig;
use crate::tools::helpers::{
    build_invalid_tool_call_feedback, build_tool_schema_map, filter_tool_calls,
};
use candle_core::{DType, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Notify;

const REQUEST_ADMISSION_DECODE_BUDGET_TOKENS: usize = 4096;

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
    kvcache_dtype: Option<KvCacheDtype>,
    device_ids: Option<Vec<usize>>,
    max_num_seqs: usize,
    block_size: usize,
    kvcache_mem_gpu: usize,
    kv_fraction: Option<f32>,
    mamba_fraction: Option<f32>,
    kvcache_mem_cpu: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    top_k: Option<isize>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    prefill_chunk_size: Option<usize>,
    yarn_scaling_factor: Option<f64>,
}

impl EngineBuilder {
    pub fn new(repo: ModelRepo) -> Self {
        Self {
            repo,
            isq: None,
            dtype: None,
            flash_attn: None,
            kvcache_dtype: None,
            device_ids: None,
            max_num_seqs: 8,
            block_size: if cfg!(feature = "cuda") { 64 } else { 32 },
            kvcache_mem_gpu: 4096,
            kv_fraction: Some(0.6),
            mamba_fraction: None,
            kvcache_mem_cpu: 128,
            temperature: None,
            top_p: None,
            min_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            prefill_chunk_size: None,
            yarn_scaling_factor: None,
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
        self.kvcache_dtype = Some(KvCacheDtype::Fp8);
        self
    }

    pub fn with_kvcache_dtype(mut self, kvcache_dtype: KvCacheDtype) -> Self {
        self.kvcache_dtype = Some(kvcache_dtype);
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

    pub fn with_kv_fraction(mut self, kv_fraction: f32) -> Self {
        self.kv_fraction = Some(kv_fraction);
        self
    }

    pub fn with_mamba_fraction(mut self, mamba_fraction: f32) -> Self {
        self.mamba_fraction = Some(mamba_fraction);
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

    pub fn with_yarn_scaling_factor(mut self, factor: f64) -> Self {
        self.yarn_scaling_factor = Some(factor);
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
            None,
            self.yarn_scaling_factor,
        ));

        // Use cached token if available, or try to load safely without prompt (api mode should not block on stdin)
        // For now, passing None will use default path or fail if not found/env var not set.
        let (paths, gguf) = loader.prepare_model_weights(None, None)?;

        let dtype = self.dtype.unwrap_or_else(|| crate::get_dtype(None));
        let kvcache_dtype_enum = self.kvcache_dtype.unwrap_or(KvCacheDtype::Auto);
        let kv_cache_dtype = if kvcache_dtype_enum.is_fp8_keys() {
            DType::U8
        } else {
            dtype
        };
        KvCacheDtype::set_global(kvcache_dtype_enum);

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
        let first_pipeline = pipelines
            .first()
            .expect("at least one pipeline must be loaded");
        let first_config = first_pipeline.get_model_config();
        let first_model_dtype = first_pipeline.dtype;
        let devices: Vec<_> = pipelines.iter().map(|pipeline| pipeline.device()).collect();
        let (kvcache_mem_gpu, mamba_cache_budget_bytes) = match self.kv_fraction {
            Some(kv_fraction) => {
                let workspace_params = crate::WorkspaceBudgetParams::from_config(
                    &first_config,
                    first_model_dtype,
                    num_shards,
                    8192,
                );
                let workspace_budget = crate::compute_workspace_budget(&workspace_params);
                let detected = crate::detect_kvcache_mem_gpu_mb_for_devices_with_workspace(
                    &devices,
                    kv_fraction,
                    Some(&workspace_budget),
                )?;
                let mut effective_kvcache_mem_gpu = detected;
                let mut mamba_cache_budget_bytes = 0usize;
                if let Some(estimate) =
                    crate::estimate_hybrid_mamba_cache(&first_config, first_model_dtype, num_shards)
                {
                    if let Some(plan) = crate::plan_hybrid_mamba_cache_with_fraction(
                        detected * 1024 * 1024,
                        estimate,
                        self.max_num_seqs,
                        false,
                        self.mamba_fraction,
                    ) {
                        let reserved_mamba_mb = plan.budget_bytes.div_ceil(1024 * 1024);
                        if reserved_mamba_mb < detected {
                            effective_kvcache_mem_gpu = detected - reserved_mamba_mb;
                            mamba_cache_budget_bytes = plan.budget_bytes;
                        }
                    }
                }
                (effective_kvcache_mem_gpu, mamba_cache_budget_bytes)
            }
            None => (self.kvcache_mem_gpu, 0),
        };

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
                let kvcache_dtype_enum = self
                    .kvcache_dtype
                    .unwrap_or(crate::openai::models::KvCacheDtype::Auto);
                let mut cache_cfg = crate::get_cache_config(
                    kvcache_mem_gpu,
                    self.kvcache_mem_cpu,
                    self.block_size,
                    &cfg,
                    kv_cache_dtype,
                    num_shards,
                    kvcache_dtype_enum,
                );
                cache_cfg.mamba_cache_budget_bytes = mamba_cache_budget_bytes;

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
            prefix_cache: PrefixCacheConfig::default(),
            mamba_cache_capacity: None,
        };

        let notify = Arc::new(Notify::new());
        // holding_time logic
        let holding_time = 100; // Default

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
            false,
        )?;

        let mut pipeline_config = PipelineConfig {
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
        };
        pipeline_config.apply_kv_cache_limit(&cache_config);

        // Return the Engine wrapper
        Ok(Engine {
            engine,
            notify,
            pipeline_config,
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

use crate::openai::requests::{ChatCompletionRequest, EmbeddingRequest};
use crate::openai::responses::{ChatCompletionResponse, EmbeddingResponse};
use crate::openai::streaming::ChatResponse;
use tracing::{info, warn};

impl Engine {
    /// Validates prompt length against model limits.
    fn validate_prompt(&self, token_ids: &[u32], request_type: &str) -> Result<()> {
        let prompt_len = token_ids.len();
        let max_model_len = self.pipeline_config.max_model_len;

        if prompt_len > max_model_len {
            warn!(
                "[{}] Prompt length {} exceeds maximum model length {}",
                request_type, prompt_len, max_model_len
            );
            return Err(candle_core::Error::msg(format!(
                "Prompt length {} exceeds maximum model length {}",
                prompt_len, max_model_len
            )));
        }

        info!(
            "[{}] Validated prompt with {} tokens (max: {})",
            request_type, prompt_len, max_model_len
        );
        Ok(())
    }

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
        mut request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        if let Messages::Chat(messages) = &mut request.messages {
            crate::openai::requests::normalize_empty_openai_tool_results(messages);
            crate::openai::requests::validate_openai_tool_messages(messages)
                .map_err(candle_core::Error::wrap)?;
        }

        let (prompt, tokenizer, image_data, resolved_tools) = {
            let e = self.engine.read();

            let tool_config = resolve_tools_for_request(&request.tools, &request.tool_choice, None)
                .map_err(candle_core::Error::wrap)?;
            let resolved_tools = tool_config.tools.clone();

            let tokenizer = e.tokenizer().clone();
            let image_config = e.image_config();
            let mut conversation = e.conversation();
            let mut image_data = None;

            // Logic to get prompt from messages
            // We need to access `messages` from request.
            match &request.messages {
                Messages::Literal(msg) => {
                    conversation.append_message("user".to_string(), msg.clone());
                }
                Messages::Chat(messages) => {
                    let (render_messages, images) =
                        build_messages_and_images(messages, image_config.as_ref())
                            .map_err(candle_core::Error::wrap)?;
                    image_data = images;
                    for message in render_messages {
                        if message.role == "system" {
                            conversation.set_system_message(Some(message.content.clone()));
                        } else {
                            conversation.append_template_message(message);
                        }
                    }
                }
                Messages::Map(messages) => {
                    for message in messages {
                        if let (Some(role), Some(content)) =
                            (message.get("role"), message.get("content"))
                        {
                            if role == "system" {
                                conversation.set_system_message(Some(content.clone()));
                            } else {
                                use crate::openai::conversation::Message;
                                conversation.append_template_message(Message {
                                    role: role.to_string(),
                                    content: content.clone(),
                                    num_images: 0,
                                    reasoning_content: None,
                                    tool_calls: None,
                                    tool_call_id: None,
                                });
                            }
                        }
                    }
                }
            };

            let enable_thinking = request.thinking.unwrap_or(true);
            let prompt = conversation.get_prompt(enable_thinking, &tool_config.tools);

            (prompt, tokenizer, image_data, resolved_tools)
        };

        let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());

        let token_ids = tokenizer
            .encode(prompt.clone(), false)
            .map_err(candle_core::Error::msg)?
            .get_ids()
            .to_vec();

        info!(
            "[generate_request] Processing request with prompt length {}",
            token_ids.len()
        );

        // Validate prompt length
        self.validate_prompt(&token_ids, "generate_request")?;

        let mut max_request_tokens = request.max_tokens.unwrap_or(16);
        let max_model_decode_tokens = self
            .pipeline_config
            .max_model_len
            .saturating_sub(token_ids.len())
            .saturating_sub(10);
        if max_request_tokens > max_model_decode_tokens {
            warn!(
                "[generate_request] Requested max_tokens {} exceeds remaining model context {}, max_tokens changed to {}",
                max_request_tokens,
                max_model_decode_tokens,
                max_model_decode_tokens
            );
            max_request_tokens = max_model_decode_tokens;
        }
        if max_request_tokens == 0 {
            return Err(candle_core::Error::msg(format!(
                "Requested prompt({} tokens) leaves no room for generated tokens within maximum model context {}",
                token_ids.len(),
                self.pipeline_config.max_model_len
            )));
        }

        let mut cached_tokens = {
            let mut e = self.engine.write();
            e.query_prefix_cache_match_tokens(&token_ids)
        };
        let mut new_tokens = token_ids.len().saturating_sub(cached_tokens);
        let minimum_decode_budget_tokens =
            max_request_tokens.min(REQUEST_ADMISSION_DECODE_BUDGET_TOKENS);
        let mut target_required_tokens = new_tokens.saturating_add(max_request_tokens);
        let mut minimum_required_tokens = new_tokens.saturating_add(minimum_decode_budget_tokens);
        let mut available_tokens = {
            let mut e = self.engine.write();
            let (available_tokens, evicted) = e.ensure_available_kv_tokens(target_required_tokens);
            if evicted > 0 {
                warn!(
                    "[generate_request] Evicted {} prefix cache block(s) to reserve {} KV tokens ({} new prompt + {} requested decode)",
                    evicted,
                    target_required_tokens,
                    new_tokens,
                    max_request_tokens
                );
            }
            available_tokens
        };
        loop {
            let refreshed_cached_tokens = {
                let mut e = self.engine.write();
                e.query_prefix_cache_match_tokens(&token_ids)
            };
            if refreshed_cached_tokens == cached_tokens {
                break;
            }

            cached_tokens = refreshed_cached_tokens;
            new_tokens = token_ids.len().saturating_sub(cached_tokens);
            target_required_tokens = new_tokens.saturating_add(max_request_tokens);
            minimum_required_tokens = new_tokens.saturating_add(minimum_decode_budget_tokens);
            let (refreshed_available_tokens, evicted) = {
                let mut e = self.engine.write();
                e.ensure_available_kv_tokens(target_required_tokens)
            };
            if evicted > 0 {
                warn!(
                    "[generate_request] Evicted {} additional prefix cache block(s) after prefix-cache hit changed; reserving {} KV tokens ({} new prompt + {} requested decode)",
                    evicted,
                    target_required_tokens,
                    new_tokens,
                    max_request_tokens
                );
            }
            available_tokens = refreshed_available_tokens;
            if evicted == 0 {
                break;
            }
        }
        if minimum_required_tokens > available_tokens {
            if available_tokens <= new_tokens {
                return Err(candle_core::Error::msg(format!(
                    "Requested prompt({} tokens, {} new after prefix cache) exceeds available KV cache capacity {}",
                    token_ids.len(),
                    new_tokens,
                    available_tokens
                )));
            }
            return Err(candle_core::Error::msg(format!(
                "Requested prompt({} tokens, {} new after prefix cache) plus {} decode budget tokens exceeds available KV cache capacity {}",
                token_ids.len(),
                new_tokens,
                minimum_decode_budget_tokens,
                available_tokens
            )));
        }
        if target_required_tokens > available_tokens {
            warn!(
                "[generate_request] Request admitted with {} KV tokens available, below requested reservation {} tokens but enough for {} new prompt tokens plus {} decode budget tokens ({} cached prompt tokens)",
                available_tokens,
                target_required_tokens,
                new_tokens,
                minimum_decode_budget_tokens,
                cached_tokens
            );
        }

        // Let's create a local notify for this request.
        let req_notify = std::sync::Arc::new(tokio::sync::Notify::new());

        // Re-add request with req_notify
        let prefilled_reasoning_end =
            crate::tools::stream_parser::detect_prefilled_reasoning_end_marker(&prompt);

        let has_tools = !resolved_tools.is_empty();
        {
            let mut e = self.engine.write();
            let mut sampling_params = SamplingParams::new(
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
                max_request_tokens,
                None,
                None,
                request.skip_special_tokens.unwrap_or(true),
                request.thinking,
            )
            .map_err(candle_core::Error::msg)?;
            sampling_params.mcp_mode = if has_tools { Some(true) } else { None };
            e.add_request(
                token_ids,
                request_id.clone(),
                std::time::SystemTime::now(),
                sampling_params,
                request.logprobs.unwrap_or(false),
                false, // is_embedding
                crate::openai::requests::EncodingFormat::default(),
                crate::openai::requests::EmbeddingType::default(),
                resolved_tools.clone(),
                image_data,
                None, // streamer
                Some(req_notify.clone()),
                false,
                prefilled_reasoning_end,
            );
            self.notify.notify_one();
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
        let response_model = if e.model_name().is_empty() {
            request.model.clone().unwrap_or("default".to_string())
        } else {
            e.model_name().to_string()
        };
        if let Some(record) = e.completion_records.get(&request_id) {
            let mut choices = record.0.clone();
            if crate::stream_as_reasoning_content() {
                for choice in &mut choices {
                    if let Some(text) = choice.message.content.take() {
                        match crate::tools::stream_parser::extract_reasoning_content(&text) {
                            Some((reasoning, remaining)) => {
                                choice.message.content = if remaining.is_empty() {
                                    None
                                } else {
                                    Some(remaining)
                                };
                                choice.message.reasoning_content = Some(reasoning);
                            }
                            None => {
                                choice.message.content = Some(text);
                            }
                        }
                    }
                }
            }
            if has_tools {
                let parser = crate::tools::parser::ToolParser::new();
                let tool_schemas = build_tool_schema_map(&resolved_tools);
                for choice in &mut choices {
                    let parsed_calls = if let Some(calls) = choice.message.tool_calls.take() {
                        calls
                    } else if let Some(content) = &choice.message.content {
                        parser.parse(content)
                    } else {
                        Vec::new()
                    };

                    if parsed_calls.is_empty() {
                        continue;
                    }

                    let (valid_calls, invalid_calls) =
                        filter_tool_calls(&parsed_calls, &tool_schemas);
                    if !invalid_calls.is_empty() {
                        tracing::warn!(
                            "Dropped {} invalid tool call(s) before response",
                            invalid_calls.len()
                        );
                    }
                    if valid_calls.is_empty() {
                        if let Some(feedback) =
                            build_invalid_tool_call_feedback(&invalid_calls, &tool_schemas, None)
                        {
                            choice.message.content = Some(feedback);
                        }
                        choice.finish_reason = Some("stop".to_string());
                        continue;
                    }

                    choice.message.tool_calls = Some(valid_calls);
                    choice.message.content = None;
                    choice.finish_reason = Some("tool_calls".to_string());
                }
            }
            Ok(ChatCompletionResponse {
                id: request_id,
                choices,
                created: record.1.created,
                model: response_model,
                object: "chat.completion",
                usage: record.1.clone(),
            })
        } else {
            Err(candle_core::Error::msg("Failed to get response"))
        }
    }

    pub fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(self.embed_async(request))
    }

    pub async fn embed_async(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let prompt_tokens = {
            let e = self.engine.read();

            let prompt_str = match &request.input {
                crate::openai::requests::EmbeddingInput::String(s) => s.clone(),
                crate::openai::requests::EmbeddingInput::MultiString(vec) => {
                    // Just take the first one for now, or join?
                    // The logic in openai_server handle list
                    if vec.is_empty() {
                        return Err(candle_core::Error::msg("Empty input"));
                    }
                    vec[0].clone()
                }
                _ => {
                    return Err(candle_core::Error::msg(
                        "Unsupported input type for internal API",
                    ))
                }
            };

            e.tokenizer()
                .encode(prompt_str, false)
                .map_err(candle_core::Error::msg)?
                .get_ids()
                .to_vec()
        };

        info!(
            "[embed_async] Processing embedding request with {} tokens",
            prompt_tokens.len()
        );

        // Validate prompt length
        self.validate_prompt(&prompt_tokens, "embed_async")?;

        let request_id = format!("embd-{}", uuid::Uuid::new_v4());

        let (tx, mut rx) = tokio::sync::mpsc::channel(1024);

        {
            let mut e = self.engine.write();
            e.add_request(
                prompt_tokens,
                request_id.clone(),
                 std::time::SystemTime::now(),
                 SamplingParams::new(
                    1, None, 0.0, 0.0, None, None, None, None, None, false, 1.0,
                    crate::openai::sampling_params::EarlyStoppingCondition::UnlikelyBetterCandidates,
                    None, Vec::new(), false, 1, None, None, true, None
                ).map_err(candle_core::Error::msg).unwrap(),
                false,
                true, // is_embedding
                request.encoding_format.clone(),
                request.embedding_type.clone(),
                Vec::new(),
                None,
                Some(std::sync::Arc::new(tx)),
                None,
                false,
                None,
            );
            self.notify.notify_one();
        }

        info!(
            "[embed_async] Request {} submitted, awaiting response",
            request_id
        );

        match rx.recv().await {
            Some(ChatResponse::Embedding(resp)) => {
                info!(
                    "[embed_async] Request {} completed successfully",
                    request_id
                );
                Ok(resp)
            }
            Some(ChatResponse::ModelError(e)) => {
                warn!(
                    "[embed_async] Request {} failed with model error: {}",
                    request_id, e
                );
                Err(candle_core::Error::msg(e.to_string()))
            }
            Some(_) => {
                warn!(
                    "[embed_async] Request {} received unexpected response type",
                    request_id
                );
                Err(candle_core::Error::msg("Unexpected response type"))
            }
            None => {
                warn!("[embed_async] Request {} channel closed", request_id);
                Err(candle_core::Error::msg("Channel closed"))
            }
        }
    }

    pub fn shutdown(&self) {
        let e = self.engine.read();
        e.exit_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.notify.notify_waiters();
    }
}
