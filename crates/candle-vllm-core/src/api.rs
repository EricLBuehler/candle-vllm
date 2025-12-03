use crate::openai::models::Config as ModelConfig;
use crate::openai::openai_server::chat_completions_with_data;
use crate::openai::pipelines::llm_engine::LLMEngine;
use crate::openai::pipelines::pipeline::{DefaultLoader, DefaultPipeline};
use crate::openai::requests::{ChatCompletionRequest, ChatMessage, Messages};
use crate::openai::responses::{ChatCompletionResponse, ChatResponder};
use crate::openai::sampling_params::{GenerationConfig, SamplingParams};
use crate::openai::{OpenAIServerData, TokenizerWrapper};
use crate::scheduler::cache_engine::{CacheConfig, CacheEngine};
use crate::scheduler::SchedulerConfig;
use candle_core::{DType, Device, Result as CandleResult};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Notify;

/// Configuration for the inference engine.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub device: Option<usize>,
    pub dtype: Option<DType>,
    pub max_batch_size: Option<usize>,
    pub max_sequence_length: Option<usize>,
    pub kv_cache_memory: Option<usize>,
    pub enable_cuda_graph: bool,
    pub enable_chunked_prefill: bool,
    pub prefill_chunk_size: Option<usize>,
}

impl EngineConfig {
    /// Create a new engine config from a model path.
    pub fn from_model_path(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            device: None,
            dtype: None,
            max_batch_size: None,
            max_sequence_length: None,
            kv_cache_memory: None,
            enable_cuda_graph: false,
            enable_chunked_prefill: false,
            prefill_chunk_size: None,
        }
    }

    /// Create a builder for EngineConfig.
    pub fn builder() -> EngineConfigBuilder {
        EngineConfigBuilder::new()
    }
}

/// Builder for EngineConfig with a fluent API.
pub struct EngineConfigBuilder {
    config: EngineConfig,
}

impl EngineConfigBuilder {
    fn new() -> Self {
        Self {
            config: EngineConfig {
                model_path: PathBuf::new(),
                device: None,
                dtype: None,
                max_batch_size: None,
                max_sequence_length: None,
                kv_cache_memory: None,
                enable_cuda_graph: false,
                enable_chunked_prefill: false,
                prefill_chunk_size: None,
            },
        }
    }

    /// Set the model path (required).
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.model_path = path.into();
        self
    }

    /// Set the device ordinal.
    pub fn device(mut self, device: usize) -> Self {
        self.config.device = Some(device);
        self
    }

    /// Set the data type.
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.config.dtype = Some(dtype);
        self
    }

    /// Set the maximum batch size.
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.config.max_batch_size = Some(size);
        self
    }

    /// Set the maximum sequence length.
    pub fn max_sequence_length(mut self, length: usize) -> Self {
        self.config.max_sequence_length = Some(length);
        self
    }

    /// Set the KV cache memory size (in MB).
    pub fn kv_cache_memory(mut self, memory: usize) -> Self {
        self.config.kv_cache_memory = Some(memory);
        self
    }

    /// Enable CUDA graph optimization.
    pub fn enable_cuda_graph(mut self, enable: bool) -> Self {
        self.config.enable_cuda_graph = enable;
        self
    }

    /// Enable chunked prefill.
    pub fn enable_chunked_prefill(mut self, enable: bool) -> Self {
        self.config.enable_chunked_prefill = enable;
        self
    }

    /// Set the prefill chunk size.
    pub fn prefill_chunk_size(mut self, size: usize) -> Self {
        self.config.prefill_chunk_size = Some(size);
        self
    }

    /// Build the EngineConfig.
    pub fn build(self) -> Result<EngineConfig> {
        if self.config.model_path.as_os_str().is_empty() {
            return Err(Error::Config("model_path is required".to_string()));
        }
        Ok(self.config)
    }
}

/// Parameters for text generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationParams {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<isize>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub logprobs: Option<usize>,
    pub seed: Option<u64>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: Some(128),
            temperature: Some(0.7),
            top_p: Some(1.0),
            top_k: Some(-1),
            repetition_penalty: Some(1.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            stop_sequences: None,
            logprobs: None,
            seed: None,
        }
    }
}

/// Statistics about a generation run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_time_ms: u128,
    pub tokens_per_second: f32,
}

/// Reason why generation stopped.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    StopSequence,
    Cancelled,
    Error(String),
}

/// Output from a text generation request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationOutput {
    pub tokens: Vec<String>,
    pub finish_reason: FinishReason,
    pub logprobs: Option<Vec<HashMap<String, f32>>>,
    pub stats: Option<GenerationStats>,
    pub text: Option<String>,
}

/// Errors that can occur when using the inference engine.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("model load error: {0}")]
    ModelLoad(String),
    #[error("tokenization error: {0}")]
    Tokenization(String),
    #[error("generation error: {0}")]
    Generation(String),
    #[error("device error: {0}")]
    Device(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("cancelled")]
    Cancelled,
    #[error("io error: {0}")]
    Io(String),
    #[error("candle error: {0}")]
    Candle(String),
    #[error("other: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// High-level inference engine for text generation.
pub struct InferenceEngine {
    pub(crate) engine: Arc<RwLock<LLMEngine>>,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    model_info: ModelInfo,
}

/// Information about the loaded model.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub model_path: PathBuf,
    pub max_sequence_length: usize,
    pub max_batch_size: usize,
    pub dtype: String, // Store as string since DType doesn't implement Serialize
}

impl InferenceEngine {
    /// Create a builder for InferenceEngine.
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::new()
    }

    /// Create a new inference engine with the given configuration.
    pub async fn new(config: EngineConfig) -> Result<Self> {
        let loader = DefaultLoader::new(
            None,
            Some(config.model_path.to_string_lossy().into_owned()),
            None,
        );
        let (paths, gguf) = loader
            .prepare_model_weights(None, None)
            .map_err(|e| Error::ModelLoad(format!("{:?}", e)))?;

        let dtype = config.dtype.unwrap_or(DType::F16);
        let kv_cache_dtype = dtype;
        let device_ids = vec![config.device.unwrap_or(0)];
        let notify = Arc::new(Notify::new());
        let (pipelines, pipeline_cfg) = loader
            .load_model(
                paths,
                dtype,
                kv_cache_dtype,
                gguf,
                None,
                64,
                config.max_batch_size.unwrap_or(8),
                device_ids.clone(),
                #[cfg(feature = "nccl")]
                None,
                None,
                None,
                #[cfg(feature = "nccl")]
                None,
                #[cfg(feature = "nccl")]
                None,
            )
            .await
            .map_err(|e| Error::ModelLoad(e.to_string()))?;

        let tokenizer = pipelines
            .first()
            .map(|p| p.tokenizer().clone())
            .ok_or_else(|| Error::ModelLoad("no tokenizer loaded".into()))?;

        let cache_config = CacheConfig {
            block_size: 64,
            num_gpu_blocks: Some(config.kv_cache_memory.unwrap_or(4096)),
            num_cpu_blocks: Some(128),
            fully_init: true,
            dtype: kv_cache_dtype,
        };

        let scheduler_config = SchedulerConfig {
            max_num_seqs: config.max_batch_size.unwrap_or(8),
        };

        let mut model_config: Option<ModelConfig> = None;
        let num_shards = 1;

        let pipelines_with_cache: HashMap<usize, (Box<DefaultPipeline>, CacheEngine)> = pipelines
            .into_iter()
            .map(|pipeline| {
                let cfg = pipeline.get_model_config();
                if model_config.is_none() {
                    model_config = Some(cfg.clone());
                }
                let cache_engine = CacheEngine::new(
                    &cfg,
                    &cache_config,
                    cache_config.dtype,
                    pipeline.device(),
                    num_shards,
                )
                .map_err(|e| Error::ModelLoad(e.to_string()))?;
                Ok((pipeline.rank(), (pipeline, cache_engine)))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let model_config =
            model_config.ok_or_else(|| Error::ModelLoad("no model config".into()))?;

        let engine = LLMEngine::new(
            pipelines_with_cache,
            scheduler_config,
            &cache_config,
            &model_config,
            notify.clone(),
            500,
            num_shards,
            false,
            #[cfg(feature = "nccl")]
            None,
            config.prefill_chunk_size,
        )
        .map_err(|e| Error::ModelLoad(e.to_string()))?;

        let model_info = ModelInfo {
            model_path: config.model_path.clone(),
            max_sequence_length: config.max_sequence_length.unwrap_or(4096),
            max_batch_size: config.max_batch_size.unwrap_or(8),
            dtype: format!("{:?}", dtype),
        };

        Ok(Self {
            engine,
            tokenizer,
            device: crate::new_device(config.device.unwrap_or(0))
                .map_err(|e| Error::Device(e.to_string()))?,
            model_info,
        })
    }

    /// Get the tokenizer instance.
    pub fn tokenizer(&self) -> tokenizers::Tokenizer {
        self.tokenizer.clone()
    }

    /// Tokenize text into token IDs.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenization(e.to_string()))
            .map(|enc| enc.get_ids().to_vec())
    }

    /// Detokenize token IDs into text.
    pub fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, false)
            .map_err(|e| Error::Tokenization(e.to_string()))
    }

    pub async fn generate(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<GenerationOutput> {
        let request = ChatCompletionRequest {
            model: "local".to_string(),
            messages: Messages::Chat(vec![ChatMessage {
                role: "user".to_string(),
                content: Some(prompt.to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }]),
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            max_tokens: params.max_tokens,
            stop: params.stop_sequences.clone().map(|v| {
                if v.len() == 1 {
                    crate::openai::requests::StopTokens::Single(v[0].clone())
                } else {
                    crate::openai::requests::StopTokens::Multi(v)
                }
            }),
            logprobs: params.logprobs.map(|v| v > 0),
            stream: None,
            min_p: None,
            n: None,
            repeat_last_n: None,
            logit_bias: None,
            user: None,
            best_of: None,
            use_beam_search: None,
            ignore_eos: None,
            skip_special_tokens: None,
            stop_token_ids: None,
            thinking: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
        };

        let pipeline_config = crate::openai::PipelineConfig {
            max_model_len: self.model_info.max_sequence_length,
            default_max_tokens: params.max_tokens.unwrap_or(128),
            generation_cfg: Some(GenerationConfig {
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                min_p: None,
                frequency_penalty: params.frequency_penalty,
                presence_penalty: params.presence_penalty,
            }),
        };
        let data = OpenAIServerData {
            model: self.engine.clone(),
            pipeline_config,
            record_conversation: false,
            device: self.device.clone(),
        };

        let responder = chat_completions_with_data(Arc::new(data), request).await;

        // Extract response from ChatResponder
        match responder {
            ChatResponder::Completion(resp) => Ok(Self::map_response(resp)),
            ChatResponder::ModelError(e) => Err(Error::Generation(e.to_string())),
            ChatResponder::InternalError(e) => Err(Error::Generation(e.to_string())),
            ChatResponder::ValidationError(e) => Err(Error::Config(e.to_string())),
            ChatResponder::Streamer(_) => {
                Err(Error::Generation("unexpected stream response".to_string()))
            }
        }
    }

    /// Generate text with streaming support.
    /// Note: Streaming is not yet fully implemented - use OpenAIAdapter for streaming support.
    pub async fn generate_stream(&self, prompt: &str, params: GenerationParams) -> Result<()> {
        let request = ChatCompletionRequest {
            model: "local".to_string(),
            messages: Messages::Chat(vec![ChatMessage {
                role: "user".to_string(),
                content: Some(prompt.to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }]),
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            max_tokens: params.max_tokens,
            stop: params.stop_sequences.clone().map(|v| {
                if v.len() == 1 {
                    crate::openai::requests::StopTokens::Single(v[0].clone())
                } else {
                    crate::openai::requests::StopTokens::Multi(v)
                }
            }),
            logprobs: params.logprobs.map(|v| v > 0),
            stream: Some(true),
            min_p: None,
            n: None,
            repeat_last_n: None,
            logit_bias: None,
            user: None,
            best_of: None,
            use_beam_search: None,
            ignore_eos: None,
            skip_special_tokens: None,
            stop_token_ids: None,
            thinking: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
        };

        let pipeline_config = crate::openai::PipelineConfig {
            max_model_len: self.model_info.max_sequence_length,
            default_max_tokens: params.max_tokens.unwrap_or(128),
            generation_cfg: Some(GenerationConfig {
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                min_p: None,
                frequency_penalty: params.frequency_penalty,
                presence_penalty: params.presence_penalty,
            }),
        };
        let data = OpenAIServerData {
            model: self.engine.clone(),
            pipeline_config,
            record_conversation: false,
            device: self.device.clone(),
        };

        let responder = chat_completions_with_data(Arc::new(data), request).await;

        // For now, streaming is not fully implemented in the API layer
        // Users should use the OpenAI adapter for streaming support
        match responder {
            ChatResponder::Streamer(_) => {
                Err(Error::Generation("streaming not yet implemented in InferenceEngine API - use OpenAIAdapter for streaming".to_string()))
            }
            ChatResponder::Completion(_) => Err(Error::Generation("unexpected completion response".to_string())),
            ChatResponder::ModelError(e) => Err(Error::Generation(e.to_string())),
            ChatResponder::InternalError(e) => Err(Error::Generation(e.to_string())),
            ChatResponder::ValidationError(e) => Err(Error::Config(e.to_string())),
        }
    }

    /// Get information about the loaded model.
    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    /// Get generation statistics (placeholder - needs implementation).
    pub fn stats(&self) -> GenerationStats {
        // TODO: Implement actual stats collection
        GenerationStats {
            prompt_tokens: 0,
            generated_tokens: 0,
            total_time_ms: 0,
            tokens_per_second: 0.0,
        }
    }

    /// Get access to the internal engine (for advanced use cases).
    pub fn engine(&self) -> &Arc<RwLock<LLMEngine>> {
        &self.engine
    }

    /// Get the device used by this engine.
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn map_response(resp: ChatCompletionResponse) -> GenerationOutput {
        let text = resp.choices.first().and_then(|c| c.message.content.clone());
        GenerationOutput {
            tokens: Vec::new(),
            finish_reason: FinishReason::Stop,
            logprobs: None,
            stats: None,
            text,
        }
    }
}

/// Builder for InferenceEngine.
pub struct InferenceEngineBuilder {
    config: Option<EngineConfig>,
}

impl InferenceEngineBuilder {
    fn new() -> Self {
        Self { config: None }
    }

    /// Set the engine configuration.
    pub fn config(mut self, config: EngineConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the InferenceEngine asynchronously.
    pub async fn build(self) -> Result<InferenceEngine> {
        let config = self
            .config
            .ok_or_else(|| Error::Config("config is required".to_string()))?;
        InferenceEngine::new(config).await
    }
}
