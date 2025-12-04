//! OpenAI-compatible inference module.
//!
//! This module provides the inference engine, model implementations, and OpenAI API types.
//!
//! The types in this module are the canonical definitions. The `candle-vllm-openai` crate
//! re-exports these types and provides additional adapters.

use self::pipelines::llm_engine::LLMEngine;
use self::responses::APIError;
use crate::openai::sampling_params::{GenerationConfig, SamplingParams};
use candle_core::Device;
use either::Either;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tokenizers::{EncodeInput, Encoding, Tokenizer};

// ============================================================================
// Modules
// ============================================================================

// Inference-related modules
#[cfg(feature = "nccl")]
pub mod communicator;
pub mod conversation;
pub mod distributed;
pub mod image_tool;
pub mod local_vision_tool;
pub mod logits_processor;
pub mod models;
pub mod openai_server;
pub mod pipelines;
pub mod requests;
pub mod responses;
pub mod sampling_params;
pub mod streaming;
pub mod tool_parser;
pub mod utils;
pub mod vision_proxy;

// ============================================================================
// Tokenizer wrapper trait
// ============================================================================

pub trait TokenizerWrapper<'s, E>
where
    E: Into<EncodeInput<'s>>,
{
    fn tokenize(&self, input: E) -> Result<Encoding, APIError>;
    fn detokenize(&self, input: &[u32]) -> Result<String, APIError>;
}

impl<'s, E> TokenizerWrapper<'s, E> for Tokenizer
where
    E: Into<EncodeInput<'s>>,
{
    fn tokenize(&self, input: E) -> Result<Encoding, APIError> {
        self.encode(input, false).map_err(APIError::from)
    }

    fn detokenize(&self, input: &[u32]) -> Result<String, APIError> {
        self.decode(input, false).map_err(APIError::from)
    }
}

// ============================================================================
// Engine-specific types
// ============================================================================

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub max_model_len: usize,
    pub default_max_tokens: usize,
    pub generation_cfg: Option<GenerationConfig>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct MaxModelLen(
    #[serde(with = "either::serde_untagged")] pub Either<Option<usize>, Option<f64>>,
);

#[derive(Clone, Debug, serde::Deserialize)]
#[allow(unused)]
pub struct TokenContent {
    content: Option<String>,
    lstrip: Option<bool>,
    normalized: Option<bool>,
    rstrip: Option<bool>,
    single_word: Option<bool>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct BosEosToken(
    #[serde(with = "either::serde_untagged")] pub Either<Option<String>, Option<TokenContent>>,
);

#[derive(Clone, Debug, serde::Deserialize)]
pub struct TokenizerConfig {
    pub model_max_length: MaxModelLen,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub chat_template: Option<String>,
    pub bos_token: Option<BosEosToken>,
    pub eos_token: Option<BosEosToken>,
}

/// Server data shared across request handlers.
///
/// The `model` field is `Arc<LLMEngine>` (no RwLock) because:
/// - LLMEngine uses internal lock-free channels for work distribution
/// - Workers own their pipelines and cache engines
/// - Shared read-only resources (tokenizer, config) are accessed without locks
/// - Per-request state uses internal RwLock where needed
pub struct OpenAIServerData {
    pub model: Arc<LLMEngine>,
    pub pipeline_config: PipelineConfig,
    pub record_conversation: bool,
    pub device: Device,
    pub vision_tool: Option<Arc<crate::openai::local_vision_tool::LocalVisionModelTool>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TaskData {
    pub seq_id: usize,
    pub group_id: usize,
    pub prompt: Vec<u32>,
    pub request_id: String,
    pub created: SystemTime,
    pub sampling_params: SamplingParams,
    pub use_logprobs: bool,
}
