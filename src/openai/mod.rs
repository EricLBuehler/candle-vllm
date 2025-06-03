use self::{pipelines::llm_engine::LLMEngine, responses::APIError};
use crate::openai::sampling_params::SamplingParams;
use candle_core::Device;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::SystemTime;
use tokenizers::{EncodeInput, Encoding, Tokenizer};
#[cfg(feature = "nccl")]
pub mod communicator;
pub mod distributed;
pub mod requests;
pub mod responses;
pub mod sampling_params;
pub mod streaming;
use either::Either;
use serde::{Deserialize, Serialize};
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

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub max_model_len: usize,
    pub default_max_tokens: usize,
    pub penalty: f32,
    pub repeat_last_n: usize,
    pub temperature: Option<f32>,
    pub top_k: Option<isize>,
    pub top_p: Option<f32>,
    pub thinking: Option<bool>,
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

pub struct OpenAIServerData {
    pub model: Arc<RwLock<LLMEngine>>,
    pub pipeline_config: PipelineConfig,
    pub record_conversation: bool,
    pub device: Device,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TaskData {
    pub seq_id: usize,
    pub group_id: usize,
    pub prompt: Encoding,
    pub request_id: String,
    pub created: SystemTime,
    pub sampling_params: SamplingParams,
    pub use_logprobs: bool,
}

pub mod conversation;
pub mod logits_processor;
pub mod models;
pub mod openai_server;
pub mod pipelines;
pub mod utils;
