use candle_core::Device;
use std::sync::Arc;
use tokenizers::{EncodeInput, Encoding, Tokenizer};
use tokio::sync::{Mutex, Notify};

use self::{pipelines::llm_engine::LLMEngine, responses::APIError};

pub mod requests;
pub mod responses;
pub mod sampling_params;
pub mod streaming;

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
    pub temperature: f32,
}

pub struct OpenAIServerData {
    pub model: Arc<Mutex<LLMEngine>>,
    pub pipeline_config: PipelineConfig,
    pub record_conversation: bool,
    pub device: Device,
    pub finish_notify: Arc<Notify>,
}

pub mod conversation;
pub mod logits_processor;
pub mod models;
pub mod openai_server;
pub mod pipelines;
pub mod utils;
