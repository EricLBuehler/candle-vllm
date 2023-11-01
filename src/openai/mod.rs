use std::sync::Mutex;

use candle_core::Device;
use tokenizers::{EncodeInput, Encoding, Tokenizer};

use self::{responses::APIError, sampling_params::SamplingParams};

pub mod requests;
pub mod responses;

pub mod conversation;

pub mod sampling_params;

pub trait TokenizerWrapper<'s, E>
where
    E: Into<EncodeInput<'s>>,
{
    fn tokenize(&self, input: E) -> Result<Encoding, APIError>;
}

impl<'s, E> TokenizerWrapper<'s, E> for Tokenizer
where
    E: Into<EncodeInput<'s>>,
{
    fn tokenize(&self, input: E) -> Result<Encoding, APIError> {
        self.encode(input, false)
            .map_err(|x| APIError::new(x.to_string()))
    }
}

pub trait ModulePipeline<'s> {
    fn forward(
        &mut self,
        xs: &Encoding,
        sampling: SamplingParams,
        device: Device,
    ) -> Result<String, APIError>;

    fn name(&self) -> String;

    fn tokenizer(&self) -> &dyn TokenizerWrapper<'s, String>;
}

pub struct PipelineConfig {
    pub max_model_len: usize,
}

pub struct OpenAIServerData<'s> {
    pub model: Mutex<Box<dyn ModulePipeline<'s>>>,
    pub pipeline_config: PipelineConfig,
    pub device: Device,
}

pub mod models;

pub mod openai_server;
