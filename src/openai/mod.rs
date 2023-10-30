use tokenizers::{EncodeInput, Encoding};

use self::responses::APIError;

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

pub trait ModulePipeline {
    fn forward(&self, xs: &Encoding) -> Result<String, APIError>;
}

pub struct PipelineConfig {
    pub max_model_len: usize,
}

pub struct OpenAIServerData<'a> {
    tokenizer: Box<dyn TokenizerWrapper<'a, String>>,
    model: Box<dyn ModulePipeline>,
    pipeline_config: PipelineConfig,
}

pub mod openai_server;
