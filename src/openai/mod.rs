use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};

use actix_web::web::Bytes;
use candle_core::{DType, Device};
use tokenizers::{EncodeInput, Encoding, Tokenizer};
use tokio::sync::mpsc::Sender;

use self::{
    responses::{APIError, ChatChoice, ChatCompletionUsageResponse},
    sampling_params::SamplingParams,
    streaming::SenderError,
};

pub mod conversation;
pub mod requests;
pub mod responses;
pub mod sampling_params;
mod streaming;

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

pub trait ModulePipeline<'s>: Send + Sync {
    fn forward(
        &mut self,
        xs: &Encoding,
        sampling: SamplingParams,
        device: Device,
        streamer: Option<Sender<Result<Bytes, SenderError>>>,
        roles: &(String, String),
    ) -> Result<(Option<Vec<ChatChoice>>, ChatCompletionUsageResponse), APIError>;

    fn name(&self) -> String;

    fn tokenizer(&self) -> &dyn TokenizerWrapper<'s, String>;
}

pub trait ModelPaths {
    fn get_weight_filenames(&self) -> &Vec<PathBuf>;
    fn get_config_filename(&self) -> &PathBuf;
    fn get_tokenizer_filename(&self) -> &PathBuf;
}

pub trait ModelLoader<'a, P: AsRef<Path>> {
    fn download_model(
        &self,
        model_id: String,
        revision: Option<String>,
        hf_token: Option<P>,
    ) -> Result<Box<dyn ModelPaths>, APIError>;

    fn load_model(
        &self,
        paths: Box<dyn ModelPaths>,
        dtype: DType,
        device: Device,
    ) -> Result<(Box<dyn ModulePipeline<'a>>, PipelineConfig), APIError>;
}

pub struct PipelineConfig {
    pub max_model_len: usize,
}

pub struct OpenAIServerData<'s> {
    pub model: Mutex<Box<dyn ModulePipeline<'s>>>,
    pub pipeline_config: PipelineConfig,
    pub device: Device,
}

pub mod pipelines;

pub mod openai_server;
