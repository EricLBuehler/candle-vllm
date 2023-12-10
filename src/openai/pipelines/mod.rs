use std::{
    collections::{HashMap, HashSet},
    env,
    path::PathBuf,
    sync::Arc,
};

use actix_web::web::Bytes;
use candle_core::{DType, Device, Tensor};
use tokenizers::Encoding;
use tokio::sync::mpsc::Sender;

use crate::{
    openai::sampling_params::SamplingType, paged_attention::input_metadata::InputMetadata,
};

use super::{
    conversation::Conversation,
    models::ConfigLike,
    responses::{APIError, ChatChoice, ChatCompletionUsageResponse},
    sampling_params::{EarlyStoppingCondition, SamplingParams},
    streaming::SenderError,
    PipelineConfig, TokenizerWrapper,
};

pub mod llama;

const PAD_SLOT_ID: usize = usize::MAX;

pub trait ModulePipeline<'s>: Send + Sync {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: Tensor,
        kv_cache: Option<Arc<Vec<(Tensor, Tensor)>>>,
        input_metadata: InputMetadata,
    ) -> Result<(Option<Vec<ChatChoice>>, ChatCompletionUsageResponse), APIError>;

    fn name(&self) -> &str;

    fn tokenizer(&self) -> &dyn TokenizerWrapper<'s, String>;

    fn get_conversation(&mut self) -> &mut dyn Conversation;

    fn get_model_config(&self) -> Box<dyn ConfigLike>;
}

// TODO(EricLBuehler): Ensure the padding token matches tokenizer
fn _make_tensor_with_pad(
    x: Vec<Vec<usize>>,
    max_len: usize,
    pad: usize,
    dtype: DType,
) -> Result<Tensor, APIError> {
    let mut padded_x = Vec::new();
    for mut x_i in x {
        assert!(x_i.len() <= max_len);
        x_i.extend(vec![pad].repeat(max_len - x_i.len()));
        let x_i = x_i.iter().map(|x| *x as f64).collect::<Vec<_>>();
        padded_x.push(
            Tensor::new(x_i, &Device::new_cuda(0).map_err(APIError::from)?)
                .map_err(APIError::from)?,
        );
    }
    Tensor::cat(&padded_x[..], 0).map_err(APIError::from)
}

pub(crate) fn read_env_var(var: String) -> Result<String, APIError> {
    env::var(var).map_err(APIError::from)
}

pub(crate) fn logical_or(a: &Tensor, b: &Tensor) -> Result<Tensor, APIError> {
    (a + b).map_err(APIError::from)
}

pub(crate) fn logical_not(xs: &Tensor) -> Result<Tensor, APIError> {
    xs.where_cond(
        &xs.zeros_like().map_err(APIError::from)?,
        &xs.ones_like().map_err(APIError::from)?,
    )
    .map_err(APIError::from)
}

pub trait ModelPaths {
    fn get_weight_filenames(&self) -> &Vec<PathBuf>;
    fn get_config_filename(&self) -> &PathBuf;
    fn get_tokenizer_filename(&self) -> &PathBuf;
}

pub trait ModelLoader<'a> {
    fn download_model(
        &self,
        model_id: String,
        revision: Option<String>,
        hf_token: Option<String>,
    ) -> Result<Box<dyn ModelPaths>, APIError>;

    fn load_model(
        &self,
        paths: Box<dyn ModelPaths>,
        dtype: DType,
        device: Device,
    ) -> Result<(Box<dyn ModulePipeline<'a>>, PipelineConfig), APIError>;
}
