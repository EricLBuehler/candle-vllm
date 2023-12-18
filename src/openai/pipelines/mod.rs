use std::{env, path::PathBuf, sync::Arc};

use candle_core::{DType, Device, Tensor, WithDType};
use candle_sampling::logits_processor::Logprobs;
use either::Either;

use crate::{paged_attention::input_metadata::InputMetadata, scheduler::sequence::Sequence};

use super::{
    conversation::Conversation, models::ConfigLike, responses::APIError,
    sampling_params::SamplingParams, PipelineConfig, TokenizerWrapper,
};

pub mod llama;
/// The LLMEngine is effectively a wrapper around a ModulePipeline. It contains a Scheduler and a CacheEngine
/// which are used to scheduler and manage the cache during generation requests, respectively.
pub mod llm_engine;

type TokenOrFinishReason = Either<Logprobs, String>;

pub trait ModulePipeline<'s>: Send + Sync {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: Tensor,
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: InputMetadata,
    ) -> Result<Tensor, APIError>;

    fn sample(
        &mut self,
        logits: Tensor,
        sampling_params: &SamplingParams,
        seqs: &[(&usize, &Arc<Sequence>)],
    ) -> Result<Vec<TokenOrFinishReason>, APIError>;

    fn name(&self) -> &str;

    fn tokenizer(&self) -> &dyn TokenizerWrapper<'s, String>;

    fn get_conversation(&mut self) -> &mut dyn Conversation;

    fn get_model_config(&self) -> Box<dyn ConfigLike>;

    fn get_dtype(&self) -> DType;
}

// TODO(EricLBuehler): Ensure the padding token matches tokenizer
fn _make_tensor_with_pad<D: WithDType>(
    x: Vec<Vec<D>>,
    max_len: usize,
    pad: D,
) -> Result<Tensor, APIError> {
    let mut padded_x = Vec::new();
    for mut x_i in x {
        assert!(x_i.len() <= max_len);
        x_i.extend([pad].repeat(max_len - x_i.len()));
        let shape = (x_i.len(),);
        padded_x.push(
            Tensor::from_vec(x_i, shape, &Device::new_cuda(0).map_err(APIError::from)?)
                .map_err(APIError::from)?,
        );
    }
    Tensor::cat(&padded_x[..], 0).map_err(APIError::from)
}

pub(crate) fn read_env_var(var: String) -> Result<String, APIError> {
    env::var(var).map_err(APIError::from)
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
