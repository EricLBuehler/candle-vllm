use crate::openai::sampling_params::Logprobs;
use candle_core::{DType, Device, Tensor, WithDType};
use dirs;
use either::Either;
use std::{env, fs, path::PathBuf, sync::Arc};

use crate::{paged_attention::input_metadata::InputMetadata, try_api};

use super::{conversation::Conversation, models::Config, responses::APIError, PipelineConfig};
use candle_examples::token_output_stream::TokenOutputStream;
/// The LLMEngine is effectively a wrapper around a ModulePipeline. It contains a Scheduler and a CacheEngine
/// which are used to scheduler and manage the cache during generation requests, respectively.
pub mod llm_engine;
pub mod pipeline;
use crate::scheduler::sequence::SequenceGroup;
type TokenOrFinishReason = Either<Logprobs, String>;
use std::collections::VecDeque;
pub trait ModulePipeline: Send + Sync {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: &Vec<Vec<usize>>,
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: InputMetadata,
    ) -> Result<Tensor, APIError>;

    fn sample(
        &mut self,
        logits: Tensor,
        groups: &VecDeque<Arc<SequenceGroup>>,
    ) -> Result<Vec<TokenOrFinishReason>, APIError>;

    fn name(&self) -> &str;

    fn tokenizer(&self) -> &TokenOutputStream;

    fn get_conversation(&mut self, with_history: bool) -> &mut dyn Conversation;

    fn get_model_config(&self) -> Config;

    fn get_dtype(&self) -> DType;

    fn device(&self) -> &Device;

    fn reset_decoder(&mut self) -> Option<String>;
}

// TODO(EricLBuehler): Ensure the padding token matches tokenizer
fn _make_tensor_with_pad<D: WithDType>(
    x: Vec<Vec<D>>,
    max_len: usize,
    pad: D,
    device: &Device,
) -> Result<Tensor, APIError> {
    let mut padded_x = Vec::new();
    for mut x_i in x {
        assert!(x_i.len() <= max_len);
        x_i.extend([pad].repeat(max_len - x_i.len()));
        let shape = (1, x_i.len());
        padded_x.push(try_api!(Tensor::from_vec(x_i, shape, device)));
    }
    Tensor::cat(&padded_x[..], 0).map_err(APIError::from)
}

pub(crate) fn get_token(
    hf_token: Option<String>,
    hf_token_path: Option<String>,
) -> Result<String, APIError> {
    Ok(match (hf_token, hf_token_path) {
        (Some(envvar), None) => try_api!(env::var(envvar)).trim().to_string(),
        (None, Some(path)) => try_api!(fs::read_to_string(path)).trim().to_string(),
        (None, None) => try_api!(fs::read_to_string(format!(
            "{}/.cache/huggingface/token",
            dirs::home_dir()
                .ok_or(APIError::new_str("No home directory"))?
                .display()
        )))
        .trim()
        .to_string(),
        _ => {
            return Err(APIError::new_str(
                "Do not specify `hf_token` and `hf_token_path` at the same time.",
            ))
        }
    })
}

pub trait ModelPaths {
    fn get_weight_filenames(&self) -> &Vec<PathBuf>;
    fn get_config_filename(&self) -> &PathBuf;
    fn get_tokenizer_filename(&self) -> &PathBuf;
}

pub trait ModelLoader {
    fn download_model(
        &self,
        model_id: String,
        revision: Option<String>,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
    ) -> Result<Box<dyn ModelPaths>, APIError>;

    fn load_model(
        &self,
        paths: Box<dyn ModelPaths>,
        dtype: DType,
        device: Device,
    ) -> Result<(Box<dyn ModulePipeline>, PipelineConfig), APIError>;
}
