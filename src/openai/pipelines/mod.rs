use super::responses::APIError;
use crate::openai::sampling_params::Logprobs;
use crate::try_api;
use candle_core::{Device, Tensor, WithDType};
use dirs;
use either::Either;
use std::{env, fs};
/// The LLMEngine is effectively a wrapper around a ModulePipeline. It contains a Scheduler and a CacheEngine
/// which are used to scheduler and manage the cache during generation requests, respectively.
pub mod llm_engine;
pub mod pipeline;
type TokenOrFinishReason = Either<Logprobs, String>;
use crate::openai::pipelines::pipeline::DefaultPipeline;

fn _make_tensor_with_pad<D: WithDType>(
    x: Vec<Vec<D>>,
    max_len: usize,
    pad: D,
    device: &Device,
) -> Result<Tensor, APIError> {
    let mut padded_x = Vec::new();
    for mut x_i in x {
        if x_i.len() < max_len {
            x_i.extend([pad].repeat(max_len - x_i.len()));
        }
        padded_x.push(x_i);
    }
    let flattened: Vec<_> = padded_x
        .iter()
        .flat_map(|slice| slice.iter())
        .map(|&xx| xx)
        .collect();
    Tensor::from_vec(flattened, (padded_x.len(), max_len), device).map_err(APIError::from)
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
