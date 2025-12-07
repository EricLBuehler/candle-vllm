use crate::openai::sampling_params::Logprobs;
use candle_core::{Device, Result, Tensor, WithDType};
use dirs;
use either::Either;
use std::collections::HashMap;
use std::{env, fs};

/// The LLMEngine using prometheus_parking_lot scheduler for resource management.
/// This is the main inference engine for candle-vllm.
pub mod llm_engine;
pub use llm_engine::LLMEngine;
pub use llm_engine::SchedulerPoolConfig;

pub mod pipeline;
pub mod work_item;
pub mod worker;

type TokenOrFinishReason = Either<Logprobs, String>;
use crate::openai::pipelines::pipeline::DefaultPipeline;

fn _make_tensor_with_pad<D: WithDType>(
    x: Vec<Vec<D>>,
    max_len: usize,
    pad: D,
    device: &Device,
) -> Result<Tensor> {
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
    Tensor::from_vec(flattened, (padded_x.len(), max_len), device)
}

#[cfg(all(feature = "cuda", feature = "graph"))]
#[macro_export]
macro_rules! graph_model_wrapper {
    ($model:expr, $device:expr, $( $variant:ident ),+ $(,)?) => {
        match &$model {
            $(
                LLMModel::$variant(m) => {
                    let model_arc = Arc::clone(&m);
                    let closure = move |input_ids: &Tensor,
                                        positions: &Tensor,
                                        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
                                        input_metadata: &InputMetadata| {
                        model_arc.forward(input_ids, positions, kv_caches, input_metadata)
                    };
                    let boxed_closure: Box<ModelFn> = Box::new(closure);
                    CudaGraphWrapper::new(boxed_closure, $device.as_cuda_device()?.clone().into())
                },
            )+
        }
    };
}

pub(crate) fn get_token(hf_token: Option<String>, hf_token_path: Option<String>) -> Result<String> {
    Ok(match (hf_token, hf_token_path) {
        (Some(envvar), None) => env::var(envvar)
            .map_err(candle_core::Error::wrap)?
            .trim()
            .to_string(),
        (None, Some(path)) => fs::read_to_string(path)
            .map_err(candle_core::Error::wrap)?
            .trim()
            .to_string(),
        (None, None) => fs::read_to_string(format!(
            "{}/.cache/huggingface/token",
            dirs::home_dir().unwrap().display()
        ))
        .map_err(candle_core::Error::wrap)?
        .trim()
        .to_string(),
        (Some(_), Some(path)) => fs::read_to_string(path)
            .map_err(candle_core::Error::wrap)?
            .trim()
            .to_string(),
    })
}

pub trait DecodeStreamTrait: Send + Sync {
    fn step(&mut self, id: u32) -> Option<String>;
}

struct StreamWithTokenizer<M, N, PT, PP, D>
where
    M: tokenizers::Model + Send + Sync + 'static,
    N: tokenizers::Normalizer + Send + Sync + 'static,
    PT: tokenizers::PreTokenizer + Send + Sync + 'static,
    PP: tokenizers::PostProcessor + Send + Sync + 'static,
    D: tokenizers::Decoder + Send + Sync + 'static,
{
    _tokenizer: Box<tokenizers::TokenizerImpl<M, N, PT, PP, D>>,
    stream: tokenizers::DecodeStream<'static, M, N, PT, PP, D>,
}

impl<M, N, PT, PP, D> DecodeStreamTrait for StreamWithTokenizer<M, N, PT, PP, D>
where
    M: tokenizers::Model + Send + Sync + 'static,
    N: tokenizers::Normalizer + Send + Sync + 'static,
    PT: tokenizers::PreTokenizer + Send + Sync + 'static,
    PP: tokenizers::PostProcessor + Send + Sync + 'static,
    D: tokenizers::Decoder + Send + Sync + 'static,
{
    fn step(&mut self, id: u32) -> Option<String> {
        self.stream.step(id).ok().flatten()
    }
}

type DecodeStreamType = Box<dyn DecodeStreamTrait + Send + Sync>;
type StreamDecoderMap = HashMap<usize, DecodeStreamType>;
