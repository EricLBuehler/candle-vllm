#![warn(clippy::cast_lossless)]

use candle_core::{DType, Tensor};
use clap::Subcommand;
use openai::pipelines::{
    llama::{LlamaLoader, LlamaSpecificConfig},
    ModelLoader,
};
use tch::Kind;

#[derive(Debug, Subcommand)]
pub enum ModelSelected {
    /// Select the llama7b model.
    Llama7b {
        #[arg(long)]
        repeat_last_n: usize,
    },

    /// Select the llama13b model.
    Llama13b {
        #[arg(long)]
        repeat_last_n: usize,
    },

    /// Select the llama70b model.
    Llama70b {
        #[arg(long)]
        repeat_last_n: usize,
    },
}

impl ToString for ModelSelected {
    fn to_string(&self) -> String {
        match self {
            ModelSelected::Llama7b { repeat_last_n: _ } => "llama7b".to_string(),
            ModelSelected::Llama13b { repeat_last_n: _ } => "llama13b".to_string(),
            ModelSelected::Llama70b { repeat_last_n: _ } => "llama70b".to_string(),
        }
    }
}

pub fn get_model_loader<'a>(selected_model: ModelSelected) -> (Box<dyn ModelLoader<'a>>, String) {
    match selected_model {
        ModelSelected::Llama7b { repeat_last_n } => (
            Box::new(LlamaLoader::new(
                LlamaSpecificConfig::new(repeat_last_n),
                "llama7b".to_string(),
            )),
            "meta-llama/Llama-27b-chat-hf".to_string(),
        ),
        ModelSelected::Llama13b { repeat_last_n } => (
            Box::new(LlamaLoader::new(
                LlamaSpecificConfig::new(repeat_last_n),
                "llama13b".to_string(),
            )),
            "meta-llama/Llama-213b-chat-hf".to_string(),
        ),
        ModelSelected::Llama70b { repeat_last_n } => (
            Box::new(LlamaLoader::new(
                LlamaSpecificConfig::new(repeat_last_n),
                "llama70b".to_string(),
            )),
            "meta-llama/Llama-270b-chat-hf".to_string(),
        ),
    }
}

pub fn log_warning(message: &str) {
    eprintln!("Warning at {:?}: '{}'", chrono::offset::Utc::now(), message);
}

fn convert_candle_to_tch(candle: &mut Tensor) -> tch::Tensor {
    let output_kind = match candle.dtype() {
        DType::BF16 => Kind::BFloat16,
        DType::F16 => Kind::Float,
        DType::F32 => Kind::Float,
        DType::F64 => Kind::Float,
        DType::I64 => Kind::Int64,
        DType::U8 => Kind::Uint8,
        DType::U32 => Kind::Int,
    };

    let mut dims = Vec::new();
    for dim in candle.dims() {
        dims.push(*dim as i64);
    }

    tch::Tensor::from_data_size(
        &candle
            .to_vec3::<u8>()
            .unwrap()
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>()[..],
        &dims[..],
        output_kind,
    )
}

fn convert_tch_to_ptr(
    tch: &mut tch::Tensor,
) -> (*mut torch_sys::C_tensor, &mut torch_sys::C_tensor) {
    (tch.as_mut_ptr(), unsafe { &mut *tch.as_mut_ptr() })
}

pub mod openai;
pub mod paged_attention;
pub mod scheduler;
