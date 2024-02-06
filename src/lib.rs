#![warn(clippy::cast_lossless)]

use clap::Subcommand;
use openai::pipelines::{
    llama::{LlamaLoader, LlamaSpecificConfig},
    ModelLoader,
};

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
            "meta-llama/Llama-2-7b-chat-hf".to_string(),
        ),
        ModelSelected::Llama13b { repeat_last_n } => (
            Box::new(LlamaLoader::new(
                LlamaSpecificConfig::new(repeat_last_n),
                "llama13b".to_string(),
            )),
            "meta-llama/Llama-2-13b-chat-hf".to_string(),
        ),
        ModelSelected::Llama70b { repeat_last_n } => (
            Box::new(LlamaLoader::new(
                LlamaSpecificConfig::new(repeat_last_n),
                "llama70b".to_string(),
            )),
            "meta-llama/Llama-2-70b-chat-hf".to_string(),
        ),
    }
}

pub fn log_warning(message: &str) {
    eprintln!("Warning at {:?}: '{}'", chrono::offset::Utc::now(), message);
}

pub mod backend;
pub mod openai;
pub mod paged_attention;
pub mod scheduler;
