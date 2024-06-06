#![warn(clippy::cast_lossless)]
use candle_core as candle;
use candle::Result;
use clap::Subcommand;
use openai::pipelines::{
    llama::{LlamaLoader, LlamaSpecificConfig},
    ModelLoader,
};

#[derive(Debug, Subcommand)]
pub enum ModelSelected {
    /// Select the llama7b model.
    Llama7b {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: usize,
    },

    /// Select the llama13b model.
    Llama13b {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: usize,
    },

    /// Select the llama70b model.
    Llama70b {
        /// Control the application of repeat penalty for the last n tokens
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

pub fn hub_load_local_safetensors(path: &String,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    println!("{:}", path.to_owned() + json_file);
    let jsfile = std::fs::File::open(path.to_owned() + json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&jsfile).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => panic!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = Vec::<std::path::PathBuf>::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(0, (path.to_owned() + file).into());
        }
    }
    Ok(safetensors_files)
}

pub mod backend;
pub mod openai;
pub mod paged_attention;
pub mod scheduler;
