#![warn(clippy::cast_lossless)]
use candle::Result;
use candle_core as candle;
use clap::Subcommand;
use openai::pipelines::{
    pipeline::{DefaultLoader, SpecificConfig},
    ModelLoader,
};

#[derive(Debug, Subcommand)]
pub enum ModelSelected {
    /// Select the llama model (default llama2-7b).
    Llama {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: usize,
    },

    /// Select the phi3 model (default 3.8b).
    Phi3 {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: usize,
    },

    /// Select the qwen model (default 1.5b).
    Qwen2 {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: usize,
    },
}

impl ToString for ModelSelected {
    fn to_string(&self) -> String {
        match self {
            ModelSelected::Llama { repeat_last_n: _ } => "llama".to_string(),
            ModelSelected::Phi3 { repeat_last_n: _ } => "phi3".to_string(),
            ModelSelected::Qwen2 { repeat_last_n: _ } => "qwen2".to_string(),
        }
    }
}

pub fn get_model_loader<'a>(
    selected_model: ModelSelected,
    model_id: Option<String>,
) -> (Box<dyn ModelLoader<'a>>, String) {
    match selected_model {
        ModelSelected::Llama { repeat_last_n } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(repeat_last_n),
                "llama".to_string(),
            )),
            if model_id.is_some() {
                model_id.unwrap()
            } else {
                "meta-llama/Llama-2-7b-chat-hf".to_string()
            },
        ),
        ModelSelected::Phi3 { repeat_last_n } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(repeat_last_n),
                "phi3".to_string(),
            )),
            if model_id.is_some() {
                model_id.unwrap()
            } else {
                "microsoft/Phi-3-mini-4k-instruct".to_string()
            },
        ),
        ModelSelected::Qwen2 { repeat_last_n } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(repeat_last_n),
                "qwen2".to_string(),
            )),
            if model_id.is_some() {
                model_id.unwrap()
            } else {
                "Qwen/Qwen2-1.5B-Instruct".to_string()
            },
        ),
    }
}

pub fn log_warning(message: &str) {
    eprintln!("Warning at {:?}: '{}'", chrono::offset::Utc::now(), message);
}

pub fn hub_load_local_safetensors(
    path: &String,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    println!("{:}", path.to_owned() + json_file);
    let jsfile = std::fs::File::open(path.to_owned() + json_file)?;
    let json: serde_json::Value = serde_json::from_reader(&jsfile).map_err(candle::Error::wrap)?;
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
