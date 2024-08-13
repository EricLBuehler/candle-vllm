#![warn(clippy::cast_lossless)]
use candle::Result;
use candle_core as candle;
use clap::Subcommand;
use openai::pipelines::{pipeline::DefaultLoader, ModelLoader};

#[derive(Debug, Subcommand)]
pub enum ModelSelected {
    /// Select the llama model (default llama2-7b).
    Llama {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    /// Select the llama3 model (default llama3.1-8b).
    Llama3 {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    /// Select the phi2 model (default 2.7b).
    Phi2 {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    /// Select the phi3 model (default 3.8b).
    Phi3 {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        top_p: Option<f64>,

        #[arg(long)]
        top_k: Option<usize>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    /// Select the qwen model (default 1.8b).
    Qwen2 {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        top_p: Option<f64>,

        #[arg(long)]
        top_k: Option<usize>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    /// Select the gemma model (default 2b).
    Gemma {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    /// Select the mistral model (default 7b).
    Mistral {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    /// Select the Yi model (default 6b).
    Yi {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    /// Select the stable-lm model (default zephyr-3b).
    StableLM {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },
}

impl ToString for ModelSelected {
    fn to_string(&self) -> String {
        match self {
            ModelSelected::Llama {
                repeat_last_n: _,
                temperature: _,
                penalty: _,
                max_gen_tokens: _,
                quant: _,
            } => "llama".to_string(),
            ModelSelected::Llama3 {
                repeat_last_n: _,
                temperature: _,
                penalty: _,
                max_gen_tokens: _,
                quant: _,
            } => "llama3".to_string(),
            ModelSelected::Phi2 {
                repeat_last_n: _,
                temperature: _,
                penalty: _,
                max_gen_tokens: _,
                quant: _,
            } => "phi2".to_string(),
            ModelSelected::Phi3 {
                repeat_last_n: _,
                temperature: _,
                top_k: _,
                top_p: _,
                penalty: _,
                max_gen_tokens: _,
                quant: _,
            } => "phi3".to_string(),
            ModelSelected::Qwen2 {
                repeat_last_n: _,
                temperature: _,
                top_k: _,
                top_p: _,
                penalty: _,
                max_gen_tokens: _,
                quant: _,
            } => "qwen2".to_string(),
            ModelSelected::Gemma {
                repeat_last_n: _,
                temperature: _,
                penalty: _,
                max_gen_tokens: _,
                quant: _,
            } => "gemma".to_string(),
            ModelSelected::Mistral {
                repeat_last_n: _,
                temperature: _,
                penalty: _,
                max_gen_tokens: _,
                quant: _,
            } => "mistral".to_string(),
            ModelSelected::Yi {
                repeat_last_n: _,
                temperature: _,
                penalty: _,
                max_gen_tokens: _,
                quant: _,
            } => "yi".to_string(),
            ModelSelected::StableLM {
                repeat_last_n: _,
                temperature: _,
                penalty: _,
                max_gen_tokens: _,
                quant: _,
            } => "stablelm".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpecificConfig {
    repeat_last_n: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f64>,
    penalty: Option<f32>,
    max_gen_tokens: Option<usize>,
    quant: Option<String>,
}

impl SpecificConfig {
    pub fn new(
        repeat_last_n: Option<usize>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        penalty: Option<f32>,
        max_gen_tokens: Option<usize>,
        quant: Option<String>,
    ) -> Self {
        Self {
            repeat_last_n,
            temperature,
            top_k,
            top_p,
            penalty,
            max_gen_tokens,
            quant,
        }
    }
}

pub fn get_model_loader(
    selected_model: ModelSelected,
    model_id: Option<String>,
) -> (Box<dyn ModelLoader>, String) {
    match selected_model {
        ModelSelected::Llama {
            repeat_last_n,
            temperature,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    None,
                    None,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "llama".to_string(),
            )),
            if model_id.is_some() {
                model_id.unwrap()
            } else {
                "meta-llama/Llama-2-7b-chat-hf".to_string()
            },
        ),
        ModelSelected::Llama3 {
            repeat_last_n,
            temperature,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    None,
                    None,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "llama3".to_string(),
            )),
            if model_id.is_some() {
                model_id.unwrap()
            } else {
                "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string()
            },
        ),
        ModelSelected::Phi2 {
            repeat_last_n,
            temperature,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    None,
                    None,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "phi2".to_string(),
            )),
            if model_id.is_some() {
                model_id.unwrap()
            } else {
                "microsoft/microsoft/phi-2".to_string()
            },
        ),
        ModelSelected::Phi3 {
            repeat_last_n,
            temperature,
            top_k,
            top_p,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    top_k,
                    top_p,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "phi3".to_string(),
            )),
            if model_id.is_some() {
                model_id.unwrap()
            } else {
                "microsoft/Phi-3-mini-4k-instruct".to_string()
            },
        ),
        ModelSelected::Qwen2 {
            repeat_last_n,
            temperature,
            top_k,
            top_p,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    top_k,
                    top_p,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "qwen2".to_string(),
            )),
            if model_id.is_some() {
                model_id.unwrap()
            } else {
                "Qwen/Qwen1.5-1.8B-Chat".to_string()
            },
        ),
        ModelSelected::Gemma {
            repeat_last_n,
            temperature,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    None,
                    None,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "gemma".to_string(),
            )),
            if model_id.is_some() {
                model_id.unwrap()
            } else {
                "google/gemma-2b-it".to_string()
            },
        ),
        ModelSelected::Mistral {
            repeat_last_n,
            temperature,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    None,
                    None,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "mistral".to_string(),
            )),
            if model_id.is_some() {
                model_id.unwrap()
            } else {
                "mistralai/Mistral-7B-Instruct-v0.3".to_string()
            },
        ),

        ModelSelected::Yi {
            repeat_last_n,
            temperature,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    None,
                    None,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "yi".to_string(),
            )),
            if model_id.is_some() {
                model_id.unwrap()
            } else {
                "01-ai/Yi-6B-Chat".to_string()
            },
        ),

        ModelSelected::StableLM {
            repeat_last_n,
            temperature,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    None,
                    None,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "stablelm".to_string(),
            )),
            if model_id.is_some() {
                model_id.unwrap()
            } else {
                "stabilityai/stablelm-zephyr-3b".to_string()
            },
        ),
    }
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
