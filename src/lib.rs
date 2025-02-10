#![warn(clippy::cast_lossless)]
use candle::utils::{cuda_is_available, metal_is_available};
use candle::{Device, Result};
use candle_core as candle;
use clap::Subcommand;
use openai::pipelines::pipeline::DefaultLoader;
use std::fmt::Display;
use std::path::Path;

pub mod backend;
pub mod openai;
pub mod paged_attention;
pub mod scheduler;
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

impl Display for ModelSelected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSelected::Llama { .. } => write!(f, "llama"),
            ModelSelected::Llama3 { .. } => write!(f, "llama3"),
            ModelSelected::Phi2 { .. } => write!(f, "phi2"),
            ModelSelected::Phi3 {
                repeat_last_n: _,
                temperature: _,
                top_k: _,
                top_p: _,
                penalty: _,
                max_gen_tokens: _,
                quant: _,
            } => write!(f, "phi3"),
            ModelSelected::Qwen2 {
                repeat_last_n: _,
                temperature: _,
                top_k: _,
                top_p: _,
                penalty: _,
                max_gen_tokens: _,
                quant: _,
            } => write!(f, "qwen2"),
            ModelSelected::Gemma { .. } => write!(f, "gemma"),
            ModelSelected::Mistral { .. } => write!(f, "mistral"),
            ModelSelected::Yi { .. } => write!(f, "yi"),
            ModelSelected::StableLM { .. } => write!(f, "stablelm"),
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
) -> (Box<DefaultLoader>, String, Option<String>) {
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
                    quant.clone(),
                ),
                "llama".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "meta-llama/Llama-2-7b-chat-hf".to_string()
            },
            quant,
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
                    quant.clone(),
                ),
                "llama3".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string()
            },
            quant,
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
                    quant.clone(),
                ),
                "phi2".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "microsoft/microsoft/phi-2".to_string()
            },
            quant,
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
                    quant.clone(),
                ),
                "phi3".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "microsoft/Phi-3-mini-4k-instruct".to_string()
            },
            quant,
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
                    quant.clone(),
                ),
                "qwen2".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "Qwen/Qwen1.5-1.8B-Chat".to_string()
            },
            quant,
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
                    quant.clone(),
                ),
                "gemma".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "google/gemma-2b-it".to_string()
            },
            quant,
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
                    quant.clone(),
                ),
                "mistral".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "mistralai/Mistral-7B-Instruct-v0.3".to_string()
            },
            quant,
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
                    quant.clone(),
                ),
                "yi".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "01-ai/Yi-6B-Chat".to_string()
            },
            quant,
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
                    quant.clone(),
                ),
                "stablelm".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "stabilityai/stablelm-zephyr-3b".to_string()
            },
            quant,
        ),
    }
}

pub fn hub_load_local_safetensors(
    path: &String,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    println!("{:}", Path::new(path).join(json_file).display());
    let jsfile = std::fs::File::open(Path::new(path).join(json_file))?;
    let json: serde_json::Value = serde_json::from_reader(&jsfile).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => panic!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = Vec::<std::path::PathBuf>::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(0, (Path::new(path).join(file)).into());
        }
    }
    Ok(safetensors_files)
}

pub fn new_device(ordinal: usize) -> Result<Device> {
    if cuda_is_available() {
        use candle_core::CudaDevice;
        let device = Device::Cuda(CudaDevice::new_with_stream(ordinal).unwrap());
        Ok(device)
    } else if metal_is_available() {
        Ok(Device::new_metal(ordinal)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}
