#![warn(clippy::cast_lossless)]
use candle::utils::{cuda_is_available, metal_is_available};
use candle::{Device, Result};
use candle_core as candle;
use clap::Subcommand;
use openai::pipelines::pipeline::DefaultLoader;
use std::fmt::Display;
use std::path::Path;
use tracing::warn;
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
        top_p: Option<f32>,

        #[arg(long)]
        top_k: Option<isize>,

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
        top_p: Option<f32>,

        #[arg(long)]
        top_k: Option<isize>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,

        #[arg(long, default_value_t = false)]
        thinking: bool,
    },

    /// Select the phi2 model (default 2.7b).
    Phi2 {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        top_p: Option<f32>,

        #[arg(long)]
        top_k: Option<isize>,

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
        top_p: Option<f32>,

        #[arg(long)]
        top_k: Option<isize>,

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
        top_p: Option<f32>,

        #[arg(long)]
        top_k: Option<isize>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    Qwen3 {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        top_p: Option<f32>,

        #[arg(long)]
        top_k: Option<isize>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,

        #[arg(long, default_value_t = false)]
        thinking: bool,
    },

    /// Select the gemma model (default 2b).
    Gemma {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        top_p: Option<f32>,

        #[arg(long)]
        top_k: Option<isize>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    Gemma3 {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        top_p: Option<f32>,

        #[arg(long)]
        top_k: Option<isize>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,

        #[arg(long, default_value_t = false)]
        thinking: bool,
    },

    /// Select the mistral model (default 7b).
    Mistral {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        top_p: Option<f32>,

        #[arg(long)]
        top_k: Option<isize>,

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
        top_p: Option<f32>,

        #[arg(long)]
        top_k: Option<isize>,

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

    GLM4 {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        top_p: Option<f32>,

        #[arg(long)]
        top_k: Option<isize>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    /// Select the deepseek model (default deepseek-v2-lite-chat).
    DeepSeek {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        top_p: Option<f32>,

        #[arg(long)]
        top_k: Option<isize>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,

        //the number of experts offloaded per rank,
        //suppose there are 256 experts in total which split into 8 devices (rank 8),
        //each rank has 32 experts, num-experts-offload-per-rank=16 means
        //half of the experts can be offloaded to cpu memory
        #[arg(long)]
        num_experts_offload_per_rank: Option<usize>,

        #[arg(long, default_value_t = false)]
        thinking: bool,
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
            ModelSelected::Qwen3 {
                repeat_last_n: _,
                temperature: _,
                top_k: _,
                top_p: _,
                penalty: _,
                max_gen_tokens: _,
                quant: _,
                thinking: _,
            } => write!(f, "qwen3"),
            ModelSelected::Gemma { .. } => write!(f, "gemma"),
            ModelSelected::Gemma3 { .. } => write!(f, "gemma3"),
            ModelSelected::Mistral { .. } => write!(f, "mistral"),
            ModelSelected::Yi { .. } => write!(f, "yi"),
            ModelSelected::StableLM { .. } => write!(f, "stablelm"),
            ModelSelected::GLM4 { .. } => write!(f, "glm4"),
            ModelSelected::DeepSeek { .. } => write!(f, "deepseek"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpecificConfig {
    repeat_last_n: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<isize>,
    top_p: Option<f32>,
    penalty: Option<f32>,
    max_gen_tokens: Option<usize>,
    quant: Option<String>,
    num_experts_offload_per_rank: Option<usize>,
    thinking: bool,
}

impl SpecificConfig {
    pub fn new(
        repeat_last_n: Option<usize>,
        temperature: Option<f32>,
        top_k: Option<isize>,
        top_p: Option<f32>,
        penalty: Option<f32>,
        max_gen_tokens: Option<usize>,
        quant: Option<String>,
        num_experts_offload_per_rank: Option<usize>,
        thinking: bool,
    ) -> Self {
        Self {
            repeat_last_n,
            temperature,
            top_k,
            top_p,
            penalty,
            max_gen_tokens,
            quant,
            num_experts_offload_per_rank,
            thinking,
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
                    None,
                    false,
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
            top_k,
            top_p,
            penalty,
            max_gen_tokens,
            quant,
            thinking,
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
                    None,
                    thinking,
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
                    None,
                    false,
                ),
                "phi2".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "microsoft/phi-2".to_string()
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
                    None,
                    false,
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
                    None,
                    false,
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
        ModelSelected::Qwen3 {
            repeat_last_n,
            temperature,
            top_k,
            top_p,
            penalty,
            max_gen_tokens,
            quant,
            thinking,
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
                    None,
                    thinking,
                ),
                "qwen3".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "Qwen/Qwen3-8B".to_string()
            },
            quant,
        ),
        ModelSelected::Gemma {
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
                    None,
                    false,
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
        ModelSelected::Gemma3 {
            repeat_last_n,
            temperature,
            top_k,
            top_p,
            penalty,
            max_gen_tokens,
            quant,
            thinking,
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
                    None,
                    thinking,
                ),
                "gemma3".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "google/gemma-3-4b-it".to_string()
            },
            quant,
        ),
        ModelSelected::Mistral {
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
                    None,
                    false,
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
                    None,
                    false,
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
                    None,
                    false,
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
        ModelSelected::GLM4 {
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
                    None,
                    false,
                ),
                "glm4".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "ZhipuAI/GLM-4-9B-0414".to_string()
            },
            quant,
        ),
        ModelSelected::DeepSeek {
            repeat_last_n,
            temperature,
            top_k,
            top_p,
            penalty,
            max_gen_tokens,
            quant,
            num_experts_offload_per_rank,
            thinking,
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
                    num_experts_offload_per_rank,
                    thinking,
                ),
                "deepseek".to_string(),
            )),
            if let Some(model_id) = model_id {
                model_id
            } else {
                "deepseek-ai/DeepSeek-V2-Lite-Chat".to_string()
            },
            quant,
        ),
    }
}

pub fn hub_load_local_safetensors(
    path: &String,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    tracing::info!("{:}", Path::new(path).join(json_file).display());
    let jsfile = std::fs::File::open(Path::new(path).join(json_file))?;
    let json: serde_json::Value = serde_json::from_reader(&jsfile).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => panic!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file);
        }
    }
    let safetensors_files: Vec<_> = safetensors_files
        .into_iter()
        .map(|v| Path::new(path).join(v))
        .collect();
    Ok(safetensors_files)
}

pub fn new_device(ordinal: usize) -> Result<Device> {
    if cuda_is_available() {
        use candle_core::CudaDevice;
        let device = Device::Cuda(CudaDevice::new_with_stream(ordinal)?);
        Ok(device)
    } else if metal_is_available() {
        Ok(Device::new_metal(ordinal)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            warn!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            warn!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}
