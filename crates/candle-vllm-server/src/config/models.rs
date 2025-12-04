use serde::{Deserialize, Serialize};
use serde_yaml;
use std::fs;

/// Parameters that mirror CLI/server options for model execution.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelParams {
    pub dtype: Option<String>,
    pub quantization: Option<String>,
    pub block_size: Option<usize>,
    pub max_num_seqs: Option<usize>,
    pub kvcache_mem_gpu: Option<usize>,
    pub kvcache_mem_cpu: Option<usize>,
    pub prefill_chunk_size: Option<usize>,
    pub multithread: Option<bool>,
    pub device_ids: Option<Vec<usize>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<isize>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub isq: Option<String>,
}

/// A friendly model definition from `models.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub name: String,
    pub hf_id: Option<String>,
    pub local_path: Option<String>,
    pub weight_file: Option<String>,
    #[serde(default)]
    pub params: ModelParams,
    pub notes: Option<String>,
}

/// Registry of models plus optional idle unload policy.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelRegistryConfig {
    #[serde(default)]
    pub models: Vec<ModelProfile>,
    pub idle_unload_secs: Option<u64>,
    /// Default model to use if no model is specified via CLI arguments
    pub default_model: Option<String>,
}

impl ModelProfile {
    pub fn has_source(&self) -> bool {
        self.hf_id.is_some() || self.local_path.is_some()
    }
}

impl ModelRegistryConfig {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let cfg: ModelRegistryConfig = serde_yaml::from_str(&content)?;
        Ok(cfg)
    }
}
