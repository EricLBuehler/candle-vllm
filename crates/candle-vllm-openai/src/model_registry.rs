use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelAlias {
    pub name: String,
    pub model_id: Option<String>,
    pub weight_path: Option<String>,
    pub weight_file: Option<String>,
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

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ModelRegistry {
    pub models: Vec<ModelAlias>,
    pub idle_unload_secs: Option<u64>,
}

impl ModelRegistry {
    pub fn load<P: AsRef<Path>>(path: P) -> Option<Self> {
        let content = fs::read_to_string(path).ok()?;
        serde_yaml::from_str(&content).ok()
    }

    pub fn find(&self, name: &str) -> Option<ModelAlias> {
        self.models.iter().find(|m| m.name == name).cloned()
    }

    pub fn list_names(&self) -> Vec<String> {
        self.models.iter().map(|m| m.name.clone()).collect()
    }
}
