use crate::config::ModelRegistryConfig;
use candle_vllm_openai::model_registry::{ModelAlias, ModelRegistry};
use candle_vllm_responses::status::{ModelLifecycleStatus, ModelStatus};
use serde::Serialize;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct ModelsState {
    pub registry: Option<ModelRegistry>,
    pub active: Arc<Mutex<Option<LoadedModel>>>,
    pub idle_unload: Option<Duration>,
    pub validation: HashMap<String, String>,
}

#[derive(Clone)]
pub struct LoadedModel {
    pub name: String,
    pub last_used: Instant,
}

impl ModelsState {
    pub fn new(
        registry: Option<ModelRegistry>,
        validation: HashMap<String, String>,
        idle_unload: Option<Duration>,
    ) -> Self {
        Self {
            registry,
            active: Arc::new(Mutex::new(None)),
            idle_unload,
            validation,
        }
    }

    pub fn list(&self) -> Vec<ModelInfo> {
        self.registry
            .as_ref()
            .map(|r| {
                r.models
                    .iter()
                    .map(|m| ModelInfo {
                        id: m.name.clone(),
                        created: 0,
                        object: "model".to_string(),
                        owned_by: "owner".to_string(),
                        status: self
                            .validation
                            .get(&m.name)
                            .cloned()
                            .unwrap_or_else(|| "valid".to_string()),
                    })
                    .collect()
            })
            .unwrap_or_else(|| {
                vec![ModelInfo {
                    id: "default".to_string(),
                    created: 0,
                    object: "model".to_string(),
                    owned_by: "owner".to_string(),
                    status: "valid".to_string(),
                }]
            })
    }

    pub fn resolve(&self, name: &str) -> Option<ModelAlias> {
        self.registry.as_ref()?.find(name)
    }

    pub async fn set_active(&self, name: String) {
        let mut active = self.active.lock().await;
        *active = Some(LoadedModel {
            name,
            last_used: Instant::now(),
        });
    }

    pub async fn status(&self) -> ModelStatus {
        let active = self.active.lock().await;
        if let Some(current) = active.as_ref() {
            ModelStatus {
                active_model: Some(current.name.clone()),
                status: ModelLifecycleStatus::Ready,
                last_error: None,
                in_flight_requests: 0,
                switch_requested_at: None,
                queue_lengths: std::collections::HashMap::new(),
            }
        } else {
            ModelStatus {
                active_model: None,
                status: ModelLifecycleStatus::Idle,
                last_error: None,
                in_flight_requests: 0,
                switch_requested_at: None,
                queue_lengths: std::collections::HashMap::new(),
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub created: u64,
    pub status: String,
}

use std::sync::Arc;

pub fn to_model_registry(cfg: &ModelRegistryConfig) -> ModelRegistry {
    let models = cfg
        .models
        .iter()
        .map(|p| ModelAlias {
            name: p.name.clone(),
            model_id: p.hf_id.clone(),
            weight_path: p.local_path.clone(),
            weight_file: p.weight_file.clone(),
            dtype: p.params.dtype.clone(),
            block_size: p.params.block_size,
            max_num_seqs: p.params.max_num_seqs,
            kvcache_mem_gpu: p.params.kvcache_mem_gpu,
            kvcache_mem_cpu: p.params.kvcache_mem_cpu,
            prefill_chunk_size: p.params.prefill_chunk_size,
            multithread: p.params.multithread,
            quantization: p.params.quantization.clone(),
            device_ids: p.params.device_ids.clone(),
            temperature: p.params.temperature,
            top_p: p.params.top_p,
            top_k: p.params.top_k,
            frequency_penalty: p.params.frequency_penalty,
            presence_penalty: p.params.presence_penalty,
            isq: p.params.isq.clone(),
        })
        .collect();
    ModelRegistry {
        models,
        idle_unload_secs: cfg.idle_unload_secs,
    }
}
