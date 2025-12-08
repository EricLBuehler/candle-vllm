use serde::{Deserialize, Serialize};
use serde_yaml;
use std::fs;

/// Worker pool configuration for parking-lot scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerPoolConfig {
    /// Number of dedicated worker threads for CPU/GPU-bound inference (default: num_cpus)
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,

    /// Maximum queue depth before rejecting requests (default: 1000)
    #[serde(default = "default_max_queue_depth_pool")]
    pub max_queue_depth: usize,

    /// Request timeout in seconds (default: 120)
    #[serde(default = "default_timeout_secs_pool")]
    pub timeout_secs: u64,
}

fn default_worker_threads() -> usize {
    num_cpus::get()
}

fn default_max_queue_depth_pool() -> usize {
    1000
}

fn default_timeout_secs_pool() -> u64 {
    120
}

impl Default for WorkerPoolConfig {
    fn default() -> Self {
        Self {
            worker_threads: default_worker_threads(),
            max_queue_depth: default_max_queue_depth_pool(),
            timeout_secs: default_timeout_secs_pool(),
        }
    }
}

/// Resource limits configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitsConfig {
    /// Maximum resource units (GPU KV-cache blocks), null = auto-derive
    pub max_units: Option<usize>,

    /// Maximum number of queued requests before rejection
    #[serde(default = "default_max_queue_depth")]
    pub max_queue_depth: usize,

    /// Request timeout in seconds
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
}

fn default_max_queue_depth() -> usize {
    1000
}

fn default_timeout_secs() -> u64 {
    120
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_units: None,
            max_queue_depth: default_max_queue_depth(),
            timeout_secs: default_timeout_secs(),
        }
    }
}

/// Queue backend configuration for models.yaml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueBackendConfig {
    /// Backend type: "memory", "postgres", "yaque"
    #[serde(default = "default_queue_backend")]
    pub backend: String,

    /// Enable persistence
    #[serde(default)]
    pub persistence: bool,

    /// PostgreSQL connection URL (if backend = "postgres")
    pub postgres_url: Option<String>,

    /// Yaque directory path (if backend = "yaque")
    pub yaque_dir: Option<String>,
}

fn default_queue_backend() -> String {
    "memory".to_string()
}

impl Default for QueueBackendConfig {
    fn default() -> Self {
        Self {
            backend: default_queue_backend(),
            persistence: false,
            postgres_url: None,
            yaque_dir: None,
        }
    }
}

/// Mailbox backend configuration for models.yaml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailboxBackendConfig {
    /// Backend type: "memory", "postgres"
    #[serde(default = "default_mailbox_backend")]
    pub backend: String,

    /// Result retention time in seconds
    #[serde(default = "default_retention_secs")]
    pub retention_secs: u64,

    /// PostgreSQL connection URL (if backend = "postgres")
    pub postgres_url: Option<String>,
}

fn default_mailbox_backend() -> String {
    "memory".to_string()
}

fn default_retention_secs() -> u64 {
    3600
}

impl Default for MailboxBackendConfig {
    fn default() -> Self {
        Self {
            backend: default_mailbox_backend(),
            retention_secs: default_retention_secs(),
            postgres_url: None,
        }
    }
}

/// Complete parking lot configuration from models.yaml.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParkingLotConfig {
    #[serde(default)]
    pub pool: WorkerPoolConfig,

    #[serde(default)]
    pub limits: LimitsConfig,

    #[serde(default)]
    pub queue: QueueBackendConfig,

    #[serde(default)]
    pub mailbox: MailboxBackendConfig,
}

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

    /// Per-model parking lot overrides
    pub parking_lot: Option<ParkingLotConfig>,
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
    /// Global parking lot scheduler configuration
    pub parking_lot: Option<ParkingLotConfig>,
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
