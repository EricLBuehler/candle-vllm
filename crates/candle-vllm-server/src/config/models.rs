use serde::{Deserialize, Serialize};
use serde_yaml;
use std::collections::HashMap;
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

// =============================================================================
// Webhook Configuration
// =============================================================================

/// Authentication method for webhook calls.
///
/// Supports multiple authentication patterns for flexibility:
/// - Bearer tokens (OAuth 2.0 / Supabase Edge Functions)
/// - API keys in custom headers
/// - HMAC signatures for payload verification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebhookAuth {
    /// Bearer token in Authorization header (OAuth 2.0 / Supabase Edge Functions)
    /// Token value supports ${ENV_VAR} interpolation.
    Bearer {
        /// The bearer token value (supports ${ENV_VAR} syntax)
        token: String,
    },

    /// API key in a custom header (e.g., X-API-Key, X-Finlight-Key)
    ApiKey {
        /// The header name to use (e.g., "X-API-Key")
        header: String,
        /// The API key value (supports ${ENV_VAR} syntax)
        key: String,
    },

    /// HMAC signature for payload verification.
    /// The signature is computed over the JSON payload and included in a header.
    Hmac {
        /// The shared secret for HMAC (supports ${ENV_VAR} syntax)
        secret: String,
        /// Hash algorithm: "sha256" or "sha512" (default: "sha256")
        #[serde(default = "default_hmac_algorithm")]
        algorithm: String,
        /// Header name for the signature (default: "X-Signature-256")
        #[serde(default = "default_hmac_header")]
        header: String,
    },

    /// No authentication
    None,
}

fn default_hmac_algorithm() -> String {
    "sha256".to_string()
}

fn default_hmac_header() -> String {
    "X-Signature-256".to_string()
}

impl Default for WebhookAuth {
    fn default() -> Self {
        Self::None
    }
}

/// Webhook configuration for callback notifications.
///
/// Webhooks can be triggered when:
/// - A client disconnects before receiving a response (on_disconnect)
/// - Any request completes (on_complete)
///
/// Example configurations:
///
/// ```yaml
/// # Simple Bearer token (Supabase Edge Functions)
/// webhook:
///   url: "https://your-project.supabase.co/functions/v1/callback"
///   enabled: true
///   on_disconnect: true
///   auth:
///     type: bearer
///     token: "${SUPABASE_SERVICE_ROLE_KEY}"
///
/// # HMAC signature (highest security)
/// webhook:
///   url: "https://secure.example.com/webhook"
///   auth:
///     type: hmac
///     secret: "${WEBHOOK_SECRET}"
///     algorithm: "sha256"
///     header: "X-Signature-256"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// The webhook endpoint URL (supports ${ENV_VAR} syntax)
    pub url: Option<String>,

    /// Master switch to enable/disable webhooks
    #[serde(default)]
    pub enabled: bool,

    /// Fire webhook when client disconnects before receiving response
    #[serde(default = "default_on_disconnect")]
    pub on_disconnect: bool,

    /// Fire webhook on every completion (in addition to normal response)
    #[serde(default)]
    pub on_complete: bool,

    /// HTTP timeout in seconds (default: 30)
    #[serde(default = "default_webhook_timeout")]
    pub timeout_secs: u64,

    /// Number of retry attempts on failure (default: 3)
    #[serde(default = "default_retry_count")]
    pub retry_count: u32,

    /// Delay between retries in milliseconds (default: 1000)
    /// Uses exponential backoff: delay * 2^(attempt-1)
    #[serde(default = "default_retry_delay")]
    pub retry_delay_ms: u64,

    /// Authentication configuration
    #[serde(default)]
    pub auth: Option<WebhookAuth>,

    /// Additional custom headers to include in webhook requests.
    /// Values support ${ENV_VAR} syntax.
    #[serde(default)]
    pub headers: HashMap<String, String>,

    /// Include HMAC signature even when using other auth methods.
    /// Useful for adding an extra layer of payload verification.
    #[serde(default)]
    pub sign_payload: bool,

    /// Secret for payload signing (if sign_payload=true and auth != Hmac)
    /// Supports ${ENV_VAR} syntax.
    pub signing_secret: Option<String>,
}

fn default_on_disconnect() -> bool {
    true
}

fn default_webhook_timeout() -> u64 {
    30
}

fn default_retry_count() -> u32 {
    3
}

fn default_retry_delay() -> u64 {
    1000
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            url: None,
            enabled: false,
            on_disconnect: true,
            on_complete: false,
            timeout_secs: default_webhook_timeout(),
            retry_count: default_retry_count(),
            retry_delay_ms: default_retry_delay(),
            auth: None,
            headers: HashMap::new(),
            sign_payload: false,
            signing_secret: None,
        }
    }
}

// =============================================================================
// Backend Configuration
// =============================================================================

/// Queue backend configuration for models.yaml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueBackendConfig {
    /// Backend type: "memory", "postgres", "sqlite", "surrealdb", "yaque"
    #[serde(default = "default_queue_backend")]
    pub backend: String,

    /// Enable persistence (where supported)
    #[serde(default)]
    pub persistence: bool,

    /// PostgreSQL connection URL (if backend = "postgres")
    pub postgres_url: Option<String>,

    /// SQLite database path (if backend = "sqlite")
    pub sqlite_path: Option<String>,

    /// SurrealDB database path (if backend = "surrealdb")
    pub surreal_path: Option<String>,

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
            sqlite_path: None,
            surreal_path: None,
            yaque_dir: None,
        }
    }
}

/// Mailbox backend configuration for models.yaml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailboxBackendConfig {
    /// Backend type: "memory", "postgres", "sqlite", "surrealdb"
    #[serde(default = "default_mailbox_backend")]
    pub backend: String,

    /// Result retention time in seconds (default: 3600 = 1 hour)
    #[serde(default = "default_retention_secs")]
    pub retention_secs: u64,

    /// PostgreSQL connection URL (if backend = "postgres")
    pub postgres_url: Option<String>,

    /// SQLite database path (if backend = "sqlite")
    pub sqlite_path: Option<String>,

    /// SurrealDB database path (if backend = "surrealdb")
    pub surreal_path: Option<String>,

    /// Webhook configuration for this mailbox
    #[serde(default)]
    pub webhook: Option<WebhookConfig>,
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
            sqlite_path: None,
            surreal_path: None,
            webhook: None,
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
    pub parking_lot: Option<ParkingLotConfig>,
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

    pub fn parking_lot_config(&self) -> Option<&ParkingLotConfig> {
        self.parking_lot
            .as_ref()
            .or_else(|| self.params.parking_lot.as_ref())
    }
}

impl ModelRegistryConfig {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let cfg: ModelRegistryConfig = serde_yaml::from_str(&content)?;
        Ok(cfg)
    }
}
