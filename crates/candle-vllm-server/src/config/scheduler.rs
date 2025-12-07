//! Scheduler configuration for the parking-lot scheduler.
//!
//! This module provides configuration loading and validation for
//! the prometheus_parking_lot scheduler integration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Root configuration for the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Named pool configurations
    pub pools: HashMap<String, PoolConfig>,

    /// Resource configuration (optional)
    #[serde(default)]
    pub resource_config: ResourceConfig,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        let mut pools = HashMap::new();
        pools.insert("default".to_string(), PoolConfig::default());
        Self {
            pools,
            resource_config: ResourceConfig::default(),
        }
    }
}

impl SchedulerConfig {
    /// Load configuration from a JSON or YAML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path_ref = path.as_ref();
        let content = std::fs::read_to_string(path_ref).map_err(|e| ConfigError::Io {
            path: path_ref.to_string_lossy().to_string(),
            source: e,
        })?;

        // Try to determine format from extension
        let path_str = path_ref.to_string_lossy();
        if path_str.ends_with(".yaml") || path_str.ends_with(".yml") {
            serde_yaml::from_str(&content).map_err(|e| ConfigError::ParseYaml {
                path: path_str.to_string(),
                source: e,
            })
        } else {
            // Assume JSON
            serde_json::from_str(&content).map_err(|e| ConfigError::ParseJson {
                path: path_str.to_string(),
                source: e,
            })
        }
    }

    /// Load configuration from environment variable or file path.
    ///
    /// Checks in order:
    /// 1. `CANDLE_VLLM_SCHEDULER_CONFIG` environment variable (JSON content or file path)
    /// 2. `--scheduler-config` CLI argument (if provided)
    /// 3. `scheduler-config.json` in current directory
    /// 4. Default configuration
    pub fn load(cli_path: Option<&str>) -> Self {
        // Try environment variable first
        if let Ok(env_value) = std::env::var("CANDLE_VLLM_SCHEDULER_CONFIG") {
            // Check if it's a file path or inline JSON
            if env_value.trim().starts_with('{') {
                if let Ok(config) = serde_json::from_str(&env_value) {
                    tracing::info!("Loaded scheduler config from CANDLE_VLLM_SCHEDULER_CONFIG env");
                    return config;
                }
            } else if Path::new(&env_value).exists() {
                if let Ok(config) = Self::from_file(&env_value) {
                    tracing::info!("Loaded scheduler config from {}", env_value);
                    return config;
                }
            }
        }

        // Try CLI argument
        if let Some(path) = cli_path {
            if let Ok(config) = Self::from_file(path) {
                tracing::info!("Loaded scheduler config from {}", path);
                return config;
            }
        }

        // Try default location
        if Path::new("scheduler-config.json").exists() {
            if let Ok(config) = Self::from_file("scheduler-config.json") {
                tracing::info!("Loaded scheduler config from scheduler-config.json");
                return config;
            }
        }

        // Fall back to default
        tracing::info!("Using default scheduler configuration");
        Self::default()
    }

    /// Get the pool configuration for a given name, or the default pool.
    pub fn get_pool(&self, name: &str) -> &PoolConfig {
        self.pools
            .get(name)
            .or_else(|| self.pools.get("default"))
            .expect("default pool should always exist")
    }
}

/// Configuration for a single resource pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Maximum resource units (GPU blocks) the pool can use
    pub max_units: usize,

    /// Maximum queue depth before rejecting requests
    pub max_queue_depth: usize,

    /// Default timeout in seconds for queued requests
    pub default_timeout_secs: u64,

    /// Queue backend configuration
    #[serde(default)]
    pub queue: QueueConfig,

    /// Mailbox configuration
    #[serde(default)]
    pub mailbox: MailboxConfig,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_units: 16384,
            max_queue_depth: 1000,
            default_timeout_secs: 120,
            queue: QueueConfig::default(),
            mailbox: MailboxConfig::default(),
        }
    }
}

/// Queue backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueueConfig {
    /// In-memory queue (default, fastest, no persistence)
    InMemory,

    /// Yaque file-backed queue (for desktop/Tauri apps)
    Yaque { path: String, stream: String },

    /// PostgreSQL with pgmq extension
    PostgresPgmq { queue_name: String },

    /// PostgreSQL custom table
    Postgres { table: String },
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self::InMemory
    }
}

/// Mailbox configuration for result storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailboxConfig {
    /// Storage backend
    #[serde(default)]
    pub storage: MailboxStorageConfig,

    /// Optional notifier configuration
    #[serde(default)]
    pub notifier: Option<MailboxNotifierConfig>,
}

impl Default for MailboxConfig {
    fn default() -> Self {
        Self {
            storage: MailboxStorageConfig::default(),
            notifier: None,
        }
    }
}

/// Mailbox storage backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MailboxStorageConfig {
    /// In-memory storage (default)
    InMemory,

    /// PostgreSQL storage
    Postgres { table: String },
}

impl Default for MailboxStorageConfig {
    fn default() -> Self {
        Self::InMemory
    }
}

/// Mailbox notifier configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MailboxNotifierConfig {
    /// HTTP callback notifier
    Http {
        base_url: String,
        #[serde(default)]
        auth_header: Option<String>,
    },
}

/// Resource configuration for cost calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Block size for KV-cache (tokens per block)
    #[serde(default = "default_block_size")]
    pub block_size: usize,

    /// Cost model to use
    #[serde(default)]
    pub cost_model: CostModel,
}

fn default_block_size() -> usize {
    16
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            cost_model: CostModel::KvCacheBlocks,
        }
    }
}

/// Cost model for resource calculation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CostModel {
    /// Cost based on KV-cache blocks (default)
    #[default]
    KvCacheBlocks,

    /// Cost based on VRAM bytes
    VramBytes,

    /// Fixed cost per request
    FixedPerRequest { units: usize },
}

/// Configuration errors.
#[derive(Debug)]
pub enum ConfigError {
    /// I/O error reading config file
    Io {
        path: String,
        source: std::io::Error,
    },

    /// Parse error in JSON config file
    ParseJson {
        path: String,
        source: serde_json::Error,
    },

    /// Parse error in YAML config file
    ParseYaml {
        path: String,
        source: serde_yaml::Error,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "failed to read config file '{}': {}", path, source)
            }
            Self::ParseJson { path, source } => {
                write!(f, "failed to parse JSON config file '{}': {}", path, source)
            }
            Self::ParseYaml { path, source } => {
                write!(f, "failed to parse YAML config file '{}': {}", path, source)
            }
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::ParseJson { source, .. } => Some(source),
            Self::ParseYaml { source, .. } => Some(source),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SchedulerConfig::default();
        assert!(config.pools.contains_key("default"));
        assert_eq!(config.get_pool("default").max_units, 16384);
    }

    #[test]
    fn test_parse_json() {
        let json = r#"{
            "pools": {
                "default": {
                    "max_units": 8192,
                    "max_queue_depth": 500,
                    "default_timeout_secs": 60,
                    "queue": { "type": "in_memory" },
                    "mailbox": { "storage": { "type": "in_memory" } }
                }
            },
            "resource_config": {
                "block_size": 32
            }
        }"#;

        let config: SchedulerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.get_pool("default").max_units, 8192);
        assert_eq!(config.resource_config.block_size, 32);
    }
}
