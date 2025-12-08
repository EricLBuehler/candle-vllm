//! Configuration for prompt caching.

use serde::{Deserialize, Serialize};

/// Configuration for prompt prefix caching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCacheConfig {
    /// Enable prompt caching
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Cache storage backend type
    #[serde(default = "default_backend")]
    pub backend: CacheBackend,

    /// Maximum cached prefixes (for in-memory backend)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_cached_prefixes: Option<usize>,

    /// Cache storage path (for sled backend)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_path: Option<String>,

    /// Redis URL (for Redis backend)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redis_url: Option<String>,

    /// TTL in seconds (for Redis backend)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,

    /// Minimum prefix length to cache (in tokens)
    #[serde(default = "default_min_prefix_length")]
    pub min_prefix_length: usize,

    /// Model fingerprint for cache invalidation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_fingerprint: Option<String>,
}

fn default_enabled() -> bool {
    false
}

fn default_backend() -> CacheBackend {
    CacheBackend::Memory
}

fn default_min_prefix_length() -> usize {
    16
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: CacheBackend::Memory,
            max_cached_prefixes: None,
            cache_path: None,
            redis_url: None,
            ttl_seconds: None,
            min_prefix_length: 16,
            model_fingerprint: None,
        }
    }
}

/// Cache storage backend type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CacheBackend {
    /// In-memory HashMap-based cache (fastest, not persistent)
    #[serde(rename = "memory")]
    Memory,
    /// Sled embedded database (persistent, file-based)
    #[serde(rename = "sled")]
    Sled,
    /// Redis distributed cache (persistent, network-based)
    #[serde(rename = "redis")]
    Redis,
}

impl std::str::FromStr for CacheBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "memory" => Ok(CacheBackend::Memory),
            "sled" => Ok(CacheBackend::Sled),
            "redis" => Ok(CacheBackend::Redis),
            _ => Err(format!("Unknown cache backend: {}", s)),
        }
    }
}
