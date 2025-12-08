//! Storage traits and types for prompt cache backends.

use async_trait::async_trait;
use candle_core::Result;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// A serialized KV cache block for a single layer.
///
/// This contains the key and value tensors for one attention layer,
/// serialized to bytes for storage in cache backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheBlock {
    /// Serialized key tensor data
    pub key_data: Vec<u8>,
    /// Serialized value tensor data
    pub value_data: Vec<u8>,
    /// Shape of the key tensor (for deserialization)
    pub key_shape: Vec<usize>,
    /// Shape of the value tensor (for deserialization)
    pub value_shape: Vec<usize>,
    /// Data type identifier
    pub dtype: u8,
}

/// Metadata associated with a cached prefix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Timestamp when the cache entry was created
    pub created_at: SystemTime,
    /// Model fingerprint (hash of model config) for cache invalidation
    pub model_fingerprint: String,
    /// Number of tokens in the cached prefix
    pub prefix_length: usize,
    /// Number of KV cache blocks
    pub block_count: usize,
}

/// A cached prompt prefix with its KV cache blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPrefix {
    /// The KV cache blocks for all layers
    pub kv_blocks: Vec<KVCacheBlock>,
    /// Metadata about the cached prefix
    pub metadata: CacheMetadata,
}

/// Result of finding a cached prefix match.
#[derive(Debug, Clone)]
pub struct CachedPrefixMatch {
    /// The cached prefix data
    pub prefix: CachedPrefix,
    /// Number of tokens that were cached (matched prefix length)
    pub cached_tokens: usize,
    /// The hash of the matched prefix
    pub prefix_hash: Vec<u8>,
}

/// Cache statistics for monitoring and debugging.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of cached prefixes
    pub cached_prefixes: usize,
    /// Total size of cached data in bytes
    pub total_size_bytes: usize,
}

impl CacheStats {
    /// Calculate hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

/// Trait for prompt cache storage backends.
///
/// This trait abstracts over different storage implementations:
/// - In-memory (HashMap-based)
/// - Persistent (sled, Redis, etc.)
#[async_trait]
pub trait PromptCacheStorage: Send + Sync {
    /// Store KV cache blocks for a prompt prefix.
    ///
    /// # Arguments
    /// * `prefix_hash` - Hash of the prompt prefix tokens (cache key)
    /// * `kv_blocks` - KV cache blocks for all layers
    /// * `metadata` - Metadata about the cached prefix
    async fn store_prefix(
        &self,
        prefix_hash: &[u8],
        kv_blocks: &[KVCacheBlock],
        metadata: CacheMetadata,
    ) -> Result<()>;

    /// Retrieve KV cache blocks for a prompt prefix.
    ///
    /// # Arguments
    /// * `prefix_hash` - Hash of the prompt prefix tokens (cache key)
    ///
    /// # Returns
    /// `Some(CachedPrefix)` if found, `None` otherwise
    async fn get_prefix(&self, prefix_hash: &[u8]) -> Result<Option<CachedPrefix>>;

    /// Check if a prefix exists in the cache (faster than full retrieval).
    ///
    /// # Arguments
    /// * `prefix_hash` - Hash of the prompt prefix tokens (cache key)
    ///
    /// # Returns
    /// `true` if the prefix exists, `false` otherwise
    async fn has_prefix(&self, prefix_hash: &[u8]) -> Result<bool>;

    /// Remove a cached prefix (for invalidation).
    ///
    /// # Arguments
    /// * `prefix_hash` - Hash of the prompt prefix tokens (cache key)
    async fn remove_prefix(&self, prefix_hash: &[u8]) -> Result<()>;

    /// Clear all cached prefixes.
    async fn clear(&self) -> Result<()>;

    /// Get cache statistics.
    async fn stats(&self) -> Result<CacheStats>;
}
