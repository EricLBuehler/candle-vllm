//! In-memory prompt cache backend using HashMap.

use crate::prompt_cache::storage::{
    CacheMetadata, CacheStats, CachedPrefix, KVCacheBlock, PromptCacheStorage,
};
use candle_core::Result;
use parking_lot::RwLock;
use std::{
    collections::HashMap,
    sync::Arc,
    time::SystemTime,
};

/// In-memory cache backend using HashMap with LRU eviction.
///
/// This backend stores cached prefixes in memory using a HashMap.
/// When the maximum number of cached prefixes is reached, it uses
/// a simple LRU eviction policy based on access timestamps.
pub struct MemoryCacheBackend {
    /// The cache storage
    cache: Arc<RwLock<HashMap<Vec<u8>, CachedPrefix>>>,
    /// Access timestamps for LRU eviction
    access_times: Arc<RwLock<HashMap<Vec<u8>, SystemTime>>>,
    /// Maximum number of cached prefixes
    max_prefixes: Option<usize>,
    /// Statistics
    stats: Arc<RwLock<CacheStats>>,
}

impl MemoryCacheBackend {
    /// Create a new in-memory cache backend.
    ///
    /// # Arguments
    /// * `max_prefixes` - Maximum number of cached prefixes (None = unlimited)
    pub fn new(max_prefixes: Option<usize>) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            access_times: Arc::new(RwLock::new(HashMap::new())),
            max_prefixes,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Evict least recently used entries if we're over the limit.
    fn evict_lru_if_needed(&self) {
        if let Some(max) = self.max_prefixes {
            let mut cache = self.cache.write();
            let mut access_times = self.access_times.write();

            if cache.len() >= max {
                // Find the oldest entry
                let oldest = access_times
                    .iter()
                    .min_by_key(|(_, &time)| time)
                    .map(|(key, _)| key.clone());

                if let Some(oldest_key) = oldest {
                    cache.remove(&oldest_key);
                    access_times.remove(&oldest_key);
                }
            }
        }
    }

    /// Update access time for a key.
    fn update_access_time(&self, key: &[u8]) {
        let mut access_times = self.access_times.write();
        access_times.insert(key.to_vec(), SystemTime::now());
    }
}

#[async_trait::async_trait]
impl PromptCacheStorage for MemoryCacheBackend {
    async fn store_prefix(
        &self,
        prefix_hash: &[u8],
        kv_blocks: &[KVCacheBlock],
        metadata: CacheMetadata,
    ) -> Result<()> {
        // Evict if needed before storing
        self.evict_lru_if_needed();

        let cached_prefix = CachedPrefix {
            kv_blocks: kv_blocks.to_vec(),
            metadata,
        };

        let key = prefix_hash.to_vec();
        let mut cache = self.cache.write();
        let mut access_times = self.access_times.write();

        cache.insert(key.clone(), cached_prefix);
        access_times.insert(key, SystemTime::now());

        // Update stats
        let mut stats = self.stats.write();
        stats.cached_prefixes = cache.len();
        stats.total_size_bytes = cache
            .values()
            .map(|p| {
                p.kv_blocks
                    .iter()
                    .map(|b| b.key_data.len() + b.value_data.len())
                    .sum::<usize>()
            })
            .sum();

        Ok(())
    }

    async fn get_prefix(&self, prefix_hash: &[u8]) -> Result<Option<CachedPrefix>> {
        let key = prefix_hash.to_vec();
        let cache = self.cache.write();
        let mut stats = self.stats.write();

        match cache.get(&key).cloned() {
            Some(prefix) => {
                // Cache hit
                stats.hits += 1;
                self.update_access_time(&key);
                Ok(Some(prefix))
            }
            None => {
                // Cache miss
                stats.misses += 1;
                Ok(None)
            }
        }
    }

    async fn has_prefix(&self, prefix_hash: &[u8]) -> Result<bool> {
        let cache = self.cache.read();
        Ok(cache.contains_key(prefix_hash))
    }

    async fn remove_prefix(&self, prefix_hash: &[u8]) -> Result<()> {
        let mut cache = self.cache.write();
        let mut access_times = self.access_times.write();
        let mut stats = self.stats.write();

        if cache.remove(prefix_hash).is_some() {
            access_times.remove(prefix_hash);
            stats.cached_prefixes = cache.len();
            stats.total_size_bytes = cache
                .values()
                .map(|p| {
                    p.kv_blocks
                        .iter()
                        .map(|b| b.key_data.len() + b.value_data.len())
                        .sum::<usize>()
                })
                .sum();
        }

        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut cache = self.cache.write();
        let mut access_times = self.access_times.write();
        let mut stats = self.stats.write();

        cache.clear();
        access_times.clear();
        stats.cached_prefixes = 0;
        stats.total_size_bytes = 0;

        Ok(())
    }

    async fn stats(&self) -> Result<CacheStats> {
        let stats = self.stats.read();
        Ok(stats.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_store_and_get() {
        let backend = MemoryCacheBackend::new(None);
        let hash = b"test_prefix";
        let kv_blocks = vec![KVCacheBlock {
            key_data: vec![1, 2, 3],
            value_data: vec![4, 5, 6],
            key_shape: vec![1, 2],
            value_shape: vec![1, 2],
            dtype: 0,
        }];
        let metadata = CacheMetadata {
            created_at: SystemTime::now(),
            model_fingerprint: "test_model".to_string(),
            prefix_length: 10,
            block_count: 1,
        };

        backend.store_prefix(hash, &kv_blocks, metadata.clone()).await.unwrap();
        let result = backend.get_prefix(hash).await.unwrap();
        assert!(result.is_some());
        let cached = result.unwrap();
        assert_eq!(cached.metadata.prefix_length, 10);
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let backend = MemoryCacheBackend::new(Some(2));
        let hash1 = b"prefix1";
        let hash2 = b"prefix2";
        let hash3 = b"prefix3";

        let kv_blocks = vec![KVCacheBlock {
            key_data: vec![1],
            value_data: vec![2],
            key_shape: vec![1],
            value_shape: vec![1],
            dtype: 0,
        }];
        let metadata = CacheMetadata {
            created_at: SystemTime::now(),
            model_fingerprint: "test".to_string(),
            prefix_length: 1,
            block_count: 1,
        };

        backend.store_prefix(hash1, &kv_blocks, metadata.clone()).await.unwrap();
        backend.store_prefix(hash2, &kv_blocks, metadata.clone()).await.unwrap();
        backend.store_prefix(hash3, &kv_blocks, metadata).await.unwrap();

        // hash1 should be evicted (oldest)
        assert!(backend.get_prefix(hash1).await.unwrap().is_none());
        assert!(backend.get_prefix(hash2).await.unwrap().is_some());
        assert!(backend.get_prefix(hash3).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_stats() {
        let backend = MemoryCacheBackend::new(None);
        let hash = b"test";
        let kv_blocks = vec![KVCacheBlock {
            key_data: vec![1, 2, 3],
            value_data: vec![4, 5, 6],
            key_shape: vec![1],
            value_shape: vec![1],
            dtype: 0,
        }];
        let metadata = CacheMetadata {
            created_at: SystemTime::now(),
            model_fingerprint: "test".to_string(),
            prefix_length: 1,
            block_count: 1,
        };

        backend.store_prefix(hash, &kv_blocks, metadata).await.unwrap();
        backend.get_prefix(hash).await.unwrap(); // Hit
        backend.get_prefix(b"nonexistent").await.unwrap(); // Miss

        let stats = backend.stats().await.unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.cached_prefixes, 1);
    }
}
