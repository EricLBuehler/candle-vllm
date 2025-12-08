//! Sled-based persistent prompt cache backend.

#[cfg(feature = "prompt-cache-sled")]
use crate::prompt_cache::storage::{
    CacheMetadata, CacheStats, CachedPrefix, KVCacheBlock, PromptCacheStorage,
};
#[cfg(feature = "prompt-cache-sled")]
use serde_json;
#[cfg(feature = "prompt-cache-sled")]
use candle_core::Result;
#[cfg(feature = "prompt-cache-sled")]
use parking_lot::RwLock;
#[cfg(feature = "prompt-cache-sled")]
use std::{
    path::PathBuf,
    sync::Arc,
    time::SystemTime,
};

/// Sled-based persistent cache backend.
///
/// This backend uses sled, an embedded database, for persistent storage
/// of cached prefixes. The cache survives process restarts and provides
/// ACID transactions.
#[cfg(feature = "prompt-cache-sled")]
pub struct SledCacheBackend {
    /// The sled database
    db: sled::Db,
    /// Statistics
    stats: Arc<RwLock<CacheStats>>,
}

#[cfg(feature = "prompt-cache-sled")]
impl SledCacheBackend {
    /// Create a new sled cache backend.
    ///
    /// # Arguments
    /// * `cache_path` - Path to the sled database directory
    ///
    /// # Errors
    /// Returns an error if the database cannot be opened
    pub fn new(cache_path: &str) -> Result<Self> {
        let path = PathBuf::from(cache_path);
        std::fs::create_dir_all(&path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create cache directory: {}", e)))?;

        let db = sled::open(&path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to open sled database: {}", e)))?;

        let stats = Arc::new(RwLock::new(CacheStats::default()));
        let backend = Self { db, stats };

        // Initialize stats from database
        backend.update_stats();

        Ok(backend)
    }

    /// Update statistics from the database.
    fn update_stats(&self) {
        let mut stats = self.stats.write();
        let mut count = 0;
        let mut total_size = 0;

        for result in self.db.iter() {
            if let Ok((_, value)) = result {
                count += 1;
                total_size += value.len();
            }
        }

        stats.cached_prefixes = count;
        stats.total_size_bytes = total_size;
    }
}

#[cfg(feature = "prompt-cache-sled")]
#[async_trait::async_trait]
impl PromptCacheStorage for SledCacheBackend {
    async fn store_prefix(
        &self,
        prefix_hash: &[u8],
        kv_blocks: &[KVCacheBlock],
        metadata: CacheMetadata,
    ) -> Result<()> {
        let cached_prefix = CachedPrefix {
            kv_blocks: kv_blocks.to_vec(),
            metadata,
        };

        let serialized = serde_json::to_vec(&cached_prefix)
            .map_err(|e| candle_core::Error::Msg(format!("Serialization failed: {}", e)))?;

        self.db
            .insert(prefix_hash, serialized.as_slice())
            .map_err(|e| candle_core::Error::Msg(format!("Failed to store prefix: {}", e)))?;

        // Flush to ensure durability
        self.db
            .flush_async()
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to flush database: {}", e)))?;

        // Update stats
        self.update_stats();

        Ok(())
    }

    async fn get_prefix(&self, prefix_hash: &[u8]) -> Result<Option<CachedPrefix>> {
        let mut stats = self.stats.write();

        match self.db.get(prefix_hash) {
            Ok(Some(serialized)) => {
                let cached_prefix: CachedPrefix = serde_json::from_slice(&serialized)
                    .map_err(|e| candle_core::Error::Msg(format!("Deserialization failed: {}", e)))?;

                stats.hits += 1;
                Ok(Some(cached_prefix))
            }
            Ok(None) => {
                stats.misses += 1;
                Ok(None)
            }
            Err(e) => Err(candle_core::Error::Msg(format!("Database error: {}", e))),
        }
    }

    async fn has_prefix(&self, prefix_hash: &[u8]) -> Result<bool> {
        match self.db.contains_key(prefix_hash) {
            Ok(exists) => Ok(exists),
            Err(e) => Err(candle_core::Error::Msg(format!("Database error: {}", e))),
        }
    }

    async fn remove_prefix(&self, prefix_hash: &[u8]) -> Result<()> {
        self.db
            .remove(prefix_hash)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to remove prefix: {}", e)))?;

        self.update_stats();

        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        self.db
            .clear()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to clear database: {}", e)))?;

        let mut stats = self.stats.write();
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
#[cfg(feature = "prompt-cache-sled")]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_store_and_get() {
        let temp_dir = TempDir::new().unwrap();
        let backend = SledCacheBackend::new(temp_dir.path().to_str().unwrap()).unwrap();

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
    async fn test_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        // Store in first instance
        {
            let backend = SledCacheBackend::new(path).unwrap();
            let hash = b"persistent_test";
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
            backend.store_prefix(hash, &kv_blocks, metadata).await.unwrap();
        }

        // Retrieve from new instance (simulating restart)
        {
            let backend = SledCacheBackend::new(path).unwrap();
            let result = backend.get_prefix(b"persistent_test").await.unwrap();
            assert!(result.is_some());
        }
    }
}
