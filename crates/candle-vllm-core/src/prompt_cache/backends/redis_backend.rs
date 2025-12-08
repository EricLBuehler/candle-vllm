//! Redis backend for prompt cache storage.
//!
//! This backend uses Redis for distributed caching, allowing cache
//! sharing across multiple instances. Supports TTL-based expiration.
#[cfg(feature = "prompt-cache-redis")]
use crate::prompt_cache::storage::{
    CacheMetadata, CacheStats, CachedPrefix, KVCacheBlock, PromptCacheStorage,
};
#[cfg(feature = "prompt-cache-redis")]
use serde_json;
use candle_core::Result;
use parking_lot::RwLock;
use std::sync::Arc;

/// Redis-based prompt cache backend.
///
/// This backend stores cached prefixes in Redis, allowing cache sharing
/// across multiple server instances. Supports TTL-based expiration.
#[cfg(feature = "prompt-cache-redis")]
pub struct RedisCacheBackend {
    /// Redis client
    client: redis::Client,
    /// TTL in seconds (None = no expiration)
    ttl_seconds: Option<u64>,
    /// Statistics
    stats: Arc<RwLock<CacheStats>>,
}

#[cfg(feature = "prompt-cache-redis")]
impl RedisCacheBackend {
    /// Create a new Redis cache backend.
    ///
    /// # Arguments
    /// * `redis_url` - Redis connection URL (e.g., "redis://localhost:6379")
    /// * `ttl_seconds` - TTL in seconds (None = no expiration)
    ///
    /// # Errors
    /// Returns an error if the Redis connection cannot be established
    pub fn new(redis_url: &str, ttl_seconds: Option<u64>) -> Result<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to connect to Redis: {}", e)))?;

        Ok(Self {
            client,
            ttl_seconds,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        })
    }

    /// Get a connection from the pool.
    async fn get_connection(&self) -> Result<redis::aio::ConnectionManager> {
        redis::aio::ConnectionManager::new(self.client.clone())
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to get Redis connection: {}", e)))
    }

    /// Update statistics from Redis.
    async fn update_stats(&self) -> Result<()> {
        // Note: Getting exact stats from Redis requires SCAN which is expensive.
        // For now, we'll just update cached_prefixes based on our tracking.
        // In a production system, you might want to use Redis INFO or maintain
        // a separate counter key.
        Ok(())
    }
}

#[cfg(feature = "prompt-cache-redis")]
#[async_trait::async_trait]
impl PromptCacheStorage for RedisCacheBackend {
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

        let mut conn = self.get_connection().await?;

        let mut cmd = redis::cmd("SET");
        cmd.arg(prefix_hash).arg(&serialized);
        if let Some(ttl) = self.ttl_seconds {
            cmd.arg("EX").arg(ttl);
        }

        cmd.query_async::<_, ()>(&mut conn)
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Redis SET failed: {}", e)))?;

        // Update stats
        let mut stats = self.stats.write();
        stats.cached_prefixes += 1;
        stats.total_size_bytes += serialized.len();

        Ok(())
    }

    async fn get_prefix(&self, prefix_hash: &[u8]) -> Result<Option<CachedPrefix>> {
        let mut conn = self.get_connection().await?;

        let result = match redis::cmd("GET")
            .arg(prefix_hash)
            .query_async::<_, Option<Vec<u8>>>(&mut conn)
            .await
        {
            Ok(Some(serialized)) => {
                let cached_prefix: CachedPrefix = serde_json::from_slice(&serialized)
                    .map_err(|e| candle_core::Error::Msg(format!("Deserialization failed: {}", e)))?;
                Ok(Some(cached_prefix))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(candle_core::Error::Msg(format!("Redis error: {}", e))),
        };

        // Update stats after the await
        let mut stats = self.stats.write();
        match &result {
            Ok(Some(_)) => stats.hits += 1,
            Ok(None) => stats.misses += 1,
            Err(_) => stats.misses += 1,
        }

        result
    }

    async fn has_prefix(&self, prefix_hash: &[u8]) -> Result<bool> {
        let mut conn = self.get_connection().await?;
        let exists: bool = redis::cmd("EXISTS")
            .arg(prefix_hash)
            .query_async(&mut conn)
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Redis EXISTS failed: {}", e)))?;
        Ok(exists)
    }

    async fn remove_prefix(&self, prefix_hash: &[u8]) -> Result<()> {
        let mut conn = self.get_connection().await?;
        redis::cmd("DEL")
            .arg(prefix_hash)
            .query_async::<_, ()>(&mut conn)
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Redis DEL failed: {}", e)))?;

        let mut stats = self.stats.write();
        if stats.cached_prefixes > 0 {
            stats.cached_prefixes -= 1;
        }

        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        // Note: This clears ALL keys in the current database.
        // In production, you might want to use a key prefix and only clear those.
        let mut conn = self.get_connection().await?;
        redis::cmd("FLUSHDB")
            .query_async::<_, ()>(&mut conn)
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Redis FLUSHDB failed: {}", e)))?;

        let mut stats = self.stats.write();
        *stats = CacheStats::default();

        Ok(())
    }

    async fn stats(&self) -> Result<CacheStats> {
        let stats = self.stats.read();
        Ok(stats.clone())
    }
}

#[cfg(test)]
#[cfg(feature = "prompt-cache-redis")]
mod tests {
    use super::*;
    use std::time::SystemTime;

    /// Helper to check if Redis is available for testing
    async fn redis_available() -> bool {
        match redis::Client::open("redis://127.0.0.1:6379") {
            Ok(client) => {
                match redis::aio::ConnectionManager::new(client).await {
                    Ok(mut conn) => {
                        redis::cmd("PING").query_async::<_, String>(&mut conn).await.is_ok()
                    }
                    Err(_) => false,
                }
            }
            Err(_) => false,
        }
    }

    #[tokio::test]
    async fn test_redis_backend_store_and_get() {
        if !redis_available().await {
            eprintln!("Skipping Redis test: Redis not available");
            return;
        }

        let backend = RedisCacheBackend::new("redis://127.0.0.1:6379", Some(60)).unwrap();

        let hash = b"test_redis_prefix";
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
        assert_eq!(cached.kv_blocks.len(), 1);

        // Cleanup
        backend.remove_prefix(hash).await.unwrap();
    }

    #[tokio::test]
    async fn test_redis_backend_ttl() {
        if !redis_available().await {
            eprintln!("Skipping Redis test: Redis not available");
            return;
        }

        let backend = RedisCacheBackend::new("redis://127.0.0.1:6379", Some(1)).unwrap();

        let hash = b"test_redis_ttl";
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
        
        // Should exist immediately
        assert!(backend.has_prefix(hash).await.unwrap());
        
        // Wait for TTL to expire
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        
        // Should be gone after TTL
        assert!(!backend.has_prefix(hash).await.unwrap());
    }
}
