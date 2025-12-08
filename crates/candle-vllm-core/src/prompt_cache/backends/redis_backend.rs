//! Redis-based distributed prompt cache backend.

#[cfg(feature = "prompt-cache-redis")]
use crate::prompt_cache::storage::{
    CacheMetadata, CacheStats, CachedPrefix, KVCacheBlock, PromptCacheStorage,
};
#[cfg(feature = "prompt-cache-redis")]
use serde_json;
#[cfg(feature = "prompt-cache-redis")]
use candle_core::Result;
#[cfg(feature = "prompt-cache-redis")]
use parking_lot::RwLock;
#[cfg(feature = "prompt-cache-redis")]
use std::sync::Arc;

/// Redis-based distributed cache backend.
///
/// This backend uses Redis for distributed caching, allowing cache
/// sharing across multiple instances. Supports TTL-based expiration.
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
        // A more sophisticated implementation could use Redis INFO or maintain
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

        if let Some(ttl) = self.ttl_seconds {
            redis::cmd("SETEX")
                .arg(prefix_hash)
                .arg(ttl)
                .arg(serialized.as_slice())
                .query_async::<_, ()>(&mut conn)
                .await
                .map_err(|e| candle_core::Error::Msg(format!("Failed to store prefix: {}", e)))?;
        } else {
            redis::cmd("SET")
                .arg(prefix_hash)
                .arg(serialized.as_slice())
                .query_async::<_, ()>(&mut conn)
                .await
                .map_err(|e| candle_core::Error::Msg(format!("Failed to store prefix: {}", e)))?;
        }

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
            .map_err(|e| candle_core::Error::Msg(format!("Redis error: {}", e)))?;

        Ok(exists)
    }

    async fn remove_prefix(&self, prefix_hash: &[u8]) -> Result<()> {
        let mut conn = self.get_connection().await?;

        redis::cmd("DEL")
            .arg(prefix_hash)
            .query_async::<_, ()>(&mut conn)
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to remove prefix: {}", e)))?;

        let mut stats = self.stats.write();
        stats.cached_prefixes = stats.cached_prefixes.saturating_sub(1);

        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        // Note: FLUSHDB clears the entire database, which may affect other keys.
        // In production, consider using a key prefix to isolate cache keys.
        let mut conn = self.get_connection().await?;

        redis::cmd("FLUSHDB")
            .query_async::<_, ()>(&mut conn)
            .await
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
