//! Prompt cache manager for prefix matching and cache operations.

#[cfg(feature = "prompt-cache-redis")]
use crate::prompt_cache::backends::RedisCacheBackend;
#[cfg(feature = "prompt-cache-sled")]
use crate::prompt_cache::backends::SledCacheBackend;
use crate::prompt_cache::{
    backends::MemoryCacheBackend,
    config::{CacheBackend, PromptCacheConfig},
    storage::{CacheMetadata, CachedPrefixMatch, KVCacheBlock, PromptCacheStorage},
};
use candle_core::Result;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::SystemTime;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

/// Manager for prompt prefix caching.
///
/// This manager handles:
/// - Finding longest matching prefix in cache
/// - Storing computed KV cache blocks
/// - Hashing prompt prefixes for cache keys
/// - Model fingerprint validation
pub struct PromptCacheManager {
    /// The storage backend
    storage: Arc<dyn PromptCacheStorage>,
    /// Tokenizer for token operations (reserved for future use)
    #[allow(dead_code)]
    tokenizer: Tokenizer,
    /// Configuration
    config: PromptCacheConfig,
}

impl PromptCacheManager {
    /// Create a new prompt cache manager.
    ///
    /// # Arguments
    /// * `config` - Cache configuration
    /// * `tokenizer` - Tokenizer instance
    ///
    /// # Errors
    /// Returns an error if the storage backend cannot be initialized
    pub fn new(config: PromptCacheConfig, tokenizer: Tokenizer) -> Result<Self> {
        if !config.enabled {
            info!("Prompt caching is disabled");
            // Return a manager with a no-op backend
            let storage: Arc<dyn PromptCacheStorage> = Arc::new(MemoryCacheBackend::new(None));
            return Ok(Self {
                storage,
                tokenizer,
                config,
            });
        }

        let storage: Arc<dyn PromptCacheStorage> = match config.backend {
            CacheBackend::Memory => {
                info!("Using in-memory prompt cache backend");
                Arc::new(MemoryCacheBackend::new(config.max_cached_prefixes))
            }
            CacheBackend::Sled => {
                #[cfg(feature = "prompt-cache-sled")]
                {
                    let path = config
                        .cache_path
                        .as_deref()
                        .unwrap_or("~/.candle-vllm/cache");
                    let expanded_path = shellexpand::tilde(path).to_string();
                    info!("Using sled prompt cache backend at: {}", expanded_path);
                    Arc::new(SledCacheBackend::new(&expanded_path)?)
                }
                #[cfg(not(feature = "prompt-cache-sled"))]
                {
                    warn!("sled backend requested but feature not enabled, falling back to memory");
                    Arc::new(MemoryCacheBackend::new(config.max_cached_prefixes))
                }
            }
            CacheBackend::Redis => {
                #[cfg(feature = "prompt-cache-redis")]
                {
                    let url = config.redis_url.as_deref().ok_or_else(|| {
                        candle_core::Error::Msg("Redis URL required for Redis backend".to_string())
                    })?;
                    info!("Using Redis prompt cache backend at: {}", url);
                    Arc::new(RedisCacheBackend::new(url, config.ttl_seconds)?)
                }
                #[cfg(not(feature = "prompt-cache-redis"))]
                {
                    warn!(
                        "Redis backend requested but feature not enabled, falling back to memory"
                    );
                    Arc::new(MemoryCacheBackend::new(config.max_cached_prefixes))
                }
            }
        };

        Ok(Self {
            storage,
            tokenizer,
            config,
        })
    }

    /// Find the longest matching prefix in the cache.
    ///
    /// This method checks all prefixes of the prompt (from longest to shortest)
    /// and returns the longest match found in the cache.
    ///
    /// # Arguments
    /// * `tokens` - The prompt tokens to search for
    ///
    /// # Returns
    /// `Some(CachedPrefixMatch)` if a match is found, `None` otherwise
    pub async fn find_cached_prefix(&self, tokens: &[u32]) -> Result<Option<CachedPrefixMatch>> {
        if !self.config.enabled || tokens.len() < self.config.min_prefix_length {
            return Ok(None);
        }

        // Check prefixes from longest to shortest
        let min_length = self.config.min_prefix_length;
        for length in (min_length..=tokens.len()).rev() {
            let prefix = &tokens[..length];
            let prefix_hash = self.hash_tokens_impl(prefix);

            // Check if this prefix exists in cache
            if let Some(cached_prefix) = self.storage.get_prefix(&prefix_hash).await? {
                // Validate model fingerprint if configured
                if let Some(ref expected_fp) = self.config.model_fingerprint {
                    if cached_prefix.metadata.model_fingerprint != *expected_fp {
                        debug!(
                            "Cache entry model fingerprint mismatch, skipping: expected={}, got={}",
                            expected_fp, cached_prefix.metadata.model_fingerprint
                        );
                        continue;
                    }
                }

                info!(
                    "Cache hit: found prefix of length {} tokens",
                    cached_prefix.metadata.prefix_length
                );

                return Ok(Some(CachedPrefixMatch {
                    prefix: cached_prefix,
                    cached_tokens: length,
                    prefix_hash,
                }));
            }
        }

        Ok(None)
    }

    /// Store computed KV cache blocks for a prompt prefix.
    ///
    /// This method stores the KV cache blocks after prefill computation,
    /// allowing future requests with the same prefix to reuse them.
    ///
    /// # Arguments
    /// * `tokens` - The prompt prefix tokens
    /// * `kv_blocks` - The computed KV cache blocks for all layers
    pub async fn store_prefix(&self, tokens: &[u32], kv_blocks: &[KVCacheBlock]) -> Result<()> {
        if !self.config.enabled || tokens.len() < self.config.min_prefix_length {
            return Ok(());
        }

        let prefix_hash = self.hash_tokens_impl(tokens);
        let metadata = CacheMetadata {
            created_at: SystemTime::now(),
            model_fingerprint: self
                .config
                .model_fingerprint
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            prefix_length: tokens.len(),
            block_count: kv_blocks.len(),
        };

        self.storage
            .store_prefix(&prefix_hash, kv_blocks, metadata)
            .await?;

        debug!(
            "Stored prefix in cache: length={} tokens, blocks={}",
            tokens.len(),
            kv_blocks.len()
        );

        Ok(())
    }

    /// Hash tokens to create a cache key.
    ///
    /// Uses SHA-256 to create a deterministic hash of the token sequence.
    ///
    /// # Arguments
    /// * `tokens` - The tokens to hash
    ///
    /// # Returns
    /// The hash as a byte vector
    #[cfg(test)]
    pub fn hash_tokens(&self, tokens: &[u32]) -> Vec<u8> {
        self.hash_tokens_impl(tokens)
    }

    /// Internal implementation of hash_tokens (non-test version).
    fn hash_tokens_impl(&self, tokens: &[u32]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        // Include model fingerprint in hash if available for better isolation
        if let Some(ref fp) = self.config.model_fingerprint {
            hasher.update(fp.as_bytes());
        }
        // Hash the tokens
        for token in tokens {
            hasher.update(&token.to_le_bytes());
        }
        hasher.finalize().to_vec()
    }

    /// Get cache statistics.
    pub async fn stats(&self) -> Result<crate::prompt_cache::storage::CacheStats> {
        self.storage.stats().await
    }

    /// Check if caching is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Tokenizer;

    fn create_test_tokenizer() -> Tokenizer {
        // Create a minimal tokenizer for testing
        use tokenizers::models::bpe::BpeBuilder;
        use tokenizers::ModelWrapper;
        let bpe = BpeBuilder::default().build().unwrap();
        Tokenizer::new(ModelWrapper::BPE(bpe))
    }

    #[tokio::test]
    async fn test_hash_tokens() {
        let config = PromptCacheConfig {
            enabled: true,
            ..Default::default()
        };
        let tokenizer = create_test_tokenizer();
        let manager = PromptCacheManager::new(config, tokenizer).unwrap();

        let tokens1 = vec![1, 2, 3, 4, 5];
        let tokens2 = vec![1, 2, 3, 4, 5];
        let tokens3 = vec![1, 2, 3, 4, 6];

        let hash1 = manager.hash_tokens(&tokens1);
        let hash2 = manager.hash_tokens(&tokens2);
        let hash3 = manager.hash_tokens(&tokens3);

        // Same tokens should produce same hash
        assert_eq!(hash1, hash2);
        // Different tokens should produce different hash
        assert_ne!(hash1, hash3);
    }

    #[tokio::test]
    async fn test_find_cached_prefix() {
        let config = PromptCacheConfig {
            enabled: true,
            min_prefix_length: 2,
            ..Default::default()
        };
        let tokenizer = create_test_tokenizer();
        let manager = PromptCacheManager::new(config, tokenizer).unwrap();

        let tokens = vec![1, 2, 3, 4, 5];
        let kv_blocks = vec![KVCacheBlock {
            key_data: vec![1, 2, 3],
            value_data: vec![4, 5, 6],
            key_shape: vec![1, 2],
            value_shape: vec![1, 2],
            dtype: 0,
        }];

        // Store a prefix
        manager
            .store_prefix(&tokens[..3], &kv_blocks)
            .await
            .unwrap();

        // Find it
        let result = manager.find_cached_prefix(&tokens).await.unwrap();
        assert!(result.is_some());
        let match_result = result.unwrap();
        assert_eq!(match_result.cached_tokens, 3);
    }

    #[tokio::test]
    async fn test_find_longest_prefix() {
        let config = PromptCacheConfig {
            enabled: true,
            min_prefix_length: 2,
            ..Default::default()
        };
        let tokenizer = create_test_tokenizer();
        let manager = PromptCacheManager::new(config, tokenizer).unwrap();

        let tokens = vec![1, 2, 3, 4, 5];
        let kv_blocks = vec![KVCacheBlock {
            key_data: vec![1],
            value_data: vec![2],
            key_shape: vec![1],
            value_shape: vec![1],
            dtype: 0,
        }];

        // Store shorter prefix
        manager
            .store_prefix(&tokens[..2], &kv_blocks)
            .await
            .unwrap();
        // Store longer prefix
        manager
            .store_prefix(&tokens[..4], &kv_blocks)
            .await
            .unwrap();

        // Should find the longer one
        let result = manager.find_cached_prefix(&tokens).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().cached_tokens, 4);
    }
}
