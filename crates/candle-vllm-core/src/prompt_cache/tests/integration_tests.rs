//! Integration tests for prompt caching with real inference.

#[cfg(test)]
mod tests {
    use crate::prompt_cache::{
        config::{CacheBackend, PromptCacheConfig},
        manager::PromptCacheManager,
        storage::KVCacheBlock,
    };
    use tokenizers::Tokenizer;

    fn create_test_tokenizer() -> Tokenizer {
        use tokenizers::models::bpe::BpeBuilder;
        use tokenizers::ModelWrapper;
        let bpe = BpeBuilder::default().build().unwrap();
        Tokenizer::new(ModelWrapper::BPE(bpe))
    }

    #[tokio::test]
    async fn test_end_to_end_caching_flow() {
        let config = PromptCacheConfig {
            enabled: true,
            min_prefix_length: 4,
            ..Default::default()
        };
        let tokenizer = create_test_tokenizer();
        let manager = PromptCacheManager::new(config, tokenizer).unwrap();

        // First request: no cache
        let tokens1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let result1 = manager.find_cached_prefix(&tokens1).await.unwrap();
        assert!(result1.is_none(), "First request should have no cache");

        // Store computed KV blocks
        let kv_blocks = vec![KVCacheBlock {
            key_data: vec![1, 2, 3, 4],
            value_data: vec![5, 6, 7, 8],
            key_shape: vec![2, 2],
            value_shape: vec![2, 2],
            dtype: 0,
        }];
        manager.store_prefix(&tokens1[..6], &kv_blocks).await.unwrap();

        // Second request with same prefix: should hit cache
        let tokens2 = vec![1, 2, 3, 4, 5, 6, 9, 10]; // Same first 6 tokens
        let result2 = manager.find_cached_prefix(&tokens2).await.unwrap();
        assert!(result2.is_some(), "Second request should hit cache");
        assert_eq!(result2.unwrap().cached_tokens, 6);
    }

    #[tokio::test]
    async fn test_partial_prefix_matching() {
        let config = PromptCacheConfig {
            enabled: true,
            min_prefix_length: 3,
            ..Default::default()
        };
        let tokenizer = create_test_tokenizer();
        let manager = PromptCacheManager::new(config, tokenizer).unwrap();

        let kv_blocks = vec![KVCacheBlock {
            key_data: vec![1],
            value_data: vec![2],
            key_shape: vec![1],
            value_shape: vec![1],
            dtype: 0,
        }];

        // Store a 5-token prefix
        let prefix = vec![10, 20, 30, 40, 50];
        manager.store_prefix(&prefix, &kv_blocks).await.unwrap();

        // Request with longer prompt that starts with the cached prefix
        let full_prompt = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let result = manager.find_cached_prefix(&full_prompt).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().cached_tokens, 5);
    }

    #[tokio::test]
    async fn test_multiple_prefix_storage() {
        let config = PromptCacheConfig {
            enabled: true,
            min_prefix_length: 2,
            ..Default::default()
        };
        let tokenizer = create_test_tokenizer();
        let manager = PromptCacheManager::new(config, tokenizer).unwrap();

        let kv_blocks = vec![KVCacheBlock {
            key_data: vec![1],
            value_data: vec![2],
            key_shape: vec![1],
            value_shape: vec![1],
            dtype: 0,
        }];

        // Store multiple different prefixes
        manager.store_prefix(&vec![1, 2, 3], &kv_blocks).await.unwrap();
        manager.store_prefix(&vec![4, 5, 6], &kv_blocks).await.unwrap();
        manager.store_prefix(&vec![7, 8, 9], &kv_blocks).await.unwrap();

        // All should be retrievable
        assert!(manager.find_cached_prefix(&vec![1, 2, 3, 10]).await.unwrap().is_some());
        assert!(manager.find_cached_prefix(&vec![4, 5, 6, 10]).await.unwrap().is_some());
        assert!(manager.find_cached_prefix(&vec![7, 8, 9, 10]).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_cache_hit_rate_tracking() {
        let config = PromptCacheConfig {
            enabled: true,
            min_prefix_length: 2,
            ..Default::default()
        };
        let tokenizer = create_test_tokenizer();
        let manager = PromptCacheManager::new(config, tokenizer).unwrap();

        let kv_blocks = vec![KVCacheBlock {
            key_data: vec![1],
            value_data: vec![2],
            key_shape: vec![1],
            value_shape: vec![1],
            dtype: 0,
        }];

        // Store a prefix
        manager.store_prefix(&vec![1, 2, 3], &kv_blocks).await.unwrap();

        // Generate hits and misses
        for _ in 0..5 {
            manager.find_cached_prefix(&vec![1, 2, 3, 4]).await.unwrap(); // Hit
        }
        for _ in 0..3 {
            manager.find_cached_prefix(&vec![10, 11, 12]).await.unwrap(); // Miss
        }

        let stats = manager.stats().await.unwrap();
        assert!(stats.hits >= 5);
        assert!(stats.misses >= 3);
        let hit_rate = stats.hit_rate();
        assert!(hit_rate > 0.0 && hit_rate <= 100.0);
    }
}
