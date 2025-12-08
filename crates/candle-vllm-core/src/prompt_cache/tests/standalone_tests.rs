//! Standalone tests for prompt caching that don't depend on other test modules.

#[cfg(test)]
mod tests {
    use crate::prompt_cache::{
        backends::MemoryCacheBackend,
        config::{CacheBackend, PromptCacheConfig},
        manager::PromptCacheManager,
        storage::{CacheMetadata, KVCacheBlock, PromptCacheStorage},
    };
    use std::time::SystemTime;
    use tokenizers::{models::bpe::BpeBuilder, ModelWrapper, Tokenizer};

    fn create_test_tokenizer() -> Tokenizer {
        let bpe = BpeBuilder::default().build().unwrap();
        Tokenizer::new(ModelWrapper::BPE(bpe))
    }

    #[tokio::test]
    async fn test_memory_backend_standalone() {
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
        assert_eq!(cached.kv_blocks.len(), 1);
    }

    #[tokio::test]
    async fn test_manager_hash_standalone() {
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

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[tokio::test]
    async fn test_manager_prefix_matching_standalone() {
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

        manager.store_prefix(&tokens[..3], &kv_blocks).await.unwrap();
        let result = manager.find_cached_prefix(&tokens).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().cached_tokens, 3);
    }
}
