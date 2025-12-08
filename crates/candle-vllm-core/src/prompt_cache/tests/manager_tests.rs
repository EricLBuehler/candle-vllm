//! Tests for prompt cache manager.

use crate::prompt_cache::{
    config::{CacheBackend, PromptCacheConfig},
    manager::PromptCacheManager,
    storage::{CacheMetadata, KVCacheBlock},
};
use std::time::SystemTime;
use tokenizers::Tokenizer;

    fn create_test_tokenizer() -> Tokenizer {
        // Create a minimal tokenizer for testing
        use tokenizers::models::bpe::BpeBuilder;
        use tokenizers::ModelWrapper;
        let bpe = BpeBuilder::default().build().unwrap();
        Tokenizer::new(ModelWrapper::BPE(bpe))
    }

#[tokio::test]
async fn test_hash_consistency() {
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
async fn test_hash_with_model_fingerprint() {
    let config1 = PromptCacheConfig {
        enabled: true,
        model_fingerprint: Some("model_v1".to_string()),
        ..Default::default()
    };
    let config2 = PromptCacheConfig {
        enabled: true,
        model_fingerprint: Some("model_v2".to_string()),
        ..Default::default()
    };

    let tokenizer = create_test_tokenizer();
    let manager1 = PromptCacheManager::new(config1, tokenizer.clone()).unwrap();
    let manager2 = PromptCacheManager::new(config2, tokenizer).unwrap();

    let tokens = vec![1, 2, 3, 4, 5];
    let hash1 = manager1.hash_tokens(&tokens);
    let hash2 = manager2.hash_tokens(&tokens);

    // Different model fingerprints should produce different hashes
    assert_ne!(hash1, hash2);
}

#[tokio::test]
async fn test_find_cached_prefix_basic() {
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
    manager.store_prefix(&tokens[..3], &kv_blocks).await.unwrap();

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
    manager.store_prefix(&tokens[..2], &kv_blocks).await.unwrap();
    // Store longer prefix
    manager.store_prefix(&tokens[..4], &kv_blocks).await.unwrap();

    // Should find the longer one
    let result = manager.find_cached_prefix(&tokens).await.unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap().cached_tokens, 4);
}

#[tokio::test]
async fn test_find_prefix_no_match() {
    let config = PromptCacheConfig {
        enabled: true,
        min_prefix_length: 2,
        ..Default::default()
    };
    let tokenizer = create_test_tokenizer();
    let manager = PromptCacheManager::new(config, tokenizer).unwrap();

    let tokens = vec![1, 2, 3, 4, 5];
    let result = manager.find_cached_prefix(&tokens).await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_find_prefix_below_min_length() {
    let config = PromptCacheConfig {
        enabled: true,
        min_prefix_length: 10,
        ..Default::default()
    };
    let tokenizer = create_test_tokenizer();
    let manager = PromptCacheManager::new(config, tokenizer).unwrap();

    let tokens = vec![1, 2, 3, 4, 5]; // Only 5 tokens, below min of 10
    let result = manager.find_cached_prefix(&tokens).await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_model_fingerprint_validation() {
    let config = PromptCacheConfig {
        enabled: true,
        min_prefix_length: 2,
        model_fingerprint: Some("model_v1".to_string()),
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

    // Store prefix (with matching fingerprint from config)
    manager.store_prefix(&tokens[..3], &kv_blocks).await.unwrap();

    // Should find it (fingerprint matches)
    let result = manager.find_cached_prefix(&tokens).await.unwrap();
    assert!(result.is_some());
}

#[tokio::test]
async fn test_cache_disabled() {
    let config = PromptCacheConfig {
        enabled: false,
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

    // Store should be no-op
    manager.store_prefix(&tokens, &kv_blocks).await.unwrap();

    // Find should return None
    let result = manager.find_cached_prefix(&tokens).await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_cache_stats() {
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

    // Store and retrieve to generate stats
    manager.store_prefix(&tokens[..3], &kv_blocks).await.unwrap();
    manager.find_cached_prefix(&tokens).await.unwrap(); // Hit
    manager.find_cached_prefix(&vec![10, 11, 12]).await.unwrap(); // Miss

    let stats = manager.stats().await.unwrap();
    assert!(stats.hits >= 1);
    assert!(stats.misses >= 1);
}
