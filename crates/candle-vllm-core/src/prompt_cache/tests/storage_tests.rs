//! Tests for storage backends.

use crate::prompt_cache::backends::MemoryCacheBackend;
use crate::prompt_cache::storage::{
    CacheMetadata, CacheStats, CachedPrefix, KVCacheBlock, PromptCacheStorage,
};
use std::time::SystemTime;

#[tokio::test]
async fn test_memory_backend_basic_operations() {
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

    // Test store
    backend.store_prefix(hash, &kv_blocks, metadata.clone()).await.unwrap();

    // Test get
    let result = backend.get_prefix(hash).await.unwrap();
    assert!(result.is_some());
    let cached = result.unwrap();
    assert_eq!(cached.metadata.prefix_length, 10);
    assert_eq!(cached.kv_blocks.len(), 1);

    // Test has_prefix
    assert!(backend.has_prefix(hash).await.unwrap());
    assert!(!backend.has_prefix(b"nonexistent").await.unwrap());

    // Test remove
    backend.remove_prefix(hash).await.unwrap();
    assert!(!backend.has_prefix(hash).await.unwrap());

    // Test clear
    backend.store_prefix(hash, &kv_blocks, metadata).await.unwrap();
    backend.clear().await.unwrap();
    assert!(!backend.has_prefix(hash).await.unwrap());
}

#[tokio::test]
async fn test_memory_backend_lru_eviction() {
    let backend = MemoryCacheBackend::new(Some(2));

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

    backend.store_prefix(b"prefix1", &kv_blocks, metadata.clone()).await.unwrap();
    backend.store_prefix(b"prefix2", &kv_blocks, metadata.clone()).await.unwrap();
    backend.store_prefix(b"prefix3", &kv_blocks, metadata).await.unwrap();

    // prefix1 should be evicted (oldest)
    assert!(!backend.has_prefix(b"prefix1").await.unwrap());
    assert!(backend.has_prefix(b"prefix2").await.unwrap());
    assert!(backend.has_prefix(b"prefix3").await.unwrap());
}

#[tokio::test]
async fn test_memory_backend_stats() {
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
    assert!(stats.hit_rate() > 0.0);
}
