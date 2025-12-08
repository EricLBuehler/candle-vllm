//! Comprehensive unit tests for ResourceAdapter.
//!
//! These tests cover all code paths including:
//! - Adapter creation from cache config
//! - Token-to-block conversion
//! - Block-to-token conversion
//! - Resource cost calculation
//! - VRAM cost calculation
//! - Edge cases (zero tokens, exact block boundaries, etc.)

use crate::parking_lot::resource_adapter::{
    calculate_resource_cost, ResourceAdapter, DEFAULT_BLOCK_SIZE,
};
use crate::scheduler::cache_engine::CacheConfig;
use candle_core::DType;
use prometheus_parking_lot::util::serde::{ResourceCost, ResourceKind};

#[test]
fn test_resource_adapter_default() {
    let adapter = ResourceAdapter::default();
    assert_eq!(adapter.block_size(), DEFAULT_BLOCK_SIZE);
    assert_eq!(adapter.max_units(), 1024);
}

#[test]
fn test_resource_adapter_new() {
    let adapter = ResourceAdapter::new(32, 2048, 128);
    assert_eq!(adapter.block_size(), 32);
    assert_eq!(adapter.max_units(), 2048);
}

#[test]
fn test_resource_adapter_from_cache_config() {
    let cache_config = CacheConfig {
        block_size: 16,
        num_gpu_blocks: Some(1024),
        num_cpu_blocks: Some(512),
        fully_init: true,
        dtype: DType::F16,
        kvcache_mem_gpu: 1024,
    };

    let adapter = ResourceAdapter::from_cache_config(&cache_config);
    assert_eq!(adapter.block_size(), 16);
    assert_eq!(adapter.max_units(), 1024);
}

#[test]
fn test_resource_adapter_from_cache_config_no_gpu_blocks() {
    let cache_config = CacheConfig {
        block_size: 16,
        num_gpu_blocks: None,
        num_cpu_blocks: Some(512),
        fully_init: false,
        dtype: DType::F16,
        kvcache_mem_gpu: 0,
    };

    let adapter = ResourceAdapter::from_cache_config(&cache_config);
    assert_eq!(adapter.max_units(), 0);
}

#[test]
fn test_tokens_to_blocks_zero() {
    let adapter = ResourceAdapter::new(16, 1024, 64);
    assert_eq!(adapter.tokens_to_blocks(0), 0);
}

#[test]
fn test_tokens_to_blocks_one() {
    let adapter = ResourceAdapter::new(16, 1024, 64);
    assert_eq!(adapter.tokens_to_blocks(1), 1);
}

#[test]
fn test_tokens_to_blocks_exact_boundary() {
    let adapter = ResourceAdapter::new(16, 1024, 64);
    assert_eq!(adapter.tokens_to_blocks(16), 1);
    assert_eq!(adapter.tokens_to_blocks(32), 2);
    assert_eq!(adapter.tokens_to_blocks(48), 3);
}

#[test]
fn test_tokens_to_blocks_round_up() {
    let adapter = ResourceAdapter::new(16, 1024, 64);
    assert_eq!(adapter.tokens_to_blocks(17), 2);
    assert_eq!(adapter.tokens_to_blocks(31), 2);
    assert_eq!(adapter.tokens_to_blocks(33), 3);
}

#[test]
fn test_tokens_to_blocks_custom_block_size() {
    let adapter = ResourceAdapter::new(32, 1024, 64);
    assert_eq!(adapter.tokens_to_blocks(0), 0);
    assert_eq!(adapter.tokens_to_blocks(1), 1);
    assert_eq!(adapter.tokens_to_blocks(32), 1);
    assert_eq!(adapter.tokens_to_blocks(33), 2);
    assert_eq!(adapter.tokens_to_blocks(64), 2);
}

#[test]
fn test_calculate_cost() {
    let adapter = ResourceAdapter::new(16, 1024, 64);

    // 100 prompt tokens + 50 max new = 150 total
    // 150 / 16 = 9.375, rounds up to 10 blocks
    let cost = adapter.calculate_cost(100, 50);
    assert_eq!(cost.units, 10);
    assert!(matches!(cost.kind, ResourceKind::GpuVram));
}

#[test]
fn test_calculate_cost_zero_prompt() {
    let adapter = ResourceAdapter::new(16, 1024, 64);
    let cost = adapter.calculate_cost(0, 50);
    // 50 / 16 = 3.125, rounds up to 4 blocks
    assert_eq!(cost.units, 4);
}

#[test]
fn test_calculate_cost_zero_new_tokens() {
    let adapter = ResourceAdapter::new(16, 1024, 64);
    let cost = adapter.calculate_cost(100, 0);
    // 100 / 16 = 6.25, rounds up to 7 blocks
    assert_eq!(cost.units, 7);
}

#[test]
fn test_calculate_cost_both_zero() {
    let adapter = ResourceAdapter::new(16, 1024, 64);
    let cost = adapter.calculate_cost(0, 0);
    assert_eq!(cost.units, 0);
}

#[test]
fn test_calculate_vram_cost() {
    let adapter = ResourceAdapter::new(16, 1024, 64);

    // 100 prompt + 50 max new = 150 total tokens
    // 150 / 16 = 10 blocks (rounded up)
    // 10 blocks * 64 bytes = 640 bytes
    let cost = adapter.calculate_vram_cost(100, 50);
    assert_eq!(cost.units, 640);
    assert!(matches!(cost.kind, ResourceKind::GpuVram));
}

#[test]
fn test_calculate_vram_cost_custom_bytes_per_block() {
    let adapter = ResourceAdapter::new(16, 1024, 128);

    // 100 prompt + 50 max new = 150 total tokens
    // 150 / 16 = 10 blocks (rounded up)
    // 10 blocks * 128 bytes = 1280 bytes
    let cost = adapter.calculate_vram_cost(100, 50);
    assert_eq!(cost.units, 1280);
}

#[test]
fn test_blocks_to_tokens() {
    let adapter = ResourceAdapter::new(16, 1024, 64);

    assert_eq!(adapter.blocks_to_tokens(0), 0);
    assert_eq!(adapter.blocks_to_tokens(1), 16);
    assert_eq!(adapter.blocks_to_tokens(10), 160);
    assert_eq!(adapter.blocks_to_tokens(100), 1600);
}

#[test]
fn test_blocks_to_tokens_custom_block_size() {
    let adapter = ResourceAdapter::new(32, 1024, 64);

    assert_eq!(adapter.blocks_to_tokens(0), 0);
    assert_eq!(adapter.blocks_to_tokens(1), 32);
    assert_eq!(adapter.blocks_to_tokens(10), 320);
}

#[test]
fn test_calculate_resource_cost_with_config() {
    let cache_config = CacheConfig {
        block_size: 16,
        num_gpu_blocks: Some(1024),
        num_cpu_blocks: Some(512),
        fully_init: true,
        dtype: DType::F16,
        kvcache_mem_gpu: 1024,
    };

    let cost = calculate_resource_cost(100, 50, Some(&cache_config));
    // Should use adapter from config
    assert!(matches!(cost.kind, ResourceKind::GpuVram));
    assert!(cost.units > 0);
}

#[test]
fn test_calculate_resource_cost_without_config() {
    let cost = calculate_resource_cost(100, 50, None);
    // Should use default adapter
    assert!(matches!(cost.kind, ResourceKind::GpuVram));
    assert!(cost.units > 0);
}

#[test]
fn test_calculate_resource_cost_zero_tokens() {
    let cost = calculate_resource_cost(0, 0, None);
    assert_eq!(cost.units, 0);
}

#[test]
fn test_calculate_resource_cost_large_values() {
    let adapter = ResourceAdapter::new(16, 1024, 64);

    // Test with very large token counts
    let cost = adapter.calculate_cost(10000, 5000);
    // 15000 / 16 = 937.5, rounds up to 938 blocks
    assert_eq!(cost.units, 938);
}

#[test]
fn test_resource_adapter_edge_cases() {
    let adapter = ResourceAdapter::new(1, 100, 1);

    // Block size of 1 means every token is its own block
    assert_eq!(adapter.tokens_to_blocks(0), 0);
    assert_eq!(adapter.tokens_to_blocks(1), 1);
    assert_eq!(adapter.tokens_to_blocks(100), 100);

    assert_eq!(adapter.blocks_to_tokens(0), 0);
    assert_eq!(adapter.blocks_to_tokens(1), 1);
    assert_eq!(adapter.blocks_to_tokens(100), 100);
}
