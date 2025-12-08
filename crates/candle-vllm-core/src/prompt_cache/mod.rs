//! Prompt prefix caching for LLM inference.
//!
//! This module provides prompt caching functionality that allows reusing
//! computed KV cache blocks for shared prompt prefixes, reducing
//! redundant computation and improving latency.

pub mod backends;
pub mod config;
pub mod manager;
pub mod storage;

#[cfg(test)]
mod tests;

pub use config::{CacheBackend, PromptCacheConfig};
pub use manager::PromptCacheManager;
pub use storage::{
    CacheMetadata, CacheStats, CachedPrefix, CachedPrefixMatch, KVCacheBlock, PromptCacheStorage,
};
