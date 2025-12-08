//! Prompt cache storage backends.

pub mod memory;

#[cfg(feature = "prompt-cache-sled")]
pub mod sled_backend;

#[cfg(feature = "prompt-cache-redis")]
pub mod redis_backend;

pub use memory::MemoryCacheBackend;

#[cfg(feature = "prompt-cache-sled")]
pub use sled_backend::SledCacheBackend;

#[cfg(feature = "prompt-cache-redis")]
pub use redis_backend::RedisCacheBackend;
