//! Top-level re-exports for candle-vllm workspace crates.
//!
//! This module provides backward-compatible re-exports of all public APIs
//! from the modular crates, ensuring existing code continues to work.

// Re-export entire crates for backward compatibility
pub use candle_vllm_core as core;
pub use candle_vllm_openai as openai;
pub use candle_vllm_responses as responses;
pub use candle_vllm_server as server;

// Re-export core API types
pub use candle_vllm_core::api::{
    EngineConfig, EngineConfigBuilder, Error, FinishReason, GenerationOutput, GenerationParams,
    GenerationStats, InferenceEngine, InferenceEngineBuilder, ModelInfo, Result,
};

// Re-export OpenAI adapter
pub use candle_vllm_openai::adapter::OpenAIAdapter;

// Re-export common types from core
pub use candle_vllm_core::openai::*;
pub use candle_vllm_core::{backend, scheduler};
