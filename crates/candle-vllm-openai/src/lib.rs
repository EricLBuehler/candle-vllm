//! OpenAI compatibility layer for candle-vllm.
//!
//! This crate provides an OpenAI-compatible API adapter that wraps the core inference engine.

pub use candle_vllm_core as core;
pub use candle_vllm_core::openai::*;
pub use candle_vllm_core::{backend, scheduler};
pub mod adapter;
pub mod model_registry;

// Re-export adapter types
pub use adapter::OpenAIAdapter;
