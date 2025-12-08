//! Shared test utilities and helpers for candle-vllm test suite.
//!
//! This module provides common functionality used across unit tests,
//! integration tests, and end-to-end tests.

pub mod mock_models;
pub mod mock_tensors;
pub mod mock_executor;
pub mod test_config;
pub mod assertions;

pub use mock_models::{create_tiny_llama_config, create_tiny_mistral_config, MockModelLoader};
pub use mock_tensors::{create_mock_tensor, create_mock_kv_cache, MockDevice};
pub use mock_executor::{MockLlmExecutor, MockInferenceJob};
pub use test_config::{test_engine_config, test_scheduler_config, test_parking_lot_config, default_sampling_params};
pub use assertions::{assert_inference_success, assert_streaming_complete};
