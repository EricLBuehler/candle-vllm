//! Test utilities for candle-vllm-core tests.

use candle_vllm_core::api::{EngineConfig, InferenceEngine};
use std::env;
use std::path::PathBuf;

/// Get the test model path from environment or use default.
pub fn get_test_model_path() -> Option<PathBuf> {
    env::var("CANDLE_VLLM_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            // Try common test model locations
            let candidates = vec![
                "./test-models/mistral-7b",
                "./models/mistral-7b",
                "../test-models/mistral-7b",
            ];
            candidates
                .into_iter()
                .map(PathBuf::from)
                .find(|p| p.exists())
        })
}

/// Get the test device ordinal (0 for CPU/default).
pub fn get_test_device_ordinal() -> usize {
    if let Ok(device_str) = env::var("CANDLE_VLLM_TEST_DEVICE") {
        match device_str.as_str() {
            "cuda" | "metal" => 0, // Use device 0 for GPU
            _ => 0, // Default to device 0
        }
    } else {
        0 // Default to device 0 (CPU or first GPU)
    }
}

/// Create a test engine configuration.
pub fn create_test_config() -> Option<EngineConfig> {
    let model_path = get_test_model_path()?;
    Some(
        EngineConfig::builder()
            .model_path(model_path)
            .device(get_test_device_ordinal())
            .max_batch_size(1)
            .kv_cache_memory(512 * 1024 * 1024) // 512MB for tests
            .build()
            .ok()?,
    )
}

/// Create a test inference engine.
pub async fn create_test_engine() -> Option<InferenceEngine> {
    let config = create_test_config()?;
    InferenceEngine::new(config).await.ok()
}

/// Skip test if no model is available.
#[macro_export]
macro_rules! skip_if_no_model {
    () => {
        if $crate::test_utils::get_test_model_path().is_none() {
            eprintln!("Skipping test: No test model available. Set CANDLE_VLLM_TEST_MODEL environment variable.");
            return;
        }
    };
}

