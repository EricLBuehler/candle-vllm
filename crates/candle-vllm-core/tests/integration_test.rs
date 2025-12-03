//! Integration tests for candle-vllm-core.
//!
//! These tests require a model to be available. Set CANDLE_VLLM_TEST_MODEL
//! environment variable to point to a model directory, or place a model
//! at ./test-models/mistral-7b

use candle_vllm_core::api::{GenerationParams, InferenceEngine};
use test_utils::{create_test_engine, get_test_model_path};

mod test_utils {
    use super::*;
    use candle_vllm_core::api::{EngineConfig, InferenceEngine};
    use std::env;
    use std::path::PathBuf;

    pub fn get_test_model_path() -> Option<PathBuf> {
        env::var("CANDLE_VLLM_TEST_MODEL")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
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

    pub fn get_test_device_ordinal() -> usize {
        if let Ok(device_str) = env::var("CANDLE_VLLM_TEST_DEVICE") {
            match device_str.as_str() {
                "cuda" | "metal" => 0, // Use device 0 for GPU
                _ => 0,
            }
        } else {
            0 // Default to device 0
        }
    }

    pub async fn create_test_engine() -> Option<InferenceEngine> {
        let model_path = get_test_model_path()?;
        let config = EngineConfig::builder()
            .model_path(model_path)
            .device(get_test_device_ordinal())
            .max_batch_size(1)
            .kv_cache_memory(512 * 1024 * 1024)
            .build()
            .ok()?;
        InferenceEngine::new(config).await.ok()
    }
}

use test_utils::*;

macro_rules! skip_if_no_model {
    () => {
        if test_utils::get_test_model_path().is_none() {
            eprintln!("Skipping test: No test model available. Set CANDLE_VLLM_TEST_MODEL environment variable.");
            return;
        }
    };
}

#[tokio::test]
async fn test_engine_creation() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await;
    assert!(engine.is_some(), "Failed to create inference engine");
    
    let engine = engine.unwrap();
    let info = engine.model_info();
    assert!(!info.model_path.as_os_str().is_empty());
    assert!(info.max_sequence_length > 0);
    assert!(info.max_batch_size > 0);
}

#[tokio::test]
async fn test_tokenization() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await.unwrap();
    
    // Test tokenization
    let text = "Hello, world!";
    let tokens = engine.tokenize(text).expect("Tokenization should succeed");
    assert!(!tokens.is_empty(), "Should produce tokens");
    
    // Test detokenization
    let detokenized = engine.detokenize(&tokens).expect("Detokenization should succeed");
    assert!(!detokenized.is_empty(), "Should produce text");
    
    // Round-trip test
    let tokens2 = engine.tokenize(&detokenized).expect("Second tokenization should succeed");
    // Note: Round-trip may not be exact due to tokenizer normalization
    assert_eq!(tokens.len(), tokens2.len(), "Round-trip should preserve token count");
}

#[tokio::test]
async fn test_generation_basic() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await.unwrap();
    
    let params = GenerationParams {
        max_tokens: Some(10),
        temperature: Some(0.7),
        ..Default::default()
    };
    
    let prompt = "The capital of France is";
    let output = engine.generate(prompt, params).await.expect("Generation should succeed");
    
    assert!(!output.tokens.is_empty(), "Should generate tokens");
    assert!(output.text.is_some(), "Should have text output");
    assert!(!output.text.as_ref().unwrap().is_empty(), "Text should not be empty");
    
    // Verify finish reason
    match output.finish_reason {
        candle_vllm_core::api::FinishReason::Stop => {},
        candle_vllm_core::api::FinishReason::Length => {},
        _ => panic!("Unexpected finish reason"),
    }
}

#[tokio::test]
async fn test_generation_with_stop_sequence() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await.unwrap();
    
    let params = GenerationParams {
        max_tokens: Some(50),
        temperature: Some(0.7),
        stop_sequences: Some(vec!["Paris".to_string()]),
        ..Default::default()
    };
    
    let prompt = "The capital of France is";
    let output = engine.generate(prompt, params).await.expect("Generation should succeed");
    
    assert!(!output.tokens.is_empty(), "Should generate tokens");
    
    // Check that stop sequence was respected (if it appeared)
    if let Some(text) = &output.text {
        // The text should not contain the stop sequence after the prompt
        let generated_part = text.strip_prefix(prompt).unwrap_or(text);
        assert!(!generated_part.contains("Paris") || output.finish_reason == candle_vllm_core::api::FinishReason::StopSequence,
            "Stop sequence should be respected");
    }
}

#[tokio::test]
async fn test_generation_temperature_effects() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await.unwrap();
    
    let prompt = "The weather is";
    
    // Low temperature (deterministic)
    let params_low = GenerationParams {
        max_tokens: Some(5),
        temperature: Some(0.1),
        ..Default::default()
    };
    
    // High temperature (more random)
    let params_high = GenerationParams {
        max_tokens: Some(5),
        temperature: Some(1.5),
        ..Default::default()
    };
    
    let output_low = engine.generate(prompt, params_low).await.expect("Low temp generation should succeed");
    let output_high = engine.generate(prompt, params_high).await.expect("High temp generation should succeed");
    
    assert!(!output_low.tokens.is_empty());
    assert!(!output_high.tokens.is_empty());
    
    // With different temperatures, outputs may differ (though not guaranteed)
    // This test mainly verifies both temperature settings work
}

#[tokio::test]
async fn test_generation_max_tokens() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await.unwrap();
    
    let params = GenerationParams {
        max_tokens: Some(5),
        temperature: Some(0.7),
        ..Default::default()
    };
    
    let prompt = "Count to ten:";
    let output = engine.generate(prompt, params).await.expect("Generation should succeed");
    
    assert!(output.tokens.len() <= 5, "Should respect max_tokens limit");
    
    if output.tokens.len() == 5 {
        assert_eq!(output.finish_reason, candle_vllm_core::api::FinishReason::Length,
            "Should finish due to length limit");
    }
}

#[tokio::test]
async fn test_model_info() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await.unwrap();
    let info = engine.model_info();
    
    assert!(!info.model_path.as_os_str().is_empty(), "Model path should be set");
    assert!(info.max_sequence_length > 0, "Max sequence length should be positive");
    assert!(info.max_batch_size > 0, "Max batch size should be positive");
    assert!(!info.dtype.is_empty(), "Dtype should be set");
}

#[tokio::test]
async fn test_generation_stats() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await.unwrap();
    
    let params = GenerationParams {
        max_tokens: Some(10),
        temperature: Some(0.7),
        ..Default::default()
    };
    
    let prompt = "Hello";
    let output = engine.generate(prompt, params).await.expect("Generation should succeed");
    
    if let Some(stats) = output.stats {
        assert!(stats.prompt_tokens > 0, "Should have prompt tokens");
        assert!(stats.generated_tokens > 0, "Should have generated tokens");
        assert!(stats.total_time_ms > 0, "Should have positive time");
        assert!(stats.tokens_per_second > 0.0, "Should have positive throughput");
    }
}

