//! Comprehensive unit tests for InferenceJob and related types.
//!
//! These tests cover all code paths including:
//! - Job creation (completion and streaming)
//! - Input metadata conversion
//! - Serialization/deserialization
//! - Edge cases (empty tokens, zero-length sequences, etc.)

use super::helpers::default_sampling_params;
use crate::openai::sampling_params::SamplingParams;
use crate::parking_lot::job::{InferenceJob, InferenceResult, StreamingTokenResult};
use candle_core::Device;

#[test]
fn test_inference_job_new_completion() {
    let job = InferenceJob::new_completion(
        "test-request-1".to_string(),
        vec![1, 2, 3, 4, 5],
        vec![0, 1, 2, 3, 4],
        default_sampling_params(),
        2048,
    );

    assert_eq!(job.request_id, "test-request-1");
    assert_eq!(job.tokens.len(), 5);
    assert_eq!(job.positions.len(), 5);
    assert!(!job.is_streaming);
    assert!(job.is_prefill);
    assert_eq!(job.max_seqlen_q, 5);
    assert_eq!(job.max_seqlen_k, 5);
    assert_eq!(job.max_context_len, 2048);
}

#[test]
fn test_inference_job_new_streaming() {
    let job = InferenceJob::new_streaming(
        "test-request-2".to_string(),
        vec![10, 20, 30],
        vec![0, 1, 2],
        default_sampling_params(),
        1234567890,
        1024,
    );

    assert_eq!(job.request_id, "test-request-2");
    assert_eq!(job.tokens.len(), 3);
    assert!(job.is_streaming);
    assert_eq!(job.created, 1234567890);
    assert_eq!(job.max_context_len, 1024);
}

#[test]
fn test_inference_job_prompt_len() {
    let job = InferenceJob::new_completion(
        "test-request".to_string(),
        vec![1, 2, 3],
        vec![0, 1, 2],
        default_sampling_params(),
        2048,
    );

    assert_eq!(job.prompt_len(), 3);
}

#[test]
fn test_inference_job_prompt_len_empty() {
    let job = InferenceJob::new_completion(
        "test-request".to_string(),
        vec![],
        vec![],
        default_sampling_params(),
        2048,
    );

    assert_eq!(job.prompt_len(), 0);
}

#[test]
fn test_inference_job_to_input_metadata() {
    let device = Device::Cpu;
    let job = InferenceJob::new_completion(
        "test-request".to_string(),
        vec![1, 2, 3],
        vec![0, 1, 2],
        default_sampling_params(),
        2048,
    );

    let metadata = job.to_input_metadata(&device).unwrap();
    assert!(metadata.is_prefill);
    assert_eq!(metadata.max_seqlen_q, 3);
    assert_eq!(metadata.max_seqlen_k, 3);
    assert_eq!(metadata.max_context_len, 2048);
}

#[test]
fn test_inference_job_to_input_metadata_empty() {
    let device = Device::Cpu;
    let job = InferenceJob::new_completion(
        "test-request".to_string(),
        vec![],
        vec![],
        default_sampling_params(),
        2048,
    );

    let metadata = job.to_input_metadata(&device).unwrap();
    assert_eq!(metadata.max_seqlen_q, 0);
    assert_eq!(metadata.max_seqlen_k, 0);
}

#[test]
fn test_inference_result_completion() {
    use crate::openai::responses::{ChatChoice, ChatChoiceData, ChatCompletionUsageResponse};

    let choices = vec![ChatChoice {
        message: ChatChoiceData {
            role: "assistant".to_string(),
            content: Some("Hello".to_string()),
            tool_calls: None,
        },
        finish_reason: Some("stop".to_string()),
        index: 0,
        logprobs: None,
    }];

    let usage = ChatCompletionUsageResponse {
        request_id: "test-request".to_string(),
        prompt_tokens: 5,
        completion_tokens: 3,
        total_tokens: 8,
        created: 1234567890,
        prompt_time_costs: 50,
        completion_time_costs: 100,
    };

    let result = InferenceResult::completion(choices.clone(), usage.clone());
    assert!(!result.is_error());
    assert!(result.error_message().is_none());
}

#[test]
fn test_inference_result_streaming() {
    let (tx, rx) = flume::unbounded();
    let result = InferenceResult::streaming("test-request".to_string(), rx);

    assert!(!result.is_error());

    // Send a token
    tx.send(Ok(StreamingTokenResult {
        text: "hello".to_string(),
        token_id: 1,
        is_finished: false,
        finish_reason: None,
        is_reasoning: false,
    }))
    .unwrap();

    // Receive should work
    match result {
        InferenceResult::Streaming { token_rx, .. } => {
            let received = token_rx.recv().unwrap();
            assert!(received.is_ok());
        }
        _ => panic!("Expected streaming result"),
    }
}

#[test]
fn test_inference_result_error() {
    let result = InferenceResult::error("Test error message");

    assert!(result.is_error());
    assert_eq!(result.error_message(), Some("Test error message"));
}

#[test]
fn test_inference_result_error_string() {
    let result = InferenceResult::error(String::from("Another error"));

    assert!(result.is_error());
    assert_eq!(result.error_message(), Some("Another error"));
}

#[test]
fn test_streaming_token_result() {
    let token = StreamingTokenResult {
        text: "world".to_string(),
        token_id: 42,
        is_finished: true,
        finish_reason: Some("stop".to_string()),
        is_reasoning: false,
    };

    assert_eq!(token.text, "world");
    assert_eq!(token.token_id, 42);
    assert!(token.is_finished);
    assert_eq!(token.finish_reason, Some("stop".to_string()));
    assert!(!token.is_reasoning);
}

#[test]
fn test_streaming_token_result_reasoning() {
    let token = StreamingTokenResult {
        text: "thinking...".to_string(),
        token_id: 100,
        is_finished: false,
        finish_reason: None,
        is_reasoning: true,
    };

    assert!(token.is_reasoning);
    assert!(!token.is_finished);
}

#[test]
fn test_inference_job_serialization_roundtrip() {
    let job = InferenceJob::new_completion(
        "test-request".to_string(),
        vec![1, 2, 3],
        vec![0, 1, 2],
        {
            let mut params = default_sampling_params();
            params.temperature = Some(0.7);
            params.top_p = Some(0.9);
            params
        },
        2048,
    );

    let json = serde_json::to_string(&job).unwrap();
    let deserialized: InferenceJob = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.request_id, job.request_id);
    assert_eq!(deserialized.tokens, job.tokens);
    assert_eq!(deserialized.positions, job.positions);
    assert_eq!(deserialized.is_streaming, job.is_streaming);
    assert_eq!(deserialized.max_context_len, job.max_context_len);
}

#[test]
fn test_inference_job_large_sequence() {
    let large_tokens: Vec<u32> = (0..1000).collect();
    let large_positions: Vec<usize> = (0..1000).collect();

    let job = InferenceJob::new_completion(
        "test-request".to_string(),
        large_tokens.clone(),
        large_positions.clone(),
        default_sampling_params(),
        4096,
    );

    assert_eq!(job.prompt_len(), 1000);
    assert_eq!(job.max_seqlen_q, 1000);
    assert_eq!(job.max_seqlen_k, 1000);
}

#[test]
fn test_inference_job_mismatched_lengths() {
    // Test with mismatched token and position lengths
    // This should still create the job, but to_input_metadata might fail
    let job = InferenceJob {
        request_id: "test".to_string(),
        tokens: vec![1, 2, 3],
        positions: vec![0, 1], // Mismatched length
        is_streaming: false,
        sampling_params: default_sampling_params(),
        created: 1234567890,
        max_context_len: 2048,
        is_prefill: true,
        max_seqlen_q: 3,
        max_seqlen_k: 3,
    };

    // Job creation should succeed
    assert_eq!(job.tokens.len(), 3);
    assert_eq!(job.positions.len(), 2);
}
