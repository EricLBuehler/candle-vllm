//! Tests for serialization of InferenceJob and SerializableInferenceResult.

use super::helpers::default_sampling_params;
use crate::openai::responses::{ChatChoice, ChatChoiceData, ChatCompletionUsageResponse};
use crate::openai::sampling_params::SamplingParams;
use crate::parking_lot::{InferenceJob, SerializableInferenceResult};

#[test]
fn test_inference_job_serialization() {
    let job = InferenceJob::new_completion(
        "test-request".to_string(),
        vec![1, 2, 3, 4, 5],
        vec![0, 1, 2, 3, 4],
        default_sampling_params(),
        2048,
    );

    // Serialize to JSON
    let json = serde_json::to_string(&job).expect("Failed to serialize");
    assert!(json.contains("test-request"));
    assert!(json.contains("\"is_streaming\":false"));

    // Deserialize back
    let deserialized: InferenceJob = serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(deserialized.request_id, "test-request");
    assert_eq!(deserialized.tokens.len(), 5);
    assert!(!deserialized.is_streaming);
}

#[test]
fn test_serializable_result_completion() {
    let choices = vec![ChatChoice {
        message: ChatChoiceData {
            role: "assistant".to_string(),
            content: Some("Hello, world!".to_string()),
            tool_calls: None,
        },
        finish_reason: Some("stop".to_string()),
        index: 0,
        logprobs: None,
    }];

    let usage = ChatCompletionUsageResponse {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
        created: 1234567890,
        completion_time_costs: 100,
    };

    let result = SerializableInferenceResult::completion(choices.clone(), usage.clone());

    // Serialize
    let json = serde_json::to_string(&result).expect("Failed to serialize");
    assert!(json.contains("Hello, world!"));
    assert!(json.contains("\"total_tokens\":15"));

    // Deserialize
    let deserialized: SerializableInferenceResult =
        serde_json::from_str(&json).expect("Failed to deserialize");

    match deserialized {
        SerializableInferenceResult::Completion {
            choices: c,
            usage: u,
        } => {
            assert_eq!(c.len(), 1);
            assert_eq!(c[0].message.content, Some("Hello, world!".to_string()));
            assert_eq!(u.total_tokens, 15);
        }
        _ => panic!("Expected Completion variant"),
    }
}

#[test]
fn test_serializable_result_streaming_channel() {
    let result = SerializableInferenceResult::streaming_channel(
        "req-123".to_string(),
        "channel-abc".to_string(),
    );

    let json = serde_json::to_string(&result).expect("Failed to serialize");
    assert!(json.contains("req-123"));
    assert!(json.contains("channel-abc"));

    let deserialized: SerializableInferenceResult =
        serde_json::from_str(&json).expect("Failed to deserialize");

    match deserialized {
        SerializableInferenceResult::StreamingChannel {
            request_id,
            channel_key,
        } => {
            assert_eq!(request_id, "req-123");
            assert_eq!(channel_key, "channel-abc");
        }
        _ => panic!("Expected StreamingChannel variant"),
    }
}

#[test]
fn test_serializable_result_error() {
    let result = SerializableInferenceResult::error("Something went wrong");

    let json = serde_json::to_string(&result).expect("Failed to serialize");
    assert!(json.contains("Something went wrong"));

    let deserialized: SerializableInferenceResult =
        serde_json::from_str(&json).expect("Failed to deserialize");

    assert!(deserialized.is_error());
    assert_eq!(deserialized.error_message(), Some("Something went wrong"));
}
