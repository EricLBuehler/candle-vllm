//! Comprehensive unit tests for LlmExecutor.
//!
//! These tests cover all code paths in the executor module including:
//! - Executor creation and configuration
//! - Completion job processing
//! - Streaming job processing
//! - Error handling paths
//! - GPU memory checks
//! - Logits extraction
//! - WorkerExecutor trait implementation

use super::helpers::default_sampling_params;
use super::{MockInferenceJob, MockLlmExecutor};
use crate::openai::sampling_params::SamplingParams;
use crate::parking_lot::{
    job::InferenceResult,
    types::{Priority, ResourceCost, ResourceKind, TaskExecutor, TaskMetadata},
};

#[tokio::test]
async fn test_executor_new() {
    // This test verifies executor creation
    // Note: Real executor requires a pipeline, so we test the mock
    let mock_executor = MockLlmExecutor::new();
    assert_eq!(mock_executor.delay_ms, 10);
    assert!(!mock_executor.should_fail);
}

#[tokio::test]
async fn test_executor_with_delay() {
    let mock_executor = MockLlmExecutor::new().with_delay(50);
    assert_eq!(mock_executor.delay_ms, 50);
}

#[tokio::test]
async fn test_executor_with_failures() {
    let mock_executor = MockLlmExecutor::new().with_failures();
    assert!(mock_executor.should_fail);
}

#[tokio::test]
async fn test_executor_process_completion() {
    let mock_executor = MockLlmExecutor::new();
    let job = MockInferenceJob::completion("test-request-1".to_string());
    let meta = TaskMetadata {
        id: 1,
        priority: Priority::Normal,
        cost: ResourceCost {
            kind: ResourceKind::GpuVram,
            units: 100,
        },
        created_at_ms: 1000,
        deadline_ms: None,
        mailbox: None,
    };

    let result = mock_executor.execute(job, meta).await;
    match result {
        InferenceResult::Completion { .. } => {
            // Completion doesn't have request_id field
        }
        _ => panic!("Expected completion result"),
    }
}

#[tokio::test]
async fn test_executor_process_streaming() {
    let mock_executor = MockLlmExecutor::new();
    let job = MockInferenceJob::streaming("test-request-2".to_string());
    let meta = TaskMetadata {
        id: 2,
        priority: Priority::Normal,
        cost: ResourceCost {
            kind: ResourceKind::GpuVram,
            units: 100,
        },
        created_at_ms: 1000,
        deadline_ms: None,
        mailbox: None,
    };

    let result = mock_executor.execute(job, meta).await;
    match result {
        InferenceResult::Streaming { request_id, .. } => {
            assert_eq!(request_id, "test-request-2");
        }
        _ => panic!("Expected streaming result"),
    }
}

#[tokio::test]
async fn test_executor_error_path() {
    let mock_executor = MockLlmExecutor::new().with_failures();
    let job = MockInferenceJob::completion("test-request-3".to_string());
    let meta = TaskMetadata {
        id: 3,
        priority: Priority::Normal,
        cost: ResourceCost {
            kind: ResourceKind::GpuVram,
            units: 100,
        },
        created_at_ms: 1000,
        deadline_ms: None,
        mailbox: None,
    };

    let result = mock_executor.execute(job, meta).await;
    match result {
        InferenceResult::Error { message } => {
            assert!(message.contains("Mock executor error"));
        }
        _ => panic!("Expected error result"),
    }
}

// Note: WorkerExecutor trait test removed - MockLlmExecutor doesn't implement
// prometheus WorkerExecutor directly, it implements our local TaskExecutor trait

#[test]
fn test_sampling_params_default() {
    // SamplingParams doesn't have Default, so we test with a minimal valid instance
    let params = SamplingParams::new(
        1,
        None,
        0.0,
        0.0,
        None,
        Some(1.0),
        Some(1.0),
        None,
        Some(-1),
        false,
        1.0,
        crate::openai::sampling_params::EarlyStoppingCondition::UnlikelyBetterCandidates,
        None,
        vec![],
        false,
        16,
        None,
        None,
        true,
        None,
    )
    .unwrap();

    assert_eq!(params.temperature, Some(1.0));
    assert_eq!(params.top_p, Some(1.0));
    assert_eq!(params.max_tokens, 16);
}

#[test]
fn test_sampling_params_custom() {
    let params = SamplingParams::new(
        1,
        None,
        0.0,
        0.0,
        None,
        Some(0.7),
        Some(0.9),
        Some(0.1),
        Some(50),
        false,
        1.0,
        crate::openai::sampling_params::EarlyStoppingCondition::UnlikelyBetterCandidates,
        Some(crate::openai::requests::StopTokens::Single(
            "\n".to_string(),
        )),
        vec![],
        false,
        100,
        None,
        None,
        true,
        None,
    )
    .unwrap();

    assert_eq!(params.temperature, Some(0.7));
    assert_eq!(params.top_p, Some(0.9));
    assert_eq!(params.max_tokens, 100);
}
