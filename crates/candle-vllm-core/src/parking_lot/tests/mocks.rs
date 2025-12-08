//! Mock implementations for parking_lot tests.

use crate::parking_lot::{
    InferenceJob, InferenceResult, TaskExecutor, TaskMetadata,
    types::{PrometheusWorkerExecutor, ParkingLotTaskMetadata},
};
use async_trait::async_trait;
use std::time::Duration;

/// Mock LLM executor for testing without actual model inference.
#[derive(Clone)]
pub struct MockLlmExecutor {
    /// Simulated processing delay
    pub delay_ms: u64,
    /// Whether to return errors
    pub should_fail: bool,
}

impl MockLlmExecutor {
    pub fn new() -> Self {
        Self {
            delay_ms: 10,
            should_fail: false,
        }
    }

    pub fn with_delay(mut self, delay_ms: u64) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    pub fn with_failures(mut self) -> Self {
        self.should_fail = true;
        self
    }
}

impl Default for MockLlmExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// Implement PrometheusWorkerExecutor for use with WorkerPool
#[async_trait]
impl PrometheusWorkerExecutor<InferenceJob, InferenceResult> for MockLlmExecutor {
    async fn execute(
        &self,
        payload: InferenceJob,
        meta: ParkingLotTaskMetadata,
    ) -> InferenceResult {
        // Convert ParkingLotTaskMetadata to TaskMetadata
        let local_meta = TaskMetadata {
            id: meta.id,
            priority: meta.priority,
            cost: meta.cost,
            created_at_ms: meta.created_at_ms,
            deadline_ms: meta.deadline_ms,
            mailbox: meta.mailbox,
        };
        self.execute_internal(payload, local_meta).await
    }
}

// Also implement TaskExecutor for backward compatibility
#[async_trait]
impl TaskExecutor<InferenceJob, InferenceResult> for MockLlmExecutor {
    async fn execute(&self, payload: InferenceJob, meta: TaskMetadata) -> InferenceResult {
        self.execute_internal(payload, meta).await
    }
}

impl MockLlmExecutor {
    /// Internal execute implementation shared by both trait implementations.
    async fn execute_internal(&self, payload: InferenceJob, _meta: TaskMetadata) -> InferenceResult {
        // Simulate processing time
        if self.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        }

        if self.should_fail {
            return InferenceResult::error("Mock executor error".to_string());
        }

        if payload.is_streaming {
            // For streaming, create a mock channel
            let (tx, rx) = flume::unbounded();

            // Send a few mock tokens
            let _ = tx.send(Ok(crate::parking_lot::StreamingTokenResult {
                text: "test".to_string(),
                token_id: 1,
                is_finished: false,
                finish_reason: None,
                is_reasoning: false,
            }));

            let _ = tx.send(Ok(crate::parking_lot::StreamingTokenResult {
                text: "output".to_string(),
                token_id: 2,
                is_finished: true,
                finish_reason: Some("length".to_string()),
                is_reasoning: false,
            }));

            drop(tx);

            InferenceResult::Streaming {
                request_id: payload.request_id,
                token_rx: rx,
            }
        } else {
            // For completion, return mock output
            use crate::openai::responses::{
                ChatChoice, ChatChoiceData, ChatCompletionUsageResponse,
            };

            InferenceResult::Completion {
                choices: vec![ChatChoice {
                    message: ChatChoiceData {
                        role: "assistant".to_string(),
                        content: Some("Mock completion output".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: Some("length".to_string()),
                    index: 0,
                    logprobs: None,
                }],
                usage: ChatCompletionUsageResponse {
                    request_id: payload.request_id.clone(),
                    prompt_tokens: 5,
                    completion_tokens: 3,
                    total_tokens: 8,
                    created: 1234567890,
                    prompt_time_costs: 50,
                    completion_time_costs: 100,
                    prompt_tokens_details: None,
                },
            }
        }
    }
}

/// Mock inference job builder for testing.
pub struct MockInferenceJob;

impl MockInferenceJob {
    pub fn completion(request_id: String) -> InferenceJob {
        use super::helpers::default_sampling_params;

        InferenceJob::new_completion(
            request_id,
            vec![1, 2, 3],
            vec![0, 1, 2],
            default_sampling_params(),
            2048,
        )
    }

    pub fn streaming(request_id: String) -> InferenceJob {
        use super::helpers::default_sampling_params;

        InferenceJob::new_streaming(
            request_id,
            vec![1, 2, 3],
            vec![0, 1, 2],
            default_sampling_params(),
            1234567890,
            2048,
        )
    }
}
