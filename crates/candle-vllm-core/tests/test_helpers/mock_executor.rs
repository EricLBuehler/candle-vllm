//! Mock executor for fast unit testing.

use candle_vllm_core::parking_lot::{
    InferenceJob, InferenceResult, LlmExecutor, TaskExecutor, TaskMetadata,
};
use async_trait::async_trait;
use std::sync::Arc;
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

#[async_trait]
impl TaskExecutor<InferenceJob, InferenceResult> for MockLlmExecutor {
    async fn execute(&self, payload: InferenceJob, _meta: TaskMetadata) -> InferenceResult {
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
            let _ = tx.send(Ok(candle_vllm_core::openai::streaming::StreamingToken {
                token_id: 1,
                token_text: "test".to_string(),
                is_finished: false,
                finish_reason: None,
            }));
            
            let _ = tx.send(Ok(candle_vllm_core::openai::streaming::StreamingToken {
                token_id: 2,
                token_text: "output".to_string(),
                is_finished: true,
                finish_reason: Some("length".to_string()),
            }));
            
            drop(tx);
            
            InferenceResult::Streaming {
                request_id: payload.request_id,
                token_rx: rx,
            }
        } else {
            // For completion, return mock output
            InferenceResult::Completion {
                request_id: payload.request_id,
                output: "Mock completion output".to_string(),
                tokens: vec![1, 2, 3],
                finish_reason: "length".to_string(),
            }
        }
    }
}

/// Mock inference job builder for testing.
pub struct MockInferenceJob;

impl MockInferenceJob {
    pub fn completion(request_id: String) -> InferenceJob {
        InferenceJob {
            request_id,
            prompt: "Test prompt".to_string(),
            is_streaming: false,
            max_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
        }
    }
    
    pub fn streaming(request_id: String) -> InferenceJob {
        InferenceJob {
            request_id,
            prompt: "Test prompt".to_string(),
            is_streaming: true,
            max_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
        }
    }
}
