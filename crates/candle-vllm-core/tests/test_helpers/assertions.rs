//! Common test assertions.

use candle_vllm_core::parking_lot::InferenceResult;

/// Assert that an inference result is successful.
pub fn assert_inference_success(result: &InferenceResult) {
    match result {
        InferenceResult::Completion { .. } => {
            // Success
        }
        InferenceResult::Streaming { .. } => {
            // Success
        }
        InferenceResult::Error { message } => {
            panic!("Expected successful inference, got error: {}", message);
        }
    }
}

/// Assert that streaming is complete.
pub fn assert_streaming_complete(result: &InferenceResult) -> bool {
    matches!(result, InferenceResult::Streaming { .. })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_assert_inference_success() {
        let result = InferenceResult::Completion {
            request_id: "test".to_string(),
            output: "output".to_string(),
            tokens: vec![1, 2, 3],
            finish_reason: "length".to_string(),
        };
        
        assert_inference_success(&result);
    }
    
    #[test]
    #[should_panic(expected = "Expected successful inference")]
    fn test_assert_inference_success_fails_on_error() {
        let result = InferenceResult::Error {
            message: "test error".to_string(),
        };
        
        assert_inference_success(&result);
    }
}
