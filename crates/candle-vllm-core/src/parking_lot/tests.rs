//! Tests for the parking-lot scheduler integration.

use crate::openai::sampling_params::{EarlyStoppingCondition, SamplingParams};

/// Helper function to create default sampling params for tests.
fn test_sampling_params() -> SamplingParams {
    SamplingParams {
        n: 1,
        best_of: 1,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repeat_last_n: None,
        temperature: Some(1.0),
        top_p: Some(1.0),
        min_p: None,
        top_k: Some(-1),
        use_beam_search: false,
        length_penalty: 1.0,
        early_stopping: EarlyStoppingCondition::UnlikelyBetterCandidates,
        stop: None,
        stop_token_ids: vec![],
        ignore_eos: false,
        max_tokens: 16,
        logprobs: None,
        prompt_logprobs: None,
        skip_special_tokens: true,
        thinking: None,
    }
}

/// Helper function to create sampling params with custom max_tokens for tests.
fn test_sampling_params_with_max_tokens(max_tokens: usize) -> SamplingParams {
    SamplingParams {
        max_tokens,
        ..test_sampling_params()
    }
}

#[cfg(test)]
mod job_tests {
    use super::*;
    use crate::parking_lot::job::{InferenceJob, InferenceResult, StreamingTokenResult};

    #[test]
    fn test_inference_job_completion_creation() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let positions = vec![0usize, 1, 2, 3, 4];
        let sampling_params = test_sampling_params();

        let job = InferenceJob::new_completion(
            "test-request-1".to_string(),
            tokens.clone(),
            positions.clone(),
            sampling_params,
            4096,
        );

        assert_eq!(job.request_id, "test-request-1");
        assert_eq!(job.tokens, tokens);
        assert_eq!(job.positions, positions);
        assert!(!job.is_streaming);
        assert_eq!(job.prompt_len(), 5);
        assert_eq!(job.max_context_len, 4096);
    }

    #[test]
    fn test_inference_job_streaming_creation() {
        let tokens = vec![1u32, 2, 3];
        let positions = vec![0usize, 1, 2];
        let sampling_params = test_sampling_params();

        let job = InferenceJob::new_streaming(
            "test-request-2".to_string(),
            tokens.clone(),
            positions.clone(),
            sampling_params,
            1234567890,
            2048,
        );

        assert_eq!(job.request_id, "test-request-2");
        assert!(job.is_streaming);
        assert_eq!(job.created, 1234567890);
        assert_eq!(job.max_context_len, 2048);
    }

    #[test]
    fn test_inference_result_error() {
        let result = InferenceResult::error("test error message");
        assert!(result.is_error());
        assert_eq!(result.error_message(), Some("test error message"));
    }

    #[test]
    fn test_streaming_token_result() {
        let token = StreamingTokenResult {
            text: "Hello".to_string(),
            token_id: 42,
            is_finished: false,
            finish_reason: None,
            is_reasoning: false,
        };

        assert_eq!(token.text, "Hello");
        assert_eq!(token.token_id, 42);
        assert!(!token.is_finished);
    }

    #[test]
    fn test_streaming_token_finished() {
        let token = StreamingTokenResult {
            text: "".to_string(),
            token_id: 0,
            is_finished: true,
            finish_reason: Some("stop".to_string()),
            is_reasoning: false,
        };

        assert!(token.is_finished);
        assert_eq!(token.finish_reason, Some("stop".to_string()));
    }
}

#[cfg(test)]
mod resource_adapter_tests {
    use crate::parking_lot::resource_adapter::{ResourceAdapter, DEFAULT_BLOCK_SIZE};

    #[test]
    fn test_default_adapter() {
        let adapter = ResourceAdapter::default();
        assert_eq!(adapter.block_size(), DEFAULT_BLOCK_SIZE);
    }

    #[test]
    fn test_tokens_to_blocks() {
        let adapter = ResourceAdapter::new(16, 1024, 64);

        // Edge cases
        assert_eq!(adapter.tokens_to_blocks(0), 0);
        assert_eq!(adapter.tokens_to_blocks(1), 1);
        assert_eq!(adapter.tokens_to_blocks(16), 1);
        assert_eq!(adapter.tokens_to_blocks(17), 2);
        assert_eq!(adapter.tokens_to_blocks(32), 2);
        assert_eq!(adapter.tokens_to_blocks(33), 3);
    }

    #[test]
    fn test_blocks_to_tokens() {
        let adapter = ResourceAdapter::new(16, 1024, 64);

        assert_eq!(adapter.blocks_to_tokens(0), 0);
        assert_eq!(adapter.blocks_to_tokens(1), 16);
        assert_eq!(adapter.blocks_to_tokens(10), 160);
    }

    #[test]
    fn test_calculate_cost() {
        let adapter = ResourceAdapter::new(16, 1024, 64);

        // 100 prompt tokens + 50 max new = 150 total
        // 150 / 16 = 9.375, rounds up to 10 blocks
        let cost = adapter.calculate_cost(100, 50);
        assert_eq!(cost.units, 10);
    }

    #[test]
    fn test_calculate_cost_small() {
        let adapter = ResourceAdapter::new(16, 1024, 64);

        // 10 prompt + 5 max = 15 total = 1 block
        let cost = adapter.calculate_cost(10, 5);
        assert_eq!(cost.units, 1);
    }

    #[test]
    fn test_calculate_cost_large() {
        let adapter = ResourceAdapter::new(16, 1024, 64);

        // 2000 prompt + 1000 max = 3000 total
        // 3000 / 16 = 187.5, rounds up to 188 blocks
        let cost = adapter.calculate_cost(2000, 1000);
        assert_eq!(cost.units, 188);
    }

    #[test]
    fn test_max_units() {
        let adapter = ResourceAdapter::new(16, 2048, 64);
        assert_eq!(adapter.max_units(), 2048);
    }

    #[test]
    fn test_calculate_vram_cost() {
        let adapter = ResourceAdapter::new(16, 1024, 128); // 128 bytes per block

        // 16 tokens = 1 block = 128 bytes
        let cost = adapter.calculate_vram_cost(16, 0);
        assert_eq!(cost.units, 128);

        // 32 tokens = 2 blocks = 256 bytes
        let cost = adapter.calculate_vram_cost(32, 0);
        assert_eq!(cost.units, 256);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::parking_lot::job::InferenceJob;
    use crate::parking_lot::resource_adapter::ResourceAdapter;

    #[test]
    fn test_job_cost_calculation() {
        let adapter = ResourceAdapter::new(16, 1024, 64);
        let sampling_params = test_sampling_params_with_max_tokens(100);

        let job = InferenceJob::new_completion(
            "test".to_string(),
            vec![1u32; 500], // 500 tokens
            (0..500).collect(),
            sampling_params,
            4096,
        );

        // 500 prompt + 100 max = 600 total
        // 600 / 16 = 37.5, rounds up to 38 blocks
        let cost = adapter.calculate_cost(job.prompt_len(), job.sampling_params.max_tokens);
        assert_eq!(cost.units, 38);
    }

    #[test]
    fn test_capacity_check() {
        let adapter = ResourceAdapter::new(16, 100, 64); // Only 100 blocks available

        // Small request should fit
        let cost = adapter.calculate_cost(100, 50);
        assert!(cost.units <= adapter.max_units() as u32);

        // Large request should not fit
        let cost = adapter.calculate_cost(2000, 1000);
        assert!(cost.units > adapter.max_units() as u32);
    }
}
