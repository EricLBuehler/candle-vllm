///! Test configuration builders.

use candle_vllm_core::scheduler::SchedulerConfig;
use candle_vllm_core::parking_lot::InferenceWorkerPoolConfig;
use candle_vllm_core::openai::sampling_params::{SamplingParams, EarlyStoppingCondition};

/// Create a test engine config.
pub fn test_engine_config() -> candle_vllm_core::api::EngineConfig {
    candle_vllm_core::api::EngineConfig {
        max_num_seqs: 16,
        max_model_len: 256,
        max_num_batched_tokens: 512,
        block_size: 16,
        gpu_memory_utilization: 0.9,
        swap_space: 4,
        cache_dtype: "auto".to_string(),
        num_gpu_blocks: None,
        num_cpu_blocks: None,
    }
}

/// Create a test scheduler config.
pub fn test_scheduler_config() -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs: 16,
        max_model_len: 256,
        max_num_batched_tokens: 512,
    }
}

/// Create a test parking lot config.
pub fn test_parking_lot_config() -> InferenceWorkerPoolConfig {
    InferenceWorkerPoolConfig {
        worker_count: 4,
        max_units: 1000,
        max_queue_depth: 100,
        timeout_secs: 60,
    }
}

/// Create default SamplingParams for testing.
/// This is a convenience function since SamplingParams doesn't implement Default.
pub fn default_sampling_params() -> SamplingParams {
    SamplingParams::new(
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
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        None,
        vec![],
        false,
        16,
        None,
        None,
        true,
        None,
    ).expect("Failed to create default SamplingParams")
}
