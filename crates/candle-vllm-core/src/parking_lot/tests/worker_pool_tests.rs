//! Tests for InferenceWorkerPool.

use crate::parking_lot::{InferenceWorkerPool, InferenceWorkerPoolConfig, StreamingRegistry};

#[test]
fn test_worker_pool_config_defaults() {
    let config = InferenceWorkerPoolConfig::default();
    assert!(config.worker_count > 0);
    assert_eq!(config.max_units, 16384);
    assert_eq!(config.max_queue_depth, 1000);
    assert_eq!(config.timeout_secs, 120);
}

#[test]
fn test_worker_pool_config_custom() {
    let config = InferenceWorkerPoolConfig::new(8, 8000, 500).with_timeout_secs(60);

    assert_eq!(config.worker_count, 8);
    assert_eq!(config.max_units, 8000);
    assert_eq!(config.max_queue_depth, 500);
    assert_eq!(config.timeout_secs, 60);
}

// Note: Full integration tests with real WorkerPool require a functional
// LlmExecutor, which needs a loaded model. These tests are in the
// integration test suite (tests/worker_pool_integration_test.rs).
