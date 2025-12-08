//! Worker pool wrapper for prometheus-parking-lot.
//!
//! This module provides `InferenceWorkerPool`, which wraps prometheus-parking-lot's
//! `WorkerPool` and integrates it with our LLM inference pipeline.
//!
//! The pool manages dedicated worker threads for CPU/GPU-bound inference work,
//! ensuring proper OS-level thread isolation from the async runtime.

use super::{
    InferenceJob, InferenceResult, LlmExecutor, SerializableInferenceResult, StreamingRegistry,
    TaskMetadata,
};
use prometheus_parking_lot::config::WorkerPoolConfig as PrometheusWorkerPoolConfig;
use prometheus_parking_lot::core::{PoolError, WorkerPool};
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info};

/// Configuration for the inference worker pool.
///
/// This wraps prometheus-parking-lot's `WorkerPoolConfig` with LLM-specific defaults.
#[derive(Debug, Clone)]
pub struct InferenceWorkerPoolConfig {
    /// Number of dedicated worker threads (default: num_cpus)
    pub worker_count: usize,

    /// Maximum resource units (GPU VRAM in MB or KV cache blocks)
    pub max_units: u32,

    /// Maximum queue depth before rejection
    pub max_queue_depth: usize,

    /// Default timeout for job execution in seconds
    pub timeout_secs: u64,
}

impl Default for InferenceWorkerPoolConfig {
    fn default() -> Self {
        Self {
            worker_count: num_cpus::get(),
            max_units: 16384, // ~256K tokens with 16-token blocks
            max_queue_depth: 1000,
            timeout_secs: 120,
        }
    }
}

impl InferenceWorkerPoolConfig {
    /// Create a new config with explicit values.
    #[must_use]
    pub fn new(worker_count: usize, max_units: u32, max_queue_depth: usize) -> Self {
        Self {
            worker_count,
            max_units,
            max_queue_depth,
            timeout_secs: 120,
        }
    }

    /// Set the timeout in seconds.
    #[must_use]
    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }
}

/// Convert to prometheus-parking-lot's WorkerPoolConfig.
impl From<InferenceWorkerPoolConfig> for PrometheusWorkerPoolConfig {
    fn from(config: InferenceWorkerPoolConfig) -> Self {
        PrometheusWorkerPoolConfig::new()
            .with_worker_count(config.worker_count)
            .with_max_units(config.max_units)
            .with_max_queue_depth(config.max_queue_depth)
    }
}

/// Worker pool for LLM inference using prometheus-parking-lot.
///
/// This pool provides:
/// - Dedicated OS threads for CPU/GPU-bound inference work
/// - Resource tracking (GPU VRAM / KV cache blocks)
/// - Priority-based queue management
/// - Graceful degradation under load
/// - Integration with StreamingRegistry for non-serializable results
pub struct InferenceWorkerPool {
    /// The underlying prometheus-parking-lot WorkerPool
    pool: Arc<WorkerPool<InferenceJob, InferenceResult, LlmExecutor>>,

    /// Streaming channel registry for non-serializable results
    streaming_registry: Arc<StreamingRegistry>,

    /// Configuration
    config: InferenceWorkerPoolConfig,
}

impl InferenceWorkerPool {
    /// Create a new inference worker pool.
    ///
    /// # Arguments
    ///
    /// * `executor` - The LLM executor that processes inference jobs
    /// * `streaming_registry` - Registry for managing streaming token channels
    /// * `config` - Pool configuration
    ///
    /// # Errors
    ///
    /// Returns error if pool creation fails (e.g., unable to spawn worker threads).
    pub fn new(
        executor: LlmExecutor,
        streaming_registry: StreamingRegistry,
        config: InferenceWorkerPoolConfig,
    ) -> Result<Self, PoolError> {
        info!(
            "🏗️ WORKER_POOL: Creating inference worker pool - workers={}, max_units={}, max_queue={}",
            config.worker_count, config.max_units, config.max_queue_depth
        );

        let pool_config: PrometheusWorkerPoolConfig = config.clone().into();
        let pool = WorkerPool::new(pool_config, executor)?;

        info!(
            "✅ WORKER_POOL: Worker pool created successfully - {} dedicated worker threads",
            config.worker_count
        );

        Ok(Self {
            pool: Arc::new(pool),
            streaming_registry: Arc::new(streaming_registry),
            config,
        })
    }

    /// Submit an inference job to the pool.
    ///
    /// This method submits a job and waits for the result. For streaming jobs,
    /// the result will contain a channel key that can be used to retrieve the
    /// streaming token channel from the registry.
    ///
    /// # Arguments
    ///
    /// * `job` - The inference job to execute
    /// * `meta` - Task metadata (priority, cost, etc.)
    ///
    /// # Returns
    ///
    /// Returns a `SerializableInferenceResult` which is either:
    /// - `Completion` - Contains the full generated text and usage info
    /// - `StreamingChannel` - Contains a channel key for retrieving tokens
    /// - `Error` - Contains error message if inference failed
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Queue is full (queue at capacity)
    /// - Job times out (exceeds configured timeout)
    /// - Pool is shutting down
    pub async fn submit(
        &self,
        job: InferenceJob,
        meta: TaskMetadata,
    ) -> Result<SerializableInferenceResult, PoolError> {
        let request_id = job.request_id.clone();
        let is_streaming = job.is_streaming;

        info!(
            "📥 WORKER_POOL: Submitting job - request_id={}, streaming={}, cost={}",
            request_id, is_streaming, meta.cost.units
        );

        // Submit job to pool and get result key
        let key = self.pool.submit_async(job, meta.into()).await?;

        info!(
            "🎫 WORKER_POOL: Job queued - request_id={}, key={:?}",
            request_id, key
        );

        // Retrieve result with timeout
        let timeout = Duration::from_secs(self.config.timeout_secs);
        let result = self.pool.retrieve_async(&key, timeout).await?;

        info!(
            "✅ WORKER_POOL: Job result retrieved - request_id={}, type={:?}",
            request_id,
            match &result {
                InferenceResult::Completion { .. } => "Completion",
                InferenceResult::Streaming { .. } => "Streaming",
                InferenceResult::Error { .. } => "Error",
            }
        );

        // Convert to serializable form
        let serializable = match result {
            InferenceResult::Completion { choices, usage } => {
                info!(
                    "📊 WORKER_POOL: Completion result - request_id={}, tokens={}",
                    request_id, usage.total_tokens
                );
                SerializableInferenceResult::completion(choices, usage)
            }
            InferenceResult::Streaming {
                request_id: req_id,
                token_rx,
            } => {
                let channel_key = uuid::Uuid::new_v4().to_string();
                info!(
                    "📡 WORKER_POOL: Streaming result, registering channel - request_id={}, key={}",
                    request_id, channel_key
                );
                self.streaming_registry
                    .register(channel_key.clone(), req_id.clone(), token_rx);
                SerializableInferenceResult::streaming_channel(req_id, channel_key)
            }
            InferenceResult::Error { message } => {
                error!(
                    "❌ WORKER_POOL: Inference error - request_id={}, error={}",
                    request_id, message
                );
                SerializableInferenceResult::error(message)
            }
        };

        Ok(serializable)
    }

    /// Get pool statistics.
    ///
    /// Returns current pool state including active tasks, queue depth,
    /// resource usage, etc.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let prometheus_stats = self.pool.stats();

        PoolStats {
            worker_threads: prometheus_stats.worker_count,
            active_tasks: prometheus_stats.active_tasks as usize,
            queued_tasks: prometheus_stats.queued_tasks as usize,
            used_units: prometheus_stats.used_units as usize,
            total_units: self.config.max_units as usize,
            completed_tasks: prometheus_stats.completed_tasks,
            failed_tasks: prometheus_stats.failed_tasks,
        }
    }

    /// Get number of available execution slots.
    #[must_use]
    pub fn available_permits(&self) -> usize {
        let stats = self.pool.stats();
        self.config
            .worker_count
            .saturating_sub(stats.active_tasks as usize)
    }

    /// Get current queue depth.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.pool.stats().queued_tasks as usize
    }

    /// Get the streaming registry.
    #[must_use]
    pub fn streaming_registry(&self) -> &Arc<StreamingRegistry> {
        &self.streaming_registry
    }
}

/// Pool statistics for monitoring.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Number of worker threads
    pub worker_threads: usize,

    /// Currently executing tasks
    pub active_tasks: usize,

    /// Tasks waiting in queue
    pub queued_tasks: usize,

    /// Used resource units (VRAM MB or KV cache blocks)
    pub used_units: usize,

    /// Total resource units
    pub total_units: usize,

    /// Tasks completed successfully
    pub completed_tasks: u64,

    /// Tasks that failed
    pub failed_tasks: u64,
}

impl PoolStats {
    /// Calculate resource utilization as a percentage (0-100).
    #[must_use]
    pub fn utilization_percent(&self) -> f64 {
        if self.total_units == 0 {
            0.0
        } else {
            (self.used_units as f64 / self.total_units as f64) * 100.0
        }
    }

    /// Check if pool is at or near capacity.
    #[must_use]
    pub fn is_near_capacity(&self) -> bool {
        self.utilization_percent() > 80.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = InferenceWorkerPoolConfig::default();
        assert!(config.worker_count > 0);
        assert_eq!(config.max_units, 16384);
        assert_eq!(config.max_queue_depth, 1000);
        assert_eq!(config.timeout_secs, 120);
    }

    #[test]
    fn test_config_custom() {
        let config = InferenceWorkerPoolConfig::new(8, 8000, 500).with_timeout_secs(60);

        assert_eq!(config.worker_count, 8);
        assert_eq!(config.max_units, 8000);
        assert_eq!(config.max_queue_depth, 500);
        assert_eq!(config.timeout_secs, 60);
    }

    #[test]
    fn test_pool_stats_utilization() {
        let stats = PoolStats {
            worker_threads: 4,
            active_tasks: 2,
            queued_tasks: 5,
            used_units: 8000,
            total_units: 10000,
            completed_tasks: 100,
            failed_tasks: 2,
        };

        assert_eq!(stats.utilization_percent(), 80.0);
        assert!(stats.is_near_capacity());
    }

    #[test]
    fn test_pool_stats_zero_capacity() {
        let stats = PoolStats {
            worker_threads: 4,
            active_tasks: 0,
            queued_tasks: 0,
            used_units: 0,
            total_units: 0,
            completed_tasks: 0,
            failed_tasks: 0,
        };

        assert_eq!(stats.utilization_percent(), 0.0);
        assert!(!stats.is_near_capacity());
    }
}
