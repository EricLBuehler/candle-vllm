//! Prometheus parking-lot scheduler integration for candle-vllm.
//!
//! This module provides integration with the prometheus_parking_lot crate
//! for resource-constrained scheduling of LLM inference requests.
//!
//! # Architecture
//!
//! The parking-lot scheduler manages GPU resources (primarily KV-cache blocks)
//! and queues excess work when capacity is exhausted:
//!
//! - `InferenceJob`: The task payload describing an inference request
//! - `InferenceResult`: The result type (completion or streaming tokens)
//! - `LlmExecutor`: The task executor that processes inference jobs
//! - `ResourceAdapter`: Maps KV-cache blocks to generic resource units
//!
//! # Design Notes
//!
//! We use prometheus_parking_lot primitives (TaskId, Priority, ResourceCost, etc.)
//! for type compatibility, but define our own `TaskExecutor` trait because LLM
//! streaming results contain channels that cannot be serialized as required by
//! the library's `TaskExecutor` trait.

pub mod executor;
pub mod job;
pub mod resource_adapter;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export executor
pub use executor::LlmExecutor;

// Re-export job types
pub use job::{InferenceJob, InferenceResult, SerializableInferenceResult, StreamingTokenResult};

// Re-export resource adapter
pub use resource_adapter::{calculate_resource_cost, ResourceAdapter};

// Re-export types
pub use types::{
    // prometheus_parking_lot primitives
    InMemoryMailbox,
    InMemoryQueue,
    Mailbox,
    MailboxKey,
    PoolLimits,
    Priority,
    ResourceCost,
    ResourceCostExt,
    ResourceKind,
    ScheduledTask,
    Spawn,
    TaskId,
    TaskQueue,
    TaskStatus,
    TokioSpawner,
    WakeState,
    now_ms,
    // Local types
    TaskExecutor,
    TaskMetadata,
    TenantId,
};
