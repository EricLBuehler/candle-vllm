# Prometheus Parking-Lot Scheduler

This document describes the inference engine architecture based on the `prometheus-parking-lot` crate for resource-aware scheduling.

## Overview

The candle-vllm inference engine provides:

- **Resource-aware scheduling**: Tracks GPU KV-cache block usage and queues requests when capacity is exhausted
- **Automatic backpressure**: Rejects requests when queue depth exceeds configured limits
- **Async-first design**: Native async/await support with tokio integration
- **Efficient capacity tracking**: Lock-free atomic operations for capacity accounting

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        LLMEngine                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐                                           │
│  │ResourceAdapter│──▶ calculates KV-cache cost per request  │
│  └──────────────┘                                           │
│          │                                                  │
│          ▼                                                  │
│  ┌──────────────┐   capacity    ┌──────────────────────────┐│
│  │ Capacity     │   check       │     LlmExecutor          ││
│  │ Tracking     │──────────────▶│ (TaskExecutor trait)     ││
│  │ (atomics)    │               │                          ││
│  └──────────────┘               └──────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **ResourceAdapter**: Maps tokens to KV-cache blocks for resource accounting
- **LlmExecutor**: Implements the `TaskExecutor` trait for inference
- **Capacity tracking**: Uses atomic operations for lock-free capacity checks
- **Request management**: Requests are rejected when capacity is exceeded

## Configuration

### SchedulerPoolConfig

```rust
pub struct SchedulerPoolConfig {
    /// Maximum resource units (GPU blocks) the pool can use
    pub max_units: usize,
    /// Maximum queue depth before rejecting requests
    pub max_queue_depth: usize,
    /// Default timeout in seconds for queued requests
    pub default_timeout_secs: u64,
}
```

Default values:
- `max_units`: Derived from cache config (`num_gpu_blocks`)
- `max_queue_depth`: 1000
- `default_timeout_secs`: 120

### Environment Variables

The scheduler respects existing configuration:
- `--mem` / `kvcache-mem-gpu`: GPU memory for KV-cache (affects `max_units`)
- `--max-num-seqs`: Maximum concurrent sequences (affects queue behavior)

## Usage

### Building

```bash
# Build for Metal (macOS)
cargo build --release --features metal

# Build for CUDA
cargo build --release --features cuda

# Run the server
cargo run --release --features metal -- --model <model-id>
```

### Library Usage

```rust
use candle_vllm_core::openai::pipelines::{LLMEngine, SchedulerPoolConfig};

// Create engine with custom pool config
let engine = LLMEngine::new(
    pipelines,
    scheduler_config,
    &cache_config,
    &model_config,
    notify,
    Some(SchedulerPoolConfig {
        max_units: 16384,
        max_queue_depth: 1000,
        default_timeout_secs: 120,
    }),
    #[cfg(feature = "nccl")]
    None,
)?;
```

### Request Submission

```rust
// Completion request (non-streaming)
let response_rx = engine.add_request(
    request_id,
    tokens,
    positions,
    sampling_params,
    max_context_len,
).await?;

// Wait for result
let result = response_rx.await?;

// Streaming request
let stream_rx = engine.add_streaming_request(
    request_id,
    tokens,
    positions,
    sampling_params,
    created,
    max_context_len,
).await?;

// Process streaming tokens
while let Ok(token_result) = stream_rx.recv_async().await {
    match token_result {
        Ok(token) => {
            // Process token
            if token.is_finished {
                break;
            }
        }
        Err(e) => {
            // Handle error
        }
    }
}
```

## Types

### InferenceJob

```rust
pub struct InferenceJob {
    pub request_id: String,
    pub tokens: Vec<u32>,
    pub positions: Vec<usize>,
    pub is_streaming: bool,
    pub sampling_params: SamplingParams,
    pub created: u64,
    pub max_context_len: usize,
}
```

### InferenceResult

```rust
pub enum InferenceResult {
    Completion {
        choices: Vec<ChatChoice>,
        usage: ChatCompletionUsageResponse,
    },
    Streaming {
        request_id: String,
        token_rx: flume::Receiver<Result<StreamingTokenResult, String>>,
    },
    Error {
        message: String,
    },
}
```

### StreamingTokenResult

```rust
pub struct StreamingTokenResult {
    pub text: String,
    pub token_id: u32,
    pub is_finished: bool,
    pub finish_reason: Option<String>,
    pub is_reasoning: bool,
}
```

## Performance Considerations

The engine provides:
- Lock-free capacity tracking using atomic operations (negligible overhead)
- O(1) resource cost calculation per request
- Async execution with tokio for efficient I/O

Benefits:
- Better resource utilization under load
- Predictable behavior when GPU memory is constrained
- Automatic request rejection instead of OOM errors

## Integration with prometheus-parking-lot

The engine uses types from `prometheus-parking-lot`:

- **TaskId**: Unique task identifier (`u64`)
- **Priority**: Task scheduling priority (Low, Normal, High, Critical)
- **ResourceCost**: Cost in resource units with kind (GpuVram, Cpu, etc.)
- **TaskMetadata**: Metadata for scheduled tasks

Custom extensions:
- **TaskExecutor**: Local trait for LLM-specific execution (doesn't require serializable results)
- **ResourceAdapter**: Maps LLM-specific resources to generic resource units
- **LlmExecutor**: Implements TaskExecutor for inference jobs

## Known Limitations

1. **Graph capture**: CUDA graph capture is not currently supported
2. **Multi-GPU**: Currently uses only the first pipeline; multi-GPU support is planned
3. **Prefill chunking**: Not yet implemented

## Future Work

- Full `ResourcePool` integration with `Mailbox` for result persistence
- Multi-GPU support via multiple executors
- Persistent task queues (Postgres, Yaque backends)
- Priority-based scheduling
- Task deadline support
