# candle-vllm Migration to prometheus-parking-lot

**Date:** December 7, 2025  
**Status:** Ready for Implementation  
**Library Version:** prometheus-parking-lot v0.1.0

---

## Executive Summary

This document provides complete implementation guidance for migrating `candle-vllm` from its custom `thread_pool.rs` implementation to the `prometheus-parking-lot` library. This migration eliminates ALL custom thread management code from candle-vllm and delegates it to the battle-tested parking-lot library.

**Key Benefits:**
1. ✅ **Zero thread management code** - candle-vllm becomes purely focused on inference logic
2. ✅ **Proper OS thread isolation** - CPU/GPU work runs on dedicated worker threads with their own Tokio runtimes
3. ✅ **Non-serializable results** - Supports streaming channels (`flume::Receiver`) without serialization hacks
4. ✅ **Built-in resource management** - GPU VRAM tracking, priority queuing, graceful degradation
5. ✅ **Cross-platform ready** - Same API works on native and WASM (when needed)
6. ✅ **Battle-tested** - 100% test coverage with dedicated candle-vllm integration tests

---

## Migration Strategy

### Phase 1: Add Dependency
### Phase 2: Implement WorkerExecutor
### Phase 3: Replace ThreadPool with WorkerPool
### Phase 4: Update HTTP Handlers
### Phase 5: Remove Legacy Code
### Phase 6: Test & Validate

---

## Phase 1: Add Dependency

### 1.1 Update `Cargo.toml`

Add to your `[dependencies]`:

```toml
[dependencies]
# ... existing dependencies ...
prometheus_parking_lot = { git = "https://github.com/prometheus-ai/prometheus-parking-lot.git", branch = "main" }
```

The library provides these key types:
- `WorkerPool<P, R, E>` - The main worker pool
- `WorkerExecutor<P, R>` - Trait your executor implements
- `WorkerPoolConfig` - Configuration builder
- `TaskMetadata` - Task metadata with resource costs
- `ResourceCost`, `ResourceKind`, `Priority` - Resource tracking types

### 1.2 Verify Installation

```bash
cargo check
```

---

## Phase 2: Implement WorkerExecutor

### 2.1 Define Your Job and Result Types

Create types that represent your inference work:

```rust
// File: src/executor.rs (NEW FILE)

use serde::{Deserialize, Serialize};

/// Inference job payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceJob {
    pub request_id: String,
    pub prompt: String,
    pub model_name: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub is_streaming: bool,
    // Add other parameters from your current Request type
}

/// Inference result - can be streaming OR completion
/// Note: This does NOT need Serialize/Deserialize!
pub enum InferenceResult {
    /// Streaming response with token channel
    Streaming {
        /// Channel for receiving generated tokens
        tokens: flume::Receiver<String>,
        /// Total expected tokens (estimate)
        estimated_tokens: usize,
    },
    /// Complete response (non-streaming)
    Completion {
        /// Generated text
        text: String,
        /// Total tokens generated
        token_count: usize,
        /// Generation time in milliseconds
        generation_time_ms: u64,
    },
}
```

### 2.2 Implement WorkerExecutor Trait

Wrap your existing inference pipeline in the `WorkerExecutor` trait:

```rust
// File: src/executor.rs (continued)

use async_trait::async_trait;
use prometheus_parking_lot::core::{WorkerExecutor, TaskMetadata};
use std::sync::Arc;
use std::time::Instant;

/// Your executor wraps your existing Pipeline and CacheEngine
#[derive(Clone)]
pub struct LlmExecutor {
    // Your existing pipeline - this is what does the actual inference
    pipeline: Arc<Pipeline>,
    cache_engine: Arc<CacheEngine>,
    // Add any other state you need (config, model registry, etc.)
}

impl LlmExecutor {
    pub fn new(pipeline: Arc<Pipeline>, cache_engine: Arc<CacheEngine>) -> Self {
        Self {
            pipeline,
            cache_engine,
        }
    }
}

#[async_trait]
impl WorkerExecutor<InferenceJob, InferenceResult> for LlmExecutor {
    async fn execute(&self, job: InferenceJob, meta: TaskMetadata) -> InferenceResult {
        // CRITICAL: This method runs in a DEDICATED WORKER THREAD!
        // You have a full single-threaded Tokio runtime here.
        // DO NOT spawn additional threads - use tokio::spawn or spawn_blocking.
        
        tracing::info!(
            request_id = %job.request_id,
            task_id = meta.id,
            priority = ?meta.priority,
            cost = meta.cost.units,
            "Starting inference"
        );
        
        let start = Instant::now();
        
        if job.is_streaming {
            // Streaming inference
            let (tx, rx) = flume::unbounded();
            
            // Clone what you need for the generation task
            let pipeline = self.pipeline.clone();
            let cache_engine = self.cache_engine.clone();
            let prompt = job.prompt.clone();
            let max_tokens = job.max_tokens;
            
            // Spawn token generation on the current runtime
            // (remember: we're already on a dedicated worker thread)
            tokio::spawn(async move {
                match pipeline.generate_streaming(&prompt, max_tokens).await {
                    Ok(mut token_stream) => {
                        while let Some(token_result) = token_stream.next().await {
                            match token_result {
                                Ok(token) => {
                                    if tx.send(token).is_err() {
                                        // Receiver dropped - client disconnected
                                        break;
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Token generation error: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Streaming generation failed: {}", e);
                    }
                }
            });
            
            InferenceResult::Streaming {
                tokens: rx,
                estimated_tokens: job.max_tokens,
            }
        } else {
            // Non-streaming (completion) inference
            match self.pipeline.generate(&job.prompt, job.max_tokens).await {
                Ok(text) => {
                    let elapsed = start.elapsed().as_millis() as u64;
                    let token_count = text.split_whitespace().count(); // Simplified
                    
                    tracing::info!(
                        request_id = %job.request_id,
                        tokens = token_count,
                        duration_ms = elapsed,
                        "Inference completed"
                    );
                    
                    InferenceResult::Completion {
                        text,
                        token_count,
                        generation_time_ms: elapsed,
                    }
                }
                Err(e) => {
                    tracing::error!("Inference failed: {}", e);
                    // You might want a Result enum or error variant in InferenceResult
                    InferenceResult::Completion {
                        text: format!("Error: {}", e),
                        token_count: 0,
                        generation_time_ms: start.elapsed().as_millis() as u64,
                    }
                }
            }
        }
    }
}
```

---

## Phase 3: Replace ThreadPool with WorkerPool

### 3.1 Remove Old ThreadPool Initialization

**BEFORE (in your main.rs or server setup):**
```rust
// OLD - DELETE THIS
let thread_pool = ThreadPool::new(num_workers)?;
let thread_pool = Arc::new(thread_pool);
```

**AFTER:**
```rust
// NEW
use prometheus_parking_lot::core::WorkerPool;
use prometheus_parking_lot::config::WorkerPoolConfig;

// Create your executor
let executor = LlmExecutor::new(pipeline.clone(), cache_engine.clone());

// Configure pool
let pool_config = WorkerPoolConfig::new()
    .with_worker_count(num_cpus::get())  // Or your preferred count
    .with_max_units(8000)                 // GPU VRAM in MB (e.g., 8GB GPU)
    .with_max_queue_depth(1000);          // Max queued requests

// Create pool
let worker_pool = WorkerPool::new(pool_config, executor)
    .expect("Failed to create worker pool");

// Wrap in Arc for sharing across handlers
let worker_pool = Arc::new(worker_pool);
```

### 3.2 Update Your Application State

**BEFORE:**
```rust
struct AppState {
    thread_pool: Arc<ThreadPool>,
    // ... other fields
}
```

**AFTER:**
```rust
struct AppState {
    worker_pool: Arc<WorkerPool<InferenceJob, InferenceResult, LlmExecutor>>,
    // ... other fields
}
```

---

## Phase 4: Update HTTP Handlers

### 4.1 Completion Endpoint (Non-Streaming)

**BEFORE (with custom thread_pool):**
```rust
async fn handle_completion(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    // OLD - custom thread management
    let handle = state.thread_pool.spawn(move || {
        // Run inference...
    });
    
    let result = handle.await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Task failed: {}", e)))?;
    
    Ok(Json(result))
}
```

**AFTER (with WorkerPool):**
```rust
use prometheus_parking_lot::util::{Priority, ResourceCost, ResourceKind};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

async fn handle_completion(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    // Create inference job
    let job = InferenceJob {
        request_id: request.id.clone(),
        prompt: request.prompt,
        model_name: request.model.unwrap_or_else(|| "default".to_string()),
        max_tokens: request.max_tokens.unwrap_or(512),
        temperature: request.temperature.unwrap_or(0.7),
        is_streaming: false,
    };
    
    // Create task metadata with resource cost
    let meta = TaskMetadata {
        id: generate_task_id(),  // Your ID generation
        mailbox: None,            // Not using mailbox for direct responses
        priority: Priority::Normal,
        cost: ResourceCost {
            kind: ResourceKind::GpuVram,
            units: estimate_vram_mb(&request),  // Your VRAM estimation function
        },
        deadline_ms: None,
        created_at_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    };
    
    // Submit to pool (async API)
    let key = state.worker_pool
        .submit_async(job, meta)
        .await
        .map_err(|e| match e {
            PoolError::QueueFull(_) => (
                StatusCode::TOO_MANY_REQUESTS,
                "Server is at capacity, please try again later".to_string()
            ),
            PoolError::Shutdown => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Server is shutting down".to_string()
            ),
            other => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to submit task: {:?}", other)
            ),
        })?;
    
    // Retrieve result with timeout
    let result = state.worker_pool
        .retrieve_async(&key, Duration::from_secs(120))
        .await
        .map_err(|e| (
            StatusCode::GATEWAY_TIMEOUT,
            format!("Inference timeout: {:?}", e)
        ))?;
    
    // Extract completion result
    match result {
        InferenceResult::Completion { text, token_count, generation_time_ms } => {
            Ok(Json(CompletionResponse {
                id: request.id,
                text,
                tokens: token_count,
                generation_time_ms,
            }))
        }
        InferenceResult::Streaming { .. } => {
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Unexpected streaming result for completion request".to_string()
            ))
        }
    }
}
```

### 4.2 Streaming Endpoint

**BEFORE:**
```rust
async fn handle_streaming(
    State(state): State<Arc<AppState>>,
    Json(request): Json<StreamRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // OLD - custom channel and thread management
    let (tx, rx) = mpsc::channel(100);
    
    state.thread_pool.spawn(move || {
        // Generate tokens and send to channel...
    });
    
    let stream = ReceiverStream::new(rx).map(|text| {
        Ok(Event::default().data(text))
    });
    
    Sse::new(stream)
}
```

**AFTER:**
```rust
use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use std::convert::Infallible;

async fn handle_streaming(
    State(state): State<Arc<AppState>>,
    Json(request): Json<StreamRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    // Create streaming job
    let job = InferenceJob {
        request_id: request.id.clone(),
        prompt: request.prompt,
        model_name: request.model.unwrap_or_else(|| "default".to_string()),
        max_tokens: request.max_tokens.unwrap_or(512),
        temperature: request.temperature.unwrap_or(0.7),
        is_streaming: true,  // IMPORTANT!
    };
    
    let meta = TaskMetadata {
        id: generate_task_id(),
        mailbox: None,
        priority: Priority::Normal,
        cost: ResourceCost {
            kind: ResourceKind::GpuVram,
            units: estimate_vram_mb(&request),
        },
        deadline_ms: None,
        created_at_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    };
    
    // Submit to pool
    let key = state.worker_pool
        .submit_async(job, meta)
        .await
        .map_err(|e| match e {
            PoolError::QueueFull(_) => (
                StatusCode::TOO_MANY_REQUESTS,
                "Server is at capacity".to_string()
            ),
            other => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to submit: {:?}", other)
            ),
        })?;
    
    // Retrieve result (which should be a streaming result)
    let result = state.worker_pool
        .retrieve_async(&key, Duration::from_secs(5))  // Quick timeout to get the channel
        .await
        .map_err(|e| (
            StatusCode::GATEWAY_TIMEOUT,
            format!("Failed to start streaming: {:?}", e)
        ))?;
    
    // Extract the streaming channel
    match result {
        InferenceResult::Streaming { tokens, .. } => {
            // Convert flume::Receiver to Stream for SSE
            let stream = async_stream::stream! {
                while let Ok(token) = tokens.recv_async().await {
                    yield Ok::<_, Infallible>(Event::default().data(token));
                }
            };
            
            Ok(Sse::new(stream))
        }
        InferenceResult::Completion { .. } => {
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Unexpected completion result for streaming request".to_string()
            ))
        }
    }
}
```

---

## Phase 5: Remove Legacy Code

### 5.1 Delete Custom Thread Pool

Remove the following files (if they exist):
- `src/thread_pool.rs`
- Any custom worker/task queue implementations
- Any custom channel/signaling code

### 5.2 Clean Up Dependencies

Remove from `Cargo.toml` if no longer needed:
- Any custom thread pool crates
- Custom channel implementations (if only used for worker pool)

### 5.3 Update Imports

Search and replace across the codebase:
```rust
// OLD
use crate::thread_pool::ThreadPool;

// NEW
use prometheus_parking_lot::core::WorkerPool;
use prometheus_parking_lot::config::WorkerPoolConfig;
```

---

## Phase 6: Test & Validate

### 6.1 Unit Tests

Add tests for your executor:

```rust
// tests/executor_test.rs

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus_parking_lot::util::{Priority, ResourceCost, ResourceKind};
    use std::time::{SystemTime, UNIX_EPOCH};
    
    fn make_meta(id: u64, vram_mb: u32) -> TaskMetadata {
        TaskMetadata {
            id,
            mailbox: None,
            priority: Priority::Normal,
            cost: ResourceCost {
                kind: ResourceKind::GpuVram,
                units: vram_mb,
            },
            deadline_ms: None,
            created_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        }
    }
    
    #[tokio::test]
    async fn test_completion_inference() {
        let executor = create_test_executor();  // Your test executor factory
        
        let job = InferenceJob {
            request_id: "test-1".to_string(),
            prompt: "Hello".to_string(),
            model_name: "test-model".to_string(),
            max_tokens: 10,
            temperature: 0.7,
            is_streaming: false,
        };
        
        let result = executor.execute(job, make_meta(1, 100)).await;
        
        match result {
            InferenceResult::Completion { text, .. } => {
                assert!(!text.is_empty());
            }
            _ => panic!("Expected completion result"),
        }
    }
    
    #[tokio::test]
    async fn test_streaming_inference() {
        let executor = create_test_executor();
        
        let job = InferenceJob {
            request_id: "test-2".to_string(),
            prompt: "Stream test".to_string(),
            model_name: "test-model".to_string(),
            max_tokens: 5,
            temperature: 0.7,
            is_streaming: true,
        };
        
        let result = executor.execute(job, make_meta(2, 100)).await;
        
        match result {
            InferenceResult::Streaming { tokens, .. } => {
                let mut count = 0;
                while let Ok(_token) = tokens.recv_async().await {
                    count += 1;
                    if count >= 5 {
                        break;  // Prevent infinite loop in test
                    }
                }
                assert!(count > 0, "Should receive at least one token");
            }
            _ => panic!("Expected streaming result"),
        }
    }
}
```

### 6.2 Integration Tests

Test the full pool integration:

```rust
// tests/pool_integration_test.rs

use prometheus_parking_lot::core::WorkerPool;
use prometheus_parking_lot::config::WorkerPoolConfig;

#[tokio::test]
async fn test_pool_capacity_enforcement() {
    let config = WorkerPoolConfig::new()
        .with_worker_count(2)
        .with_max_units(200)  // 200MB total VRAM
        .with_max_queue_depth(10);
    
    let pool = WorkerPool::new(config, executor).unwrap();
    
    // Submit task requiring 150MB
    let meta1 = make_meta(1, 150);
    let key1 = pool.submit_async(job1, meta1).await.unwrap();
    
    // Submit task requiring 100MB - should queue (150 + 100 > 200)
    let meta2 = make_meta(2, 100);
    let key2 = pool.submit_async(job2, meta2).await.unwrap();
    
    // Check that second task is queued
    let stats = pool.stats();
    assert!(stats.queued_tasks > 0 || stats.active_tasks == 2);
    
    // Both should complete
    let _ = pool.retrieve_async(&key1, Duration::from_secs(10)).await.unwrap();
    let _ = pool.retrieve_async(&key2, Duration::from_secs(10)).await.unwrap();
}

#[tokio::test]
async fn test_graceful_degradation() {
    let config = WorkerPoolConfig::new()
        .with_worker_count(1)
        .with_max_units(100)
        .with_max_queue_depth(2);  // Very small queue
    
    let pool = WorkerPool::new(config, executor).unwrap();
    
    // Fill queue completely
    for i in 0..3 {
        let _ = pool.submit_async(job.clone(), make_meta(i, 50)).await;
    }
    
    // Next submission should fail with QueueFull
    let result = pool.submit_async(job.clone(), make_meta(999, 50)).await;
    assert!(matches!(result, Err(PoolError::QueueFull(_))));
}
```

### 6.3 Load Testing

Test under realistic load:

```bash
# Use your existing load test tools
# Verify:
# 1. Throughput matches or exceeds old thread pool
# 2. Latency p50, p95, p99 are acceptable
# 3. No memory leaks under sustained load
# 4. Graceful queue overflow handling
```

---

## Configuration Reference

### WorkerPoolConfig Options

```rust
WorkerPoolConfig::new()
    .with_worker_count(num_workers)      // Number of OS threads (default: num_cpus)
    .with_max_units(vram_mb)              // Total resource capacity
    .with_max_queue_depth(max_queued)     // Max queued tasks before rejection
```

### Resource Cost Estimation

Helper function to estimate VRAM usage:

```rust
fn estimate_vram_mb(request: &InferenceRequest) -> u32 {
    // Base model size
    let model_size_mb = 4000;  // e.g., 4GB for 7B model
    
    // KV cache for context
    let context_tokens = request.prompt.split_whitespace().count();
    let kv_cache_mb = (context_tokens * 2) / 1024;  // Rough estimate
    
    // Generation buffer
    let generation_mb = (request.max_tokens.unwrap_or(512) * 2) / 1024;
    
    model_size_mb + kv_cache_mb + generation_mb
}
```

---

## Performance Tuning

### Worker Count

```rust
// For CPU-bound workloads:
let workers = num_cpus::get();

// For GPU-bound workloads (single GPU):
let workers = 1;  // Only one worker can use GPU at a time

// For GPU-bound workloads (multi-GPU):
let workers = num_gpus;  // One worker per GPU
```

### Queue Depth

```rust
// For low-latency services:
let max_queue = worker_count * 10;  // Reject quickly if overloaded

// For batch processing:
let max_queue = 10_000;  // Accept large backlogs
```

### Resource Units

```rust
// GPU VRAM (in MB):
ResourceCost {
    kind: ResourceKind::GpuVram,
    units: 4096,  // 4GB
}

// CPU (abstract units):
ResourceCost {
    kind: ResourceKind::Cpu,
    units: 100,  // Your cost model
}
```

---

## Troubleshooting

### Issue: Tasks Not Starting

**Symptom:** Tasks submitted but never execute.

**Solution:** Check resource costs. If every task costs more than `max_units`, they'll queue forever.

```rust
// Bad: task costs 5000MB, pool max is 4000MB
let meta = TaskMetadata { cost: ResourceCost { units: 5000, .. }, .. };

// Fix: reduce cost or increase pool capacity
let config = config.with_max_units(8000);
```

### Issue: QueueFull Errors

**Symptom:** `PoolError::QueueFull` under load.

**Solutions:**
1. Increase `max_queue_depth`
2. Add more workers
3. Implement backpressure in your API (return 429 Too Many Requests)

### Issue: Slow Throughput

**Symptom:** Lower throughput than old thread pool.

**Debug:**
```rust
let stats = pool.stats();
println!("Active: {}, Queued: {}, Completed: {}", 
    stats.active_tasks, 
    stats.queued_tasks,
    stats.completed_tasks
);
```

**Solutions:**
1. Increase worker count if CPU/GPU not saturated
2. Reduce per-task resource cost if overestimated
3. Check for deadlocks in executor code

### Issue: Memory Leaks

**Symptom:** Memory grows over time.

**Solutions:**
1. Ensure you're calling `retrieve_async` for every submitted task
2. Check that streaming channels are fully consumed or dropped
3. Use `pool.stats()` to verify task cleanup

---

## Migration Checklist

- [ ] Add `prometheus_parking_lot` dependency to `Cargo.toml`
- [ ] Create `InferenceJob` and `InferenceResult` types
- [ ] Implement `WorkerExecutor` trait for `LlmExecutor`
- [ ] Replace `ThreadPool` initialization with `WorkerPool`
- [ ] Update `AppState` to use `WorkerPool`
- [ ] Migrate completion endpoint to use `submit_async`/`retrieve_async`
- [ ] Migrate streaming endpoint to use `submit_async`/`retrieve_async`
- [ ] Add VRAM estimation function
- [ ] Delete `src/thread_pool.rs` and related code
- [ ] Update imports across codebase
- [ ] Add unit tests for executor
- [ ] Add integration tests for pool
- [ ] Run load tests
- [ ] Verify metrics and observability
- [ ] Update documentation
- [ ] Deploy to staging
- [ ] Monitor production metrics

---

## Example: Complete Before/After

### BEFORE (src/main.rs)

```rust
mod thread_pool;
use thread_pool::ThreadPool;

#[tokio::main]
async fn main() {
    let thread_pool = Arc::new(ThreadPool::new(4).unwrap());
    
    let app_state = Arc::new(AppState {
        thread_pool,
        pipeline,
        cache,
    });
    
    let app = Router::new()
        .route("/v1/completions", post(handle_completion))
        .with_state(app_state);
    
    // ...
}

async fn handle_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Request>,
) -> Result<Json<Response>, (StatusCode, String)> {
    let handle = state.thread_pool.spawn(move || {
        // inference...
    });
    let result = handle.await.unwrap();
    Ok(Json(result))
}
```

### AFTER (src/main.rs)

```rust
mod executor;
use executor::{LlmExecutor, InferenceJob, InferenceResult};
use prometheus_parking_lot::core::WorkerPool;
use prometheus_parking_lot::config::WorkerPoolConfig;

#[tokio::main]
async fn main() {
    let executor = LlmExecutor::new(pipeline.clone(), cache.clone());
    
    let pool_config = WorkerPoolConfig::new()
        .with_worker_count(4)
        .with_max_units(8000)
        .with_max_queue_depth(1000);
    
    let worker_pool = Arc::new(
        WorkerPool::new(pool_config, executor).unwrap()
    );
    
    let app_state = Arc::new(AppState {
        worker_pool,
        pipeline,
        cache,
    });
    
    let app = Router::new()
        .route("/v1/completions", post(handle_completion))
        .with_state(app_state);
    
    // ...
}

async fn handle_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Request>,
) -> Result<Json<Response>, (StatusCode, String)> {
    let job = InferenceJob::from(req);
    let meta = TaskMetadata::new(/* ... */);
    
    let key = state.worker_pool.submit_async(job, meta).await?;
    let result = state.worker_pool.retrieve_async(&key, Duration::from_secs(120)).await?;
    
    match result {
        InferenceResult::Completion { text, .. } => Ok(Json(Response { text })),
        _ => Err((StatusCode::INTERNAL_SERVER_ERROR, "Unexpected result".into())),
    }
}
```

---

## Additional Resources

- **Library Tests:** See `prometheus-parking-lot/tests/candle_vllm/` for working examples
- **Certification Report:** `prometheus-parking-lot/CERTIFICATION_REPORT.md`
- **API Documentation:** `prometheus-parking-lot/README.md`

---

## Support

If you encounter issues during migration:

1. Check the library's test suite for working examples
2. Verify your resource cost estimates are reasonable
3. Use `pool.stats()` to debug queue/execution state
4. Check tracing logs for detailed execution flow

The library has been tested specifically with candle-vllm patterns and is ready for production use.
