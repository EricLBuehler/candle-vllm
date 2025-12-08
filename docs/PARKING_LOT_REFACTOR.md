# Parking Lot Scheduler Refactor Plan

## Problem

The current implementation does NOT properly use `prometheus-parking-lot`'s `ResourcePool` for thread pool management. Instead, it:

1. **Manually spawns tokio tasks** for each request
2. **Blocks the async runtime** with CPU/GPU-bound model inference 
3. **No actual thread pool** - just direct executor calls
4. **Missing ResourcePool configuration** - no JSON config for pool limits

This causes:
- Requests hanging because CPU-bound inference blocks async runtime
- No proper thread pool for parallel request processing
- Streaming and completion requests competing for same async threads

## Solution: Proper ResourcePool Integration

### Phase 1: Add Configuration Schema

Extend `models.yaml` to include parking-lot configuration:

```yaml
# Global parking lot scheduler configuration
parking_lot:
  # Thread pool configuration
  pool:
    worker_threads: 4           # Number of worker threads for CPU-bound inference
    max_blocking_threads: 512   # Max blocking threads for tokio runtime
    thread_stack_size: 2097152  # Stack size per thread (2MB default)
  
  # Resource limits (per-model or global)
  limits:
    max_units: null             # null = derive from cache config
    max_queue_depth: 1000       # Max queued requests before rejection
    timeout_secs: 120           # Request timeout in seconds
  
  # Queue configuration  
  queue:
    backend: "memory"           # "memory" | "postgres" | "yaque"
    persistence: false          # Enable persistent queue
    
  # Mailbox configuration (for result delivery)
  mailbox:
    backend: "memory"           # "memory" | "postgres"
    retention_secs: 3600        # How long to keep results

# Default model configuration
default_model: mistral-3-ministral-3B-reasoning

models:
  - name: mistral-3-ministral-3B-reasoning
    hf_id: mistralai/Ministral-3-3B-Reasoning-2512
    params:
      dtype: f16
      kvcache_mem_gpu: 8192
      max_num_seqs: 128
      
    # Per-model parking lot overrides (optional)
    parking_lot:
      limits:
        max_units: 1260         # Override global limit
        max_queue_depth: 500    # Lower queue for this model
```

### Phase 2: Create ResourcePool with Proper Configuration

```rust
use prometheus_parking_lot::core::{ResourcePool, PoolConfig};
use prometheus_parking_lot::infra::mailbox::memory::InMemoryMailbox;
use prometheus_parking_lot::infra::queue::memory::InMemoryQueue;
use prometheus_parking_lot::runtime::tokio_spawner::TokioSpawner;

// Create pool configuration from YAML
let pool_config = PoolConfig {
    max_units: config.parking_lot.limits.max_units.unwrap_or(cache_config.num_gpu_blocks),
    max_queue_depth: config.parking_lot.limits.max_queue_depth,
    timeout_ms: config.parking_lot.limits.timeout_secs * 1000,
    worker_threads: config.parking_lot.pool.worker_threads,
};

// Create the resource pool with proper backends
let pool = ResourcePool::new(
    pool_config,
    InMemoryQueue::new(),      // Task queue
    InMemoryMailbox::new(),    // Result mailbox
    TokioSpawner::new(),       // Runtime spawner
)?;
```

### Phase 3: Implement Serializable TaskExecutor

The current custom `TaskExecutor` trait avoids serialization. We need to:

**Option A**: Use Mailbox pattern (recommended by prometheus-parking-lot)
```rust
// Submit task and get mailbox key
let mailbox_key = pool.submit(job, meta).await?;

// Poll mailbox for result
let result = pool.mailbox.retrieve(&mailbox_key).await?;
```

**Option B**: Hybrid approach (keep channels but use pool)
```rust
// LlmExecutor implements prometheus_parking_lot's TaskExecutor
impl prometheus_parking_lot::TaskExecutor for LlmExecutor {
    type Payload = SerializableInferenceJob;
    type Result = SerializableInferenceResult;
    
    async fn execute(&self, payload: Self::Payload) -> Self::Result {
        // Convert to internal types
        let job = InferenceJob::from(payload);
        
        // Do inference (this runs in worker thread pool)
        let result = self.process_job(&job);
        
        // Convert back to serializable
        SerializableInferenceResult::from(result)
    }
}
```

### Phase 4: Fix Thread Pool Usage

**Current (WRONG)**:
```rust
// Directly awaiting in async context - blocks runtime!
let result = executor.execute(job, meta).await;
```

**Correct**:
```rust
// Submit to ResourcePool - it manages worker threads
let task_id = resource_pool.submit(
    job,
    meta,
    executor,
).await?;

// Get result from mailbox (non-blocking)
let result = resource_pool.retrieve(task_id).await?;
```

### Phase 5: Update models.yaml Schema

Add to existing configuration:

```yaml
# example.models.yaml

# ============================================================================
# Parking Lot Scheduler Configuration (Optional)
# ============================================================================
# Configure the thread pool and resource scheduler.
# If omitted, sensible defaults are used based on system resources.

parking_lot:
  # Thread pool for CPU-bound inference work
  pool:
    # Number of worker threads (default: num_cpus)
    worker_threads: 4
    
    # Maximum blocking threads for tokio runtime (default: 512)
    max_blocking_threads: 512
    
    # Stack size per worker thread in bytes (default: 2MB)
    thread_stack_size: 2097152
    
  # Resource limits and queue configuration
  limits:
    # Maximum resource units (GPU KV-cache blocks)
    # null = auto-derive from kvcache_mem_gpu
    max_units: null
    
    # Maximum number of queued requests before rejection
    max_queue_depth: 1000
    
    # Request timeout in seconds
    timeout_secs: 120
    
  # Queue backend configuration
  queue:
    # Backend type: "memory" | "postgres" | "yaque"
    backend: "memory"
    
    # Enable persistent queue (survives restarts)
    persistence: false
    
    # PostgreSQL connection string (if backend = "postgres")
    # postgres_url: "postgresql://user:pass@localhost/candle_vllm"
    
    # Yaque queue directory (if backend = "yaque")
    # yaque_dir: "./queue"
    
  # Result mailbox configuration
  mailbox:
    # Backend type: "memory" | "postgres"
    backend: "memory"
    
    # How long to retain completed results (seconds)
    retention_secs: 3600
    
    # PostgreSQL connection string (if backend = "postgres")
    # postgres_url: "postgresql://user:pass@localhost/candle_vllm"

# models configuration continues as before...
models:
  - name: mistral-7b
    hf_id: mistralai/Mistral-7B-Instruct-v0.3
    params:
      dtype: f16
      kvcache_mem_gpu: 8192
      max_num_seqs: 256
    
    # Per-model parking lot overrides (optional)
    # These override the global parking_lot settings for this specific model
    parking_lot:
      limits:
        max_units: 2048        # Different limit for this model
        max_queue_depth: 500   # Smaller queue
        timeout_secs: 60       # Shorter timeout
```

## Implementation Steps

### Step 1: Update Configuration Structs

Add to `crates/candle-vllm-server/src/models_config.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParkingLotConfig {
    #[serde(default)]
    pub pool: ThreadPoolConfig,
    
    #[serde(default)]
    pub limits: LimitsConfig,
    
    #[serde(default)]
    pub queue: QueueConfig,
    
    #[serde(default)]
    pub mailbox: MailboxConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,
    
    #[serde(default = "default_max_blocking_threads")]
    pub max_blocking_threads: usize,
    
    #[serde(default = "default_thread_stack_size")]
    pub thread_stack_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitsConfig {
    pub max_units: Option<usize>,
    
    #[serde(default = "default_max_queue_depth")]
    pub max_queue_depth: usize,
    
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    #[serde(default = "default_queue_backend")]
    pub backend: String,
    
    #[serde(default)]
    pub persistence: bool,
    
    pub postgres_url: Option<String>,
    pub yaque_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailboxConfig {
    #[serde(default = "default_mailbox_backend")]
    pub backend: String,
    
    #[serde(default = "default_retention_secs")]
    pub retention_secs: u64,
    
    pub postgres_url: Option<String>,
}
```

### Step 2: Integrate ResourcePool

In `crates/candle-vllm-core/src/openai/pipelines/llm_engine.rs`:

```rust
pub struct LLMEngine {
    // ... existing fields ...
    
    // ADD: ResourcePool for proper thread management
    resource_pool: Arc<ResourcePool<
        SerializableInferenceJob,
        SerializableInferenceResult,
        LlmExecutor,
    >>,
}

impl LLMEngine {
    pub fn new(..., parking_lot_config: ParkingLotConfig) -> Result<Self> {
        // Create resource pool with configuration
        let pool = create_resource_pool(parking_lot_config, executor)?;
        
        Ok(Self {
            // ... existing fields ...
            resource_pool: Arc::new(pool),
        })
    }
    
    pub async fn add_streaming_request(...) -> Result<...> {
        // Use resource pool instead of direct executor call
        let task_id = self.resource_pool.submit(job, meta).await?;
        
        // Get result from mailbox (non-blocking)
        let result = self.resource_pool.retrieve(task_id).await?;
        
        // ... rest of logic ...
    }
}
```

### Step 3: Create Serializable Wrappers

Add to `crates/candle-vllm-core/src/parking_lot/job.rs`:

```rust
/// Serializable version of InferenceJob for ResourcePool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableInferenceJob {
    pub request_id: String,
    pub tokens: Vec<u32>,
    pub positions: Vec<usize>,
    pub is_streaming: bool,
    // ... other fields ...
}

/// Serializable version of InferenceResult
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializableInferenceResult {
    Completion {
        // Store completed data that can be serialized
    },
    StreamingStarted {
        // Return a mailbox key for streaming channel
        channel_key: String,
    },
    Error {
        message: String,
    },
}
```

## Testing

After refactoring:

```bash
# Test with default configuration
cargo run --release --features metal -- --ui-server

# Test with custom parking lot config
# Create test.models.yaml with parking_lot section
cargo run --release --features metal -- --models-config test.models.yaml --ui-server

# Verify thread pool is created
# Look for logs like:
# "Parking lot pool initialized with 4 worker threads"
```

## Success Criteria

- ✅ Requests don't hang
- ✅ Multiple concurrent requests process in parallel
- ✅ Worker thread pool is created from configuration
- ✅ CPU-bound inference doesn't block async runtime
- ✅ Streaming and completion both work correctly
- ✅ Proper logging shows pool activity

## References

- prometheus-parking-lot: https://github.com/Prometheus-AGS/prometheus-parking-lot-rs
- Current docs: `docs/PARKING_LOT_SCHEDULER.md`
- Configuration: `example.models.yaml`
