# prometheus-parking-lot Requirements for candle-vllm

## Executive Summary

The `candle-vllm` project needs `prometheus-parking-lot` to provide a **proper thread pool with blocking worker threads** for CPU/GPU-bound inference work. The current primitives (TaskQueue, Mailbox, TokioSpawner) are available, but we need a **ResourcePool** implementation that manages dedicated worker threads.

## Current Imports from prometheus-parking-lot

```rust
// From prometheus_parking_lot::core
pub use prometheus_parking_lot::core::{
    Mailbox,
    PoolLimits,
    ScheduledTask,
    Spawn,
    TaskMetadata,
    TaskQueue,
    TaskStatus,
    WakeState,
};

// From prometheus_parking_lot::infra
pub use prometheus_parking_lot::infra::mailbox::memory::InMemoryMailbox;
pub use prometheus_parking_lot::infra::queue::memory::InMemoryQueue;

// From prometheus_parking_lot::runtime
pub use prometheus_parking_lot::runtime::tokio_spawner::TokioSpawner;

// From prometheus_parking_lot::util
pub use prometheus_parking_lot::util::clock::now_ms;
pub use prometheus_parking_lot::util::serde::{
    MailboxKey,
    Priority,
    ResourceCost,
    ResourceKind,
    TaskId,
};
```

## What We Need: ResourcePool with Worker Thread Pool

### Required API

```rust
/// Resource pool that manages worker threads for CPU-bound tasks
pub struct ResourcePool<P, R, E>
where
    P: Serialize + DeserializeOwned + Send + 'static,  // Payload
    R: Serialize + DeserializeOwned + Send + 'static,  // Result
    E: TaskExecutor<P, R> + Clone + Send + Sync + 'static,  // Executor
{
    // Internal implementation
}

/// Task executor trait that the pool uses
#[async_trait::async_trait]
pub trait TaskExecutor<P, R>: Send + Sync
where
    P: Send + 'static,
    R: Send + 'static,
{
    /// Execute a task payload.
    /// 
    /// CRITICAL: This should be called from a dedicated worker thread,
    /// NOT from the async runtime's thread pool.
    async fn execute(&self, payload: P, meta: TaskMetadata) -> R;
}

impl<P, R, E> ResourcePool<P, R, E>
where
    P: Serialize + DeserializeOwned + Send + 'static,
    R: Serialize + DeserializeOwned + Send + 'static,
    E: TaskExecutor<P, R> + Clone + Send + Sync + 'static,
{
    /// Create a new resource pool with configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Pool configuration (limits, timeouts, etc.)
    /// * `executor` - The task executor implementation
    /// * `queue` - Task queue backend (InMemoryQueue, PostgresQueue, etc.)
    /// * `mailbox` - Result mailbox backend (InMemoryMailbox, PostgresMailbox, etc.)
    /// * `spawner` - Runtime spawner (TokioSpawner)
    ///
    /// # Thread Pool Behavior
    ///
    /// The pool MUST create dedicated worker threads (not async tasks) with:
    /// - Number of threads: config.worker_threads (default: num_cpus)
    /// - Stack size: config.thread_stack_size (default: 2MB)
    /// - Each worker processes jobs from the queue in a loop
    /// - Workers call executor.execute() in their thread context
    /// - Results are stored in the mailbox
    ///
    pub fn new(
        config: PoolConfig,
        executor: E,
        queue: Box<dyn TaskQueue<P>>,
        mailbox: Box<dyn Mailbox<R>>,
        spawner: Box<dyn Spawn>,
    ) -> Result<Self, PoolError>;

    /// Submit a task to the pool.
    ///
    /// # Returns
    ///
    /// Returns a mailbox key that can be used to retrieve the result.
    ///
    /// # Behavior
    ///
    /// 1. Checks capacity (resources + queue depth)
    /// 2. If capacity available: enqueues task, returns mailbox key
    /// 3. If capacity exceeded: returns error
    /// 4. Task is picked up by next available worker thread
    /// 5. Worker calls executor.execute() with the payload
    /// 6. Result is stored in mailbox with the returned key
    ///
    pub async fn submit(
        &self,
        payload: P,
        meta: TaskMetadata,
    ) -> Result<MailboxKey, PoolError>;

    /// Retrieve a result from the mailbox.
    ///
    /// # Returns
    ///
    /// Returns the result if available, or waits up to timeout.
    ///
    pub async fn retrieve(
        &self,
        key: &MailboxKey,
        timeout: Duration,
    ) -> Result<R, PoolError>;

    /// Get current pool statistics.
    pub fn stats(&self) -> PoolStats;
}

/// Pool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Number of dedicated worker threads (default: num_cpus)
    pub worker_threads: usize,
    
    /// Stack size per worker thread in bytes (default: 2MB)
    pub thread_stack_size: usize,
    
    /// Maximum resource units (e.g., GPU blocks)
    pub max_units: usize,
    
    /// Maximum queue depth before rejection
    pub max_queue_depth: usize,
    
    /// Default timeout in milliseconds
    pub default_timeout_ms: u64,
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
    
    /// Used resource units
    pub used_units: usize,
    
    /// Total resource units
    pub total_units: usize,
    
    /// Tasks completed
    pub completed_tasks: u64,
    
    /// Tasks failed
    pub failed_tasks: u64,
}

/// Pool errors.
#[derive(Debug)]
pub enum PoolError {
    /// Queue is full
    QueueFull,
    
    /// Insufficient resources
    InsufficientResources { needed: usize, available: usize },
    
    /// Task timed out
    Timeout,
    
    /// Result not found in mailbox
    ResultNotFound,
    
    /// Internal error
    Internal(String),
}
```

## Critical Implementation Details

### Worker Thread Behavior

Each worker thread MUST:

1. **Run in a dedicated OS thread** (not tokio async task)
   ```rust
   std::thread::Builder::new()
       .name(format!("parking-lot-worker-{}", worker_id))
       .stack_size(config.thread_stack_size)
       .spawn(move || {
           // Worker loop
           loop {
               let task = queue.dequeue_blocking(); // Blocks until task available
               let result = executor.execute(task.payload, task.meta).await;
               mailbox.store(task.mailbox_key, result);
           }
       })
   ```

2. **Use a tokio runtime PER WORKER THREAD** for async executor.execute():
   ```rust
   // Inside worker thread
   let rt = tokio::runtime::Builder::new_current_thread()
       .enable_all()
       .build()
       .unwrap();
   
   loop {
       let task = queue.dequeue_blocking();
       let result = rt.block_on(async {
           executor.execute(task.payload, task.meta).await
       });
       mailbox.store(task.mailbox_key, result);
   }
   ```

3. **NOT use tokio::task::spawn_blocking()** - that still uses the main runtime!

### Queue Behavior

The TaskQueue MUST support:

```rust
pub trait TaskQueue<P>: Send + Sync {
    /// Enqueue a task (async, non-blocking)
    async fn enqueue(&self, task: ScheduledTask<P>) -> Result<(), QueueError>;
    
    /// Dequeue a task (blocking until task available or shutdown)
    /// This is called from worker threads and should block the thread,
    /// not an async task.
    fn dequeue_blocking(&self) -> Option<ScheduledTask<P>>;
    
    /// Dequeue with timeout
    fn dequeue_blocking_timeout(&self, timeout: Duration) -> Option<ScheduledTask<P>>;
    
    /// Get current queue length
    fn len(&self) -> usize;
}
```

### Mailbox Behavior

The Mailbox MUST support:

```rust
pub trait Mailbox<R>: Send + Sync {
    /// Store a result with the given key
    fn store(&self, key: &MailboxKey, result: R) -> Result<(), MailboxError>;
    
    /// Retrieve a result (async, returns immediately if available)
    async fn retrieve(&self, key: &MailboxKey) -> Result<Option<R>, MailboxError>;
    
    /// Wait for a result with timeout
    async fn retrieve_wait(
        &self,
        key: &MailboxKey,
        timeout: Duration,
    ) -> Result<R, MailboxError>;
    
    /// Remove a result from the mailbox
    fn remove(&self, key: &MailboxKey) -> Option<R>;
}
```

## Why This Matters for candle-vllm

### The Problem

LLM inference involves:
1. **CPU-bound tokenization** (milliseconds)
2. **GPU-bound forward passes** (10-100ms per token)
3. **CPU-bound sampling** (milliseconds)
4. **Repeated in a loop** (for 100-1000+ tokens)

If this runs in tokio's async runtime:
- Blocks async worker threads
- Prevents other async I/O from making progress
- Causes requests to queue up and hang
- No true parallelism for concurrent requests

### The Solution

Dedicated worker threads:
- Each thread runs one inference request start-to-finish
- True OS-level parallelism (4-8 concurrent inferences)
- Async runtime stays free for I/O and request handling
- Tokio only used for queue/mailbox coordination

## Configuration Example

```json
{
  "pools": {
    "default": {
      "worker_threads": 4,
      "thread_stack_size": 2097152,
      "max_units": 16384,
      "max_queue_depth": 1000,
      "default_timeout_ms": 120000,
      "queue": { "type": "in_memory" },
      "mailbox": { 
        "storage": { "type": "in_memory" },
        "retention_secs": 3600
      }
    }
  }
}
```

## Usage Pattern in candle-vllm

```rust
// Create pool at server startup
let pool = ResourcePool::new(
    config,
    llm_executor,  // Implements TaskExecutor
    Box::new(InMemoryQueue::new()),
    Box::new(InMemoryMailbox::new()),
    Box::new(TokioSpawner::new()),
)?;

// Submit inference job (from async context)
let meta = TaskMetadata::new(request_id, ResourceCost::gpu_vram(100));
let mailbox_key = pool.submit(job, meta).await?;

// Retrieve result (async, non-blocking wait)
let result = pool.retrieve(&mailbox_key, Duration::from_secs(120)).await?;

// The actual inference runs in a dedicated worker thread,
// not blocking the async runtime!
```

## Proposed API Addition to prometheus-parking-lot

Add to `prometheus_parking_lot/src/pool/mod.rs` (or similar):

```rust
pub mod resource_pool;
pub use resource_pool::{ResourcePool, PoolConfig, PoolStats, PoolError};
```

The implementation should:
1. Create N worker threads on pool initialization
2. Each worker has its own single-threaded tokio runtime
3. Workers dequeue tasks and call executor.execute()
4. Results go into the mailbox
5. Main async runtime only coordinates via queue/mailbox

## Benefits

- **Separation of concerns**: parking-lot manages threads, candle-vllm does inference
- **Reusability**: Other CPU-bound tasks can use the same pool
- **Correctness**: No risk of blocking async runtime with sync work
- **Performance**: True parallelism for concurrent requests
- **Monitoring**: Pool stats expose thread utilization, queue depth, etc.

## Alternative: Simple spawn_blocking Wrapper

If a full ResourcePool is too complex initially, at minimum provide:

```rust
pub struct BlockingTaskPool {
    semaphore: Semaphore,
    queue_depth: AtomicUsize,
    config: PoolConfig,
}

impl BlockingTaskPool {
    pub async fn execute<F, R>(&self, f: F) -> Result<R, PoolError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // Check capacity
        // Acquire semaphore
        // Run in tokio::task::spawn_blocking
        // Return result
    }
}
```

This is simpler but still provides:
- Concurrency limiting
- Queue depth tracking
- Proper thread isolation

## Recommendation

Implement the full **ResourcePool** for maximum flexibility and reusability across projects.

---

**Next Steps for candle-vllm** (after prometheus-parking-lot provides ResourcePool):

1. Import ResourcePool
2. Create pool at server startup with configuration
3. Replace direct executor calls with pool.submit/retrieve
4. Add streaming registry for non-serializable channels
5. Test concurrent request handling
