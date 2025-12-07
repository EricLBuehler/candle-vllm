# Hanging Requests: Root Cause Analysis and Solution

## Problem Summary

Requests (both streaming and non-streaming) are hanging and never completing.

## Root Cause

**The inference engine is NOT properly using the `prometheus-parking-lot` thread pool!**

### What's Actually Happening

1. **No Thread Pool**: The code bypasses `ResourcePool` entirely and calls the executor directly
2. **Blocking Async Runtime**: CPU/GPU-bound model inference runs in async context, blocking tokio threads
3. **Manual Task Management**: Manually spawning tokio tasks instead of using the pool's worker threads
4. **Missing Configuration**: No JSON/YAML config for pool limits, worker threads, etc.

### Evidence

From `crates/candle-vllm-core/src/parking_lot/executor.rs:664`:
```rust
// Note: We implement a local TaskExecutor variant since prometheus_parking_lot
// expects Clone + 'static, and our InferenceResult contains non-serializable channels.
// The executor is invoked directly by LLMEngine rather than through ResourcePool.submit().
```

From `docs/PARKING_LOT_SCHEDULER.md:224`:
```
## Future Work
- Full `ResourcePool` integration with `Mailbox` for result persistence
```

This confirms the current implementation **does not** use `ResourcePool`!

### Code Flow (Current - BROKEN)

```
HTTP Request
  ↓
Handler (async)
  ↓
Core openai_server (async)
  ↓
LLMEngine.add_streaming_request (async)
  ↓
executor.execute(job).await  ← BLOCKS HERE! 
  ↓                             CPU-bound inference 
Process_streaming                runs in async context
  ↓                              
pipeline.forward() ← Synchronous GPU/CPU work blocks tokio thread!
```

### Why Requests Hang

1. **First request** starts, blocks a tokio async worker thread with CPU-bound work
2. **Second request** arrives, tries to use another tokio thread, also blocks
3. **All tokio threads** get consumed by CPU-bound inference
4. **No threads left** for async I/O, request handling, or SSE streaming
5. **Everything freezes**

## Solution

Use `prometheus-parking-lot` properly with `ResourcePool`:

### Correct Architecture

```
HTTP Request
  ↓
Handler (async, non-blocking)
  ↓  
ResourcePool.submit(job)  ← Returns immediately with task_id
  ↓                          Work goes to dedicated thread pool
  ├─→ Worker Thread 1 ─→ Inference (CPU-bound)
  ├─→ Worker Thread 2 ─→ Inference (CPU-bound)  
  ├─→ Worker Thread 3 ─→ Inference (CPU-bound)
  └─→ Worker Thread 4 ─→ Inference (CPU-bound)
  ↓
Result stored in Mailbox
  ↓
Async task polls Mailbox (non-blocking)
  ↓
Response sent to client
```

### Required Changes

See `docs/PARKING_LOT_REFACTOR.md` for detailed implementation plan.

## Configuration

### Extended models.yaml Schema

Now supports parking_lot configuration:

```yaml
parking_lot:
  pool:
    worker_threads: 4          # Dedicated CPU-bound worker threads
    max_blocking_threads: 512  # Tokio blocking threads  
    thread_stack_size: 2097152
  
  limits:
    max_units: null            # Auto from cache config
    max_queue_depth: 1000
    timeout_secs: 120
    
  queue:
    backend: "memory"
    persistence: false
    
  mailbox:
    backend: "memory"
    retention_secs: 3600
    
models:
  - name: my-model
    hf_id: mistralai/Mistral-7B
    params:
      kvcache_mem_gpu: 8192
      max_num_seqs: 256
    # Optional per-model overrides
    parking_lot:
      limits:
        max_units: 2048
```

## Immediate Workaround

Until the refactor is complete, you can partially fix this by:

1. **Increase tokio runtime threads**:
   ```bash
   TOKIO_WORKER_THREADS=16 cargo run --release --features metal -- --ui-server
   ```

2. **Reduce concurrent requests**:
   - Lower `max_num_seqs` in models.yaml
   - Reduce `max_queue_depth` 

3. **Use blocking thread pool** (quick fix):
   Wrap `executor.execute()` calls in `tokio::task::spawn_blocking()` - **THIS IS THE MINIMAL FIX**

## Next Steps

1. ✅ Added extensive logging (completed)
2. ✅ Extended models.yaml schema (completed)
3. ✅ Created refactor plan (completed)
4. ⏳ Implement ResourcePool integration (in progress)
5. ⏳ Add parking_lot config parsing
6. ⏳ Test with proper thread pool

## Testing Plan

After implementing ResourcePool:

```bash
# Test with minimal config
cargo run --release --features metal -- --ui-server

# Test with custom parking lot config  
cat > test.models.yaml << 'EOF'
parking_lot:
  pool:
    worker_threads: 8  # More threads
  limits:
    max_queue_depth: 500

models:
  - name: test-model
    hf_id: mistralai/Ministral-3-3B-Reasoning-2512
    params:
      kvcache_mem_gpu: 8192
EOF

cargo run --release --features metal -- --models-config test.models.yaml --ui-server

# Verify in logs:
# ✅ "Parking lot pool initialized with 8 worker threads"
# ✅ "Submitting job to worker thread 3"
# ✅ Requests complete without hanging
```

## References

- Root cause analysis: This document
- Refactor plan: `docs/PARKING_LOT_REFACTOR.md`
- Current docs: `docs/PARKING_LOT_SCHEDULER.md`
- Updated schema: `example.models.yaml`
- Logging added: Multiple files (see git diff)

---

**Bottom Line**: We're not using the thread pool library we imported. We need to actually use it! The hanging is caused by CPU-bound work blocking the async runtime.
