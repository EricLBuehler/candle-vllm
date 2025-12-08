# Streaming Audit and Implementation Summary

## Executive Summary

This document summarizes the audit of Candle-vLLM's streaming infrastructure and provides a comprehensive implementation plan for AG-UI style tool call streaming and reasoning token handling.

## Current State

### ✅ What's Working

1. **Basic Streaming Infrastructure**
   - Producer-consumer pattern with channels (token generation → HTTP SSE)
   - Separate thread for streaming generation
   - Proper channel-based bridging
   - Token-by-token streaming for regular content
   - Reasoning token detection and streaming

2. **Non-Streaming Path**
   - Synchronous completion in worker thread
   - Blocking response via `completion_records`
   - Proper resource cleanup

3. **Reasoning Tokens**
   - Detection based on model type and token patterns
   - Separate `reasoning` field in deltas
   - Support for models: DeepSeek-R1, QwQ, Ministral-reasoning

### ⚠️ Areas Requiring Implementation

1. **Tool Call Streaming**
   - ❌ No incremental tool call deltas (start/args/end pattern)
   - ❌ Tool calls currently detected only after full generation
   - ❌ Missing AG-UI style ToolCallStart/Args/End events
   - ❌ No state machine for tracking active tool calls

2. **Non-Streaming Extensions**
   - ❌ No `extensions` field in completion responses
   - ❌ Reasoning chunks not collected for non-streaming
   - ❌ Tool call events not captured for non-streaming
   - ❌ Missing chunk collector infrastructure

3. **Performance Concerns**
   - ⚠️ Unbounded channels (memory growth risk)
   - ⚠️ Per-request thread spawning (no pooling)
   - ⚠️ No concurrency limiting (thread exhaustion risk)
   - ⚠️ No backpressure mechanism

## AG-UI Protocol Compliance

### Event Types to Implement

#### Tool Call Events (AG-UI Standard)

```typescript
// Start
{
  type: "TOOL_CALL_START",
  toolCallId: "call_abc123",
  toolCallName: "get_weather",
  parentMessageId?: "msg-123"
}

// Arguments (incremental)
{
  type: "TOOL_CALL_ARGS",
  toolCallId: "call_abc123",
  delta: "{\"location\":"
}

// End
{
  type: "TOOL_CALL_END",
  toolCallId: "call_abc123"
}
```

#### OpenAI API Equivalent (Streaming)

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "choices": [{
    "index": 0,
    "delta": {
      "tool_calls": [{
        "index": 0,
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\":"
        }
      }]
    }
  }]
}
```

#### Reasoning Events (AG-UI Draft Proposal)

```typescript
// Start
{
  type: "REASONING_START",
  messageId: "reasoning-001"
}

// Content (incremental)
{
  type: "REASONING_MESSAGE_CONTENT",
  messageId: "reasoning-001",
  delta: "Let me think..."
}

// End
{
  type: "REASONING_END",
  messageId: "reasoning-001"
}
```

#### OpenAI API Equivalent (Extended)

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "choices": [{
    "index": 0,
    "delta": {
      "reasoning": "Let me think..."
    }
  }]
}
```

### Non-Streaming Extensions Format

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "choices": [...],
  "usage": {...},
  "extensions": {
    "reasoning_chunks": [
      "Step 1: Analyze the problem",
      "Step 2: Consider alternatives",
      "Step 3: Reach conclusion"
    ],
    "tool_call_chunks": [
      {
        "type": "start",
        "tool_call_id": "call_abc123",
        "tool_name": "get_weather"
      },
      {
        "type": "args",
        "tool_call_id": "call_abc123",
        "delta": "{\"location\": \"NYC\"}"
      },
      {
        "type": "end",
        "tool_call_id": "call_abc123"
      }
    ]
  }
}
```

## Implementation Plan

### Phase 1: Core Data Structures (1-2 days)

**Priority: HIGH**

- [ ] Create `ToolCallStreamState` state machine
- [ ] Create `ChunkCollector` for non-streaming
- [ ] Add `extensions` field to `ChatCompletionResponse`
- [ ] Add helper methods to `ToolCallDelta`

**Files to Create/Modify:**
- `candle-vllm-core/src/openai/tool_streaming.rs` (new)
- `candle-vllm-core/src/openai/chunk_collector.rs` (new)
- `candle-vllm-core/src/openai/responses.rs` (modify)

### Phase 2: Streaming Enhancement (2-3 days)

**Priority: HIGH**

- [ ] Enhance streaming bridge with tool call detection
- [ ] Implement `IncrementalToolParser` trait
- [ ] Add tool call state machine to streaming loop
- [ ] Emit proper tool call deltas (start/args/end)

**Files to Modify:**
- `candle-vllm-core/src/openai/openai_server.rs` (line ~440, streaming bridge)
- `candle-vllm-core/src/openai/tool_parser.rs` (add incremental parsing)

### Phase 3: Non-Streaming Extensions (1-2 days)

**Priority: MEDIUM**

- [ ] Add chunk collection to completion handler
- [ ] Store collected chunks in model state
- [ ] Add extensions to final response
- [ ] Test extension field population

**Files to Modify:**
- `candle-vllm-core/src/openai/openai_server.rs` (line ~580, completion handler)
- `candle-vllm-core/src/parking_lot/executor.rs` (add collection)

### Phase 4: Worker Integration (1 day)

**Priority: MEDIUM**

- [ ] Add chunk collector to worker
- [ ] Implement chunk storage mechanism
- [ ] Update completion path with collection

**Files to Modify:**
- `candle-vllm-core/src/parking_lot/executor.rs`
- `candle-vllm-core/src/parking_lot/mod.rs` (chunk storage)

### Phase 5: Testing (2-3 days)

**Priority: HIGH**

- [ ] Unit tests for state machine
- [ ] Unit tests for collector
- [ ] Integration tests for streaming with tool calls
- [ ] Integration tests for non-streaming with extensions
- [ ] Performance/load tests

**Files to Create:**
- `candle-vllm-core/tests/tool_streaming_tests.rs` (new)
- `candle-vllm-core/tests/chunk_collector_tests.rs` (new)

### Phase 6: Performance Optimization (1-2 days)

**Priority: MEDIUM**

- [ ] Replace unbounded channels with bounded
- [ ] Add thread pool for streaming
- [ ] Add semaphore for concurrency control
- [ ] Benchmark and tune buffer sizes

**Files to Modify:**
- `candle-vllm-core/src/openai/openai_server.rs`
- `candle-vllm-core/src/openai/streaming.rs`

### Phase 7: Documentation (1 day)

**Priority: MEDIUM**

- [ ] Update API documentation
- [ ] Update configuration guide
- [ ] Add examples for tool call streaming
- [ ] Add examples for reasoning models

**Files to Create/Modify:**
- `docs/API.md` (update)
- `docs/TOOL_CALLING.md` (new)
- `docs/REASONING_MODELS.md` (new)
- `examples/tool_streaming.rs` (new)

## Technical Architecture

### Streaming Flow

```
Request → Token Generation → Classification → State Machine → Delta Emission → SSE
          (Worker Thread)   (Token Type)    (Tool/Reasoning) (Chunk Format)  (HTTP)
```

### Token Classification

```rust
for each token {
    if is_reasoning_token() {
        emit ChoiceData { reasoning: token.text }
    } else if in_tool_call() {
        match tool_parser.parse_incremental() {
            ToolParseState::Start(name) => {
                emit ToolCallDelta::start(id, name)
            }
            ToolParseState::Arguments(args) => {
                emit ToolCallDelta::arguments(args)
            }
            ToolParseState::Complete => {
                emit ToolCallDelta::end()
            }
        }
    } else {
        emit ChoiceData { content: token.text }
    }
}
```

### Non-Streaming Collection

```rust
let mut collector = ChunkCollector::new();

for each token {
    if is_reasoning {
        collector.add_reasoning(token.text)
    } else if is_tool_call {
        collector.add_tool_call_event(event)
    } else {
        collector.add_content(token.text)
    }
}

response.extensions = Some(collector.to_extensions())
```

## Performance Considerations

### Current Issues

1. **Unbounded Channels**
   - Risk: Memory growth if client is slow
   - Solution: Bounded channels with size 1000
   ```rust
   let (tx, rx) = flume::bounded::<ChatResponse>(1000);
   ```

2. **Thread Spawning**
   - Risk: Thread exhaustion with 1000+ concurrent requests
   - Solution: Rayon thread pool (100 threads)
   ```rust
   static STREAMING_POOL: Lazy<ThreadPool> = ...;
   ```

3. **No Concurrency Limiting**
   - Risk: Resource exhaustion
   - Solution: Tokio semaphore (100 permits)
   ```rust
   static STREAMING_SEM: Lazy<Semaphore> = Semaphore::new(100);
   ```

### Recommended Configuration

```rust
// Buffer sizes
const STREAMING_BUFFER_SIZE: usize = 1000;
const TOOL_CALL_BUFFER_SIZE: usize = 4096; // For JSON accumulation

// Thread pool
const STREAMING_THREADS: usize = 100;

// Concurrency limits
const MAX_CONCURRENT_STREAMS: usize = 100;
```

## Tool Call Format Support

### Supported Formats

1. **Mistral**: `[TOOL_CALLS] [{"name": "...", "arguments": {...}}]`
2. **Llama**: `<function=func_name>{"arg": "value"}</function>`
3. **Qwen**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
4. **Generic JSON**: `{"name": "...", "arguments": {...}}`

### Incremental Parsing Strategy

```rust
// Buffer tokens until pattern is clear
let mut buffer = String::new();

for token in tokens {
    buffer.push_str(&token.text);
    
    match parser.parse_incremental(&buffer) {
        ToolParseState::NotToolCall => {
            // Flush buffer as content
            emit_content(buffer);
            buffer.clear();
        }
        ToolParseState::InProgress(partial) => {
            // Keep accumulating
            if partial.has_name() {
                emit_tool_start(partial.name);
            }
            if partial.has_new_args() {
                emit_tool_args(partial.new_args);
            }
        }
        ToolParseState::Complete(tool) => {
            emit_tool_end();
            buffer.clear();
        }
    }
}
```

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_tool_call_state_machine() {
    // Test start → args → end flow
}

#[test]
fn test_chunk_collector() {
    // Test reasoning + tool call collection
}

#[test]
fn test_incremental_tool_parsing() {
    // Test all format parsers
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_streaming_tool_calls() {
    // End-to-end with Mistral format
    // End-to-end with Llama format
    // End-to-end with Qwen format
}

#[tokio::test]
async fn test_non_streaming_extensions() {
    // Verify extensions field
    // Verify reasoning chunks
    // Verify tool call events
}

#[tokio::test]
async fn test_mixed_streaming() {
    // Reasoning + content
    // Tool calls + content
    // All three types
}
```

### Performance Tests

```rust
#[tokio::test]
async fn test_concurrent_streaming() {
    // 1000 concurrent streams
    // Measure thread count
    // Measure memory usage
}

#[tokio::test]
async fn test_slow_client() {
    // Client reads slowly
    // Verify backpressure
    // Verify no memory leak
}
```

## Risk Assessment

### High Risk
- **Thread exhaustion**: Per-request spawning without limits
  - **Mitigation**: Thread pool + semaphore (Phase 6)
- **Memory growth**: Unbounded channels
  - **Mitigation**: Bounded channels (Phase 6)

### Medium Risk
- **Tool call parsing errors**: Malformed output
  - **Mitigation**: Robust error handling, fallback to content
- **Client disconnection**: Mid-stream cleanup
  - **Mitigation**: Already handled by channel disconnection

### Low Risk
- **Reasoning detection false positives**: Non-reasoning treated as reasoning
  - **Mitigation**: Strict pattern matching, model whitelist

## Success Criteria

### Must Have (MVP)
- [x] Reasoning tokens stream correctly
- [ ] Tool calls emit start/args/end deltas
- [ ] Non-streaming includes extensions field
- [ ] All supported tool formats work
- [ ] Unit tests pass
- [ ] Integration tests pass

### Should Have (V1)
- [ ] Bounded channels implemented
- [ ] Thread pool implemented
- [ ] Performance tests pass
- [ ] Documentation complete

### Nice to Have (Future)
- [ ] AG-UI MetaEvents support
- [ ] Activity snapshots for long operations
- [ ] Reasoning encryption support
- [ ] Multi-tool parallel execution

## Timeline Estimate

- **Phase 1-2 (Core + Streaming)**: 3-5 days
- **Phase 3-4 (Non-streaming + Worker)**: 2-3 days
- **Phase 5 (Testing)**: 2-3 days
- **Phase 6 (Performance)**: 1-2 days
- **Phase 7 (Documentation)**: 1 day

**Total: 9-14 days** (1.5-3 weeks)

## Next Steps

1. Review and approve this plan
2. Create feature branch: `feature/tool-call-streaming`
3. Implement Phase 1 (core structures)
4. Implement Phase 2 (streaming enhancement)
5. Add tests incrementally
6. Performance tune
7. Document and merge

## References

- **Specification**: `TOOL_CALL_STREAMING_SPEC.md`
- **Implementation Details**: `TOOL_CALL_STREAMING_IMPL.md`
- **Pipeline Audit**: `PIPELINE_AUDIT.md`
- **AG-UI Events**: https://docs.ag-ui.com/concepts/events
- **AG-UI Reasoning**: https://docs.ag-ui.com/drafts/reasoning
- **OpenAI Streaming**: https://platform.openai.com/docs/guides/streaming-responses
- **OpenAI Function Calling**: https://platform.openai.com/docs/guides/function-calling

---

**Document Version**: 1.0  
**Date**: 2025-01-XX  
**Author**: System Architect  
**Status**: Ready for Implementation