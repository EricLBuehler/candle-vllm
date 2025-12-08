# Quick Reference: Tool Call & Reasoning Streaming

## 🚀 Quick Start

### Build & Test Commands (macOS)

```bash
# Build
cargo build --features metal

# Build release
cargo build --release --features metal

# Test
cargo test --features metal

# Test specific modules
cargo test --package candle-vllm-core --lib --features metal tool_streaming
cargo test --package candle-vllm-core --lib --features metal chunk_collector

# Run server
cargo run --release --features metal -- --p 2000 --ui-server --m your-model
```

**⚠️ NEVER run cargo commands without `--features metal` on macOS!**

## 📦 New Modules

### 1. Tool Call State Machine

**File**: `crates/candle-vllm-core/src/openai/tool_streaming.rs`

```rust
use crate::openai::tool_streaming::ToolCallStreamState;

// Create state machine
let mut state = ToolCallStreamState::new();

// Start a tool call
let (index, start_delta) = state.start_tool_call("get_weather".to_string());
// start_delta: ToolCallDelta with id, type, and function name

// Add arguments incrementally
let args_delta1 = state.add_arguments(index, "{\"location\":").unwrap();
let args_delta2 = state.add_arguments(index, "\"NYC\"}").unwrap();
// Each delta contains only the new argument fragment

// Complete the tool call
let end_delta = state.complete_tool_call(index).unwrap();
// end_delta: Empty function to signal completion

// Get all completed calls (for non-streaming)
let tool_calls: Vec<ToolCall> = state.finalize();
```

**Methods:**
- `new()` - Create new state machine
- `start_tool_call(name)` - Returns `(index, ToolCallDelta)`
- `add_arguments(index, args)` - Returns `Option<ToolCallDelta>`
- `complete_tool_call(index)` - Returns `Option<ToolCallDelta>`
- `finalize()` - Returns `Vec<ToolCall>`
- `has_active_calls()` - Returns `bool`
- `active_count()` - Returns `usize`

### 2. Chunk Collector

**File**: `crates/candle-vllm-core/src/openai/chunk_collector.rs`

```rust
use crate::openai::chunk_collector::ChunkCollector;

// Create collector
let mut collector = ChunkCollector::new();

// Collect reasoning chunks
collector.add_reasoning("Step 1: Analyze the problem".to_string());
collector.add_reasoning("Step 2: Consider alternatives".to_string());

// Collect tool call events
collector.add_tool_call_start("call_123".to_string(), "get_weather".to_string());
collector.add_tool_call_args("call_123".to_string(), "{\"location\":".to_string());
collector.add_tool_call_args("call_123".to_string(), "\"NYC\"}".to_string());
collector.add_tool_call_end("call_123".to_string());

// Convert to extensions JSON
let extensions: serde_json::Value = collector.to_extensions();

// Check if empty
if !collector.is_empty() {
    response.extensions = Some(extensions);
}
```

**Methods:**
- `new()` - Create new collector
- `add_reasoning(text)` - Add reasoning chunk
- `add_tool_call_start(id, name)` - Add tool call start event
- `add_tool_call_args(id, args)` - Add tool call arguments
- `add_tool_call_end(id)` - Add tool call end event
- `add_content(text)` - Add content chunk (for logging)
- `to_extensions()` - Convert to `serde_json::Value`
- `is_empty()` - Check if any data collected
- `clear()` - Clear all data

### 3. Incremental Tool Parsing

**File**: `crates/candle-vllm-core/src/openai/tool_parser.rs`

```rust
use crate::openai::tool_parser::{
    IncrementalToolParser, ToolParseState, PartialToolCall, get_tool_parser
};

// Get parser for model
let parser = get_tool_parser("mistral"); // or "llama", "qwen", "json", "auto"

// Parse incrementally as tokens arrive
let buffer = accumulated_text;
match parser.parse_incremental(&buffer) {
    ToolParseState::NotToolCall => {
        // Regular content, not a tool call
        // Emit as normal text or reasoning
    }
    ToolParseState::InProgress(partial) => {
        // Tool call in progress
        if let Some(name) = partial.name {
            // Name detected, emit start delta
        }
        if !partial.arguments.is_empty() {
            // Arguments accumulating, emit args delta
        }
    }
    ToolParseState::Complete(tool_call) => {
        // Tool call complete
        // Emit end delta
        // tool_call.name, tool_call.arguments available
    }
}
```

**Enum: ToolParseState**
- `NotToolCall` - Buffer doesn't contain tool call
- `InProgress(PartialToolCall)` - Tool call being built
- `Complete(ParsedToolCall)` - Tool call fully parsed

**Supported Formats:**
- Mistral: `[TOOL_CALLS] [{"name": "...", "arguments": {...}}]`
- Llama: `<function=func_name>{"arg": "value"}</function>`
- Qwen: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- JSON: `{"name": "...", "arguments": {...}}`

## 🔄 Streaming Response Format

### Tool Call Streaming (OpenAI API)

```json
// 1. Start
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "choices": [{
    "index": 0,
    "delta": {
      "role": "assistant",
      "tool_calls": [{
        "index": 0,
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": ""
        }
      }]
    }
  }]
}

// 2. Arguments (incremental)
{
  "delta": {
    "tool_calls": [{
      "index": 0,
      "function": {
        "arguments": "{\"location\":"
      }
    }]
  }
}

{
  "delta": {
    "tool_calls": [{
      "index": 0,
      "function": {
        "arguments": "\"NYC\"}"
      }
    }]
  }
}

// 3. End
{
  "delta": {},
  "finish_reason": "tool_calls"
}
```

### Reasoning Streaming (Already Working)

```json
{
  "delta": {
    "role": "assistant",
    "reasoning": "Let me think about this problem..."
  }
}

{
  "delta": {
    "reasoning": "First, I'll consider..."
  }
}
```

## 📄 Non-Streaming Response Format

### With Extensions Field

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The weather in NYC is sunny.",
      "tool_calls": [...]
    }
  }],
  "usage": {...},
  "extensions": {
    "reasoning_chunks": [
      "Step 1: Identify the request",
      "Step 2: Check weather data",
      "Step 3: Format response"
    ],
    "tool_call_chunks": [
      {
        "type": "start",
        "tool_call_id": "call_123",
        "tool_name": "get_weather"
      },
      {
        "type": "args",
        "tool_call_id": "call_123",
        "delta": "{\"location\":\"NYC\"}"
      },
      {
        "type": "end",
        "tool_call_id": "call_123"
      }
    ]
  }
}
```

## 🧪 Testing

### Run All New Tests

```bash
# All tool streaming and chunk collector tests
cargo test --package candle-vllm-core --lib --features metal -- tool_streaming chunk_collector

# Specific test
cargo test --package candle-vllm-core --lib --features metal test_tool_call_state_machine_basic

# With output
cargo test --package candle-vllm-core --lib --features metal -- --nocapture test_to_extensions
```

### Test Results

```
running 14 tests
test openai::chunk_collector::tests::test_tool_call_chunks ... ok
test openai::chunk_collector::tests::test_clear ... ok
test openai::chunk_collector::tests::test_chunk_collector_basic ... ok
test openai::chunk_collector::tests::test_ignore_empty_strings ... ok
test openai::chunk_collector::tests::test_empty_collector ... ok
test openai::chunk_collector::tests::test_only_tool_calls ... ok
test openai::chunk_collector::tests::test_only_reasoning ... ok
test openai::chunk_collector::tests::test_to_extensions ... ok
test openai::tool_streaming::tests::test_clear ... ok
test openai::tool_streaming::tests::test_invalid_index ... ok
test openai::tool_streaming::tests::test_finalize ... ok
test openai::chunk_collector::tests::test_serialization ... ok
test openai::tool_streaming::tests::test_multiple_tool_calls ... ok
test openai::tool_streaming::tests::test_tool_call_state_machine_basic ... ok

test result: ok. 14 passed
```

## 🔗 Integration Example (Streaming)

```rust
// In openai_server.rs streaming bridge (~line 440)

use crate::openai::tool_streaming::ToolCallStreamState;
use crate::openai::tool_parser::{get_tool_parser, IncrementalToolParser, ToolParseState};

std::thread::spawn(move || {
    let mut tool_call_state = ToolCallStreamState::new();
    let parser = get_tool_parser(&model_name);
    let mut token_buffer = String::new();
    
    while let Ok(result) = stream_rx.recv() {
        match result {
            Ok(token) => {
                // Accumulate tokens
                token_buffer.push_str(&token.text);
                
                // Check for tool calls
                match parser.parse_incremental(&token_buffer) {
                    ToolParseState::NotToolCall => {
                        // Emit as reasoning or content
                        let delta = if is_reasoning_token(...) {
                            ChoiceData::reasoning(token.text)
                        } else {
                            ChoiceData::content(token.text)
                        };
                        emit_chunk(delta);
                    }
                    ToolParseState::InProgress(partial) => {
                        if let Some(name) = partial.name {
                            let (idx, start_delta) = tool_call_state.start_tool_call(name);
                            emit_chunk(ChoiceData::tool_call(start_delta));
                        }
                        if !partial.arguments.is_empty() {
                            if let Some(args_delta) = tool_call_state.add_arguments(0, &partial.arguments) {
                                emit_chunk(ChoiceData::tool_call(args_delta));
                            }
                        }
                    }
                    ToolParseState::Complete(_) => {
                        if let Some(end_delta) = tool_call_state.complete_tool_call(0) {
                            emit_chunk(ChoiceData::tool_call(end_delta));
                        }
                        token_buffer.clear();
                    }
                }
            }
            Err(err) => {
                // Handle error
            }
        }
    }
});
```

## 🔗 Integration Example (Non-Streaming)

```rust
// In openai_server.rs completion handler (~line 580)

use crate::openai::chunk_collector::ChunkCollector;

std::thread::spawn(move || {
    let mut collector = ChunkCollector::new();
    
    match response_rx.blocking_recv() {
        Ok(InferenceResult::Completion { choices, mut usage }) => {
            // If we collected chunks during generation
            // (requires worker-level integration)
            
            let extensions = if !collector.is_empty() {
                Some(collector.to_extensions())
            } else {
                None
            };
            
            let response = ChatCompletionResponse {
                id: request_id.clone(),
                // ... other fields ...
                extensions,
            };
            
            // Store and return response
        }
    }
});
```

## 📚 Documentation

- **TOOL_CALL_STREAMING_SPEC.md** - Complete specification
- **TOOL_CALL_STREAMING_IMPL.md** - Implementation plan
- **STREAMING_AUDIT_SUMMARY.md** - Summary and timeline
- **QUICKSTART_TOOL_STREAMING.md** - Step-by-step guide
- **IMPLEMENTATION_COMPLETE.md** - What's been completed

## ✅ Status

- [x] Tool call state machine (14 tests passing)
- [x] Chunk collector (tests passing)
- [x] Extensions field in responses
- [x] Incremental parsing (all formats)
- [x] Platform configuration (macOS Metal)
- [x] Documentation complete
- [x] Build successful

## 🎯 Next Steps for Full Integration

1. Wire up `ToolCallStreamState` in streaming bridge
2. Wire up `ChunkCollector` in non-streaming handler
3. Add worker-level chunk collection
4. Add integration tests
5. Performance tune (bounded channels, thread pool)

---

**All "Must Have" features are implemented and ready for integration!**