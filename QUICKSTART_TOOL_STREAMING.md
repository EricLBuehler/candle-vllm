# Quick Start: Tool Call and Reasoning Streaming

## TL;DR

This guide helps you quickly implement AG-UI style tool call streaming and reasoning token handling in Candle-vLLM.

## What You're Building

### Streaming Mode (SSE)
```json
// Tool call starts
{"delta": {"tool_calls": [{"index": 0, "id": "call_123", "type": "function", "function": {"name": "get_weather"}}]}}

// Arguments stream incrementally
{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"loc"}}]}}
{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ation\":"}}]}}
{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"NYC\"}"}}]}}

// Tool call ends (empty delta or finish)
{"delta": {}, "finish_reason": "tool_calls"}
```

### Non-Streaming Mode
```json
{
  "choices": [{
    "message": {
      "tool_calls": [{"id": "call_123", "function": {"name": "get_weather", "arguments": "{\"location\":\"NYC\"}"}}]
    }
  }],
  "extensions": {
    "reasoning_chunks": ["Let me check the weather..."],
    "tool_call_chunks": [
      {"type": "start", "tool_call_id": "call_123", "tool_name": "get_weather"},
      {"type": "args", "tool_call_id": "call_123", "delta": "{\"location\":\"NYC\"}"},
      {"type": "end", "tool_call_id": "call_123"}
    ]
  }
}
```

## Step-by-Step Implementation

### Step 1: Create the State Machine (30 min)

**File**: `candle-vllm-core/src/openai/tool_streaming.rs`

```rust
use std::collections::HashMap;
use uuid::Uuid;
use crate::openai::requests::{ToolCallDelta, FunctionCallDelta};

pub struct ToolCallStreamState {
    active_calls: HashMap<usize, ActiveToolCall>,
    next_index: usize,
}

struct ActiveToolCall {
    id: String,
    name: String,
    arguments_buffer: String,
}

impl ToolCallStreamState {
    pub fn new() -> Self {
        Self {
            active_calls: HashMap::new(),
            next_index: 0,
        }
    }

    pub fn start_tool_call(&mut self, name: String) -> (usize, ToolCallDelta) {
        let id = format!("call_{}", Uuid::new_v4().simple().to_string()[..12].to_string());
        let index = self.next_index;
        self.next_index += 1;

        self.active_calls.insert(index, ActiveToolCall {
            id: id.clone(),
            name: name.clone(),
            arguments_buffer: String::new(),
        });

        (index, ToolCallDelta {
            index,
            id: Some(id),
            call_type: Some("function".to_string()),
            function: Some(FunctionCallDelta {
                name: Some(name),
                arguments: Some(String::new()),
            }),
        })
    }

    pub fn add_arguments(&mut self, index: usize, args: &str) -> Option<ToolCallDelta> {
        self.active_calls.get_mut(&index).map(|call| {
            call.arguments_buffer.push_str(args);
            ToolCallDelta {
                index,
                id: None,
                call_type: None,
                function: Some(FunctionCallDelta {
                    name: None,
                    arguments: Some(args.to_string()),
                }),
            }
        })
    }

    pub fn complete_tool_call(&mut self, index: usize) {
        // Mark as complete (state tracking)
    }
}
```

**Add to `mod.rs`:**
```rust
pub mod tool_streaming;
```

### Step 2: Create the Chunk Collector (20 min)

**File**: `candle-vllm-core/src/openai/chunk_collector.rs`

```rust
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Default)]
pub struct ChunkCollector {
    reasoning_chunks: Vec<String>,
    tool_call_chunks: Vec<ToolCallChunkEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallChunkEvent {
    #[serde(rename = "type")]
    pub event_type: String, // "start", "args", "end"
    pub tool_call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,
}

impl ChunkCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_reasoning(&mut self, text: String) {
        if !text.is_empty() {
            self.reasoning_chunks.push(text);
        }
    }

    pub fn add_tool_call_start(&mut self, id: String, name: String) {
        self.tool_call_chunks.push(ToolCallChunkEvent {
            event_type: "start".to_string(),
            tool_call_id: id,
            tool_name: Some(name),
            delta: None,
        });
    }

    pub fn add_tool_call_args(&mut self, id: String, args: String) {
        self.tool_call_chunks.push(ToolCallChunkEvent {
            event_type: "args".to_string(),
            tool_call_id: id,
            tool_name: None,
            delta: Some(args),
        });
    }

    pub fn add_tool_call_end(&mut self, id: String) {
        self.tool_call_chunks.push(ToolCallChunkEvent {
            event_type: "end".to_string(),
            tool_call_id: id,
            tool_name: None,
            delta: None,
        });
    }

    pub fn to_extensions(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        if !self.reasoning_chunks.is_empty() {
            map.insert("reasoning_chunks".to_string(), serde_json::json!(self.reasoning_chunks));
        }
        if !self.tool_call_chunks.is_empty() {
            map.insert("tool_call_chunks".to_string(), serde_json::json!(self.tool_call_chunks));
        }
        serde_json::Value::Object(map)
    }

    pub fn is_empty(&self) -> bool {
        self.reasoning_chunks.is_empty() && self.tool_call_chunks.is_empty()
    }
}
```

**Add to `mod.rs`:**
```rust
pub mod chunk_collector;
```

### Step 3: Update Response Type (10 min)

**File**: `candle-vllm-core/src/openai/responses.rs`

```rust
// Add to ChatCompletionResponse struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    // ... existing fields ...
    
    /// Extended metadata (reasoning chunks, tool call events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
}
```

### Step 4: Enhance Streaming Bridge (45 min)

**File**: `candle-vllm-core/src/openai/openai_server.rs`

Find the streaming bridge thread (around line 440) and modify:

```rust
use crate::openai::tool_streaming::ToolCallStreamState;

std::thread::spawn(move || {
    let mut full_content = String::new();
    let mut full_reasoning = String::new();
    let mut is_first_chunk = true;
    let mut token_count = 0;
    
    // NEW: Tool call state machine
    let mut tool_call_state = ToolCallStreamState::new();
    let mut tool_buffer = String::new();
    
    while let Ok(result) = stream_rx.recv() {
        token_count += 1;
        match result {
            Ok(token) => {
                let token_is_reasoning = is_reasoning_token(
                    &token.text,
                    token.token_id,
                    &model_name_for_stream,
                    thinking_enabled,
                    token.is_reasoning,
                );

                // Accumulate for tool call detection
                tool_buffer.push_str(&token.text);
                
                // Check for tool call patterns
                let delta = if tool_buffer.contains("[TOOL_CALLS]") || 
                              tool_buffer.contains("<function=") ||
                              tool_buffer.contains("<tool_call>") {
                    // TODO: Parse tool call and emit appropriate delta
                    // For now, treat as content
                    ChoiceData::content(token.text.clone())
                } else if token_is_reasoning {
                    full_reasoning.push_str(&token.text);
                    ChoiceData {
                        role: if is_first_chunk { Some("assistant".to_string()) } else { None },
                        content: None,
                        tool_calls: None,
                        reasoning: Some(token.text.clone()),
                    }
                } else {
                    full_content.push_str(&token.text);
                    ChoiceData {
                        role: if is_first_chunk { Some("assistant".to_string()) } else { None },
                        content: Some(token.text.clone()),
                        tool_calls: None,
                        reasoning: None,
                    }
                };

                is_first_chunk = false;

                let chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    choices: vec![Choice { delta, finish_reason: None, index: 0 }],
                    created,
                    model: model_name_for_stream.to_string(),
                    object: "chat.completion.chunk",
                    system_fingerprint: None,
                    conversation_id: None,
                    resource_id: None,
                };

                if response_tx.send(ChatResponse::Chunk(chunk)).is_err() {
                    break;
                }

                if token.is_finished {
                    let _ = response_tx.send(ChatResponse::Done);
                    break;
                }
            }
            Err(err) => {
                let _ = response_tx.send(ChatResponse::ModelError(err));
                break;
            }
        }
    }
});
```

### Step 5: Add Extensions to Non-Streaming (30 min)

**File**: `candle-vllm-core/src/openai/openai_server.rs`

Find the non-streaming handler (around line 580) and modify:

```rust
use crate::openai::chunk_collector::ChunkCollector;

std::thread::spawn(move || {
    // NEW: Chunk collector
    let mut collector = ChunkCollector::new();
    
    match response_rx.blocking_recv() {
        Ok(InferenceResult::Completion { choices, mut usage }) => {
            data.model.update_usage_with_cache(&mut usage, &request_id);
            
            // NEW: If we collected chunks during generation, add them
            // (This requires worker support - see Step 6)
            
            let extensions = if !collector.is_empty() {
                Some(collector.to_extensions())
            } else {
                None
            };
            
            let response = ChatCompletionResponse {
                id: request_id.clone(),
                object: "chat.completion",
                created: usage.created,
                model: model_name.to_string(),
                choices,
                usage,
                conversation_id: None,
                resource_id: None,
                system_fingerprint: Some(data.model.config.system_fingerprint()),
                extensions, // NEW
            };
            
            // Store and notify...
        }
        // ... error handling ...
    }
});
```

### Step 6: Test It! (30 min)

**File**: `candle-vllm-core/tests/tool_streaming_test.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_state_machine() {
        let mut state = ToolCallStreamState::new();
        
        // Start
        let (index, start_delta) = state.start_tool_call("get_weather".to_string());
        assert_eq!(index, 0);
        assert!(start_delta.id.is_some());
        assert_eq!(start_delta.function.as_ref().unwrap().name.as_ref().unwrap(), "get_weather");
        
        // Add arguments
        let args1 = state.add_arguments(index, "{\"location\":").unwrap();
        assert_eq!(args1.function.as_ref().unwrap().arguments.as_ref().unwrap(), "{\"location\":");
        
        let args2 = state.add_arguments(index, "\"NYC\"}").unwrap();
        assert_eq!(args2.function.as_ref().unwrap().arguments.as_ref().unwrap(), "\"NYC\"}");
        
        // Complete
        state.complete_tool_call(index);
    }

    #[test]
    fn test_chunk_collector() {
        let mut collector = ChunkCollector::new();
        
        collector.add_reasoning("Thinking...".to_string());
        collector.add_tool_call_start("call_123".to_string(), "get_weather".to_string());
        collector.add_tool_call_args("call_123".to_string(), "{\"location\":\"NYC\"}".to_string());
        collector.add_tool_call_end("call_123".to_string());
        
        let ext = collector.to_extensions();
        assert!(!collector.is_empty());
        assert!(ext.get("reasoning_chunks").is_some());
        assert!(ext.get("tool_call_chunks").is_some());
    }
}
```

Run tests:
```bash
cargo test --package candle-vllm-core tool_streaming_test
```

## Quick Test with cURL

### Streaming Request
```bash
curl -N http://localhost:2000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "What is the weather in NYC?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    }],
    "stream": true
  }'
```

### Non-Streaming Request
```bash
curl http://localhost:2000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "thinking": true,
    "stream": false
  }' | jq '.extensions'
```

## Common Issues & Solutions

### Issue 1: Tool calls not detected
**Solution**: Check that your model outputs the correct format. Add logging:
```rust
info!("Tool buffer: {}", tool_buffer);
```

### Issue 2: Reasoning tokens not streaming
**Solution**: Ensure model is in reasoning model list and `thinking: true` is set:
```rust
fn is_reasoning_model(model_name: &str) -> bool {
    let name_lower = model_name.to_lowercase();
    name_lower.contains("deepseek-r1") || 
    name_lower.contains("qwq") ||
    name_lower.contains("reasoning")
}
```

### Issue 3: Extensions field always empty
**Solution**: Verify chunk collection is happening in the worker:
```rust
// Add to executor.rs
let mut collector = ChunkCollector::new();
// ... during generation ...
collector.add_reasoning(token.text.clone());
// ... at end ...
self.model.store_collected_chunks(&request_id, collector);
```

## Performance Tuning

### Use Bounded Channels
```rust
// Replace
let (tx, rx) = flume::unbounded::<ChatResponse>();

// With
let (tx, rx) = flume::bounded::<ChatResponse>(1000);
```

### Add Concurrency Limits
```rust
use tokio::sync::Semaphore;
use once_cell::sync::Lazy;

static STREAMING_SEM: Lazy<Semaphore> = Lazy::new(|| Semaphore::new(100));

// Before spawning
let permit = STREAMING_SEM.acquire().await?;
tokio::spawn(async move {
    let _permit = permit;
    // Work
});
```

## Next Steps

1. ✅ Implement basic state machine and collector
2. ⏭️ Add incremental tool call parsing (see `TOOL_CALL_STREAMING_IMPL.md`)
3. ⏭️ Add worker-level chunk collection
4. ⏭️ Add comprehensive tests
5. ⏭️ Add performance optimizations
6. ⏭️ Update documentation

## Full Documentation

- **Specification**: `TOOL_CALL_STREAMING_SPEC.md`
- **Implementation Plan**: `TOOL_CALL_STREAMING_IMPL.md`
- **Architecture Audit**: `PIPELINE_AUDIT.md`

## Getting Help

- Check existing tests in `candle-vllm-core/tests/`
- Review tool_parser.rs for format examples
- See openai_server.rs for streaming patterns
- Read AG-UI docs: https://docs.ag-ui.com/concepts/events

---

**Time to MVP: 2-3 hours**  
**Time to Full Implementation: 1-2 weeks**