# Tool Call and Reasoning Streaming Implementation Plan

## Executive Summary

This document outlines the implementation plan for adding proper AG-UI style tool call streaming and reasoning token handling to Candle-vLLM. The implementation will ensure:

1. **Tool calls stream incrementally** with start/args/end deltas
2. **Reasoning tokens stream separately** from content
3. **Non-streaming mode collects chunks** and adds them to `extensions`
4. **Proper state management** across streaming pipeline
5. **OpenAI API compatibility** with AG-UI enhancements

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        HTTP Request Layer                        │
│  (openai_server.rs - chat_completions_with_data)                │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Token Classification                         │
│  - Reasoning detection (is_reasoning_token)                     │
│  - Tool call detection (ToolCallParser)                         │
│  - Content classification                                        │
└───────────────────┬─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  Streaming Mode  │    │ Non-Streaming   │
│                  │    │     Mode         │
│ - Emit deltas    │    │ - Collect chunks│
│ - Send SSE       │    │ - Add extensions│
└──────────────────┘    └──────────────────┘
```

## Phase 1: Core Data Structures

### 1.1 Tool Call State Machine

**File**: `candle-vllm/crates/candle-vllm-core/src/openai/tool_streaming.rs`

```rust
/// State machine for tracking tool call streaming
#[derive(Debug, Clone)]
pub struct ToolCallStreamState {
    /// Current active tool calls (index -> state)
    active_calls: HashMap<usize, ActiveToolCall>,
    /// Next available tool call index
    next_index: usize,
}

#[derive(Debug, Clone)]
struct ActiveToolCall {
    /// Unique ID for this tool call
    id: String,
    /// Tool/function name
    name: String,
    /// Accumulated arguments (JSON fragments)
    arguments_buffer: String,
    /// State of this tool call
    state: ToolCallState,
}

#[derive(Debug, Clone, PartialEq)]
enum ToolCallState {
    /// Just started, need to emit start event
    Started,
    /// Emitting arguments
    Arguments,
    /// Completed, need to emit end event
    Completed,
}

impl ToolCallStreamState {
    /// Create a new tool call state machine
    pub fn new() -> Self {
        Self {
            active_calls: HashMap::new(),
            next_index: 0,
        }
    }

    /// Start a new tool call
    pub fn start_tool_call(&mut self, name: String) -> (usize, ToolCallDelta) {
        let id = format!("call_{}", Uuid::new_v4().to_string().replace("-", "")[..12].to_string());
        let index = self.next_index;
        self.next_index += 1;

        self.active_calls.insert(
            index,
            ActiveToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments_buffer: String::new(),
                state: ToolCallState::Started,
            },
        );

        // Create start delta
        let delta = ToolCallDelta {
            index,
            id: Some(id),
            call_type: Some("function".to_string()),
            function: Some(FunctionCallDelta {
                name: Some(name),
                arguments: Some(String::new()),
            }),
        };

        (index, delta)
    }

    /// Add arguments to an active tool call
    pub fn add_arguments(&mut self, index: usize, args: &str) -> Option<ToolCallDelta> {
        if let Some(call) = self.active_calls.get_mut(&index) {
            call.arguments_buffer.push_str(args);
            call.state = ToolCallState::Arguments;

            Some(ToolCallDelta {
                index,
                id: None,
                call_type: None,
                function: Some(FunctionCallDelta {
                    name: None,
                    arguments: Some(args.to_string()),
                }),
            })
        } else {
            None
        }
    }

    /// Complete a tool call
    pub fn complete_tool_call(&mut self, index: usize) -> Option<ToolCallDelta> {
        if let Some(call) = self.active_calls.get_mut(&index) {
            call.state = ToolCallState::Completed;

            // Final delta (empty to signal completion)
            Some(ToolCallDelta {
                index,
                id: None,
                call_type: None,
                function: Some(FunctionCallDelta {
                    name: None,
                    arguments: Some(String::new()),
                }),
            })
        } else {
            None
        }
    }

    /// Get all completed tool calls for final response
    pub fn finalize(&self) -> Vec<ToolCall> {
        self.active_calls
            .values()
            .filter(|c| c.state == ToolCallState::Completed)
            .map(|c| ToolCall {
                id: c.id.clone(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: c.name.clone(),
                    arguments: c.arguments_buffer.clone(),
                },
            })
            .collect()
    }
}
```

### 1.2 Chunk Collector for Non-Streaming

**File**: `candle-vllm/crates/candle-vllm-core/src/openai/chunk_collector.rs`

```rust
/// Collector for chunks in non-streaming mode
#[derive(Debug, Clone, Default)]
pub struct ChunkCollector {
    /// Reasoning chunks
    reasoning_chunks: Vec<String>,
    /// Tool call event chunks
    tool_call_chunks: Vec<ToolCallChunkEvent>,
    /// Content chunks (for debugging/logging)
    content_chunks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallChunkEvent {
    #[serde(rename = "type")]
    pub event_type: ToolCallEventType,
    pub tool_call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolCallEventType {
    Start,
    Args,
    End,
}

impl ChunkCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a reasoning chunk
    pub fn add_reasoning(&mut self, text: String) {
        if !text.is_empty() {
            self.reasoning_chunks.push(text);
        }
    }

    /// Add a tool call start event
    pub fn add_tool_call_start(&mut self, id: String, name: String) {
        self.tool_call_chunks.push(ToolCallChunkEvent {
            event_type: ToolCallEventType::Start,
            tool_call_id: id,
            tool_name: Some(name),
            delta: None,
        });
    }

    /// Add tool call arguments
    pub fn add_tool_call_args(&mut self, id: String, args: String) {
        self.tool_call_chunks.push(ToolCallChunkEvent {
            event_type: ToolCallEventType::Args,
            tool_call_id: id,
            tool_name: None,
            delta: Some(args),
        });
    }

    /// Add tool call end event
    pub fn add_tool_call_end(&mut self, id: String) {
        self.tool_call_chunks.push(ToolCallChunkEvent {
            event_type: ToolCallEventType::End,
            tool_call_id: id,
            tool_name: None,
            delta: None,
        });
    }

    /// Add content chunk
    pub fn add_content(&mut self, text: String) {
        if !text.is_empty() {
            self.content_chunks.push(text);
        }
    }

    /// Convert to extensions JSON
    pub fn to_extensions(&self) -> serde_json::Value {
        let mut extensions = serde_json::Map::new();

        if !self.reasoning_chunks.is_empty() {
            extensions.insert(
                "reasoning_chunks".to_string(),
                serde_json::json!(self.reasoning_chunks),
            );
        }

        if !self.tool_call_chunks.is_empty() {
            extensions.insert(
                "tool_call_chunks".to_string(),
                serde_json::json!(self.tool_call_chunks),
            );
        }

        serde_json::Value::Object(extensions)
    }

    /// Check if collector has any data
    pub fn is_empty(&self) -> bool {
        self.reasoning_chunks.is_empty() && self.tool_call_chunks.is_empty()
    }
}
```

### 1.3 Enhanced Response Types

**File**: `candle-vllm/crates/candle-vllm-core/src/openai/responses.rs` (additions)

```rust
// Add to ChatCompletionResponse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    // ... existing fields ...
    
    /// Extended metadata (reasoning chunks, tool call events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
}

// Add helper methods to ToolCallDelta
impl ToolCallDelta {
    /// Create a start delta
    pub fn start(index: usize, id: String, name: String) -> Self {
        Self {
            index,
            id: Some(id),
            call_type: Some("function".to_string()),
            function: Some(FunctionCallDelta {
                name: Some(name),
                arguments: Some(String::new()),
            }),
        }
    }

    /// Create an arguments delta
    pub fn arguments(index: usize, args: String) -> Self {
        Self {
            index,
            id: None,
            call_type: None,
            function: Some(FunctionCallDelta {
                name: None,
                arguments: Some(args),
            }),
        }
    }

    /// Create an end delta (empty function)
    pub fn end(index: usize) -> Self {
        Self {
            index,
            id: None,
            call_type: None,
            function: None,
        }
    }
}
```

## Phase 2: Streaming Implementation

### 2.1 Enhanced Streaming Bridge

**File**: `candle-vllm/crates/candle-vllm-core/src/openai/openai_server.rs` (modifications)

```rust
// Inside the streaming bridge thread (line ~440)
std::thread::spawn(move || {
    let mut full_content = String::new();
    let mut full_reasoning = String::new();
    let mut is_first_chunk = true;
    let mut token_count = 0;
    
    // NEW: Tool call state machine
    let mut tool_call_state = ToolCallStreamState::new();
    let mut current_tool_buffer = String::new();
    let mut in_tool_call = false;
    
    // NEW: Tool parser for detecting tool calls
    let tool_parser = get_tool_parser(&model_name_for_stream);

    info!(
        "👂 CORE: Streaming bridge thread started with tool call tracking - request_id={}",
        request_id
    );
    
    while let Ok(result) = stream_rx.recv() {
        token_count += 1;
        match result {
            Ok(token) => {
                // Log progress
                if token_count == 1 {
                    info!("🎉 CORE: Received FIRST streaming token - request_id={}", request_id);
                }
                
                // Classify token type
                let token_is_reasoning = is_reasoning_token(
                    &token.text,
                    token.token_id,
                    &model_name_for_stream,
                    thinking_enabled,
                    token.is_reasoning,
                );

                // NEW: Check for tool call patterns
                current_tool_buffer.push_str(&token.text);
                let tool_parse_result = tool_parser.parse_incremental(&current_tool_buffer);
                
                let delta = match tool_parse_result {
                    ToolParseState::InProgress(partial) => {
                        // We're building a tool call
                        if !in_tool_call && partial.name.is_some() {
                            // Start of new tool call
                            in_tool_call = true;
                            let (index, start_delta) = tool_call_state.start_tool_call(
                                partial.name.unwrap()
                            );
                            
                            ChoiceData {
                                role: if is_first_chunk { Some("assistant".to_string()) } else { None },
                                content: None,
                                tool_calls: Some(vec![start_delta]),
                                reasoning: None,
                            }
                        } else if in_tool_call && !partial.arguments.is_empty() {
                            // Arguments being streamed
                            if let Some(args_delta) = tool_call_state.add_arguments(0, &partial.arguments) {
                                ChoiceData {
                                    role: None,
                                    content: None,
                                    tool_calls: Some(vec![args_delta]),
                                    reasoning: None,
                                }
                            } else {
                                // Error state, treat as content
                                ChoiceData::content(token.text.clone())
                            }
                        } else {
                            // Still accumulating, emit regular content
                            ChoiceData::content(token.text.clone())
                        }
                    }
                    ToolParseState::Complete(tool_call) => {
                        // Tool call complete
                        if let Some(end_delta) = tool_call_state.complete_tool_call(0) {
                            current_tool_buffer.clear();
                            in_tool_call = false;
                            
                            ChoiceData {
                                role: None,
                                content: None,
                                tool_calls: Some(vec![end_delta]),
                                reasoning: None,
                            }
                        } else {
                            ChoiceData::content(token.text.clone())
                        }
                    }
                    ToolParseState::NotToolCall => {
                        // Regular token classification
                        if token_is_reasoning {
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
                        }
                    }
                };

                is_first_chunk = false;

                let chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    choices: vec![Choice {
                        delta,
                        finish_reason: if token.is_finished {
                            token.finish_reason.clone()
                        } else {
                            None
                        },
                        index: 0,
                    }],
                    created,
                    model: model_name_for_stream.to_string(),
                    object: "chat.completion.chunk",
                    system_fingerprint: None,
                    conversation_id: None,
                    resource_id: None,
                };

                if response_tx.send(ChatResponse::Chunk(chunk)).is_err() {
                    error!("❌ CORE: Failed to send chunk, client disconnected - request_id={}", request_id);
                    break;
                }

                if token.is_finished {
                    info!("🏁 CORE: Streaming finished - request_id={}, total_tokens={}", 
                        request_id, token_count);
                    let _ = response_tx.send(ChatResponse::Done);
                    sync_notify_clone.notify_one();
                    break;
                }
            }
            Err(err) => {
                error!("❌ CORE: Streaming error - request_id={}, error={}", request_id, err);
                let _ = response_tx.send(ChatResponse::ModelError(err));
                sync_notify_clone.notify_one();
                break;
            }
        }
    }
});
```

### 2.2 Tool Parser Enhancements

**File**: `candle-vllm/crates/candle-vllm-core/src/openai/tool_parser.rs` (additions)

```rust
/// Result of incremental tool call parsing
pub enum ToolParseState {
    /// Not a tool call (regular content)
    NotToolCall,
    /// Tool call in progress (partial data)
    InProgress(PartialToolCall),
    /// Tool call complete
    Complete(ToolCallInfo),
}

#[derive(Debug, Clone)]
pub struct PartialToolCall {
    /// Tool name (if detected)
    pub name: Option<String>,
    /// Partial arguments accumulated so far
    pub arguments: String,
}

pub trait IncrementalToolParser {
    /// Parse incrementally as tokens arrive
    fn parse_incremental(&self, buffer: &str) -> ToolParseState;
}

// Implement for each format
impl IncrementalToolParser for MistralToolParser {
    fn parse_incremental(&self, buffer: &str) -> ToolParseState {
        // Look for [TOOL_CALLS] marker
        if !buffer.contains("[TOOL_CALLS]") {
            return ToolParseState::NotToolCall;
        }
        
        // Extract JSON after marker
        if let Some(json_start) = buffer.find('[', buffer.find("[TOOL_CALLS]").unwrap()) {
            let json_part = &buffer[json_start..];
            
            // Try to parse complete JSON
            if let Ok(tools) = serde_json::from_str::<Vec<serde_json::Value>>(json_part) {
                // Complete!
                return ToolParseState::Complete(/* ... */);
            }
            
            // Partial JSON, try to extract what we can
            // ... (partial parsing logic)
            return ToolParseState::InProgress(/* ... */);
        }
        
        ToolParseState::NotToolCall
    }
}
```

## Phase 3: Non-Streaming Implementation

### 3.1 Enhanced Completion Handler

**File**: `candle-vllm/crates/candle-vllm-core/src/openai/openai_server.rs` (modifications)

```rust
// Inside non-streaming response handler (line ~580)
std::thread::spawn(move || {
    use crate::parking_lot::InferenceResult;
    
    // NEW: Chunk collector for extensions
    let mut collector = ChunkCollector::new();
    
    info!("👂 CORE: Waiting for completion result from engine - request_id={}", request_id);
    
    match response_rx.blocking_recv() {
        Ok(InferenceResult::Completion { choices, mut usage }) => {
            info!("✅ CORE: Received completion result - request_id={}, choices={}", 
                request_id, choices.len());
            
            // Update usage with cached token information
            data.model.update_usage_with_cache(&mut usage, &request_id);
            
            // NEW: If the generation included streaming tokens, collect them
            // This happens when we internally collected chunks during generation
            if let Some(chunks) = data.model.get_collected_chunks(&request_id) {
                for chunk in chunks {
                    match chunk {
                        CollectedChunk::Reasoning(text) => {
                            collector.add_reasoning(text);
                        }
                        CollectedChunk::ToolCallStart { id, name } => {
                            collector.add_tool_call_start(id, name);
                        }
                        CollectedChunk::ToolCallArgs { id, args } => {
                            collector.add_tool_call_args(id, args);
                        }
                        CollectedChunk::ToolCallEnd { id } => {
                            collector.add_tool_call_end(id);
                        }
                        CollectedChunk::Content(_) => {
                            // Regular content, already in choices
                        }
                    }
                }
            }
            
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
                extensions, // NEW: Add collected chunks
            };
            
            // Store in completion_records
            let mut records = data.model.completion_records.write();
            records.insert(
                request_id.clone(),
                (response.choices.clone(), response.usage.clone()),
            );
            
            sync_notify_clone.notify_one();
        }
        // ... error handling ...
    }
});
```

## Phase 4: Worker/Executor Enhancements

### 4.1 Chunk Collection in Worker

**File**: `candle-vllm/crates/candle-vllm-core/src/parking_lot/executor.rs` (modifications)

For non-streaming requests, we need to collect chunks during generation:

```rust
// In process_completion (non-streaming path)
fn process_completion(&mut self, request: WorkItem) -> Result<()> {
    // ... existing logic ...
    
    // NEW: Chunk collector for this request
    let mut collector = ChunkCollector::new();
    let model_name = self.model.config.model_name();
    let thinking_enabled = request.sampling_params.thinking.unwrap_or(false);
    
    // During token generation loop
    for step in 0..max_tokens {
        let token = self.generate_token(&request)?;
        
        // Classify token
        let is_reasoning = is_reasoning_token(
            &token.text,
            token.id,
            &model_name,
            thinking_enabled,
            token.is_reasoning,
        );
        
        if is_reasoning {
            collector.add_reasoning(token.text.clone());
        } else {
            collector.add_content(token.text.clone());
        }
        
        // Tool call detection
        // ... (similar to streaming logic)
        
        output.push_str(&token.text);
        
        if token.is_eos {
            break;
        }
    }
    
    // Store collected chunks for later retrieval
    self.model.store_collected_chunks(&request.request_id, collector);
    
    // ... rest of completion logic ...
}
```

## Phase 5: Testing Strategy

### 5.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_state_machine() {
        let mut state = ToolCallStreamState::new();
        
        // Start tool call
        let (index, start_delta) = state.start_tool_call("get_weather".to_string());
        assert_eq!(index, 0);
        assert!(start_delta.id.is_some());
        assert_eq!(start_delta.function.as_ref().unwrap().name, Some("get_weather".to_string()));
        
        // Add arguments
        let args_delta1 = state.add_arguments(index, "{\"location\":").unwrap();
        assert_eq!(args_delta1.function.as_ref().unwrap().arguments, Some("{\"location\":".to_string()));
        
        let args_delta2 = state.add_arguments(index, " \"NYC\"}").unwrap();
        assert_eq!(args_delta2.function.as_ref().unwrap().arguments, Some(" \"NYC\"}".to_string()));
        
        // Complete
        let end_delta = state.complete_tool_call(index).unwrap();
        assert!(end_delta.function.is_none());
        
        // Finalize
        let calls = state.finalize();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"location\": \"NYC\"}");
    }

    #[test]
    fn test_chunk_collector() {
        let mut collector = ChunkCollector::new();
        
        collector.add_reasoning("Thinking step 1".to_string());
        collector.add_reasoning("Thinking step 2".to_string());
        
        collector.add_tool_call_start("call_123".to_string(), "calculate".to_string());
        collector.add_tool_call_args("call_123".to_string(), "{\"expr\":".to_string());
        collector.add_tool_call_args("call_123".to_string(), "\"2+2\"}".to_string());
        collector.add_tool_call_end("call_123".to_string());
        
        let extensions = collector.to_extensions();
        assert!(!collector.is_empty());
        
        let reasoning = extensions.get("reasoning_chunks").unwrap().as_array().unwrap();
        assert_eq!(reasoning.len(), 2);
        
        let tool_chunks = extensions.get("tool_call_chunks").unwrap().as_array().unwrap();
        assert_eq!(tool_chunks.len(), 4); // start + 2 args + end
    }
}
```

### 5.2 Integration Tests

```rust
#[tokio::test]
async fn test_streaming_with_tool_calls() {
    // Setup test server
    let data = setup_test_server().await;
    
    // Request with tools
    let request = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: Messages::Chat(vec![/* ... */]),
        tools: Some(vec![/* tool definitions */]),
        stream: Some(true),
        ..Default::default()
    };
    
    let response = chat_completions_with_data(data, request).await;
    
    // Verify streaming chunks
    match response {
        ChatResponder::Streamer(stream) => {
            let chunks: Vec<_> = stream.collect().await;
            
            // Should have: role chunk, tool call start, args chunks, tool call end, done
            assert!(chunks.len() >= 4);
            
            // Verify tool call deltas
            // ... (assertions)
        }
        _ => panic!("Expected streamer"),
    }
}

#[tokio::test]
async fn test_non_streaming_with_extensions() {
    let data = setup_test_server().await;
    
    let request = ChatCompletionRequest {
        model: "deepseek-r1".to_string(),
        messages: Messages::Chat(vec![/* ... */]),
        thinking: Some(true),
        stream: Some(false),
        ..Default::default()
    };
    
    let response = chat_completions_with_data(data, request).await;
    
    // Should get completion with extensions
    match response {
        ChatResponder::Completion(completion) => {
            assert!(completion.extensions.is_some());
            
            let ext = completion.extensions.unwrap();
            assert!(ext.get("reasoning_chunks").is_some());
            
            let reasoning = ext.get("reasoning_chunks").unwrap().as_array().unwrap();
            assert!(!reasoning.is_empty());
        }
        _ => panic!("Expected completion"),
    }
}
```

## Phase 6: Performance Optimizations

### 6.1 Bounded Channels

```rust
// Replace unbounded channels
const STREAMING_BUFFER_SIZE: usize = 1000;
let (response_tx, rx) = flume::bounded::<ChatResponse>(STREAMING_BUFFER_SIZE);
```

### 6.2 Thread Pool for Streaming

```rust
use once_cell::sync::Lazy;
use rayon::ThreadPool;

static STREAMING_POOL: Lazy<ThreadPool> = Lazy::new(|| {
    rayon::ThreadPoolBuilder::new()
        .num_threads(100)
        .thread_name(|i| format!("streaming-{}", i))
        .build()
        .unwrap()
});

// Use pool instead of spawn
STREAMING_POOL.spawn(move || {
    // Streaming work
});
```

### 6.3 Semaphore for Concurrency Control

```rust
use tokio::sync::Semaphore;
use once_cell::sync::Lazy;

static STREAMING_SEMAPHORE: Lazy<Semaphore> = Lazy::new(|| {
    Semaphore::new(100) // Max 100 concurrent streams
});

// Acquire permit before spawning
let permit = STREAMING_SEMAPHORE.acquire().await?;
tokio::spawn(async move {
    let _permit = permit; // Hold until done
    // Work
});
```

## Phase 7: Documentation Updates

### 7.1 API Documentation

Update `docs/API.md` with:
- New `extensions` field in non-streaming responses
- Tool call streaming format
- Reasoning token streaming format

### 7.2 Configuration Guide

Update `docs/CONFIGURATION.md` with:
- `thinking` parameter for reasoning models
- Tool call streaming behavior
- Performance tuning options

### 7.3 Examples

Add examples for:
- Streaming with tool calls
- Reasoning model usage
- Extension field parsing

## Implementation Checklist

### Phase 1: Core Structures
- [ ] Implement `ToolCallStreamState`
- [ ] Implement `ChunkCollector`
- [ ] Add `extensions` field to `ChatCompletionResponse`
- [ ] Add helper methods to `ToolCallDelta`

### Phase 2: Streaming
- [ ] Enhance streaming bridge with tool call detection
- [ ] Implement incremental tool parsing
- [ ] Add tool call state machine to streaming loop
- [ ] Test streaming with multiple tool calls

### Phase 3: Non-Streaming
- [ ] Add chunk collection to completion handler
- [ ] Store collected chunks in model state
- [ ] Add extensions to final response
- [ ] Test non-streaming with extensions

### Phase 4: Worker Updates
- [ ] Add chunk collector to worker
- [ ] Implement chunk storage mechanism
- [ ] Update completion path with collection
- [ ] Test worker chunk collection

### Phase 5: Testing
- [ ] Unit tests for state machine
- [ ] Unit tests for collector
- [ ] Integration tests for streaming
- [ ] Integration tests for non-streaming
- [ ] Performance/load tests

### Phase 6: Performance
- [ ] Replace unbounded channels with bounded
- [ ] Add thread pool for streaming
- [ ] Add semaphore for concurrency control
- [ ] Benchmark and tune

### Phase 7: Documentation
- [ ] Update API documentation
- [ ] Update configuration guide
- [ ] Ad