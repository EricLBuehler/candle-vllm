# Tool Call and Reasoning Streaming Specification

## Overview

This document specifies how Candle-vLLM emits streaming chunks for tool calls, reasoning/thinking tokens, and regular content in both streaming and non-streaming modes, following AG-UI protocol patterns and OpenAI API compatibility.

## Event Types

### 1. Tool Call Events (AG-UI Style)

Based on AG-UI protocol, tool call streaming follows a **Start → Args → End** pattern:

#### ToolCallStart
Emitted when a tool call is first detected.

**OpenAI Streaming Format:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "model-name",
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
    },
    "finish_reason": null
  }]
}
```

**AG-UI Event Equivalent:**
```typescript
{
  type: "TOOL_CALL_START",
  toolCallId: "call_abc123",
  toolCallName: "get_weather",
  parentMessageId: "msg-123"
}
```

#### ToolCallArgs (Incremental Arguments)
Emitted as tool arguments are streamed.

**OpenAI Streaming Format:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "model-name",
  "choices": [{
    "index": 0,
    "delta": {
      "tool_calls": [{
        "index": 0,
        "function": {
          "arguments": "{\"location\":"
        }
      }]
    },
    "finish_reason": null
  }]
}
```

**AG-UI Event Equivalent:**
```typescript
{
  type: "TOOL_CALL_ARGS",
  toolCallId: "call_abc123",
  delta: "{\"location\":"
}
```

#### ToolCallEnd
Emitted when tool call arguments are complete.

**OpenAI Streaming Format:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "model-name",
  "choices": [{
    "index": 0,
    "delta": {
      "tool_calls": [{
        "index": 0,
        "function": {
          "arguments": "\"}"
        }
      }]
    },
    "finish_reason": null
  }]
}
```

**AG-UI Event Equivalent:**
```typescript
{
  type: "TOOL_CALL_END",
  toolCallId: "call_abc123"
}
```

### 2. Reasoning/Thinking Events

For models that emit reasoning tokens (DeepSeek-R1, QwQ, etc.), we emit reasoning chunks separately from content.

#### ReasoningStart
Emitted when reasoning begins.

**OpenAI Streaming Format (Extended):**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "deepseek-r1",
  "choices": [{
    "index": 0,
    "delta": {
      "role": "assistant",
      "reasoning": "<think>Let me analyze this problem..."
    },
    "finish_reason": null
  }]
}
```

**AG-UI Event Equivalent:**
```typescript
{
  type: "REASONING_START",
  messageId: "reasoning-001"
}
```

#### ReasoningContent (Incremental Reasoning)
Emitted as reasoning tokens are generated.

**OpenAI Streaming Format:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "deepseek-r1",
  "choices": [{
    "index": 0,
    "delta": {
      "reasoning": "First, I need to consider..."
    },
    "finish_reason": null
  }]
}
```

**AG-UI Event Equivalent:**
```typescript
{
  type: "REASONING_MESSAGE_CONTENT",
  messageId: "reasoning-001",
  delta: "First, I need to consider..."
}
```

#### ReasoningEnd
Emitted when reasoning is complete.

**OpenAI Streaming Format:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "deepseek-r1",
  "choices": [{
    "index": 0,
    "delta": {
      "reasoning": "</think>"
    },
    "finish_reason": null
  }]
}
```

**AG-UI Event Equivalent:**
```typescript
{
  type: "REASONING_END",
  messageId: "reasoning-001"
}
```

### 3. Regular Content Events

Standard text content streaming.

**OpenAI Streaming Format:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "model-name",
  "choices": [{
    "index": 0,
    "delta": {
      "content": "The weather in"
    },
    "finish_reason": null
  }]
}
```

**AG-UI Event Equivalent:**
```typescript
{
  type: "TEXT_MESSAGE_CONTENT",
  messageId: "msg-123",
  delta: "The weather in"
}
```

## Detection Logic

### Reasoning Token Detection

Reasoning tokens are identified by:

1. **Model Type Check**: Model must be in known reasoning model list
   - `deepseek-r1`, `deepseek-r1-*`
   - `qwq-*`
   - Models with `reasoning` or `thinking` in name
   - `ministral-*-reasoning`

2. **Thinking Flag**: Request must have `thinking: true`

3. **Token Patterns**: Token text contains reasoning markers
   - `<think>`, `</think>`
   - `<reasoning>`, `</reasoning>`
   - `<thought>`, `</thought>`

4. **Worker Flag**: Token has `is_reasoning: true` from worker

```rust
fn is_reasoning_token(
    token_text: &str,
    token_id: u32,
    model_name: &str,
    thinking_enabled: bool,
    token_is_reasoning: bool,
) -> bool {
    // Worker override
    if token_is_reasoning {
        return true;
    }

    // Must be enabled and supported
    if !thinking_enabled || !is_reasoning_model(model_name) {
        return false;
    }

    // Pattern detection
    let text = token_text.to_lowercase();
    text.contains("<think>")
        || text.contains("</think>")
        || text.contains("<reasoning>")
        || text.contains("</reasoning>")
        || text.contains("<thought>")
        || text.contains("</thought>")
}
```

### Tool Call Detection

Tool calls are identified by parsing model output for specific formats:

1. **Mistral Format**: `[TOOL_CALLS] [{"name": "...", "arguments": {...}}]`
2. **Llama Format**: `<function=func_name>{"arg": "value"}</function>`
3. **Qwen Format**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
4. **Generic JSON**: `{"name": "...", "arguments": {...}}`

Tool call streaming should emit incremental deltas as the JSON arguments are generated.

## Non-Streaming (Blocking) Mode

For non-streaming requests, all chunks (reasoning, tool calls, content) are collected and added to the final response's `extensions` property.

### Response Structure

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "deepseek-r1",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The answer is 42.",
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "calculate",
            "arguments": "{\"expression\": \"6*7\"}"
          }
        }
      ]
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150,
    "prompt_tokens_details": {
      "cached_tokens": 20
    }
  },
  "extensions": {
    "reasoning_chunks": [
      "Let me think about this...",
      "First, I'll calculate 6 times 7...",
      "The result is 42."
    ],
    "tool_call_chunks": [
      {
        "type": "start",
        "tool_call_id": "call_abc123",
        "tool_name": "calculate"
      },
      {
        "type": "args",
        "tool_call_id": "call_abc123",
        "delta": "{\"expression\": \"6*7\"}"
      },
      {
        "type": "end",
        "tool_call_id": "call_abc123"
      }
    ]
  }
}
```

## Implementation Requirements

### Streaming Path

1. **Token Classification**: Each token must be classified as:
   - Reasoning token
   - Tool call token (part of function call JSON)
   - Regular content token

2. **Delta Emission**: Emit appropriate delta based on token type:
   ```rust
   let delta = if token_is_reasoning {
       ChoiceData {
           role: if is_first_chunk { Some("assistant".to_string()) } else { None },
           content: None,
           tool_calls: None,
           reasoning: Some(token.text.clone()),
       }
   } else if in_tool_call {
       // TODO: Implement tool call streaming
       ChoiceData {
           role: None,
           content: None,
           tool_calls: Some(vec![tool_call_delta]),
           reasoning: None,
       }
   } else {
       ChoiceData {
           role: if is_first_chunk { Some("assistant".to_string()) } else { None },
           content: Some(token.text.clone()),
           tool_calls: None,
           reasoning: None,
       }
   };
   ```

3. **State Tracking**: Maintain state for:
   - Current tool call ID
   - Tool call arguments buffer
   - Reasoning state (in/out of reasoning)
   - First chunk flag

### Non-Streaming Path

1. **Chunk Collection**: Collect all chunks during generation:
   ```rust
   struct CompletionCollector {
       reasoning_chunks: Vec<String>,
       tool_call_chunks: Vec<ToolCallChunk>,
       content_chunks: Vec<String>,
   }
   ```

2. **Extensions Addition**: Add collected chunks to `extensions` field:
   ```rust
   let extensions = serde_json::json!({
       "reasoning_chunks": collector.reasoning_chunks,
       "tool_call_chunks": collector.tool_call_chunks,
   });
   ```

## Error Handling

### Tool Call Errors

If tool call parsing fails mid-stream:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "model-name",
  "choices": [{
    "index": 0,
    "delta": {},
    "finish_reason": "error"
  }],
  "error": {
    "message": "Tool call parsing failed: Invalid JSON",
    "type": "tool_call_parse_error"
  }
}
```

### Reasoning Token Errors

If reasoning detection fails, fall back to treating as regular content.

## Testing Requirements

### Unit Tests

1. ✅ Test reasoning token detection
2. ✅ Test tool call delta creation
3. ✅ Test chunk serialization
4. ⚠️ Test tool call streaming with all formats (Mistral, Llama, Qwen, generic)
5. ⚠️ Test mixed reasoning + content streaming
6. ⚠️ Test mixed tool calls + content streaming

### Integration Tests

1. ⚠️ End-to-end streaming with reasoning model
2. ⚠️ End-to-end streaming with tool calls
3. ⚠️ Non-streaming with extensions collection
4. ⚠️ Client disconnection during streaming
5. ⚠️ Tool call streaming across multiple tool calls

## Performance Considerations

### Channel Backpressure

Currently using unbounded channels:
```rust
let (response_tx, rx) = flume::unbounded::<ChatResponse>();
```

**Recommendation**: Use bounded channels with appropriate size:
```rust
let (response_tx, rx) = flume::bounded::<ChatResponse>(1000);
```

This prevents memory growth if client is slow.

### Thread Pool Management

Currently spawning threads per request:
```rust
std::thread::spawn(move || {
    // Streaming bridge
});
```

**Recommendation**: Use a dedicated thread pool or tokio runtime with semaphore:
```rust
use tokio::sync::Semaphore;

static STREAMING_SEMAPHORE: Lazy<Semaphore> = Lazy::new(|| Semaphore::new(100));

let permit = STREAMING_SEMAPHORE.acquire().await?;
tokio::spawn(async move {
    let _permit = permit; // Hold until done
    // Streaming work
});
```

### Token Buffer Size

For tool call streaming, buffer management is critical:
- Too small: Frequent partial JSON updates (inefficient)
- Too large: Delayed feedback to client

**Recommendation**: Adaptive buffering based on content type:
- Regular content: Emit immediately (token-by-token)
- Tool call JSON: Buffer until valid JSON fragment (bracket/quote boundaries)
- Reasoning: Emit by sentence or thought boundaries

## OpenAI API Compatibility

### Standard Fields

All chunks must include:
- `id`: Request ID (consistent across stream)
- `object`: `"chat.completion.chunk"`
- `created`: Unix timestamp
- `model`: Model name
- `choices`: Array of choice deltas

### Extended Fields (Candle-vLLM Specific)

- `choices[].delta.reasoning`: Reasoning/thinking content
- `extensions`: Non-streaming mode metadata

### Finish Reasons

- `"stop"`: Natural completion
- `"length"`: Max tokens reached
- `"tool_calls"`: Tool calls present
- `"content_filter"`: Content filtered (if applicable)
- `"error"`: Generation error

## Future Enhancements

1. **Tool Execution Results**: Stream tool results back to model
2. **Multi-Tool Streaming**: Handle multiple simultaneous tool calls
3. **Reasoning Summaries**: Emit condensed reasoning in extensions
4. **Activity Events**: AG-UI ActivitySnapshot/ActivityDelta for long operations
5. **Reasoning Encryption**: Support encrypted reasoning for privacy (AG-UI spec)

## References

- [AG-UI Events Documentation](https://docs.ag-ui.com/concepts/events)
- [AG-UI Reasoning Proposal](https://docs.ag-ui.com/drafts/reasoning)
- [OpenAI Streaming API](https://platform.openai.com/docs/guides/streaming-responses)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI Reasoning Models](https://platform.openai.com/docs/guides/reasoning)

## Changelog

- **2025-01-XX**: Initial specification
- **TODO**: Implement tool call streaming state machine
- **TODO**: Add extensions collection for non-streaming mode
- **TODO**: Implement bounded channels and thread pool
- **TODO**: Add comprehensive integration tests