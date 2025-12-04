# Data Model: Library-First Architecture

**Branch**: `001-library-api-implementation`  
**Date**: December 3, 2025  
**Spec**: [spec.md](./spec.md)

## Entity Relationship Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            candle-vllm-core                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │ InferenceEngine  │──────│    Scheduler     │──────│   CacheEngine    │  │
│  │                  │      │                  │      │                  │  │
│  │ - tokenizer      │      │ - sequences      │      │ - gpu_cache      │  │
│  │ - pipelines      │      │ - running        │      │ - cpu_cache      │  │
│  │ - config         │      │ - waiting        │      │ - block_size     │  │
│  └────────┬─────────┘      └──────────────────┘      └──────────────────┘  │
│           │                                                                  │
│           │ generates                                                        │
│           ▼                                                                  │
│  ┌──────────────────┐      ┌──────────────────┐                             │
│  │ GenerationOutput │      │ GenerationParams │                             │
│  │                  │      │                  │                             │
│  │ - tokens         │      │ - max_tokens     │                             │
│  │ - finish_reason  │      │ - temperature    │                             │
│  │ - logprobs       │      │ - top_p/top_k    │                             │
│  │ - stats          │      │ - penalties      │                             │
│  └──────────────────┘      └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           candle-vllm-openai                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │  OpenAIAdapter   │──────│ConversationMgr   │──────│   ToolParser     │  │
│  │                  │      │                  │      │                  │  │
│  │ - engine         │      │ - templates      │      │ - patterns       │  │
│  │ - conv_manager   │      │ - history        │      │ - extractors     │  │
│  │ - tool_parser    │      │                  │      │                  │  │
│  └────────┬─────────┘      └──────────────────┘      └──────────────────┘  │
│           │                                                                  │
│           │ handles                                                          │
│           ▼                                                                  │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │ChatCompletion    │      │ChatCompletion    │      │     Message      │  │
│  │    Request       │──────│   Response       │      │                  │  │
│  │                  │      │                  │      │ - role           │  │
│  │ - model          │      │ - choices        │      │ - content        │  │
│  │ - messages       │      │ - usage          │      │ - tool_calls     │  │
│  │ - tools          │      │ - id             │      │ - tool_call_id   │  │
│  └──────────────────┘      └──────────────────┘      └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         candle-vllm-responses                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │ResponsesSession  │──────│    McpClient     │──────│McpServerConfig   │  │
│  │                  │      │                  │      │                  │  │
│  │ - adapter        │      │ - client         │      │ - url            │  │
│  │ - mcp_clients    │      │ - config         │      │ - auth           │  │
│  │ - options        │      │                  │      │ - timeout        │  │
│  └────────┬─────────┘      └──────────────────┘      └──────────────────┘  │
│           │                                                                  │
│           │ produces                                                         │
│           ▼                                                                  │
│  ┌──────────────────┐      ┌──────────────────┐                             │
│  │ConversationResult│      │ConversationOpts  │                             │
│  │                  │      │                  │                             │
│  │ - final_message  │      │ - max_turns      │                             │
│  │ - tool_calls     │      │ - allowed_tools  │                             │
│  │ - turns_taken    │      │                  │                             │
│  └──────────────────┘      └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          candle-vllm-server                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │  ModelManager    │──────│  RequestQueue    │──────│  QueuedRequest   │  │
│  │                  │      │                  │      │                  │  │
│  │ - status (FSM)   │      │ - requests       │      │ - id             │  │
│  │ - active_model   │      │ - max_size       │      │ - model          │  │
│  │ - last_error     │      │ - model          │      │ - stream         │  │
│  │ - queue          │      │                  │      │ - tx             │  │
│  └──────────────────┘      └──────────────────┘      └──────────────────┘  │
│           │                                                                  │
│           │ FSM states                                                       │
│           ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │           Idle ──► Ready ──► Switching ──► Loading ──► Ready        │  │
│  │                                   │                      │           │  │
│  │                                   └──────► Error ◄───────┘           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Entities

### InferenceEngine

The primary abstraction for model inference, wrapping internal pipeline management.

| Field | Type | Description |
|-------|------|-------------|
| tokenizer | Tokenizer | HuggingFace tokenizer for the loaded model |
| pipelines | HashMap<usize, (Pipeline, CacheEngine)> | Per-device pipeline instances |
| scheduler | Scheduler | Request scheduling and batching |
| config | EngineConfig | Configuration used to initialize |
| model_info | ModelInfo | Metadata about the loaded model |

**Invariants:**
- At least one pipeline must be present after initialization
- Tokenizer must match the model's vocabulary
- All pipelines share the same model architecture

**State Transitions:**
- Uninitialized → Initialized (via `new()` or builder)
- Initialized → Generating (via `generate()`)
- Generating → Initialized (on completion/error)

### EngineConfig

Configuration for creating an InferenceEngine.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| model_path | String | required | Path to model or HuggingFace ID |
| device | Device | auto-detect | CUDA(idx), Metal, or CPU |
| dtype | DType | BF16 | Model weight precision |
| max_batch_size | usize | 16 | Maximum concurrent sequences |
| max_sequence_length | usize | model default | Maximum context length |
| kv_cache_memory | usize | 4GB | GPU memory for KV cache |
| enable_cuda_graph | bool | false | Enable CUDA graph capture |
| enable_chunked_prefill | bool | false | Enable chunked prefill |
| prefill_chunk_size | usize | 1024 | Chunk size when enabled |

**Validation Rules:**
- `model_path` must be non-empty
- `max_batch_size` must be ≥ 1
- `prefill_chunk_size` must be divisible by 1024 if chunked prefill enabled
- `kv_cache_memory` must be positive

### GenerationParams

Parameters for a single generation request.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| max_tokens | usize | 256 | Maximum tokens to generate |
| temperature | f32 | 1.0 | Sampling temperature (0 = greedy) |
| top_p | f32 | 1.0 | Nucleus sampling threshold |
| top_k | Option<usize> | None | Top-k sampling limit |
| repetition_penalty | f32 | 1.0 | Penalty for repeated tokens |
| frequency_penalty | f32 | 0.0 | Penalty based on frequency |
| presence_penalty | f32 | 0.0 | Penalty based on presence |
| stop_sequences | Vec<String> | [] | Sequences that stop generation |
| logprobs | bool | false | Whether to return log probabilities |
| seed | Option<u64> | None | Random seed for reproducibility |

**Validation Rules:**
- `temperature` must be ≥ 0.0
- `top_p` must be in (0.0, 1.0]
- `repetition_penalty` must be > 0.0

### GenerationOutput

Result of a generation operation.

| Field | Type | Description |
|-------|------|-------------|
| tokens | Vec<u32> | Generated token IDs |
| finish_reason | FinishReason | Why generation stopped |
| logprobs | Option<Vec<f32>> | Per-token log probabilities |
| stats | GenerationStats | Performance metrics |

### FinishReason

Enumeration of generation completion reasons.

| Variant | Description |
|---------|-------------|
| Stop | Natural completion (EOS token) |
| Length | Reached max_tokens limit |
| StopSequence(String) | Hit a stop sequence |
| Cancelled | Request was cancelled |
| Error(String) | An error occurred |

### GenerationStats

Statistics about a generation operation.

| Field | Type | Description |
|-------|------|-------------|
| prompt_tokens | usize | Number of input tokens |
| generated_tokens | usize | Number of output tokens |
| total_time_ms | u64 | Total generation time |
| tokens_per_second | f32 | Generation throughput |

---

## OpenAI Compatibility Entities

### ChatCompletionRequest

OpenAI-compatible chat completion request (existing in `requests.rs`).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| model | String | yes | Model identifier |
| messages | Vec<Message> | yes | Conversation history |
| temperature | Option<f32> | no | Sampling temperature |
| top_p | Option<f32> | no | Nucleus sampling |
| max_tokens | Option<usize> | no | Maximum completion tokens |
| stream | Option<bool> | no | Enable streaming |
| tools | Option<Vec<Tool>> | no | Available tools |
| tool_choice | Option<ToolChoice> | no | Tool selection mode |
| stop | Option<StopCondition> | no | Stop sequences |

### ChatCompletionResponse

OpenAI-compatible chat completion response (existing in `responses.rs`).

| Field | Type | Description |
|-------|------|-------------|
| id | String | Unique response ID |
| object | String | Always "chat.completion" |
| created | i64 | Unix timestamp |
| model | String | Model used |
| choices | Vec<ChatChoice> | Generated completions |
| usage | Usage | Token usage statistics |

### Message

Chat message in a conversation.

| Field | Type | Description |
|-------|------|-------------|
| role | Role | system, user, assistant, tool |
| content | Option<String> | Text content |
| tool_calls | Option<Vec<ToolCall>> | Tool calls (assistant only) |
| tool_call_id | Option<String> | ID of tool call (tool role only) |
| name | Option<String> | Tool name (tool role only) |

### Tool

Tool definition for function calling.

| Field | Type | Description |
|-------|------|-------------|
| type | String | Always "function" |
| function | FunctionDefinition | Function details |

### FunctionDefinition

Function definition within a tool.

| Field | Type | Description |
|-------|------|-------------|
| name | String | Function name |
| description | Option<String> | Function description |
| parameters | Option<Value> | JSON Schema for parameters |

### ToolCall

Tool call generated by the model.

| Field | Type | Description |
|-------|------|-------------|
| id | String | Unique call ID |
| type | String | Always "function" |
| function | FunctionCall | Function call details |

### FunctionCall

Function call details.

| Field | Type | Description |
|-------|------|-------------|
| name | String | Function name |
| arguments | String | JSON arguments string |

---

## MCP Integration Entities

### McpServerConfig

Configuration for an MCP server connection.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| url | String | required | Base URL of MCP server |
| auth | Option<String> | None | Authorization header value |
| timeout_secs | u64 | 30 | Request timeout |

### McpClient

Client for MCP server communication (existing in `mcp_client.rs`).

| Field | Type | Description |
|-------|------|-------------|
| config | McpServerConfig | Server configuration |
| client | reqwest::Client | HTTP client |

### ResponsesSession

High-level session for multi-turn conversations.

| Field | Type | Description |
|-------|------|-------------|
| adapter | OpenAIAdapter | Underlying adapter |
| mcp_clients | HashMap<String, McpClient> | Connected MCP servers |

### ConversationOptions

Options for running a conversation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| max_turns | usize | 10 | Maximum conversation turns |
| allowed_tools | Option<Vec<String>> | None | Allowed tool names (None = all) |

### ConversationResult

Result of a completed conversation.

| Field | Type | Description |
|-------|------|-------------|
| final_message | String | Final assistant response |
| tool_calls_executed | Vec<ToolCall> | All tool calls made |
| turns_taken | usize | Number of turns completed |

---

## Server Entities

### ModelManager

Manager for model lifecycle and switching (existing in `model_manager.rs`).

| Field | Type | Description |
|-------|------|-------------|
| status | ModelLifecycleStatus | Current FSM state |
| active_model | Option<String> | Currently loaded model |
| last_error | Option<String> | Last error message |
| switch_requested_at | Option<Instant> | When switch started |
| queue | VecDeque<ModelSwitchRequest> | Pending switch requests |

### ModelLifecycleStatus

FSM states for model lifecycle (existing in `status.rs`).

| State | Description |
|-------|-------------|
| Idle | No model loaded |
| Ready | Model loaded and ready |
| Switching | Preparing to switch models |
| Loading | Model loading in progress |
| Error | Error occurred during operation |

### RequestQueue (NEW)

Per-model queue for pending requests.

| Field | Type | Description |
|-------|------|-------------|
| requests | VecDeque<QueuedRequest> | Pending requests |
| max_size | usize | Maximum queue size (default 10) |
| model | String | Target model for this queue |

### QueuedRequest (NEW)

A request waiting in queue.

| Field | Type | Description |
|-------|------|-------------|
| id | String | Request ID |
| model | String | Target model |
| stream | bool | Whether streaming is requested |
| tx | Option<Sender<StreamChunk>> | Channel for status updates |
| enqueued_at | Instant | When request was queued |
| timeout | Duration | Request timeout (default 30s) |

### QueueStatus (NEW)

Status information for streaming updates.

| Field | Type | Description |
|-------|------|-------------|
| status | String | "queued", "waiting_for_model", "loading" |
| position | usize | Position in queue |
| model | String | Target model |
| eta_ms | Option<u64> | Estimated time (optional) |

---

## Validation Rules Summary

### Engine Configuration
1. Model path must exist or be valid HuggingFace ID
2. Device must be available (CUDA index valid, Metal available)
3. Memory limits must be positive and realistic

### Generation Parameters
1. Temperature ≥ 0 (0 means greedy)
2. Top-p in (0, 1]
3. Penalties must be reasonable ranges
4. Stop sequences must be non-empty strings

### Request Handling
1. Queue size ≤ max_queue (reject with 503 if exceeded)
2. Non-streaming timeout ≤ configured limit (503 if exceeded)
3. Tool calls must reference defined tools
4. Message roles must be valid

### MCP Integration
1. Server URLs must be valid HTTP(S) URLs
2. Tool names must match server-declared tools
3. Tool arguments must validate against declared schema

---

## State Machine: Model Lifecycle

```
        ┌──────────────────────────────────────────────────────────────┐
        │                        INITIAL                                │
        └──────────────────────────┬───────────────────────────────────┘
                                   │ startup
                                   ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                         IDLE                                  │
        │  - No model loaded                                           │
        │  - Accepting switch requests                                 │
        └──────────────────────────┬───────────────────────────────────┘
                                   │ switch request received
                                   ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                       SWITCHING                               │
        │  - Previous model being unloaded                             │
        │  - Requests queued                                           │
        └──────────────────────────┬───────────────────────────────────┘
                                   │ unload complete
                                   ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                        LOADING                                │
        │  - New model loading                                         │
        │  - Streaming status updates sent                             │
        └──────────────────────┬───────────────────┬───────────────────┘
                               │ success           │ failure
                               ▼                   ▼
        ┌──────────────────────────────┐   ┌──────────────────────────┐
        │            READY             │   │          ERROR           │
        │  - Model loaded              │   │  - Error recorded        │
        │  - Serving requests          │   │  - Queued requests       │
        │  - Queue draining            │   │    failed                │
        └──────────────────────────────┘   └──────────────────────────┘
                     │                               │
                     │ switch request               │ retry
                     └──────────────┬───────────────┘
                                    ▼
                               SWITCHING
```

