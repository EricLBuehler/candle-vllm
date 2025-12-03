# Research: Library-First Architecture Implementation

**Branch**: `001-library-api-implementation`  
**Date**: December 3, 2025  
**Spec**: [spec.md](./spec.md)

## Executive Summary

This research resolves all technical unknowns identified during planning for the library-first restructuring of candle-vllm. The codebase already has a partial workspace structure in place, but significant work remains to fully implement the library API, fix compilation issues, remove stubs/TODOs, and implement missing MCP orchestration and request queuing features.

---

## Research Area 1: Current Codebase State Analysis

### Decision
The existing workspace structure provides a solid foundation with four crates already defined:
- `candle-vllm-core`: Contains model implementations, scheduler, backend, and OpenAI types
- `candle-vllm-openai`: Re-exports from core, adds model registry
- `candle-vllm-responses`: MCP client and session management
- `candle-vllm-server`: HTTP server with model manager state machine

### Findings

**Compilation Issues Identified:**
1. `candle-vllm-core/src/backend/cache.rs`: Missing `metal` crate import under Metal feature flag (lines 394, 441)
2. Several `unimplemented!()` macros in `paged_attention/attn_bias.rs` (lines 67, 228, 231, 234, 240)
3. One `todo!()` in `pipelines/llm_engine.rs` (line 474)
4. Two TODO comments (cosmetic but should be addressed)

**Architecture Gaps:**
1. `InferenceEngine` wrapper API not yet implemented (LIBRARY_API.md specifies builder pattern API)
2. `OpenAIAdapter` not yet implemented (currently uses direct `OpenAIServerData`)
3. Request queuing with streaming status chunks not implemented
4. Multi-turn conversation orchestrator (`run_conversation`) not implemented

### Alternatives Considered
- **Option A**: Complete rewrite from scratch - Rejected (too risky, loses existing functionality)
- **Option B**: Incremental refactoring with façade pattern - Selected (maintains compatibility)

---

## Research Area 2: Library API Design Patterns

### Decision
Implement the LIBRARY_API.md specification using the Builder pattern for configuration and wrapper types to provide a clean abstraction over existing internals.

### Rationale
1. **Builder Pattern**: Rust ecosystem standard for complex configuration (e.g., `reqwest::Client`, `tokio::runtime::Builder`)
2. **Wrapper Types**: Allow internal refactoring without breaking public API
3. **Async-First**: All generation methods should be async for streaming support

### Key Design Decisions

**InferenceEngine**:
- Wraps existing `LLMEngine` + `Pipeline` + `CacheEngine` combination
- Provides simplified tokenize/generate/detokenize interface
- Manages lifecycle internally

**OpenAIAdapter**:
- Wraps `InferenceEngine` and handles conversation templating
- Uses existing `tool_parser.rs` for tool call extraction
- Converts between `ChatCompletionRequest`/`Response` and internal types

**ResponsesSession**:
- Already partially implemented
- Needs `run_conversation` method for automatic multi-turn orchestration

---

## Research Area 3: MCP Integration Protocol

### Decision
Use HTTP-based MCP protocol as currently implemented in `mcp_client.rs`. The rmcp crate is already a dependency for future expansion.

### Rationale
1. HTTP transport is simpler and widely supported
2. rmcp crate provides future path to other transports (stdio, SSE)
3. Current implementation already connects and lists tools

### Protocol Details
- **Tool Listing**: `GET /tools` returns array of tool definitions
- **Tool Execution**: `POST /tools/{name}` with JSON payload
- **Tool Format Conversion**: Already implemented in `requests.rs` (`Tool::from_mcp_list`)

---

## Research Area 4: Request Queuing & Model Switch FSM

### Decision
Implement per-model request queues with streaming status updates as specified in docs/REMAINDER.md.

### Implementation Approach

**Queue Structure**:
```rust
struct RequestQueue {
    requests: VecDeque<QueuedRequest>,
    max_size: usize,  // default 10
}

struct QueuedRequest {
    id: String,
    model: String,
    stream: bool,
    tx: Option<Sender<StreamChunk>>,  // for streaming status updates
    enqueued_at: Instant,
}
```

**FSM States** (already implemented in `ModelManager`):
- Idle → Ready → Switching → Loading → Ready/Error

**Status Chunk Format** (OpenAI-compatible extension):
```json
{
  "choices": [{"delta": {"content": ""}}],
  "candle_metadata": {
    "status": "queued",
    "position": 3,
    "model": "mistral-7b"
  }
}
```

---

## Research Area 5: Tool Call Parsing Patterns

### Decision
Use existing `tool_parser.rs` patterns which support multiple model families.

### Supported Patterns (from existing code analysis)

1. **Mistral/Mixtral**: `[TOOL_CALLS]` JSON block
2. **Llama 3.x**: `<|python_tag|>` function calls
3. **Qwen**: `<tool_call>` XML-style blocks
4. **Generic**: JSON function_call format

### Coverage
Current implementation covers the 3 model families required by SC-003.

---

## Research Area 6: Streaming Implementation

### Decision
Extend existing `streaming.rs` to support:
1. Queue status chunks (new)
2. Tool call delta streaming (existing)
3. Content delta streaming (existing)

### Implementation Details
- Use `futures::Stream` trait for async streaming
- `ChatCompletionStream` wraps underlying generation stream
- Inject status updates before generation begins for queued requests

---

## Research Area 7: Backward Compatibility

### Decision
Maintain 100% backward compatibility through the `candle-vllm-server` crate.

### Strategy
1. Keep all CLI arguments in `Args` struct unchanged
2. Keep all HTTP endpoints at existing paths
3. Existing `main.rs` delegates to new library APIs internally
4. Ensure `models.yaml` and `mcp.json` formats unchanged

---

## Research Area 8: Testing Strategy

### Decision
Focus on CPU-safe tests with feature-gated GPU tests.

### Test Categories
1. **Unit Tests**: Core types, parsers, conversions (no GPU)
2. **Integration Tests**: End-to-end flows with mock models (CPU)
3. **GPU Tests**: Feature-gated behind `#[cfg(feature = "cuda")]` or `#[cfg(feature = "metal")]`

### Test Locations
- `crates/candle-vllm-core/tests/`
- `crates/candle-vllm-openai/tests/`
- `crates/candle-vllm-responses/tests/`
- `crates/candle-vllm-server/tests/`

---

## Resolved Unknowns Summary

| Unknown | Resolution |
|---------|------------|
| Library API pattern | Builder pattern + wrapper types |
| MCP transport | HTTP (existing), rmcp for future |
| Queue implementation | VecDeque per model, streaming status |
| Tool parsing coverage | 3 families already implemented |
| Streaming approach | futures::Stream with status injection |
| Backward compatibility | Server crate maintains CLI/API |
| Testing approach | CPU-safe unit/integration, GPU feature-gated |

---

## Dependencies to Add/Update

No new external dependencies required. Existing `workspace.dependencies` sufficient:
- `candle-core`, `candle-nn`, `attention-rs`: Model inference
- `tokio`, `futures`: Async runtime and streaming
- `serde`, `serde_json`: Serialization
- `axum`, `tower-http`: HTTP server
- `rmcp`: MCP protocol (already present)
- `reqwest`: HTTP client for MCP

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Metal feature compilation | Fix missing import in cache.rs |
| Unimplemented blocks | Implement or guard with proper error returns |
| Breaking API changes | Wrapper types isolate internals |
| Performance regression | Use existing optimized codepaths |

