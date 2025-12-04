# Implementation Plan: Library-First Architecture

**Branch**: `001-library-api-implementation` | **Date**: December 3, 2025 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/001-library-api-implementation/spec.md`

## Summary

Transform candle-vllm from a binary-first project to a library-first architecture that can be embedded in Tauri apps, AI gateways, and agent frameworks. The implementation includes:

1. **Core Library API** - Clean `InferenceEngine` wrapper with builder pattern
2. **OpenAI Compatibility** - `OpenAIAdapter` for chat completions with tool calling
3. **MCP Integration** - HTTP-based MCP client and multi-turn conversation orchestration
4. **Request Queuing** - Per-model queues with streaming status updates
5. **Backward Compatibility** - Existing binary and API unchanged

## Technical Context

**Language/Version**: Rust 1.83+  
**Primary Dependencies**: 
- candle-core/candle-nn/attention-rs (ML inference)
- tokio (async runtime)
- serde/serde_json (serialization)
- axum/tower-http (HTTP server)
- rmcp/reqwest (MCP client)
- minijinja (chat templates)

**Storage**: Local filesystem (models, HF cache, configs)  
**Testing**: cargo test (CPU-safe tests), feature-gated GPU tests  
**Target Platform**: Linux (CUDA), macOS (Metal), CPU fallback  
**Project Type**: Rust workspace with 4 crates  
**Performance Goals**: Maintain existing throughput, no regression  
**Constraints**: Single active model, GPU memory limits, no concurrent model loads  
**Scale/Scope**: Support 16+ concurrent sequences, multiple MCP servers

## Constitution Check

*GATE: All items pass - proceeding with implementation.*

| Principle | Status | Notes |
|-----------|--------|-------|
| Library-First | ✅ PASS | Primary goal - creating embeddable library |
| Clean API | ✅ PASS | Builder pattern, wrapper types |
| Test-First | ✅ PASS | Unit tests for all new code |
| Backward Compatibility | ✅ PASS | Existing CLI/API unchanged |
| Observability | ✅ PASS | Structured logging via tracing |
| Simplicity | ✅ PASS | Incremental refactoring, façade pattern |

## Project Structure

### Documentation (this feature)

```text
specs/001-library-api-implementation/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Technical research and decisions
├── data-model.md        # Entity definitions and relationships
├── quickstart.md        # Developer quickstart guide
├── contracts/
│   ├── openai-api.yaml  # OpenAPI specification
│   └── library-api.rs   # Rust trait definitions
├── checklists/
│   └── requirements.md  # Validation checklist
└── tasks.md             # Implementation tasks (Phase 2)
```

### Source Code (repository root)

```text
crates/
├── candle-vllm-core/           # Core inference engine
│   ├── src/
│   │   ├── lib.rs              # Public API exports
│   │   ├── api.rs              # NEW: InferenceEngine wrapper
│   │   ├── backend/            # Existing backend code
│   │   ├── openai/             # Existing OpenAI types
│   │   ├── scheduler/          # Existing scheduler
│   │   └── paged_attention/    # Existing attention impl
│   └── tests/
│       ├── engine_test.rs      # NEW: Engine API tests
│       └── generation_test.rs  # NEW: Generation tests
│
├── candle-vllm-openai/         # OpenAI compatibility
│   ├── src/
│   │   ├── lib.rs              # Re-exports + adapter
│   │   ├── adapter.rs          # NEW: OpenAIAdapter
│   │   ├── model_registry.rs   # Existing registry
│   │   └── streaming.rs        # NEW: Stream helpers
│   └── tests/
│       └── adapter_test.rs     # NEW: Adapter tests
│
├── candle-vllm-responses/      # MCP + Responses API
│   ├── src/
│   │   ├── lib.rs              # Public exports
│   │   ├── mcp_client.rs       # Existing MCP client
│   │   ├── session.rs          # Existing + run_conversation
│   │   ├── orchestrator.rs     # NEW: Conversation loop
│   │   └── status.rs           # Existing status types
│   └── tests/
│       ├── mcp_test.rs         # NEW: MCP client tests
│       └── session_test.rs     # NEW: Session tests
│
├── candle-vllm-server/         # HTTP server
│   ├── src/
│   │   ├── lib.rs              # Server entry + run()
│   │   ├── main.rs             # Binary entry
│   │   ├── routes.rs           # Existing routes
│   │   ├── config/             # Existing config
│   │   ├── state/
│   │   │   ├── mod.rs
│   │   │   ├── model_manager.rs # Existing FSM
│   │   │   └── request_queue.rs # NEW: Per-model queues
│   │   └── models_config.rs    # Existing
│   └── tests/
│       ├── queue_test.rs       # NEW: Queue tests
│       └── integration_test.rs # NEW: E2E tests
│
src/
├── lib.rs                      # Root re-exports all crates
└── main.rs                     # Delegates to candle-vllm-server

tests/
└── backward_compat_test.rs     # NEW: CLI/API compatibility
```

**Structure Decision**: Workspace with 4 crates as already established. New files marked with `# NEW`. Existing files receive fixes and additions.

## Implementation Phases

### Phase 1: Fix Compilation Issues (Priority: P0)

Must be done first to enable all other work.

1. **Fix Metal feature compilation** in `candle-vllm-core/src/backend/cache.rs`
   - Add `use metal::NSUInteger` under `#[cfg(feature = "metal")]`
   - Lines 394, 441

2. **Resolve `unimplemented!()` in paged_attention**
   - `candle-vllm-core/src/paged_attention/attn_bias.rs` lines 67, 228, 231, 234, 240
   - Either implement or return appropriate errors

3. **Remove `todo!()` in llm_engine.rs**
   - Line 474 - implement or error

4. **Address TODO comments**
   - Line 168 in `scheduler/sequence.rs`
   - Line 390 in `backend/gguf.rs`

### Phase 2: Core Library API (Priority: P1)

Implement the `InferenceEngine` wrapper as specified in LIBRARY_API.md.

1. **Create `api.rs`** with:
   - `InferenceEngine` struct wrapping LLMEngine
   - `InferenceEngineBuilder` with builder pattern
   - `EngineConfig`, `GenerationParams`, `GenerationOutput` types
   - `tokenize()`, `detokenize()`, `generate()`, `generate_stream()` methods

2. **Update `lib.rs`** to export:
   - Public API types
   - Re-export necessary internal types

3. **Add tests** for:
   - Builder configuration
   - Tokenization round-trip
   - Generation parameters validation

### Phase 3: OpenAI Adapter (Priority: P1)

Implement the `OpenAIAdapter` wrapper.

1. **Create `adapter.rs`** in candle-vllm-openai:
   - `OpenAIAdapter` struct wrapping InferenceEngine
   - `chat_completion()` and `chat_completion_stream()` methods
   - Conversation templating using existing `ConversationManager`
   - Tool call parsing using existing `ToolParser`

2. **Create `streaming.rs`** for stream utilities:
   - `ChatCompletionStream` type
   - Delta accumulation helpers
   - Tool call delta handling

3. **Add tests** for:
   - Request to response conversion
   - Tool call parsing (Mistral, Llama, Qwen patterns)
   - Streaming chunk format

### Phase 4: Responses Session & MCP (Priority: P2)

Complete the multi-turn conversation orchestration.

1. **Implement `run_conversation()`** in session.rs:
   - Conversation loop with max_turns
   - Automatic tool execution
   - Result aggregation

2. **Create `orchestrator.rs`**:
   - Tool call routing to correct MCP server
   - Result injection back into conversation
   - Allowed tools filtering

3. **Add tests** for:
   - Multi-turn conversation completion
   - Tool filtering
   - Error handling

### Phase 5: Request Queuing (Priority: P3)

Implement per-model request queues with streaming status.

1. **Create `request_queue.rs`** in candle-vllm-server:
   - `RequestQueue` struct per model
   - `QueuedRequest` with streaming channel
   - Queue size limits and timeout handling

2. **Update `model_manager.rs`**:
   - Integrate queues with FSM
   - Drain queue on model switch complete
   - Coalesce requests during switch

3. **Update streaming** to emit status chunks:
   - Queue position updates
   - Model loading status
   - `candle_metadata` extension field

4. **Add `POST /v1/models/select`** endpoint

5. **Add tests** for:
   - Queue overflow → 503
   - Timeout handling
   - Status chunk format

### Phase 6: Backward Compatibility & Documentation (Priority: P1)

1. **Verify CLI compatibility**:
   - All existing Args work
   - Same default behavior

2. **Verify API compatibility**:
   - All endpoints return same format
   - Streaming unchanged (except new metadata)

3. **Update documentation**:
   - README.md with library usage
   - Doc comments on all public items

4. **Add integration tests**:
   - End-to-end request flow
   - Model switching flow

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking changes | Façade pattern isolates internals |
| Performance regression | Profile before/after, use existing optimized paths |
| Feature flag complexity | Clear cfg attributes, CI tests per feature |
| MCP server unavailability | Timeout and error handling, graceful degradation |

## Complexity Tracking

No constitution violations requiring justification. Structure follows established workspace pattern.

## Success Criteria Mapping

| Success Criteria | Implementation Phase |
|------------------|---------------------|
| SC-001: Embeddable engine | Phase 2 |
| SC-002: OpenAI schema validation | Phase 3 |
| SC-003: Tool parsing (3 families) | Phase 3 (existing code) |
| SC-004: MCP integration | Phase 4 |
| SC-005: Multi-turn agent | Phase 4 |
| SC-006: Model switch queue | Phase 5 |
| SC-007: Backward compatibility | Phase 6 |
| SC-008: Zero clippy warnings | Phase 1 |
| SC-009: cargo fmt clean | All phases |
| SC-010: All tests pass | All phases |
| SC-011: No stubs/TODOs | Phase 1 |
| SC-012: Complete documentation | Phase 6 |

## Next Steps

Run `/speckit.tasks` to break this plan into detailed implementation tasks with file-level granularity.
