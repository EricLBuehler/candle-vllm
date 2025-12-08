# Tasks: Library-First Architecture Implementation

**Input**: Design documents from `/specs/001-library-api-implementation/`  
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Unit and integration tests included per user story requirements.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, etc.)
- Include exact file paths in descriptions

## Path Conventions

- **Workspace root**: `/Users/gqadonis/Projects/references/candle-vllm/`
- **Crates**: `crates/candle-vllm-{core,openai,responses,server}/`
- **Tests**: `crates/<crate>/tests/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Verify project structure and dependencies

- [X] T001 Verify workspace structure in Cargo.toml at repository root
- [X] T002 [P] Verify candle-vllm-core Cargo.toml dependencies in crates/candle-vllm-core/Cargo.toml
- [X] T003 [P] Verify candle-vllm-openai Cargo.toml dependencies in crates/candle-vllm-openai/Cargo.toml
- [X] T004 [P] Verify candle-vllm-responses Cargo.toml dependencies in crates/candle-vllm-responses/Cargo.toml
- [X] T005 [P] Verify candle-vllm-server Cargo.toml dependencies in crates/candle-vllm-server/Cargo.toml
- [X] T006 Create tests directories structure: crates/candle-vllm-core/tests/, crates/candle-vllm-openai/tests/, crates/candle-vllm-responses/tests/, crates/candle-vllm-server/tests/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Fix compilation issues that BLOCK all user story implementation

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Fix Metal Feature Compilation

- [X] T007 Add `use metal::NSUInteger` import under `#[cfg(feature = "metal")]` in crates/candle-vllm-core/src/backend/cache.rs line 394
- [X] T008 Fix metal type reference at line 441 in crates/candle-vllm-core/src/backend/cache.rs

### Resolve Unimplemented Macros

- [X] T009 Implement or replace `unimplemented!()` at line 67 in crates/candle-vllm-core/src/paged_attention/attn_bias.rs
- [X] T010 [P] Implement or replace `unimplemented!()` at line 228 in crates/candle-vllm-core/src/paged_attention/attn_bias.rs
- [X] T011 [P] Implement or replace `unimplemented!()` at line 231 in crates/candle-vllm-core/src/paged_attention/attn_bias.rs
- [X] T012 [P] Implement or replace `unimplemented!()` at line 234 in crates/candle-vllm-core/src/paged_attention/attn_bias.rs
- [X] T013 [P] Implement or replace `unimplemented!()` at line 240 in crates/candle-vllm-core/src/paged_attention/attn_bias.rs

### Remove TODO and todo!() Macros

- [X] T014 Implement or replace `todo!()` at line 474 in crates/candle-vllm-core/src/openai/pipelines/llm_engine.rs
- [X] T015 [P] Address TODO comment at line 168 in crates/candle-vllm-core/src/scheduler/sequence.rs
- [X] T016 [P] Address TODO comment at line 390 in crates/candle-vllm-core/src/backend/gguf.rs

### Verify Compilation

- [X] T017 Run `cargo check --all-targets` to verify basic compilation passes
- [ ] T018 Run `cargo check --features metal` to verify Metal feature compiles (macOS only)
- [ ] T019 Run `cargo fmt --all --check` to verify formatting
- [ ] T020 Run `cargo clippy --all-targets -D warnings` to verify no clippy warnings

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Embed Inference Engine in Custom Application (Priority: P1) 🎯 MVP

**Goal**: Developers can embed the inference engine in standalone applications without HTTP server

**Independent Test**: Create a minimal application that imports the core library, loads a model config, tokenizes a prompt, and calls generation methods.

### Core Types for US1

- [X] T021 [P] [US1] Create EngineConfig struct with builder pattern in crates/candle-vllm-core/src/api/config.rs
- [X] T022 [P] [US1] Create GenerationParams struct with Default impl in crates/candle-vllm-core/src/api/params.rs
- [X] T023 [P] [US1] Create GenerationOutput struct in crates/candle-vllm-core/src/api/output.rs
- [X] T024 [P] [US1] Create FinishReason enum in crates/candle-vllm-core/src/api/output.rs
- [X] T025 [P] [US1] Create GenerationStats struct in crates/candle-vllm-core/src/api/output.rs
- [X] T026 [P] [US1] Create Error enum and Result type in crates/candle-vllm-core/src/api/error.rs

### InferenceEngine Implementation

- [X] T027 [US1] Create api/ module directory and mod.rs in crates/candle-vllm-core/src/api/mod.rs
- [X] T028 [US1] Implement InferenceEngineBuilder struct in crates/candle-vllm-core/src/api/engine.rs
- [X] T029 [US1] Implement InferenceEngine struct wrapping LLMEngine in crates/candle-vllm-core/src/api/engine.rs
- [X] T030 [US1] Implement InferenceEngine::builder() method in crates/candle-vllm-core/src/api/engine.rs
- [X] T031 [US1] Implement InferenceEngine::new(config) async constructor in crates/candle-vllm-core/src/api/engine.rs
- [X] T032 [US1] Implement InferenceEngine::tokenize(&self, text) method in crates/candle-vllm-core/src/api/engine.rs
- [X] T033 [US1] Implement InferenceEngine::detokenize(&self, tokens) method in crates/candle-vllm-core/src/api/engine.rs
- [X] T034 [US1] Implement InferenceEngine::generate(&mut self, prompt, params) async method in crates/candle-vllm-core/src/api/engine.rs
- [X] T035 [US1] Implement InferenceEngine::generate_stream(&mut self, prompt, params) async method in crates/candle-vllm-core/src/api/engine.rs
- [X] T036 [US1] Implement InferenceEngine::model_info(&self) method in crates/candle-vllm-core/src/api/engine.rs
- [X] T037 [US1] Implement InferenceEngine::stats(&self) method in crates/candle-vllm-core/src/api/engine.rs

### Public API Exports

- [X] T038 [US1] Update crates/candle-vllm-core/src/lib.rs to export api module and public types
- [ ] T039 [US1] Add doc comments for all public types in crates/candle-vllm-core/src/api/

### Tests for US1

- [ ] T040 [P] [US1] Create unit test for EngineConfig builder validation in crates/candle-vllm-core/tests/config_test.rs
- [ ] T041 [P] [US1] Create unit test for GenerationParams defaults in crates/candle-vllm-core/tests/params_test.rs
- [ ] T042 [US1] Create integration test for tokenize/detokenize round-trip in crates/candle-vllm-core/tests/engine_test.rs

**Checkpoint**: User Story 1 complete - InferenceEngine can be embedded in applications

---

## Phase 4: User Story 2 - OpenAI-Compatible Chat API (Priority: P1)

**Goal**: Developers can use OpenAI chat completion format with local models

**Independent Test**: Create OpenAIAdapter, send ChatCompletionRequest, verify response format matches OpenAI spec.

### Includes User Story 3 (Tool Calling) since both are in candle-vllm-openai

### OpenAIAdapter Implementation

- [X] T043 [US2] Create OpenAIAdapter struct in crates/candle-vllm-openai/src/adapter.rs
- [X] T044 [US2] Implement OpenAIAdapter::new(engine) constructor in crates/candle-vllm-openai/src/adapter.rs
- [X] T045 [US2] Implement request-to-params conversion in crates/candle-vllm-openai/src/adapter.rs
- [X] T046 [US2] Implement chat_completion(&mut self, request) async method in crates/candle-vllm-openai/src/adapter.rs
- [X] T047 [US2] Implement response building with usage statistics in crates/candle-vllm-openai/src/adapter.rs
- [X] T048 [US2] Integrate ConversationManager for chat template formatting in crates/candle-vllm-openai/src/adapter.rs

### Streaming Support

- [X] T049 [P] [US2] Create ChatCompletionStream type in crates/candle-vllm-openai/src/streaming.rs
- [X] T050 [US2] Implement chat_completion_stream(&mut self, request) async method in crates/candle-vllm-openai/src/adapter.rs
- [ ] T051 [US2] Implement delta content streaming in crates/candle-vllm-openai/src/streaming.rs

### Tool Calling (US3)

- [X] T052 [P] [US3] Verify Tool::from_mcp_list conversion works in crates/candle-vllm-core/src/openai/requests.rs
- [X] T053 [US3] Integrate ToolParser for tool call extraction in crates/candle-vllm-openai/src/adapter.rs
- [X] T054 [US3] Implement tool call response formatting in crates/candle-vllm-openai/src/adapter.rs
- [X] T055 [US3] Implement tool_choice handling (none, auto, required, specific) in crates/candle-vllm-openai/src/adapter.rs
- [ ] T056 [US3] Implement tool call delta streaming in crates/candle-vllm-openai/src/streaming.rs

### Public API Exports

- [X] T057 [US2] Update crates/candle-vllm-openai/src/lib.rs to export OpenAIAdapter and streaming types
- [X] T058 [US2] Add doc comments for OpenAIAdapter public methods in crates/candle-vllm-openai/src/adapter.rs

### Tests for US2/US3

- [ ] T059 [P] [US2] Create unit test for request-to-params conversion in crates/candle-vllm-openai/tests/adapter_test.rs
- [ ] T060 [P] [US2] Create unit test for response format validation in crates/candle-vllm-openai/tests/adapter_test.rs
- [ ] T061 [P] [US3] Create unit test for Mistral tool call parsing pattern in crates/candle-vllm-openai/tests/tool_parsing_test.rs
- [ ] T062 [P] [US3] Create unit test for Llama tool call parsing pattern in crates/candle-vllm-openai/tests/tool_parsing_test.rs
- [ ] T063 [P] [US3] Create unit test for Qwen tool call parsing pattern in crates/candle-vllm-openai/tests/tool_parsing_test.rs
- [ ] T064 [US2] Create streaming chunk format test in crates/candle-vllm-openai/tests/streaming_test.rs

**Checkpoint**: User Stories 2 & 3 complete - OpenAI-compatible API with tool calling works

---

## Phase 5: User Story 7 - Maintain Backward Compatibility (Priority: P1)

**Goal**: Existing binary users experience no breaking changes

**Independent Test**: Run rebuilt binary with existing CLI arguments and verify identical behavior.

### CLI Verification

- [X] T065 [US7] Verify all Args fields unchanged in crates/candle-vllm-server/src/lib.rs
- [X] T066 [US7] Verify default values match existing behavior in crates/candle-vllm-server/src/lib.rs
- [X] T067 [US7] Update run() function to use new library APIs internally in crates/candle-vllm-server/src/lib.rs

### API Endpoint Verification

- [X] T068 [P] [US7] Verify /v1/chat/completions endpoint unchanged in crates/candle-vllm-server/src/routes.rs
- [X] T069 [P] [US7] Verify /v1/models endpoint unchanged in crates/candle-vllm-server/src/routes.rs
- [X] T070 [US7] Verify streaming response format unchanged in crates/candle-vllm-server/src/routes.rs

### Root Package Updates

- [X] T071 [US7] Update src/lib.rs to re-export all crate public APIs
- [X] T072 [US7] Verify src/main.rs delegates to candle-vllm-server correctly

### Tests for US7

- [ ] T073 [US7] Create CLI argument parsing test in tests/backward_compat_test.rs
- [ ] T074 [US7] Create API response format test in tests/backward_compat_test.rs

**Checkpoint**: User Story 7 complete - Existing users can upgrade without changes

---

## Phase 6: User Story 4 - Integrate MCP Servers (Priority: P2)

**Goal**: Connect MCP servers and use their tools in chat requests

**Independent Test**: Connect to MCP server, list tools, convert to OpenAI format, execute tool call.

### MCP Client Enhancements

- [X] T075 [P] [US4] Verify McpClient::connect works in crates/candle-vllm-responses/src/mcp_client.rs
- [X] T076 [P] [US4] Verify McpClient::list_tools returns proper format in crates/candle-vllm-responses/src/mcp_client.rs
- [X] T077 [US4] Verify McpClient::call_tool routes correctly in crates/candle-vllm-responses/src/mcp_client.rs
- [X] T078 [US4] Add error handling for MCP server timeouts in crates/candle-vllm-responses/src/mcp_client.rs
- [X] T079 [US4] Add error handling for MCP connection failures in crates/candle-vllm-responses/src/mcp_client.rs

### Tool Format Conversion

- [X] T080 [US4] Verify Tool::from_mcp_list handles all MCP tool fields in crates/candle-vllm-core/src/openai/requests.rs
- [X] T081 [US4] Implement ToolCall::to_mcp_call conversion in crates/candle-vllm-core/src/openai/requests.rs
- [X] T082 [US4] Implement Message::from_mcp_result conversion in crates/candle-vllm-core/src/openai/requests.rs

### Session MCP Integration

- [X] T083 [US4] Verify ResponsesSession::add_mcp_server works in crates/candle-vllm-responses/src/session.rs
- [X] T084 [US4] Verify ResponsesSession::list_openai_tools works in crates/candle-vllm-responses/src/session.rs
- [X] T085 [US4] Verify ResponsesSession::call_tool routes to correct server in crates/candle-vllm-responses/src/session.rs

### Tests for US4

- [ ] T086 [P] [US4] Create unit test for MCP tool format conversion in crates/candle-vllm-responses/tests/mcp_test.rs
- [ ] T087 [P] [US4] Create unit test for tool call routing by server name in crates/candle-vllm-responses/tests/mcp_test.rs
- [ ] T088 [US4] Create error handling test for MCP server unavailable in crates/candle-vllm-responses/tests/mcp_test.rs

**Checkpoint**: User Story 4 complete - MCP servers can provide tools to models

---

## Phase 7: User Story 5 - Run Multi-Turn Agent Conversations (Priority: P2)

**Goal**: Automated multi-turn conversations with automatic tool execution

**Independent Test**: Create session, provide task, verify automatic tool calls and completion.

### Orchestrator Implementation

- [X] T089 [US5] Create Orchestrator struct in crates/candle-vllm-responses/src/orchestrator.rs
- [X] T090 [US5] Implement tool call routing logic in crates/candle-vllm-responses/src/orchestrator.rs
- [X] T091 [US5] Implement result injection back to conversation in crates/candle-vllm-responses/src/orchestrator.rs
- [X] T092 [US5] Implement allowed_tools filtering in crates/candle-vllm-responses/src/orchestrator.rs

### ResponsesSession run_conversation

- [X] T093 [US5] Create ConversationResult struct in crates/candle-vllm-responses/src/session.rs
- [X] T094 [US5] Implement run_conversation(&mut self, messages, options) async method in crates/candle-vllm-responses/src/session.rs
- [X] T095 [US5] Implement conversation loop with max_turns limit in crates/candle-vllm-responses/src/session.rs
- [X] T096 [US5] Implement tool call execution and result aggregation in crates/candle-vllm-responses/src/session.rs

### ResponsesSessionBuilder

- [X] T097 [US5] Create ResponsesSessionBuilder struct in crates/candle-vllm-responses/src/session.rs
- [X] T098 [US5] Implement builder pattern with model_path, device, mcp_servers in crates/candle-vllm-responses/src/session.rs

### Public API Exports

- [X] T099 [US5] Update crates/candle-vllm-responses/src/lib.rs to export session builder and result types
- [X] T100 [US5] Add doc comments for ResponsesSession public methods in crates/candle-vllm-responses/src/session.rs

### Tests for US5

- [ ] T101 [P] [US5] Create unit test for max_turns limit in crates/candle-vllm-responses/tests/session_test.rs
- [ ] T102 [P] [US5] Create unit test for allowed_tools filtering in crates/candle-vllm-responses/tests/session_test.rs
- [ ] T103 [US5] Create conversation result aggregation test in crates/candle-vllm-responses/tests/session_test.rs

**Checkpoint**: User Story 5 complete - Multi-turn agent conversations work automatically

---

## Phase 8: User Story 6 - Queue Requests During Model Switching (Priority: P3)

**Goal**: Intelligent request queuing with streaming status during model switches

**Independent Test**: Send request for inactive model, verify queue status, receive response after load.

### RequestQueue Implementation

- [X] T104 [P] [US6] Create RequestQueue struct in crates/candle-vllm-server/src/state/request_queue.rs
- [X] T105 [P] [US6] Create QueuedRequest struct in crates/candle-vllm-server/src/state/request_queue.rs
- [X] T106 [US6] Implement queue size limits and 503 rejection in crates/candle-vllm-server/src/state/request_queue.rs
- [X] T107 [US6] Implement timeout handling for non-streaming requests in crates/candle-vllm-server/src/state/request_queue.rs

### ModelManager Queue Integration

- [X] T108 [US6] Add per-model queues to ModelManager in crates/candle-vllm-server/src/state/model_manager.rs
- [X] T109 [US6] Implement queue draining on model switch complete in crates/candle-vllm-server/src/state/model_manager.rs
- [X] T110 [US6] Implement request coalescing during switch in crates/candle-vllm-server/src/state/model_manager.rs

### Streaming Status Chunks

- [X] T111 [P] [US6] Create QueueStatus struct in crates/candle-vllm-responses/src/status.rs
- [ ] T112 [US6] Implement streaming status chunk emission in crates/candle-vllm-server/src/routes.rs
- [ ] T113 [US6] Add candle_metadata extension field to streaming chunks in crates/candle-vllm-server/src/routes.rs

### New Endpoints

- [X] T114 [US6] Implement POST /v1/models/select endpoint in crates/candle-vllm-server/src/routes.rs
- [X] T115 [US6] Update GET /v1/models/status to include queue lengths in crates/candle-vllm-server/src/routes.rs

### Configuration

- [X] T116 [US6] Add queue_size config option (default 10) in crates/candle-vllm-server/src/config/mod.rs
- [X] T117 [US6] Add request_timeout config option (default 30s) in crates/candle-vllm-server/src/config/mod.rs

### Tests for US6

- [ ] T118 [P] [US6] Create unit test for queue overflow → 503 in crates/candle-vllm-server/tests/queue_test.rs
- [ ] T119 [P] [US6] Create unit test for timeout → 503 in crates/candle-vllm-server/tests/queue_test.rs
- [ ] T120 [US6] Create streaming status chunk format test in crates/candle-vllm-server/tests/queue_test.rs
- [ ] T121 [US6] Create model switch with queue drain test in crates/candle-vllm-server/tests/queue_test.rs

**Checkpoint**: User Story 6 complete - Request queuing with status updates works

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, final verification, cleanup

### Documentation

- [X] T122 [P] Update README.md with library usage section at repository root
- [ ] T123 [P] Update LIBRARY_API.md with final API signatures at repository root
- [X] T124 [P] Create crates/candle-vllm-core/README.md with crate documentation
- [X] T125 [P] Create crates/candle-vllm-openai/README.md with crate documentation
- [X] T126 [P] Create crates/candle-vllm-responses/README.md with crate documentation
- [X] T127 [P] Create crates/candle-vllm-server/README.md with crate documentation

### Final Verification

- [X] T128 Run `cargo fmt --all` to format all code
- [ ] T129 Run `cargo clippy --all-targets --all-features -D warnings` and fix any warnings
- [ ] T130 Run `cargo test --all --all-features` and ensure all tests pass
- [X] T131 Verify no `todo!()`, `unimplemented!()`, or `TODO` comments remain in crates/
- [X] T132 Run `cargo doc --all --no-deps` and verify documentation builds
- [ ] T133 Validate quickstart.md examples compile and run

### Examples

- [X] T134 [P] Create examples/tauri_app/README.md with Tauri integration example
- [X] T135 [P] Create examples/ai_gateway/README.md with custom gateway example
- [X] T136 [P] Create examples/agent_framework/README.md with agent example

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - **BLOCKS all user stories**
- **User Story 1 (Phase 3)**: Depends on Foundational - Core engine API
- **User Story 2+3 (Phase 4)**: Depends on US1 - OpenAI adapter wraps engine
- **User Story 7 (Phase 5)**: Depends on US2 - Backward compat uses adapter
- **User Story 4 (Phase 6)**: Depends on US2 - MCP uses OpenAI tools format
- **User Story 5 (Phase 7)**: Depends on US4 - Multi-turn uses MCP
- **User Story 6 (Phase 8)**: Depends on US7 - Queuing is server feature
- **Polish (Phase 9)**: Depends on all user stories complete

### User Story Dependencies

```
           ┌─────────────────────────────────────────────────┐
           │              Foundational (Phase 2)             │
           │       Fix compilation, remove stubs/TODOs       │
           └─────────────────────────┬───────────────────────┘
                                     │
                                     ▼
           ┌─────────────────────────────────────────────────┐
           │         US1: Core Inference Engine (P1)         │
           │            InferenceEngine wrapper              │
           └─────────────────────────┬───────────────────────┘
                                     │
                                     ▼
           ┌─────────────────────────────────────────────────┐
           │    US2+US3: OpenAI Adapter + Tool Calling (P1)  │
           │         OpenAIAdapter, ChatCompletion           │
           └───────────────┬─────────────────┬───────────────┘
                           │                 │
              ┌────────────┘                 └────────────┐
              │                                           │
              ▼                                           ▼
┌─────────────────────────────┐             ┌─────────────────────────────┐
│  US7: Backward Compat (P1)  │             │   US4: MCP Integration (P2) │
│    Server uses adapter      │             │     McpClient + tools       │
└─────────────┬───────────────┘             └─────────────┬───────────────┘
              │                                           │
              │                                           ▼
              │                             ┌─────────────────────────────┐
              │                             │ US5: Multi-Turn Agent (P2)  │
              │                             │   run_conversation loop     │
              │                             └─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  US6: Request Queuing (P3)  │
│   Per-model queues, FSM     │
└─────────────────────────────┘
```

### Parallel Opportunities

**Within Phase 2 (Foundational):**
- T010-T013 can run in parallel (different lines in same file)
- T015-T016 can run in parallel (different files)
- T017-T020 must be sequential (verification steps)

**Within Phase 3 (US1):**
- T021-T026 can run in parallel (different type files)
- T040-T041 can run in parallel (different test files)

**Within Phase 4 (US2/US3):**
- T049 can run parallel with T043-T048
- T059-T063 can run in parallel (different test files)

**Within Phase 6 (US4):**
- T075-T076 can run in parallel
- T086-T087 can run in parallel

**Within Phase 8 (US6):**
- T104-T105 can run in parallel
- T118-T119 can run in parallel

**Within Phase 9 (Polish):**
- T122-T127 all can run in parallel
- T134-T136 all can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all type definitions in parallel:
Task: "Create EngineConfig struct in crates/candle-vllm-core/src/api/config.rs"
Task: "Create GenerationParams struct in crates/candle-vllm-core/src/api/params.rs"
Task: "Create GenerationOutput struct in crates/candle-vllm-core/src/api/output.rs"
Task: "Create FinishReason enum in crates/candle-vllm-core/src/api/output.rs"
Task: "Create GenerationStats struct in crates/candle-vllm-core/src/api/output.rs"
Task: "Create Error enum in crates/candle-vllm-core/src/api/error.rs"

# Then sequentially implement InferenceEngine (depends on types):
Task: "Create api/ module in crates/candle-vllm-core/src/api/mod.rs"
Task: "Implement InferenceEngineBuilder in crates/candle-vllm-core/src/api/engine.rs"
# ... remaining engine implementation
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup verification
2. Complete Phase 2: Foundational fixes (**CRITICAL**)
3. Complete Phase 3: User Story 1 - Core InferenceEngine
4. **STOP and VALIDATE**: Test embedding engine in minimal app
5. Deploy/demo if ready

### Incremental Delivery

| Milestone | Stories Complete | Deliverable |
|-----------|-----------------|-------------|
| Foundation | Phase 2 | Code compiles cleanly |
| MVP | US1 | Embeddable inference engine |
| OpenAI Compat | US1, US2, US3 | Drop-in OpenAI replacement |
| Server Ready | US1-3, US7 | Backward-compatible server |
| MCP Ready | US1-5, US7 | Full agent capabilities |
| Complete | All | Production-ready library |

### Suggested MVP Scope

**Minimum Viable Product**: User Story 1 (Core Inference Engine)
- Enables: Embedding in Tauri apps, custom gateways
- Validates: Library-first architecture works
- Risk: Lowest (core functionality only)

### Success Criteria Verification by Task

| Success Criteria | Verification Tasks |
|------------------|-------------------|
| SC-001: Embeddable engine | T042, T133 |
| SC-002: OpenAI schema | T059, T060 |
| SC-003: Tool parsing (3 families) | T061, T062, T063 |
| SC-004: MCP integration | T086, T087, T088 |
| SC-005: Multi-turn agent | T101, T102, T103 |
| SC-006: Model switch queue | T118, T119, T120, T121 |
| SC-007: Backward compatibility | T073, T074 |
| SC-008: Zero clippy warnings | T020, T129 |
| SC-009: cargo fmt clean | T019, T128 |
| SC-010: All tests pass | T130 |
| SC-011: No stubs/TODOs | T131 |
| SC-012: Complete documentation | T122-T127, T132 |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- **CRITICAL**: Phase 2 (Foundational) must complete before any user story work begins
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- Total tasks: 136

