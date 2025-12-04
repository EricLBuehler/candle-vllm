# Feature Specification: Library-First Architecture Implementation

**Feature Branch**: `001-library-api-implementation`  
**Created**: December 3, 2025  
**Status**: Draft  
**Input**: User description: "Implement library-first architecture for candle-vllm with core inference engine, OpenAI compatibility layer, MCP integration, and Responses API support. Complete implementation with no stubs, no TODOs, compiling perfectly with no errors or warnings."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Embed Inference Engine in Custom Application (Priority: P1)

A developer wants to embed the candle-vllm inference engine directly into their custom application (such as a Tauri desktop app, custom HTTP gateway, or agent framework) without running a separate server process. They need programmatic access to model loading, tokenization, and text generation through a clean library API.

**Why this priority**: This is the core value proposition of the library-first restructuring. Without embeddable inference capabilities, none of the other features (OpenAI compatibility, MCP integration) can function. This enables all downstream use cases.

**Independent Test**: Can be fully tested by creating a minimal application that imports the core library, loads a model, tokenizes a prompt, generates tokens, and detokenizes the output—all without any HTTP server involvement.

**Acceptance Scenarios**:

1. **Given** a developer has added the core library as a dependency, **When** they configure an inference engine with a model path and device settings, **Then** the engine initializes successfully and reports model information.

2. **Given** an initialized inference engine, **When** the developer calls tokenize with a text string, **Then** they receive a vector of token IDs.

3. **Given** tokenized input and generation parameters, **When** the developer calls generate, **Then** they receive generated tokens with finish reason and statistics.

4. **Given** generated token IDs, **When** the developer calls detokenize, **Then** they receive the reconstructed text string.

5. **Given** the engine is configured for streaming, **When** the developer calls generate_stream, **Then** they receive an async stream of tokens that can be consumed incrementally.

---

### User Story 2 - Use OpenAI-Compatible Chat API (Priority: P1)

A developer wants to use the familiar OpenAI chat completion API format with their local models. They need to send ChatCompletionRequest objects and receive ChatCompletionResponse objects that match the OpenAI specification, enabling drop-in replacement for OpenAI API calls.

**Why this priority**: OpenAI compatibility is essential for ecosystem integration. Most AI tools, SDKs, and frameworks expect OpenAI-format APIs. This enables developers to switch between cloud and local inference without code changes.

**Independent Test**: Can be fully tested by creating an OpenAI adapter, sending a chat completion request with messages, and verifying the response matches OpenAI's response format including choices, message content, and usage statistics.

**Acceptance Scenarios**:

1. **Given** an OpenAI adapter wrapping an inference engine, **When** a ChatCompletionRequest is submitted with user messages, **Then** a ChatCompletionResponse is returned with generated content in the expected format.

2. **Given** a chat request with system, user, and assistant message history, **When** the adapter processes the request, **Then** the conversation context is properly applied using the model's chat template.

3. **Given** a streaming chat request, **When** the adapter processes the request, **Then** ChatCompletionChunk objects are streamed with delta content matching OpenAI's streaming format.

4. **Given** generation parameters (temperature, top_p, max_tokens, stop sequences), **When** included in the request, **Then** they are correctly applied to influence generation behavior.

---

### User Story 3 - Execute Tool Calls with Function Calling (Priority: P2)

A developer wants their model to decide when to call external tools/functions and generate the appropriate call arguments. They need to provide tool definitions in the request and receive structured tool call responses that can be executed and fed back to the model.

**Why this priority**: Tool calling is the foundation for agentic workflows. It enables models to interact with external systems, databases, and APIs—essential for building practical AI applications beyond simple chat.

**Independent Test**: Can be fully tested by sending a chat request with tool definitions, receiving a response with tool_calls, executing a mock tool, sending the tool result back, and receiving the model's final response.

**Acceptance Scenarios**:

1. **Given** a request with tool definitions (function name, description, parameters schema), **When** the model decides to call a tool, **Then** the response includes tool_calls with function name and JSON arguments.

2. **Given** a tool call response, **When** the developer executes the tool and sends back a tool result message, **Then** the model incorporates the result into its next response.

3. **Given** multiple tools are defined, **When** the model needs to call multiple tools, **Then** the response includes all tool calls that should be executed.

4. **Given** tool_choice is set to "auto", **When** the model determines a tool isn't needed, **Then** the response contains regular content without tool_calls.

5. **Given** tool_choice is set to a specific function, **When** the request is processed, **Then** the model is forced to call that specific function.

---

### User Story 4 - Integrate MCP Servers for External Capabilities (Priority: P2)

A developer wants to connect MCP (Model Context Protocol) servers to provide external tools and resources to their model. They need to discover available tools from MCP servers, convert them to the OpenAI tool format, execute tool calls through MCP, and handle responses.

**Why this priority**: MCP integration enables standardized tool connectivity. Rather than building custom integrations for each external service, developers can leverage the growing ecosystem of MCP servers for file systems, databases, APIs, and more.

**Independent Test**: Can be fully tested by connecting to an MCP server, listing available tools, including them in a chat request, and verifying tool calls are correctly routed to the MCP server.

**Acceptance Scenarios**:

1. **Given** an MCP server URL and optional authentication, **When** connecting the MCP client, **Then** the connection is established and the client reports available tools.

2. **Given** a connected MCP client, **When** listing tools, **Then** tool definitions are returned with names, descriptions, and parameter schemas.

3. **Given** MCP tool definitions, **When** converting to OpenAI format, **Then** they can be used directly in ChatCompletionRequest tool definitions.

4. **Given** a model generates a tool call for an MCP tool, **When** the call is executed, **Then** it is correctly routed to the MCP server and the result is returned.

5. **Given** an MCP tool execution fails, **When** handling the error, **Then** an appropriate error message is returned that can be fed back to the model.

---

### User Story 5 - Run Multi-Turn Agent Conversations (Priority: P2)

A developer wants to run automated multi-turn conversations where the model can make multiple tool calls across several turns to complete complex tasks. They need a high-level API that handles the conversation loop, tool execution, and result feeding automatically.

**Why this priority**: Multi-turn agent conversations are the end goal for most AI applications. This provides the highest-level abstraction that makes building agents simple while leveraging all the lower-level capabilities.

**Independent Test**: Can be fully tested by creating a ResponsesSession with MCP servers, providing an initial task message, and verifying the session automatically executes tools and completes the task across multiple turns.

**Acceptance Scenarios**:

1. **Given** a ResponsesSession configured with MCP servers, **When** a conversation is started with an initial message, **Then** the session runs until completion or max turns reached.

2. **Given** the model generates tool calls during conversation, **When** the session processes them, **Then** tools are automatically executed and results fed back.

3. **Given** conversation options specify max_turns, **When** that limit is reached, **Then** the conversation stops and returns partial results.

4. **Given** conversation options specify allowed_tools, **When** the model tries to call a disallowed tool, **Then** the call is blocked and the model is informed.

5. **Given** a completed conversation, **When** accessing results, **Then** the final message, all tool calls executed, and turn count are available.

---

### User Story 6 - Queue Requests During Model Switching (Priority: P3)

A developer operating a server with multiple models wants requests to be queued intelligently when a model switch is needed. They need clear feedback about queue status, model loading progress, and graceful handling of timeouts.

**Why this priority**: In multi-model deployments, model switching is inevitable. Without proper queuing and feedback, users experience cryptic failures. This enables reliable multi-model serving.

**Independent Test**: Can be fully tested by sending a request for an inactive model, verifying the request is queued, observing status updates during model loading, and receiving the response once the model is ready.

**Acceptance Scenarios**:

1. **Given** a request targets an inactive model, **When** submitted to the server, **Then** the request is queued and the client receives status indicating queued position.

2. **Given** a streaming request is queued, **When** waiting for model switch, **Then** streaming status chunks indicate "queued", "waiting_for_model", and "loading" states with metadata.

3. **Given** model loading completes, **When** queued requests exist, **Then** they are processed in order and responses stream normally.

4. **Given** a non-streaming request is queued, **When** wait exceeds configured timeout, **Then** a 503 response with server_busy error is returned.

5. **Given** per-model queue has a configured maximum size, **When** that limit is exceeded, **Then** new requests receive immediate 503 rejection.

---

### User Story 7 - Maintain Backward Compatibility with Existing Binary (Priority: P1)

An existing user of the candle-vllm binary wants to continue using the server exactly as before. The command-line interface, configuration options, and HTTP API endpoints must work identically after the restructuring.

**Why this priority**: Breaking existing deployments would cause immediate user impact and prevent adoption. Backward compatibility ensures smooth migration paths.

**Independent Test**: Can be fully tested by running the rebuilt binary with existing command-line arguments and configuration, then sending requests to all existing endpoints and verifying identical responses.

**Acceptance Scenarios**:

1. **Given** an existing command-line invocation, **When** running the rebuilt binary, **Then** it starts successfully with the same behavior.

2. **Given** existing API calls to /v1/chat/completions, **When** sent to the rebuilt server, **Then** responses are identical in structure and content.

3. **Given** existing streaming requests, **When** sent to the rebuilt server, **Then** stream chunks have identical format.

4. **Given** existing tool calling requests, **When** sent to the rebuilt server, **Then** tool calls are parsed and formatted identically.

---

### Edge Cases

- What happens when a model fails to load? The system reports a clear error through the appropriate channel (library error type, HTTP status, or streaming error chunk) with actionable information.
- How does the system handle device unavailability (requested CUDA device doesn't exist)? Configuration validation fails early with a descriptive error before attempting model load.
- What happens when KV cache memory is exhausted? New requests are rejected with capacity errors; existing requests complete or timeout gracefully.
- How does streaming handle client disconnection? Generation is cancelled, resources are freed, and the request is removed from any queues.
- What happens when an MCP server becomes unavailable mid-conversation? Tool calls to that server fail with timeout/connection errors that can be reported back to the model.
- How does the system handle malformed tool call arguments from the model? Parse errors are returned as tool results so the model can correct itself.
- What happens when a queued request is cancelled by the client? The request is removed from the queue and resources are freed.

## Requirements *(mandatory)*

### Functional Requirements

#### Core Library (candle-vllm-core)

- **FR-001**: Library MUST provide an InferenceEngine that can load models from local paths or HuggingFace model IDs.
- **FR-002**: Library MUST support device selection (CUDA with device index, Metal, CPU) through configuration.
- **FR-003**: Library MUST provide tokenization and detokenization through the engine interface.
- **FR-004**: Library MUST support configurable generation parameters including max_tokens, temperature, top_p, top_k, repetition_penalty, frequency_penalty, presence_penalty, stop_sequences, and seed.
- **FR-005**: Library MUST provide both synchronous generation and async streaming generation methods.
- **FR-006**: Library MUST support request batching with configurable maximum batch size.
- **FR-007**: Library MUST provide request cancellation through request handles.
- **FR-008**: Library MUST expose generation statistics including prompt tokens, generated tokens, total time, and tokens per second.
- **FR-009**: Library MUST support KV cache configuration with memory limits.
- **FR-010**: Library MUST support CUDA graph optimization when enabled.
- **FR-011**: Library MUST support chunked prefill for long prompts when enabled.

#### OpenAI Compatibility (candle-vllm-openai)

- **FR-012**: Library MUST accept ChatCompletionRequest matching OpenAI's specification.
- **FR-013**: Library MUST return ChatCompletionResponse matching OpenAI's specification.
- **FR-014**: Library MUST support message types: system, user, assistant, and tool.
- **FR-015**: Library MUST apply model-specific chat templates to format conversations.
- **FR-016**: Library MUST support tool definitions with function name, description, and JSON Schema parameters.
- **FR-017**: Library MUST support tool_choice options: "none", "auto", "required", and specific function selection.
- **FR-018**: Library MUST parse model output for tool calls according to model-specific formats (e.g., Mistral, Llama, Qwen patterns).
- **FR-019**: Library MUST return tool calls with unique IDs, function names, and JSON argument strings.
- **FR-020**: Library MUST support tool result messages that reference tool call IDs.
- **FR-021**: Library MUST support streaming with ChatCompletionChunk format including delta content and tool call deltas.
- **FR-022**: Library MUST calculate and return usage statistics (prompt_tokens, completion_tokens, total_tokens).

#### MCP Integration (candle-vllm-responses)

- **FR-023**: Library MUST provide an MCP client that connects to MCP servers via HTTP.
- **FR-024**: MCP client MUST support optional authentication headers.
- **FR-025**: MCP client MUST list available tools from connected servers.
- **FR-026**: MCP client MUST execute tool calls and return results.
- **FR-027**: Library MUST provide conversion between MCP tool format and OpenAI tool format.
- **FR-028**: Library MUST provide conversion between OpenAI tool calls and MCP tool call format.
- **FR-029**: Library MUST provide conversion from MCP tool results to OpenAI tool message format.

#### Responses API Session (candle-vllm-responses)

- **FR-030**: Library MUST provide ResponsesSession that manages multi-turn conversations.
- **FR-031**: ResponsesSession MUST support registering multiple MCP servers by name.
- **FR-032**: ResponsesSession MUST automatically execute tool calls and feed results back.
- **FR-033**: ResponsesSession MUST respect max_turns configuration.
- **FR-034**: ResponsesSession MUST support allowed_tools filtering.
- **FR-035**: ResponsesSession MUST return ConversationResult with final message, executed tool calls, and turn count.

#### Model Management & Queuing (candle-vllm-server)

- **FR-036**: Server MUST maintain a finite state machine for model status (Ready, Switching, Loading, Error).
- **FR-037**: Server MUST allow only one model to be active at a time.
- **FR-038**: Server MUST queue requests targeting inactive models.
- **FR-039**: Server MUST enforce per-model queue size limits (configurable, default 10).
- **FR-040**: Server MUST reject requests with 503 when queue is full.
- **FR-041**: Server MUST reject non-streaming requests with 503 when wait exceeds timeout (configurable, default 30s).
- **FR-042**: Server MUST emit streaming status chunks for queued requests with position and status metadata.
- **FR-043**: Server MUST serialize model switches (no concurrent loads).
- **FR-044**: Server MUST drain the active model's queue before switching to a new model.
- **FR-045**: Server MUST coalesce requests targeting the same pending model during a switch.
- **FR-046**: Server MUST provide /v1/models/status endpoint showing FSM state, active model, queue lengths.

#### Backward Compatibility

- **FR-047**: Binary MUST accept all existing command-line arguments with identical behavior.
- **FR-048**: Server MUST serve all existing API endpoints with identical request/response formats.
- **FR-049**: Server MUST support existing configuration file formats.

### Key Entities

- **InferenceEngine**: The core abstraction for model inference. Holds loaded model weights, tokenizer, scheduler, and cache. Provides tokenization, generation, and detokenization operations.

- **GenerationParams**: Configuration for a single generation request. Includes sampling parameters (temperature, top_p, top_k), penalties (repetition, frequency, presence), limits (max_tokens), and control options (stop_sequences, seed).

- **GenerationOutput**: Result of a generation operation. Contains generated token IDs, finish reason, optional logprobs, and generation statistics.

- **OpenAIAdapter**: Wrapper that translates OpenAI-format requests to core engine calls and back. Manages conversation templating and tool call parsing.

- **ChatCompletionRequest**: OpenAI-format chat request. Contains model identifier, message history, optional tools, generation parameters, and streaming flag.

- **ChatCompletionResponse**: OpenAI-format chat response. Contains choices with messages (content and/or tool_calls), finish reason, and usage statistics.

- **McpClient**: Client for communicating with MCP servers. Handles connection, tool listing, and tool execution.

- **ResponsesSession**: High-level session manager for multi-turn agent conversations. Orchestrates conversation flow, tool execution, and result aggregation.

- **ModelManager**: Server component managing model lifecycle. Tracks active model, handles load/unload operations, and maintains the state machine.

- **RequestQueue**: Per-model queue of pending requests. Enforces size limits and provides position tracking for status updates.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Developers can embed the inference engine in a standalone application and generate text without any HTTP server code, verified by compiling and running the Tauri example.

- **SC-002**: The OpenAI adapter produces responses that validate against OpenAI's API schema for chat completions, verified by schema validation tests.

- **SC-003**: Tool calls are correctly parsed for at least 3 model families (Mistral, Llama, Qwen patterns), verified by unit tests with known model outputs.

- **SC-004**: MCP server integration successfully lists tools and executes tool calls with a reference MCP server, verified by integration tests.

- **SC-005**: Multi-turn agent conversations complete complex tasks involving 3+ tool calls across 5+ turns, verified by agent framework example.

- **SC-006**: Model switch requests are queued and processed in order with correct status updates, verified by integration tests simulating concurrent requests.

- **SC-007**: Existing binary users experience no breaking changes when upgrading, verified by running existing command patterns and comparing outputs.

- **SC-008**: All library code compiles with zero errors and zero warnings under `cargo clippy --all-targets --all-features -D warnings`.

- **SC-009**: All library code passes `cargo fmt --all --check` with no formatting differences.

- **SC-010**: All tests pass under `cargo test --all --all-features`.

- **SC-011**: No stub implementations or TODO comments remain in production code.

- **SC-012**: Each crate provides complete public API documentation with doc comments.

## Assumptions

- Models are accessed from local filesystem paths or downloaded via HuggingFace hub; no custom model registry integration is needed.
- MCP servers expose HTTP endpoints at standard paths (/tools/list, /tools/call); no WebSocket MCP transport is required initially.
- Chat templates are available via the tokenizer's built-in template or model-specific defaults; no custom template language beyond Jinja2 is needed.
- Performance parity with the existing implementation is acceptable; no performance regression testing against benchmarks is in scope.
- The idle_unload policy mentioned in the remainder document is optional and can be deferred if not critical to core functionality.
- Worker pool sizing is single-threaded for inference (GPU constraint); the pool abstraction primarily manages request lifecycle.
