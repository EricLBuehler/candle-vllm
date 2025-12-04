# Candle-vLLM Architecture: Library-First Design

## Overview

This document outlines the architectural restructuring of candle-vllm to support:
1. **Library-first design** - Core inference as a reusable library
2. **Embedded applications** - Tauri desktop apps, AI gateways
3. **OpenAI Responses API** - Direct MCP server integration
4. **Custom agents** - Build agent systems on top of the inference engine

## Current Architecture Problems

```
candle-vllm (binary)
├── main.rs (HTTP server entry point)
├── lib.rs (minimal exports)
└── openai/
    ├── openai_server.rs (HTTP handlers tightly coupled)
    ├── pipelines/ (inference engine)
    └── models/ (model implementations)
```

**Issues:**
- HTTP server logic mixed with inference logic
- Hard to embed in custom applications
- Can't easily swap out the HTTP layer
- Difficult to use from Tauri or other frameworks

## Proposed Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  (HTTP servers, Tauri apps, AI gateways, agents)       │
│                                                          │
│  Examples:                                              │
│  - candle-vllm-server (Axum HTTP server)              │
│  - Your Tauri desktop app                             │
│  - Your AI gateway (custom Rust server)               │
│  - Agent frameworks (LangChain-style)                 │
└─────────────────────────────────────────────────────────┘
                          │
                          │ uses
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   API Adapter Layer                      │
│       (OpenAI compatibility, MCP integration)           │
│                                                          │
│  - OpenAI request/response types                       │
│  - Tool calling (function calling)                     │
│  - MCP server integration                              │
│  - Streaming adapters                                  │
│  - Conversation management                             │
└─────────────────────────────────────────────────────────┘
                          │
                          │ uses
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Core Inference Engine                   │
│          (Model-agnostic inference library)             │
│                                                          │
│  - LLMEngine (inference orchestration)                 │
│  - Scheduler (request scheduling)                      │
│  - CacheEngine (KV cache management)                   │
│  - PagedAttention (attention kernels)                  │
│  - Model implementations (Mistral, Llama, etc.)        │
└─────────────────────────────────────────────────────────┘
```

## Crate Structure

### Option 1: Workspace with Multiple Crates (Recommended)

```
candle-vllm/
├── Cargo.toml (workspace)
├── crates/
│   ├── candle-vllm-core/          # Core inference engine
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── engine.rs          # LLMEngine
│   │   │   ├── scheduler/
│   │   │   ├── cache/
│   │   │   ├── models/
│   │   │   └── attention/
│   │   └── Cargo.toml
│   │
│   ├── candle-vllm-openai/        # OpenAI API compatibility
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── types.rs           # Request/Response types
│   │   │   ├── conversation.rs
│   │   │   ├── tool_calling.rs
│   │   │   ├── streaming.rs
│   │   │   └── mcp.rs             # MCP integration
│   │   └── Cargo.toml
│   │
│   ├── candle-vllm-server/        # HTTP server (optional)
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── routes.rs
│   │   │   └── middleware.rs
│   │   └── Cargo.toml
│   │
│   └── candle-vllm-responses/     # Responses API support
│       ├── src/
│       │   ├── lib.rs
│       │   ├── session.rs
│       │   ├── mcp_client.rs
│       │   └── orchestrator.rs
│       └── Cargo.toml
│
└── examples/
    ├── tauri_app/                  # Tauri desktop example
    ├── ai_gateway/                 # Custom gateway example
    └── agent_framework/            # Agent system example
```

**Benefits:**
- Clean separation of concerns
- Users can depend on only what they need
- Independent versioning
- Easier to maintain and test

### Option 2: Feature Flags (Simpler, Less Ideal)

Keep single crate but use features:
```toml
[features]
default = ["core"]
core = []              # Just the inference engine
openai = ["core"]      # + OpenAI compatibility
server = ["openai", "axum"]  # + HTTP server
responses-api = ["openai", "mcp-client"]
full = ["server", "responses-api"]
```

## Core API Design

### 1. Core Inference Engine (`candle-vllm-core`)

```rust
// Simple, model-agnostic inference API
pub struct InferenceEngine {
    scheduler: Scheduler,
    cache_engine: CacheEngine,
    model: Box<dyn Model>,
}

impl InferenceEngine {
    pub async fn new(config: EngineConfig) -> Result<Self>;
    
    pub async fn generate(
        &mut self,
        prompt: Vec<u32>,
        params: GenerationParams,
    ) -> Result<GenerationOutput>;
    
    pub async fn generate_stream(
        &mut self,
        prompt: Vec<u32>,
        params: GenerationParams,
    ) -> Result<impl Stream<Item = GenerationChunk>>;
    
    pub fn add_request(&mut self, request: InferenceRequest) -> RequestId;
    pub fn cancel_request(&mut self, id: RequestId);
    pub fn step(&mut self) -> Result<Vec<GenerationOutput>>;
}

pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stop_sequences: Vec<String>,
    // No OpenAI-specific types here
}
```

### 2. OpenAI Compatibility Layer (`candle-vllm-openai`)

```rust
// Converts OpenAI requests to core inference requests
pub struct OpenAIAdapter {
    engine: Arc<Mutex<InferenceEngine>>,
    conversation_manager: ConversationManager,
    tool_parser: ToolParser,
}

impl OpenAIAdapter {
    pub async fn chat_completion(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse>;
    
    pub async fn chat_completion_stream(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Result<impl Stream<Item = ChatCompletionChunk>>;
    
    // Tool calling support
    pub async fn chat_completion_with_tools(
        &mut self,
        request: ChatCompletionRequest,
        tools: Vec<Tool>,
    ) -> Result<ChatCompletionResponse>;
}

// Standalone conversion functions
pub fn to_generation_params(req: &ChatCompletionRequest) -> GenerationParams;
pub fn to_chat_response(output: GenerationOutput) -> ChatCompletionResponse;
```

### 3. MCP Integration (`candle-vllm-responses`)

```rust
// OpenAI Responses API with MCP server support
pub struct ResponsesSession {
    adapter: OpenAIAdapter,
    mcp_servers: HashMap<String, McpServerConnection>,
}

impl ResponsesSession {
    pub async fn new(config: SessionConfig) -> Result<Self>;
    
    // Connect to an MCP server
    pub async fn add_mcp_server(
        &mut self,
        name: &str,
        url: &str,
        auth: Option<String>,
    ) -> Result<()>;
    
    // Run a conversation with automatic tool execution
    pub async fn run_conversation(
        &mut self,
        messages: Vec<Message>,
        options: ConversationOptions,
    ) -> Result<ConversationResult>;
    
    // Get available tools from all connected MCP servers
    pub async fn list_available_tools(&self) -> Result<Vec<Tool>>;
    
    // Execute a tool via MCP
    async fn execute_tool_call(
        &mut self,
        tool_call: &ToolCall,
    ) -> Result<ToolResult>;
}

pub struct ConversationOptions {
    pub tools: ToolSelection,
    pub max_turns: usize,
    pub allowed_tools: Option<Vec<String>>,
}

pub enum ToolSelection {
    Auto,           // Let model decide
    Required,       // Model must use tools
    None,          // Disable tools
    Specific(Vec<String>),  // Only these tools
}
```

## Usage Examples

### Example 1: Tauri Desktop App

```rust
// In your Tauri app's main.rs
use candle_vllm_core::{InferenceEngine, EngineConfig};
use candle_vllm_openai::OpenAIAdapter;
use tauri::State;

struct AppState {
    adapter: Arc<Mutex<OpenAIAdapter>>,
}

#[tauri::command]
async fn chat(
    message: String,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let adapter = state.adapter.lock().await;
    
    let request = ChatCompletionRequest {
        model: "local".into(),
        messages: vec![Message::user(message)],
        ..Default::default()
    };
    
    let response = adapter.chat_completion(request).await
        .map_err(|e| e.to_string())?;
    
    Ok(response.choices[0].message.content.clone())
}

fn main() {
    // Initialize engine on startup
    let engine = InferenceEngine::new(EngineConfig {
        model_path: "./models/mistral-7b",
        device: Device::Cuda(0),
        ..Default::default()
    }).await.unwrap();
    
    let adapter = OpenAIAdapter::new(engine);
    
    tauri::Builder::default()
        .manage(AppState {
            adapter: Arc::new(Mutex::new(adapter)),
        })
        .invoke_handler(tauri::generate_handler![chat])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### Example 2: Custom AI Gateway

```rust
// Your own Rust HTTP server (not using candle-vllm-server)
use axum::{Router, Json};
use candle_vllm_openai::OpenAIAdapter;

async fn custom_handler(
    Json(req): Json<MyCustomRequest>,
    adapter: Extension<Arc<Mutex<OpenAIAdapter>>>,
) -> Json<MyCustomResponse> {
    // Convert your custom format to OpenAI format
    let openai_req = convert_to_openai(req);
    
    let adapter = adapter.lock().await;
    let response = adapter.chat_completion(openai_req).await.unwrap();
    
    // Convert back to your format
    Json(convert_from_openai(response))
}

#[tokio::main]
async fn main() {
    let engine = InferenceEngine::new(config).await.unwrap();
    let adapter = Arc::new(Mutex::new(OpenAIAdapter::new(engine)));
    
    let app = Router::new()
        .route("/my-api/chat", post(custom_handler))
        .layer(Extension(adapter));
    
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

### Example 3: Agent Framework with MCP

```rust
use candle_vllm_responses::{ResponsesSession, ConversationOptions};

#[tokio::main]
async fn main() {
    let mut session = ResponsesSession::new(SessionConfig {
        model_path: "./models/mistral-7b",
        ..Default::default()
    }).await.unwrap();
    
    // Connect to MCP servers
    session.add_mcp_server(
        "filesystem",
        "http://localhost:3001/mcp",
        None,
    ).await.unwrap();
    
    session.add_mcp_server(
        "github",
        "http://localhost:3002/mcp",
        Some("Bearer YOUR_TOKEN".into()),
    ).await.unwrap();
    
    // Run multi-turn conversation with automatic tool execution
    let result = session.run_conversation(
        vec![Message::user("Read README.md and create a GitHub issue summarizing it")],
        ConversationOptions {
            tools: ToolSelection::Auto,
            max_turns: 10,
            allowed_tools: Some(vec![
                "read_file".into(),
                "create_github_issue".into(),
            ]),
        },
    ).await.unwrap();
    
    println!("Final response: {}", result.final_message);
    println!("Tools used: {:?}", result.tool_calls_made);
}
```

## Migration Path

### Phase 1: Refactor Current Code (No Breaking Changes)
1. Keep existing `candle-vllm` crate working
2. Extract core types to internal modules
3. Add new public API alongside existing one
4. Mark old APIs as deprecated

### Phase 2: Create Library Crates
1. Create `candle-vllm-core` crate
2. Move inference engine logic
3. Create `candle-vllm-openai` adapter
4. Update `candle-vllm` binary to use new crates

### Phase 3: Add Responses API Support
1. Create `candle-vllm-responses` crate
2. Implement MCP client
3. Build conversation orchestrator
4. Add examples

### Phase 4: Polish and Document
1. Add comprehensive examples
2. Write integration guides
3. Create starter templates (Tauri, gateway, etc.)
4. Performance benchmarks

## Benefits of This Architecture

### For Library Users
- **Minimal dependencies**: Only include what you need
- **No HTTP overhead**: Direct function calls
- **Easy embedding**: Works in any Rust application
- **Flexible**: Adapt to your own API design

### For Application Developers
- **Own your endpoints**: Full control over HTTP layer
- **Custom middleware**: Add auth, rate limiting, etc.
- **Framework agnostic**: Use with Axum, Actix, Tauri, etc.
- **Agent building**: MCP support built-in

### For Maintainers
- **Cleaner code**: Separation of concerns
- **Easier testing**: Test layers independently
- **Better docs**: Clear API boundaries
- **More flexible**: Can change HTTP server without breaking core

## Open Questions

1. **Backward Compatibility**: Keep old API or breaking change?
   - Recommendation: Keep old API with deprecation warnings

2. **Async Runtime**: Require tokio or stay runtime-agnostic?
   - Recommendation: Provide both sync and async APIs

3. **Error Handling**: Custom error types or use anyhow?
   - Recommendation: Custom error types for library, anyhow for binary

4. **Feature Flags**: How granular should they be?
   - Recommendation: Start simple, add more as needed

## Implementation Checklist

- [ ] Create workspace structure
- [ ] Extract `candle-vllm-core` with minimal API
- [ ] Create `candle-vllm-openai` adapter layer
- [ ] Add tool calling support to adapter
- [ ] Create `candle-vllm-responses` with MCP integration
- [ ] Update main binary to use new crates
- [ ] Write migration guide
- [ ] Add Tauri example
- [ ] Add AI gateway example
- [ ] Add agent framework example
- [ ] Update documentation
- [ ] Publish to crates.io

## References

- OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
- MCP Specification: https://modelcontextprotocol.io/
- Tauri Framework: https://tauri.app/