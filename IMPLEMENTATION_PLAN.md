# Implementation Plan: Library-First Restructuring

This document provides a step-by-step plan to restructure candle-vllm into a library-first architecture while maintaining backward compatibility.

## Overview

**Goal**: Transform candle-vllm from a binary-first project to a library that can be embedded in Tauri apps, AI gateways, and agent frameworks, while adding OpenAI Responses API support with MCP integration.

**Strategy**: Incremental refactoring without breaking existing functionality.

## Phase 1: Prepare for Refactoring (Week 1)

### Step 1.1: Create Workspace Structure

**Action**: Convert to a Cargo workspace

**Files to modify:**
```
candle-vllm/Cargo.toml
```

**Changes:**
```toml
[workspace]
members = [
    "crates/candle-vllm-core",
    "crates/candle-vllm-openai",
    "crates/candle-vllm-server",
    "crates/candle-vllm-responses",
]
resolver = "2"

[workspace.dependencies]
# Shared dependencies
candle-core = { git = "https://github.com/guoqingbao/candle.git", version = "0.8.3", rev = "27cfdef" }
candle-nn = { git = "https://github.com/guoqingbao/candle.git", version = "0.8.3", rev = "27cfdef" }
attention-rs = { git = "https://github.com/guoqingbao/attention.rs", version="0.1.7", rev = "753f0a6" }
tokenizers = "0.21.2"
serde = { version = "1.0.190", features = ["serde_derive"] }
serde_json = "1.0.108"
tokio = { version = "1.38.0", features = ["sync"] }
anyhow = "1.0.75"
thiserror = "1.0.58"
tracing = "0.1.40"
uuid = { version = "1.5.0", features = ["v4"] }
```

### Step 1.2: Create Crate Directories

**Action**: Create directory structure

```bash
mkdir -p crates/candle-vllm-core/src
mkdir -p crates/candle-vllm-openai/src
mkdir -p crates/candle-vllm-server/src
mkdir -p crates/candle-vllm-responses/src
mkdir -p examples/tauri_app
mkdir -p examples/ai_gateway
mkdir -p examples/agent_framework
```

### Step 1.3: Create Initial Cargo.toml Files

**Files to create:**
- `crates/candle-vllm-core/Cargo.toml`
- `crates/candle-vllm-openai/Cargo.toml`
- `crates/candle-vllm-server/Cargo.toml`
- `crates/candle-vllm-responses/Cargo.toml`

**Content for each** (see detailed specs below)

---

## Phase 2: Extract Core Inference Engine (Week 2)

### Step 2.1: Create candle-vllm-core Crate

**File**: `crates/candle-vllm-core/Cargo.toml`

```toml
[package]
name = "candle-vllm-core"
version = "0.5.0"
edition = "2021"
description = "Core inference engine for candle-vllm"
license = "MIT OR Apache-2.0"

[dependencies]
candle-core = { workspace = true }
candle-nn = { workspace = true }
attention-rs = { workspace = true }
tokenizers = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
anyhow = { workspace = true }
thiserror = { workspace = true }
tracing = { workspace = true }
parking_lot = "0.12"
half = { version = "2.5.0", features = ["num-traits", "use-intrinsics", "rand_distr"] }
rand = "0.9.0"

[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda", "attention-rs/cuda"]
metal = ["candle-core/metal", "candle-nn/metal", "attention-rs/metal"]
flash-attn = ["cuda", "attention-rs/flash-attn"]
flash-decoding = ["attention-rs/flash-decoding"]
graph = ["attention-rs/graph", "candle-core/graph"]
```

**File**: `crates/candle-vllm-core/src/lib.rs`

```rust
//! Core inference engine for candle-vllm
//! 
//! This crate provides the foundational inference capabilities without
//! any HTTP server or OpenAI-specific logic.

pub mod engine;
pub mod scheduler;
pub mod cache;
pub mod models;
pub mod config;
pub mod error;

// Re-export main types
pub use engine::{InferenceEngine, InferenceEngineBuilder};
pub use config::{EngineConfig, GenerationParams};
pub use error::{Error, Result};
pub use scheduler::{Scheduler, SchedulerConfig};
pub use cache::{CacheEngine, CacheConfig};

// Re-export from attention-rs
pub use attention_rs::{InputMetadata, PagedAttention};

// Common types
pub use candle_core::{Device, DType, Result as CandleResult};
pub use tokenizers::Tokenizer;
```

### Step 2.2: Move Existing Code to Core Crate

**Actions**:

1. **Move scheduler** → `crates/candle-vllm-core/src/scheduler/`
   - Copy `candle-vllm/src/scheduler/` → new location
   - Remove OpenAI-specific dependencies
   - Export clean API

2. **Move backend** → `crates/candle-vllm-core/src/backend/`
   - Copy `candle-vllm/src/backend/` → new location
   - Rename to avoid confusion

3. **Extract models** → `crates/candle-vllm-core/src/models/`
   - Copy `candle-vllm/src/openai/models/` → new location
   - Remove OpenAI coupling

4. **Create engine module** → `crates/candle-vllm-core/src/engine.rs`
   - Extract from `openai/pipelines/llm_engine.rs`
   - Remove OpenAI types
   - Create clean interface

**Files to create:**
```
crates/candle-vllm-core/src/
├── lib.rs
├── engine.rs              # InferenceEngine (extracted from LLMEngine)
├── config.rs              # EngineConfig, GenerationParams
├── error.rs               # Error types
├── scheduler/
│   ├── mod.rs
│   ├── sequence.rs        # From candle-vllm/src/scheduler/
│   └── cache_engine.rs
├── models/
│   ├── mod.rs
│   ├── config.rs          # Model configs
│   ├── mistral.rs
│   ├── llama.rs
│   └── ... (other models)
└── backend/
    ├── mod.rs
    └── ... (backend code)
```

### Step 2.3: Create Core API Types

**File**: `crates/candle-vllm-core/src/config.rs`

```rust
use serde::{Deserialize, Serialize};
use candle_core::{Device, DType};

/// Configuration for the inference engine
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: String,
    pub device: Device,
    pub dtype: DType,
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub kv_cache_memory: usize,
    pub enable_cuda_graph: bool,
    pub enable_chunked_prefill: bool,
    pub prefill_chunk_size: usize,
}

impl EngineConfig {
    pub fn builder() -> EngineConfigBuilder {
        EngineConfigBuilder::default()
    }
}

/// Parameters for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub stop_sequences: Vec<String>,
    pub logprobs: bool,
    pub seed: Option<u64>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_sequences: vec![],
            logprobs: false,
            seed: None,
        }
    }
}
```

**File**: `crates/candle-vllm-core/src/error.rs`

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Model loading error: {0}")]
    ModelLoad(String),
    
    #[error("Tokenization error: {0}")]
    Tokenization(String),
    
    #[error("Generation error: {0}")]
    Generation(String),
    
    #[error("Device error: {0}")]
    Device(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Request cancelled")]
    Cancelled,
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

---

## Phase 3: Create OpenAI Compatibility Layer (Week 3)

### Step 3.1: Create candle-vllm-openai Crate

**File**: `crates/candle-vllm-openai/Cargo.toml`

```toml
[package]
name = "candle-vllm-openai"
version = "0.5.0"
edition = "2021"
description = "OpenAI-compatible API layer for candle-vllm"
license = "MIT OR Apache-2.0"

[dependencies]
candle-vllm-core = { path = "../candle-vllm-core" }
serde = { workspace = true }
serde_json = { workspace = true }
tokio = { workspace = true }
uuid = { workspace = true }
thiserror = { workspace = true }
futures = "0.3.29"
minijinja = { version = "2.10.2", features = ["builtins", "json"] }
minijinja-contrib = { version = "2.10.2", features = ["pycompat"] }
chrono = "0.4.41"
regex = "1.10"

[features]
default = []
```

**File**: `crates/candle-vllm-openai/src/lib.rs`

```rust
//! OpenAI-compatible API layer for candle-vllm

pub mod adapter;
pub mod types;
pub mod conversation;
pub mod tool_calling;
pub mod streaming;

// Re-export main types
pub use adapter::OpenAIAdapter;
pub use types::{
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    Tool,
    ToolChoice,
    ToolCall,
    FunctionDefinition,
};
pub use conversation::{Conversation, ConversationManager};
pub use streaming::ChatCompletionStream;
```

### Step 3.2: Move OpenAI Types

**Actions**:

1. **Move requests/responses** → `crates/candle-vllm-openai/src/types/`
   - Copy from `candle-vllm/src/openai/requests.rs`
   - Copy from `candle-vllm/src/openai/responses.rs`
   - Keep all the MCP-related helpers we added

2. **Move conversation** → `crates/candle-vllm-openai/src/conversation/`
   - Copy from `candle-vllm/src/openai/conversation/`

3. **Move tool calling** → `crates/candle-vllm-openai/src/tool_calling/`
   - Copy from `candle-vllm/src/openai/tool_parser.rs`

**Files to create:**
```
crates/candle-vllm-openai/src/
├── lib.rs
├── adapter.rs             # OpenAIAdapter (wraps InferenceEngine)
├── types/
│   ├── mod.rs
│   ├── requests.rs        # ChatCompletionRequest, etc.
│   ├── responses.rs       # ChatCompletionResponse, etc.
│   └── tools.rs           # Tool, ToolCall, etc.
├── conversation/
│   ├── mod.rs
│   ├── manager.rs
│   └── templates.rs
├── tool_calling/
│   ├── mod.rs
│   ├── parser.rs          # Tool call parsers
│   └── formatters.rs      # Tool result formatters
└── streaming.rs           # Streaming support
```

### Step 3.3: Create OpenAIAdapter

**File**: `crates/candle-vllm-openai/src/adapter.rs`

```rust
use candle_vllm_core::{InferenceEngine, GenerationParams, Error as CoreError};
use crate::types::{ChatCompletionRequest, ChatCompletionResponse};
use crate::conversation::ConversationManager;
use crate::tool_calling::ToolParser;

pub struct OpenAIAdapter {
    engine: InferenceEngine,
    conversation_manager: ConversationManager,
    tool_parser: ToolParser,
}

impl OpenAIAdapter {
    pub fn new(engine: InferenceEngine) -> Self {
        Self {
            engine,
            conversation_manager: ConversationManager::new(),
            tool_parser: ToolParser::new(),
        }
    }
    
    pub async fn chat_completion(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, CoreError> {
        // 1. Convert messages to prompt using conversation manager
        let prompt = self.conversation_manager.build_prompt(&request)?;
        
        // 2. Tokenize
        let tokens = self.engine.tokenize(&prompt)?;
        
        // 3. Convert request params to GenerationParams
        let params = self.convert_params(&request);
        
        // 4. Generate
        let output = self.engine.generate(tokens, params).await?;
        
        // 5. Detokenize
        let text = self.engine.detokenize(&output.tokens)?;
        
        // 6. Parse for tool calls if tools are present
        let parsed = if request.has_tools() {
            self.tool_parser.parse(&text, &request.model)
        } else {
            ParsedOutput::Text(text)
        };
        
        // 7. Convert to response
        Ok(self.build_response(request, parsed, output))
    }
    
    fn convert_params(&self, request: &ChatCompletionRequest) -> GenerationParams {
        GenerationParams {
            max_tokens: request.max_tokens.unwrap_or(256),
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            // ... map other fields
            ..Default::default()
        }
    }
}
```

---

## Phase 4: Add Responses API with MCP (Week 4)

### Step 4.1: Create candle-vllm-responses Crate

**File**: `crates/candle-vllm-responses/Cargo.toml`

```toml
[package]
name = "candle-vllm-responses"
version = "0.5.0"
edition = "2021"
description = "OpenAI Responses API with MCP server integration"
license = "MIT OR Apache-2.0"

[dependencies]
candle-vllm-core = { path = "../candle-vllm-core" }
candle-vllm-openai = { path = "../candle-vllm-openai" }
serde = { workspace = true }
serde_json = { workspace = true }
tokio = { workspace = true }
reqwest = { version = "0.11", features = ["json"] }
async-trait = "0.1"

[features]
default = []
```

**File**: `crates/candle-vllm-responses/src/lib.rs`

```rust
//! OpenAI Responses API implementation with MCP support

pub mod session;
pub mod mcp_client;
pub mod orchestrator;

pub use session::{ResponsesSession, SessionConfig, ConversationOptions};
pub use mcp_client::{McpClient, McpServerConfig};
```

### Step 4.2: Implement MCP Client

**File**: `crates/candle-vllm-responses/src/mcp_client.rs`

```rust
use serde_json::Value;
use reqwest::Client;

pub struct McpServerConfig {
    pub url: String,
    pub auth: Option<String>,
    pub timeout_secs: u64,
}

pub struct McpClient {
    config: McpServerConfig,
    client: Client,
}

impl McpClient {
    pub async fn connect(config: McpServerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()?;
        
        Ok(Self { config, client })
    }
    
    pub async fn list_tools(&self) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
        let url = format!("{}/tools/list", self.config.url);
        let mut req = self.client.get(&url);
        
        if let Some(auth) = &self.config.auth {
            req = req.header("Authorization", auth);
        }
        
        let response = req.send().await?;
        let tools: Vec<Value> = response.json().await?;
        Ok(tools)
    }
    
    pub async fn call_tool(&self, tool_call: Value) -> Result<Value, Box<dyn std::error::Error>> {
        let url = format!("{}/tools/call", self.config.url);
        let mut req = self.client.post(&url).json(&tool_call);
        
        if let Some(auth) = &self.config.auth {
            req = req.header("Authorization", auth);
        }
        
        let response = req.send().await?;
        let result: Value = response.json().await?;
        Ok(result)
    }
}
```

### Step 4.3: Implement ResponsesSession

**File**: `crates/candle-vllm-responses/src/session.rs`

```rust
use candle_vllm_openai::{OpenAIAdapter, Message, Tool, ToolCall};
use crate::mcp_client::{McpClient, McpServerConfig};
use std::collections::HashMap;

pub struct SessionConfig {
    pub model_path: String,
    pub device: candle_vllm_core::Device,
    // ... other config
}

pub struct ConversationOptions {
    pub max_turns: usize,
    pub allowed_tools: Option<Vec<String>>,
}

impl Default for ConversationOptions {
    fn default() -> Self {
        Self {
            max_turns: 10,
            allowed_tools: None,
        }
    }
}

pub struct ResponsesSession {
    adapter: OpenAIAdapter,
    mcp_servers: HashMap<String, McpClient>,
}

impl ResponsesSession {
    pub async fn new(config: SessionConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize engine
        let engine_config = candle_vllm_core::EngineConfig::builder()
            .model_path(config.model_path)
            .device(config.device)
            .build()?;
        
        let engine = candle_vllm_core::InferenceEngine::new(engine_config).await?;
        let adapter = OpenAIAdapter::new(engine);
        
        Ok(Self {
            adapter,
            mcp_servers: HashMap::new(),
        })
    }
    
    pub async fn add_mcp_server(
        &mut self,
        name: &str,
        url: &str,
        auth: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let client = McpClient::connect(McpServerConfig {
            url: url.to_string(),
            auth,
            timeout_secs: 30,
        }).await?;
        
        self.mcp_servers.insert(name.to_string(), client);
        Ok(())
    }
    
    pub async fn run_conversation(
        &mut self,
        initial_messages: Vec<Message>,
        options: ConversationOptions,
    ) -> Result<ConversationResult, Box<dyn std::error::Error>> {
        // Multi-turn conversation with automatic tool execution
        // (Implementation details...)
        todo!()
    }
}

pub struct ConversationResult {
    pub final_message: String,
    pub tool_calls_executed: Vec<ToolCall>,
    pub turns_taken: usize,
}
```

---

## Phase 5: Update Server Binary (Week 5)

### Step 5.1: Create candle-vllm-server Crate

**File**: `crates/candle-vllm-server/Cargo.toml`

```toml
[package]
name = "candle-vllm-server"
version = "0.5.0"
edition = "2021"
description = "HTTP server for candle-vllm"

[dependencies]
candle-vllm-core = { path = "../candle-vllm-core" }
candle-vllm-openai = { path = "../candle-vllm-openai" }
axum = { version = "0.7.4", features = ["tokio"] }
tower-http = { version = "0.5.1", features = ["cors"] }
tokio = { workspace = true, features = ["full"] }
clap = { version = "4.4.7", features = ["derive"] }
```

**File**: `crates/candle-vllm-server/src/main.rs`

```rust
// This becomes the new main.rs for the server binary
// Reuse existing server logic but with new library APIs

use candle_vllm_core::{InferenceEngine, EngineConfig};
use candle_vllm_openai::OpenAIAdapter;

#[tokio::main]
async fn main() {
    // Parse args, initialize engine, start server
    // Similar to current main.rs but using library APIs
}
```

### Step 5.2: Update Root Cargo.toml

**File**: `candle-vllm/Cargo.toml` (keep for backward compatibility)

```toml
[package]
name = "candle-vllm"
version = "0.5.0"
edition = "2021"

[dependencies]
candle-vllm-server = { path = "crates/candle-vllm-server" }

# Re-export for library users
[dependencies.candle-vllm-core]
path = "crates/candle-vllm-core"

[dependencies.candle-vllm-openai]
path = "crates/candle-vllm-openai"

[dependencies.candle-vllm-responses]
path = "crates/candle-vllm-responses"

[[bin]]
name = "candle-vllm"
path = "crates/candle-vllm-server/src/main.rs"
```

---

## Phase 6: Create Examples (Week 6)

### Step 6.1: Tauri Example

**File**: `examples/tauri_app/src-tauri/Cargo.toml`

```toml
[package]
name = "candle-vllm-tauri-example"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-vllm-core = { path = "../../../crates/candle-vllm-core" }
candle-vllm-openai = { path = "../../../crates/candle-vllm-openai" }
tauri = { version = "1.5", features = [] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
```

**File**: `examples/tauri_app/src-tauri/src/main.rs`

```rust
// See LIBRARY_API.md Example 1 for full code
```

### Step 6.2: AI Gateway Example

**File**: `examples/ai_gateway/Cargo.toml`

```toml
[package]
name = "candle-vllm-gateway-example"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-vllm-core = { path = "../../crates/candle-vllm-core" }
candle-vllm-openai = { path = "../../crates/candle-vllm-openai" }
axum = "0.7"
tokio = { version = "1", features = ["full"] }
```

### Step 6.3: Agent Framework Example

**File**: `examples/agent_framework/Cargo.toml`

```toml
[package]
name = "candle-vllm-agent-example"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-vllm-responses = { path = "../../crates/candle-vllm-responses" }
tokio = { version = "1", features = ["full"] }
```

---

## Phase 7: Documentation & Testing (Week 7)

### Step 7.1: Update Documentation

**Files to update:**
- `README.md` - Add library usage section
- `ARCHITECTURE.md` - Already created
- `LIBRARY_API.md` - Already created
- Each crate's `README.md`

### Step 7.2: Add Tests

**For each crate**, add:
```
crates/candle-vllm-core/tests/
crates/candle-vllm-openai/tests/
crates/candle-vllm-responses/tests/
```

### Step 7.3: Add Integration Tests

**File**: `tests/integration_test.rs` (workspace root)

```rust
#[tokio::test]
async fn test_full_stack() {
    // Test that everything works together
}
```

---

## Migration Guide for Users

### For Existing Users (Using Binary)

**No changes needed!** The binary still works exactly the same:

```bash
cargo build --release --features cuda
./target/release/candle-vllm --model mistral-7b
```

### For Library Users (New)

**Add to Cargo.toml:**

```toml
[dependencies]
candle-vllm-core = "0.5"
candle-vllm-openai = "0.5"  # Optional
```

**Use in code:**

```rust
use candle_vllm_core::InferenceEngine;

let engine = InferenceEngine::new(config).await?;
```

---

## Testing Checklist

- [ ] Core engine works standalone
- [ ] OpenAI adapter converts requests correctly
- [ ] Tool calling works end-to-end
- [ ] MCP client can connect and call tools
- [ ] ResponsesSession orchestrates multi-turn conversations
- [ ] Binary server still works as before
- [ ] Tauri example compiles and runs
- [ ] AI gateway example works
- [ ] Agent framework example completes tasks
- [ ] All tests pass
- [ ] Documentation is complete

---

## Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Prepare | Workspace structure created |
| 2 | Core | candle-vllm-core crate working |
| 3 | OpenAI | candle-vllm-openai adapter working |
| 4 | Responses | MCP integration complete |
| 5 | Server | Binary updated to use libraries |
| 6 | Examples | All examples working |
| 7 | Polish | Tests, docs, release |

---

## Success Criteria

1. ✅ Existing binary users see no breaking changes
2. ✅ Library can be embedded in Tauri apps
3. ✅ Library can be used in custom HTTP servers
4. ✅ MCP servers can be integrated easily
5. ✅ Tool calling works across all model formats
6. ✅ Performance is equal or better than before
7. ✅ Documentation is comprehensive
8. ✅ Examples demonstrate all use cases

---

## Future Enhancements (Post-Launch)

- Add Python bindings (PyO3)
- Add WebAssembly support
- Add LangChain Rust adapter
- Add more MCP server examples
- Performance optimizations
- Additional model architectures