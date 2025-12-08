# Candle-vLLM Library API Specification

This document defines the public API for using candle-vllm as a library in your Rust applications (Tauri, AI gateways, agent frameworks, etc.).

## Table of Contents

1. [Core Inference API](#core-inference-api)
2. [OpenAI Compatibility Layer](#openai-compatibility-layer)
3. [Tool Calling & MCP Integration](#tool-calling--mcp-integration)
4. [Streaming Support](#streaming-support)
5. [Error Handling](#error-handling)
6. [Configuration](#configuration)
7. [Examples](#examples)

---

## Core Inference API

### Basic Usage

```rust
use candle_vllm::InferenceEngine;
use candle_vllm::GenerationParams;

// Initialize the engine
let mut engine = InferenceEngine::builder()
    .model_path("./models/mistral-7b")
    .device(Device::Cuda(0))
    .max_batch_size(16)
    .build()
    .await?;

// Generate text
let prompt = engine.tokenize("Hello, how are you?")?;
let output = engine.generate(prompt, GenerationParams::default()).await?;
let text = engine.detokenize(&output.tokens)?;

println!("Response: {}", text);
```

### InferenceEngine

The core inference engine that handles model execution, scheduling, and caching.

```rust
pub struct InferenceEngine {
    // Internal fields are private
}

impl InferenceEngine {
    /// Create a new builder for configuring the engine
    pub fn builder() -> InferenceEngineBuilder;
    
    /// Load a model and create the engine
    pub async fn new(config: EngineConfig) -> Result<Self>;
    
    /// Tokenize text input
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
    
    /// Detokenize token IDs back to text
    pub fn detokenize(&self, tokens: &[u32]) -> Result<String>;
    
    /// Generate tokens from a prompt (non-streaming)
    pub async fn generate(
        &mut self,
        prompt: Vec<u32>,
        params: GenerationParams,
    ) -> Result<GenerationOutput>;
    
    /// Generate tokens with streaming
    pub async fn generate_stream(
        &mut self,
        prompt: Vec<u32>,
        params: GenerationParams,
    ) -> Result<GenerationStream>;
    
    /// Add a generation request to the queue (for batching)
    pub fn add_request(
        &mut self,
        request: InferenceRequest,
    ) -> RequestHandle;
    
    /// Cancel an in-flight request
    pub fn cancel_request(&mut self, handle: RequestHandle);
    
    /// Execute one step of generation (for custom scheduling)
    pub fn step(&mut self) -> Result<Vec<RequestUpdate>>;
    
    /// Get model information
    pub fn model_info(&self) -> &ModelInfo;
    
    /// Get current statistics
    pub fn stats(&self) -> EngineStats;
}
```

### GenerationParams

```rust
pub struct GenerationParams {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    
    /// Temperature for sampling (0.0 = greedy)
    pub temperature: f32,
    
    /// Top-p nucleus sampling
    pub top_p: f32,
    
    /// Top-k sampling
    pub top_k: Option<usize>,
    
    /// Repetition penalty
    pub repetition_penalty: f32,
    
    /// Frequency penalty
    pub frequency_penalty: f32,
    
    /// Presence penalty
    pub presence_penalty: f32,
    
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    
    /// Whether to include logprobs
    pub logprobs: bool,
    
    /// Random seed for reproducibility
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

### GenerationOutput

```rust
pub struct GenerationOutput {
    /// Generated token IDs
    pub tokens: Vec<u32>,
    
    /// Finish reason
    pub finish_reason: FinishReason,
    
    /// Log probabilities (if requested)
    pub logprobs: Option<Vec<f32>>,
    
    /// Generation statistics
    pub stats: GenerationStats,
}

pub enum FinishReason {
    /// Generation completed naturally
    Stop,
    
    /// Reached max_tokens limit
    Length,
    
    /// Hit a stop sequence
    StopSequence(String),
    
    /// Request was cancelled
    Cancelled,
    
    /// An error occurred
    Error(String),
}

pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_time_ms: u64,
    pub tokens_per_second: f32,
}
```

---

## OpenAI Compatibility Layer

### OpenAIAdapter

Converts OpenAI-style requests to core inference and back.

```rust
use candle_vllm::openai::{OpenAIAdapter, ChatCompletionRequest, ChatCompletionResponse};

let engine = InferenceEngine::new(config).await?;
let mut adapter = OpenAIAdapter::new(engine);

// Use OpenAI-compatible API
let request = ChatCompletionRequest {
    model: "mistral-7b".into(),
    messages: vec![
        Message::user("What is Rust?"),
    ],
    temperature: Some(0.7),
    max_tokens: Some(256),
    ..Default::default()
};

let response = adapter.chat_completion(request).await?;
println!("{}", response.choices[0].message.content);
```

### OpenAIAdapter API

```rust
pub struct OpenAIAdapter {
    engine: InferenceEngine,
    conversation_manager: ConversationManager,
}

impl OpenAIAdapter {
    /// Create a new adapter from an inference engine
    pub fn new(engine: InferenceEngine) -> Self;
    
    /// Create with custom configuration
    pub fn with_config(engine: InferenceEngine, config: AdapterConfig) -> Self;
    
    /// Handle a chat completion request
    pub async fn chat_completion(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse>;
    
    /// Handle a streaming chat completion request
    pub async fn chat_completion_stream(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream>;
    
    /// Get the underlying engine (for custom operations)
    pub fn engine(&self) -> &InferenceEngine;
    
    /// Get mutable access to the engine
    pub fn engine_mut(&mut self) -> &mut InferenceEngine;
}
```

### Request/Response Types

All OpenAI-compatible types are re-exported:

```rust
use candle_vllm::openai::{
    // Request types
    ChatCompletionRequest,
    Message,
    Tool,
    ToolChoice,
    FunctionDefinition,
    
    // Response types
    ChatCompletionResponse,
    ChatChoice,
    ToolCall,
    FunctionCall,
    
    // Streaming types
    ChatCompletionChunk,
    ChatCompletionStream,
};
```

---

## Tool Calling & MCP Integration

### Tool Calling with OpenAI Format

```rust
use candle_vllm::openai::{Tool, ToolChoice, FunctionDefinition};

let tools = vec![
    Tool::function(
        FunctionDefinition::new("get_weather")
            .with_description("Get current weather")
            .with_parameters(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }))
    ),
];

let request = ChatCompletionRequest {
    model: "mistral-7b".into(),
    messages: vec![Message::user("What's the weather in Tokyo?")],
    tools: Some(tools),
    tool_choice: Some(ToolChoice::Mode("auto".into())),
    ..Default::default()
};

let response = adapter.chat_completion(request).await?;

// Check if model wants to call a tool
if let Some(tool_calls) = &response.choices[0].message.tool_calls {
    for tool_call in tool_calls {
        println!("Tool: {}", tool_call.function.name);
        println!("Args: {}", tool_call.function.arguments);
        
        // Execute the tool (your implementation)
        let result = execute_tool(tool_call).await?;
        
        // Send result back
        let follow_up = ChatCompletionRequest {
            messages: vec![
                // ... previous messages
                Message::assistant_with_tool_calls(tool_calls.clone()),
                Message::tool_with_name(
                    &tool_call.id,
                    &result,
                    &tool_call.function.name,
                ),
            ],
            ..request
        };
        
        let final_response = adapter.chat_completion(follow_up).await?;
    }
}
```

### MCP Server Integration

```rust
use candle_vllm::mcp::{McpClient, McpServerConfig};
use candle_vllm::openai::Tool;

// Connect to an MCP server
let mcp_client = McpClient::connect(McpServerConfig {
    url: "http://localhost:3001/mcp".into(),
    auth: None,
    timeout_secs: 30,
}).await?;

// List available tools
let mcp_tools = mcp_client.list_tools().await?;

// Convert MCP tools to OpenAI format
let tools = Tool::from_mcp_list(&mcp_tools);

// Use in chat completion
let request = ChatCompletionRequest {
    messages: vec![Message::user("Read the README file")],
    tools: Some(tools),
    tool_choice: Some(ToolChoice::Mode("auto".into())),
    ..Default::default()
};

let response = adapter.chat_completion(request).await?;

// If model calls a tool, execute via MCP
if let Some(tool_calls) = &response.choices[0].message.tool_calls {
    for tool_call in tool_calls {
        // Convert to MCP format
        let mcp_call = tool_call.to_mcp_call();
        
        // Execute via MCP server
        let result = mcp_client.call_tool(mcp_call).await?;
        
        // Convert result back to OpenAI format
        let tool_message = Message::from_mcp_result(
            &tool_call.id,
            &result,
            Some(tool_call.function.name.clone()),
        );
    }
}
```

### ResponsesSession (High-Level MCP API)

For easier multi-turn conversations with automatic tool execution:

```rust
use candle_vllm::responses::{ResponsesSession, SessionConfig, ConversationOptions};

// Create a session with MCP servers
let mut session = ResponsesSession::builder()
    .model_path("./models/mistral-7b")
    .add_mcp_server("filesystem", "http://localhost:3001/mcp", None)
    .add_mcp_server("github", "http://localhost:3002/mcp", Some("Bearer TOKEN"))
    .build()
    .await?;

// Run a conversation with automatic tool execution
let result = session.run_conversation(
    vec![Message::user("Read README.md and create a GitHub issue")],
    ConversationOptions {
        max_turns: 10,
        allowed_tools: Some(vec!["read_file".into(), "create_issue".into()]),
        ..Default::default()
    },
).await?;

println!("Final response: {}", result.final_message);
println!("Used {} tools across {} turns", 
    result.tool_calls_executed.len(), 
    result.turns_taken
);
```

---

## Streaming Support

### Basic Streaming

```rust
use futures::StreamExt;

let stream = adapter.chat_completion_stream(request).await?;

while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    if let Some(content) = &chunk.choices[0].delta.content {
        print!("{}", content);
        std::io::stdout().flush()?;
    }
}
println!();
```

### Streaming with Tool Calls

```rust
let mut stream = adapter.chat_completion_stream(request).await?;
let mut tool_calls: Vec<ToolCall> = vec![];

while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    let delta = &chunk.choices[0].delta;
    
    // Accumulate content
    if let Some(content) = &delta.content {
        print!("{}", content);
    }
    
    // Accumulate tool calls
    if let Some(tool_call_deltas) = &delta.tool_calls {
        for tc_delta in tool_call_deltas {
            // Merge deltas into tool_calls
            merge_tool_call_delta(&mut tool_calls, tc_delta);
        }
    }
    
    // Check finish reason
    if chunk.choices[0].finish_reason.is_some() {
        break;
    }
}

// Execute tools if any were called
if !tool_calls.is_empty() {
    for tool_call in &tool_calls {
        let result = execute_tool(tool_call).await?;
        // Send back and continue...
    }
}
```

---

## Error Handling

### Error Types

```rust
use candle_vllm::Error;

pub enum Error {
    /// Model loading error
    ModelLoad(String),
    
    /// Tokenization error
    Tokenization(String),
    
    /// Generation error
    Generation(String),
    
    /// Device error (CUDA/Metal)
    Device(String),
    
    /// Configuration error
    Config(String),
    
    /// MCP connection error
    McpConnection(String),
    
    /// MCP tool execution error
    McpToolExecution(String),
    
    /// Request cancelled
    Cancelled,
    
    /// IO error
    Io(std::io::Error),
    
    /// Other errors
    Other(String),
}

impl std::error::Error for Error {}
impl std::fmt::Display for Error { /* ... */ }
impl From<std::io::Error> for Error { /* ... */ }
```

### Result Type

```rust
pub type Result<T> = std::result::Result<T, Error>;
```

---

## Configuration

### EngineConfig

```rust
pub struct EngineConfig {
    /// Path to model directory or HuggingFace model ID
    pub model_path: String,
    
    /// Device to use (CUDA, Metal, CPU)
    pub device: Device,
    
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// Maximum sequence length
    pub max_sequence_length: usize,
    
    /// GPU memory for KV cache (in bytes)
    pub kv_cache_memory: usize,
    
    /// Data type (BF16, FP16, FP32)
    pub dtype: DType,
    
    /// Quantization settings
    pub quantization: Option<QuantizationConfig>,
    
    /// Enable CUDA graph optimization
    pub enable_cuda_graph: bool,
    
    /// Enable chunked prefill
    pub enable_chunked_prefill: bool,
    
    /// Chunk size for prefill
    pub prefill_chunk_size: usize,
}

impl EngineConfig {
    pub fn from_model_path(path: impl Into<String>) -> Self;
}
```

### Builder Pattern

```rust
let config = EngineConfig::builder()
    .model_path("./models/mistral-7b")
    .device(Device::Cuda(0))
    .max_batch_size(16)
    .kv_cache_memory(4 * 1024 * 1024 * 1024) // 4GB
    .dtype(DType::BF16)
    .enable_cuda_graph(true)
    .build()?;

let engine = InferenceEngine::new(config).await?;
```

---

## Examples

### Example 1: Simple Tauri Chat App

```rust
// src-tauri/src/main.rs
use candle_vllm::{InferenceEngine, EngineConfig};
use candle_vllm::openai::{OpenAIAdapter, ChatCompletionRequest, Message};
use tauri::{State, Manager};
use tokio::sync::Mutex;
use std::sync::Arc;

struct AppState {
    adapter: Arc<Mutex<OpenAIAdapter>>,
}

#[tauri::command]
async fn send_message(
    message: String,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let mut adapter = state.adapter.lock().await;
    
    let request = ChatCompletionRequest {
        model: "local".into(),
        messages: vec![Message::user(message)],
        max_tokens: Some(512),
        ..Default::default()
    };
    
    let response = adapter.chat_completion(request).await
        .map_err(|e| e.to_string())?;
    
    Ok(response.choices[0].message.content.clone().unwrap_or_default())
}

#[tokio::main]
async fn main() {
    // Initialize engine on startup
    let config = EngineConfig::builder()
        .model_path("./models/mistral-7b")
        .device(Device::best_available())
        .max_batch_size(1)
        .build()
        .unwrap();
    
    let engine = InferenceEngine::new(config).await.unwrap();
    let adapter = OpenAIAdapter::new(engine);
    
    tauri::Builder::default()
        .manage(AppState {
            adapter: Arc::new(Mutex::new(adapter)),
        })
        .invoke_handler(tauri::generate_handler![send_message])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### Example 2: Custom AI Gateway

```rust
// Custom HTTP server with your own API format
use axum::{
    routing::post,
    Router, Json, Extension,
};
use candle_vllm::openai::OpenAIAdapter;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Deserialize)]
struct MyRequest {
    prompt: String,
    options: MyOptions,
}

#[derive(Deserialize)]
struct MyOptions {
    temperature: f32,
    max_length: usize,
}

#[derive(Serialize)]
struct MyResponse {
    text: String,
    tokens_used: usize,
}

async fn custom_generate(
    Json(req): Json<MyRequest>,
    Extension(adapter): Extension<Arc<Mutex<OpenAIAdapter>>>,
) -> Json<MyResponse> {
    let mut adapter = adapter.lock().await;
    
    // Convert to OpenAI format
    let openai_req = ChatCompletionRequest {
        model: "local".into(),
        messages: vec![Message::user(req.prompt)],
        temperature: Some(req.options.temperature),
        max_tokens: Some(req.options.max_length),
        ..Default::default()
    };
    
    let response = adapter.chat_completion(openai_req).await.unwrap();
    
    // Convert back to your format
    Json(MyResponse {
        text: response.choices[0].message.content.clone().unwrap_or_default(),
        tokens_used: response.usage.total_tokens,
    })
}

#[tokio::main]
async fn main() {
    let engine = InferenceEngine::new(config).await.unwrap();
    let adapter = Arc::new(Mutex::new(OpenAIAdapter::new(engine)));
    
    let app = Router::new()
        .route("/generate", post(custom_generate))
        .layer(Extension(adapter));
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();
    
    axum::serve(listener, app).await.unwrap();
}
```

### Example 3: Agent with MCP Tools

```rust
use candle_vllm::responses::{ResponsesSession, SessionConfig};
use candle_vllm::openai::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create session with multiple MCP servers
    let mut session = ResponsesSession::builder()
        .model_path("./models/mistral-7b")
        .add_mcp_server("fs", "http://localhost:3001/mcp", None)
        .add_mcp_server("github", "http://localhost:3002/mcp", Some("Bearer TOKEN"))
        .add_mcp_server("slack", "http://localhost:3003/mcp", Some("Bearer TOKEN"))
        .build()
        .await?;
    
    // Define a complex task
    let task = "Read the README.md file, summarize it, \
                create a GitHub issue with the summary, \
                and post a message in the #dev Slack channel.";
    
    // Let the agent work through it
    let result = session.run_conversation(
        vec![Message::user(task)],
        ConversationOptions::default()
            .with_max_turns(20)
            .with_allowed_tools(vec![
                "read_file".into(),
                "create_issue".into(),
                "post_message".into(),
            ]),
    ).await?;
    
    println!("Task completed!");
    println!("Final response: {}", result.final_message);
    println!("\nTools executed:");
    for (idx, tool_call) in result.tool_calls_executed.iter().enumerate() {
        println!("  {}. {}", idx + 1, tool_call.function.name);
    }
    
    Ok(())
}
```

### Example 4: Batch Processing

```rust
use candle_vllm::{InferenceEngine, GenerationParams};
use tokio::task::JoinSet;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = InferenceEngine::new(config).await?;
    
    let prompts = vec![
        "What is Rust?",
        "Explain async/await",
        "What are lifetimes?",
        "How do I use tokio?",
    ];
    
    // Process in batches
    let mut handles = Vec::new();
    for prompt in prompts {
        let tokens = engine.tokenize(prompt)?;
        let handle = engine.add_request(InferenceRequest {
            prompt: tokens,
            params: GenerationParams::default(),
        });
        handles.push(handle);
    }
    
    // Collect results
    let mut results = Vec::new();
    loop {
        let updates = engine.step()?;
        for update in updates {
            if let RequestUpdate::Completed(output) = update {
                let text = engine.detokenize(&output.tokens)?;
                results.push(text);
            }
        }
        
        if results.len() == prompts.len() {
            break;
        }
    }
    
    for (prompt, result) in prompts.iter().zip(results.iter()) {
        println!("Q: {}", prompt);
        println!("A: {}\n", result);
    }
    
    Ok(())
}
```

---

## Thread Safety

All public types are designed to be thread-safe:

- `InferenceEngine`: Requires `&mut self` for generation (not Send)
- `OpenAIAdapter`: Wraps engine, requires `&mut self`
- `ResponsesSession`: Wraps adapter, requires `&mut self`

For multi-threaded use, wrap in `Arc<Mutex<T>>`:

```rust
use std::sync::Arc;
use tokio::sync::Mutex;

let adapter = Arc::new(Mutex::new(OpenAIAdapter::new(engine)));

// Clone and share across threads/tasks
let adapter_clone = Arc::clone(&adapter);
tokio::spawn(async move {
    let mut adapter = adapter_clone.lock().await;
    // Use adapter...
});
```

---

## Feature Flags

Control what gets compiled:

```toml
[dependencies]
candle-vllm = { version = "0.5", default-features = false, features = ["openai", "mcp"] }
```

Available features:
- `core` (default): Core inference engine
- `openai`: OpenAI compatibility layer
- `mcp`: MCP server integration
- `responses-api`: High-level Responses API
- `cuda`: CUDA support
- `metal`: Metal (Apple Silicon) support
- `flash-attn`: Flash Attention kernels
- `server`: Built-in HTTP server (optional)

---

## Performance Tips

1. **Batch requests** when possible for better throughput
2. **Use CUDA graph** for faster repeated generation patterns
3. **Enable chunked prefill** for long prompts
4. **Tune KV cache memory** based on your use case
5. **Use BF16/FP16** for better performance vs FP32
6. **Consider quantization** (Q4K, Q8) for memory-constrained environments

---

## Troubleshooting

### Common Issues

**Issue**: Model fails to load
```rust
// Solution: Check model path and format
let config = EngineConfig::builder()
    .model_path("./models/mistral-7b")  // Must contain safetensors
    .build()?;
```

**Issue**: Out of memory
```rust
// Solution: Reduce KV cache or batch size
let config = EngineConfig::builder()
    .kv_cache_memory(2 * 1024 * 1024 * 1024)  // 2GB instead of 4GB
    .max_batch_size(8)  // Reduce from 16
    .build()?;
```

**Issue**: Slow generation
```rust
// Solution: Enable optimizations
let config = EngineConfig::builder()
    .enable_cuda_graph(true)
    .enable_chunked_prefill(true)
    .build()?;
```

---

## API Stability

- **Core API** (`InferenceEngine`, `GenerationParams`): Stable, semantic versioning
- **OpenAI Layer** (`OpenAIAdapter`, types): Stable, follows OpenAI spec
- **MCP Integration** (`McpClient`, `ResponsesSession`): Evolving, may change
- **Internal APIs**: No stability guarantees

---

## Next Steps

- Read the [Architecture Document](./ARCHITECTURE.md)
- Check out [Examples](./examples/)
- Join our Discord for help
- Contribute on GitHub
