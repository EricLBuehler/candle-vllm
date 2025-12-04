# Quickstart Guide: candle-vllm Library

**Branch**: `001-library-api-implementation`  
**Date**: December 3, 2025

## Overview

This guide shows how to use candle-vllm as a library in your Rust applications. The library provides three levels of abstraction:

1. **Core Engine** (`candle-vllm-core`): Low-level inference with tokenization and generation
2. **OpenAI Adapter** (`candle-vllm-openai`): OpenAI-compatible chat completion API
3. **Responses Session** (`candle-vllm-responses`): High-level agent conversations with MCP tools

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
# Core only (minimal)
candle-vllm-core = "0.5"

# With OpenAI compatibility
candle-vllm-openai = "0.5"

# Full stack with MCP support
candle-vllm-responses = "0.5"

# Required for async
tokio = { version = "1", features = ["full"] }
```

### Feature Flags

```toml
[dependencies.candle-vllm-core]
version = "0.5"
features = ["metal"]  # or "cuda" for NVIDIA GPUs
```

Available features:
- `cuda` - NVIDIA GPU support
- `metal` - Apple Silicon GPU support
- `flash-attn` - Flash Attention kernels (CUDA only)
- `flash-decoding` - Flash Decoding optimization

## Quick Examples

### 1. Basic Text Generation (Core)

```rust
use candle_vllm_core::{InferenceEngine, EngineConfig, GenerationParams};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize engine
    let engine = InferenceEngine::builder()
        .model_path("mistralai/Mistral-7B-Instruct-v0.3")
        .device(Device::best_available())
        .max_batch_size(1)
        .kv_cache_memory(4 * 1024 * 1024 * 1024) // 4GB
        .build()
        .await?;

    // Generate text
    let prompt = engine.tokenize("Hello, how are you?")?;
    let output = engine.generate(prompt, GenerationParams::default()).await?;
    let text = engine.detokenize(&output.tokens)?;

    println!("Generated: {}", text);
    Ok(())
}
```

### 2. OpenAI-Compatible Chat (OpenAI Adapter)

```rust
use candle_vllm_openai::{OpenAIAdapter, ChatCompletionRequest, Message};
use candle_vllm_core::{InferenceEngine, EngineConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize
    let engine = InferenceEngine::builder()
        .model_path("mistralai/Mistral-7B-Instruct-v0.3")
        .build()
        .await?;
    let mut adapter = OpenAIAdapter::new(engine);

    // Chat completion
    let request = ChatCompletionRequest {
        model: "mistral-7b".into(),
        messages: vec![
            Message::system("You are a helpful assistant."),
            Message::user("What is Rust?"),
        ],
        temperature: Some(0.7),
        max_tokens: Some(256),
        ..Default::default()
    };

    let response = adapter.chat_completion(request).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
```

### 3. Tool Calling

```rust
use candle_vllm_openai::{
    OpenAIAdapter, ChatCompletionRequest, Message, 
    Tool, FunctionDefinition,
};
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut adapter = /* ... initialize ... */;

    // Define tools
    let tools = vec![
        Tool::function(
            FunctionDefinition::new("get_weather")
                .with_description("Get current weather")
                .with_parameters(json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }))
        ),
    ];

    // Request with tools
    let request = ChatCompletionRequest {
        model: "mistral-7b".into(),
        messages: vec![Message::user("What's the weather in Tokyo?")],
        tools: Some(tools),
        tool_choice: Some(ToolChoice::Auto),
        ..Default::default()
    };

    let response = adapter.chat_completion(request).await?;

    // Check for tool calls
    if let Some(tool_calls) = &response.choices[0].message.tool_calls {
        for tool_call in tool_calls {
            println!("Tool: {}", tool_call.function.name);
            println!("Args: {}", tool_call.function.arguments);
            
            // Execute tool and continue conversation...
        }
    }
    Ok(())
}
```

### 4. Streaming

```rust
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut adapter = /* ... initialize ... */;

    let request = ChatCompletionRequest {
        model: "mistral-7b".into(),
        messages: vec![Message::user("Tell me a story.")],
        stream: Some(true),
        ..Default::default()
    };

    let mut stream = adapter.chat_completion_stream(request).await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(content) = &chunk.choices[0].delta.content {
            print!("{}", content);
            std::io::stdout().flush()?;
        }
    }
    println!();
    Ok(())
}
```

### 5. MCP Server Integration

```rust
use candle_vllm_responses::{ResponsesSession, ConversationOptions, Message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create session with MCP servers
    let mut session = ResponsesSession::builder()
        .model_path("mistralai/Mistral-7B-Instruct-v0.3")
        .add_mcp_server("filesystem", "http://localhost:3001/mcp", None)
        .add_mcp_server("github", "http://localhost:3002/mcp", Some("Bearer TOKEN"))
        .build()
        .await?;

    // Run multi-turn conversation with automatic tool execution
    let result = session.run_conversation(
        vec![Message::user("Read the README.md file and summarize it.")],
        ConversationOptions {
            max_turns: 10,
            allowed_tools: Some(vec!["read_file".into()]),
            ..Default::default()
        },
    ).await?;

    println!("Final: {}", result.final_message);
    println!("Tools used: {}", result.tool_calls_executed.len());
    println!("Turns: {}", result.turns_taken);
    Ok(())
}
```

## Configuration Reference

### EngineConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_path` | String | required | HuggingFace model ID or local path |
| `device` | Device | auto | `Device::Cuda(0)`, `Device::Metal`, `Device::Cpu` |
| `dtype` | DType | BF16 | `DType::BF16`, `DType::F16`, `DType::F32` |
| `max_batch_size` | usize | 16 | Maximum concurrent requests |
| `kv_cache_memory` | usize | 4GB | GPU memory for KV cache |
| `enable_cuda_graph` | bool | false | Enable CUDA graph optimization |
| `enable_chunked_prefill` | bool | false | Enable chunked prefill |
| `prefill_chunk_size` | usize | 1024 | Chunk size (must be divisible by 1024) |

### GenerationParams

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_tokens` | usize | 256 | Maximum tokens to generate |
| `temperature` | f32 | 1.0 | Sampling temperature (0 = greedy) |
| `top_p` | f32 | 1.0 | Nucleus sampling threshold |
| `top_k` | Option<usize> | None | Top-k sampling |
| `repetition_penalty` | f32 | 1.0 | Penalty for repeated tokens |
| `frequency_penalty` | f32 | 0.0 | Frequency-based penalty |
| `presence_penalty` | f32 | 0.0 | Presence-based penalty |
| `stop_sequences` | Vec<String> | [] | Stop generation on these sequences |
| `seed` | Option<u64> | None | Random seed |

## Thread Safety

The library uses `&mut self` for generation methods, so wrap in `Arc<Mutex>` for concurrent access:

```rust
use std::sync::Arc;
use tokio::sync::Mutex;

let adapter = Arc::new(Mutex::new(OpenAIAdapter::new(engine)));

// In async handlers:
let adapter_clone = Arc::clone(&adapter);
tokio::spawn(async move {
    let mut adapter = adapter_clone.lock().await;
    let response = adapter.chat_completion(request).await?;
    // ...
});
```

## Error Handling

```rust
use candle_vllm_core::Error;

match engine.generate(prompt, params).await {
    Ok(output) => println!("Success: {:?}", output),
    Err(Error::ModelLoad(msg)) => eprintln!("Model load failed: {}", msg),
    Err(Error::Device(msg)) => eprintln!("Device error: {}", msg),
    Err(Error::Generation(msg)) => eprintln!("Generation failed: {}", msg),
    Err(Error::Cancelled) => eprintln!("Request cancelled"),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Running the HTTP Server

If you don't need to embed the library, you can still run the standalone server:

```bash
# Build
cargo build --release --features metal  # or cuda

# Run
./target/release/candle-vllm \
    -m mistralai/Mistral-7B-Instruct-v0.3 \
    --mem 4096 \
    --port 2000
```

The server exposes the same OpenAI-compatible API at `http://localhost:2000/v1/`.

## Next Steps

- See [LIBRARY_API.md](../../LIBRARY_API.md) for full API documentation
- Check [examples/](../../examples/) for complete applications
- Read [ARCHITECTURE.md](../../ARCHITECTURE.md) for system design

