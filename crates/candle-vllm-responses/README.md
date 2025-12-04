# candle-vllm-responses

High-level API for multi-turn agentic conversations with MCP (Model Context Protocol) integration.

## Features

- Multi-turn conversation management
- Automatic tool execution
- MCP server integration (loadable via `ResponsesSession::from_config_*` or server-side `CANDLE_VLLM_MCP_CONFIG`)
- Tool call routing and result injection
- Conversation result tracking

## Usage

```rust
use candle_vllm_responses::{ResponsesSession, ResponsesSessionBuilder, ConversationOptions};
use candle_vllm_core::{InferenceEngine, EngineConfig};
use candle_core::Device;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create inference engine
    let engine = InferenceEngine::builder()
        .model_path("mistralai/Mistral-7B-Instruct-v0.3")
        .device(Device::Cpu)
        .build()
        .await?;

    // Build session with engine and MCP servers
    let mut session = ResponsesSessionBuilder::new()
        .engine(engine)
        .add_mcp_server(
            "weather".to_string(),
            candle_vllm_responses::mcp_client::McpServerConfig {
                url: "http://localhost:8080".to_string(),
                auth: None,
                timeout_secs: Some(30),
            },
        )
        .build()
        .await?;

    // Run conversation with automatic tool execution
    let options = ConversationOptions {
        max_turns: 10,
        allowed_tools: None,
    };

    let messages = vec![
        candle_vllm_core::openai::requests::ChatMessage {
            role: "user".to_string(),
            content: Some("What's the weather in San Francisco?".to_string()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    ];

    let result = session.run_conversation(messages, options).await?;
    println!("Final message: {}", result.final_message);
    println!("Tool calls made: {}", result.tool_calls.len());

    Ok(())
}
```

## API Overview

- `ResponsesSession`: Manages conversations and MCP clients
- `ResponsesSessionBuilder`: Builder for configuring sessions
- `ConversationOptions`: Options for conversation behavior
- `ConversationResult`: Result of a multi-turn conversation
- `Orchestrator`: Routes and executes tool calls

See the [API documentation](https://docs.rs/candle-vllm-responses) and the SDK guide ([`docs/SDK.md`](../../docs/SDK.md)) for detailed integration patterns (Axum, Tauri, Electron, Cherry Studio, etc.).

