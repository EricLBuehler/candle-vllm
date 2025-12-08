# Agent Framework Example

This example demonstrates how to build an AI agent with tool calling capabilities using candle-vllm.

## Features

- **Tool definitions**: Define functions the model can call
- **Tool parsing**: Extract tool calls from model output
- **Agent loop**: Execute tools and continue conversation
- **Multi-turn handling**: Maintain context across iterations
- **Model-agnostic**: Works with Mistral, Llama, Qwen, and other models

## Building

```bash
# CPU only
cargo build --release

# For NVIDIA GPUs
cargo build --release --features cuda

# For Apple Silicon
cargo build --release --features metal
```

## Running

```bash
cargo run --release
```

## Available Tools

### get_weather

Get current weather for a location.

```json
{
  "name": "get_weather",
  "arguments": {
    "location": "Paris, France",
    "unit": "celsius"
  }
}
```

### calculator

Perform mathematical calculations.

```json
{
  "name": "calculator",
  "arguments": {
    "expression": "15 * 7 + 23"
  }
}
```

### search

Search for information.

```json
{
  "name": "search",
  "arguments": {
    "query": "Rust programming language"
  }
}
```

## Agent Loop

```
┌─────────────────────────────────────────────┐
│                 Agent Loop                   │
├─────────────────────────────────────────────┤
│                                             │
│   User Input                                │
│       │                                     │
│       ▼                                     │
│   ┌─────────┐                              │
│   │  Model  │ ◄──────┐                     │
│   └────┬────┘        │                     │
│        │             │                     │
│        ▼             │                     │
│   ┌─────────────┐    │                     │
│   │ Parse Output │    │                     │
│   └──────┬──────┘    │                     │
│          │           │                     │
│    ┌─────┴─────┐     │                     │
│    │           │     │                     │
│    ▼           ▼     │                     │
│ ┌──────┐   ┌──────┐  │                     │
│ │ Text │   │ Tool │  │                     │
│ │      │   │Calls │  │                     │
│ └──┬───┘   └──┬───┘  │                     │
│    │          │      │                     │
│    │          ▼      │                     │
│    │    ┌─────────┐  │                     │
│    │    │ Execute │  │                     │
│    │    │  Tools  │  │                     │
│    │    └────┬────┘  │                     │
│    │         │       │                     │
│    │         ▼       │                     │
│    │    ┌─────────┐  │                     │
│    │    │ Add to  │──┘                     │
│    │    │  Conv   │                        │
│    │    └─────────┘                        │
│    │                                       │
│    ▼                                       │
│  Response                                  │
│                                            │
└────────────────────────────────────────────┘
```

## Integration with Real Models

To use with actual model inference:

```rust
use candle_vllm_openai::{OpenAIAdapter, ChatCompletionRequest, Messages, Tool};
use candle_vllm_core::InferenceEngineBuilder;

// 1. Build the inference engine
let engine = InferenceEngineBuilder::new()
    .model_path("mistralai/Mistral-7B-Instruct-v0.3")
    .build()
    .await?;

// 2. Create the adapter
let mut adapter = OpenAIAdapter::new(engine);

// 3. Define tools
let tools = vec![
    Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: Some("Get weather for a location".to_string()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            })),
            strict: None,
        },
    },
];

// 4. Make request with tools
let request = ChatCompletionRequest {
    model: "mistral".to_string(),
    messages: Messages::Chat(vec![
        ChatMessage::user("What's the weather in Paris?".to_string())
    ]),
    tools: Some(tools),
    ..Default::default()
};

// 5. Get response and check for tool calls
let response = adapter.chat_completion(request).await?;

if let Some(tool_calls) = response.choices[0].message.tool_calls {
    // Execute tools and continue conversation
    for call in tool_calls {
        let result = execute_tool(&call.function.name, &call.function.arguments)?;
        // Add tool result to conversation...
    }
}
```

## MCP Integration

For MCP (Model Context Protocol) integration, use the `candle-vllm-responses` crate:

```rust
use candle_vllm_responses::McpClient;

// Connect to MCP server
let mcp_client = McpClient::connect("ws://localhost:8080/mcp").await?;

// Get tools from MCP server
let tools = mcp_client.list_tools().await?;

// Execute tool via MCP
let result = mcp_client.call_tool("get_weather", args).await?;
```

## Testing

Run the tests:

```bash
cargo test
```

