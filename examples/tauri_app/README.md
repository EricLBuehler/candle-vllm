# Tauri App Example for candle-vllm

This example demonstrates how to use candle-vllm as a library in an embedded context, such as a Tauri desktop application, **without requiring an HTTP server**.

## Library-First Design

The candle-vllm project supports a "library-first" architecture where you can:

1. **Use OpenAI-compatible types** directly from `candle-vllm-openai`
2. **Run inference** without starting an HTTP server
3. **Embed in desktop apps** like Tauri, Electron (via NAPI), or native GUI frameworks
4. **Stream responses** via channels instead of HTTP SSE

## Building

### For NVIDIA GPUs (CUDA)

```bash
cargo build --release --features cuda
```

### For Apple Silicon (Metal)

```bash
cargo build --release --features metal
```

### CPU Only

```bash
cargo build --release
```

## Running

```bash
cargo run --release --features metal  # or cuda
```

## Key Concepts

### 1. OpenAI-Compatible Types

```rust
use candle_vllm_openai::{
    ChatCompletionRequest, ChatMessage, Messages, Tool,
    ChatCompletionResponse, APIError,
};

// Build a request
let request = ChatCompletionRequest {
    model: "mistral".to_string(),
    messages: Messages::Chat(vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("Hello!"),
    ]),
    temperature: Some(0.7),
    max_tokens: Some(100),
    ..Default::default()
};
```

### 2. Tool Calling

```rust
use candle_vllm_openai::{Tool, FunctionDefinition, get_tool_parser};

// Define tools
let tool = Tool::function(
    FunctionDefinition::new("get_weather")
        .with_description("Get weather for a location")
        .with_parameters(serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }))
);

// Parse tool calls from model output
let parser = get_tool_parser("mistral");
let parsed = parser.parse(model_output);
```

### 3. Conversation Management

```rust
use candle_vllm_openai::conversation::ToolConversationBuilder;

let mut builder = ToolConversationBuilder::new("mistral");
builder
    .add_user_message("What's the weather?")
    .add_assistant_tool_calls(vec![/* tool calls */])
    .add_tool_result("call_123", r#"{"temp": 20}"#, Some("get_weather".to_string()))
    .add_assistant_response("It's 20°C.");
```

### 4. Streaming Responses

In a Tauri app, use channels for streaming instead of HTTP SSE:

```rust
use tokio::sync::mpsc;

let (tx, rx) = mpsc::channel(32);

// Send chunks to frontend
for chunk in response_stream {
    tx.send(chunk).await?;
}

// In Tauri, emit events instead:
// app.emit_all("chat-chunk", chunk)?;
```

## Integration with Tauri

To use this in a real Tauri application:

1. **Add Tauri dependencies**:
   ```toml
   [dependencies]
   tauri = { version = "1.5", features = [] }
   ```

2. **Create Tauri commands**:
   ```rust
   #[tauri::command]
   async fn chat(
       state: tauri::State<'_, AppState>,
       message: String,
   ) -> Result<String, String> {
       // Use candle-vllm-openai types here
   }
   ```

3. **Stream via events**:
   ```rust
   #[tauri::command]
   async fn chat_stream(
       app: tauri::AppHandle,
       message: String,
   ) -> Result<(), String> {
       for chunk in response_stream {
           app.emit_all("chat-chunk", &chunk)?;
       }
       Ok(())
   }
   ```

4. **Manage model lifecycle**:
   ```rust
   fn main() {
       tauri::Builder::default()
           .manage(AppState::new())
           .invoke_handler(tauri::generate_handler![chat, chat_stream])
           .run(tauri::generate_context!())
           .expect("error while running tauri application");
   }
   ```

## File Structure

```
examples/tauri_app/
├── Cargo.toml          # Rust dependencies
├── src/
│   └── main.rs         # Example implementation
└── README.md           # This file
```

## Benefits of Library-First Approach

1. **No HTTP overhead** - Direct function calls instead of network requests
2. **Type safety** - Full Rust type checking across the stack
3. **Smaller binary** - No HTTP server dependencies needed
4. **Better integration** - Native OS features accessible
5. **Offline capable** - No network requirements for inference

## Related Crates

- `candle-vllm-core` - Core inference engine
- `candle-vllm-openai` - OpenAI-compatible types and adapters
- `candle-vllm-responses` - MCP integration for multi-turn conversations

