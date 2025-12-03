# candle-vllm-openai

OpenAI-compatible API adapter for candle-vllm. Provides a drop-in replacement for OpenAI's chat completion API.

## Features

- OpenAI-compatible request/response formats
- Chat completion API
- Tool calling support
- Streaming support with incremental tool-call deltas
- Conversation management

## Usage

```rust
use candle_vllm_openai::{OpenAIAdapter, ChatCompletionRequest};
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

    // Create OpenAI adapter
    let mut adapter = OpenAIAdapter::new(engine);

    // Create chat completion request
    let request = ChatCompletionRequest {
        model: "local".to_string(),
        messages: candle_vllm_core::openai::requests::Messages::Chat(vec![
            candle_vllm_core::openai::requests::ChatMessage {
                role: "user".to_string(),
                content: Some("Hello!".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }
        ]),
        max_tokens: Some(100),
        temperature: Some(0.7),
        ..Default::default()
    };

    // Get completion
    let response = adapter.chat_completion(request).await?;
    println!("Response: {:?}", response.choices[0].message.content);

    Ok(())
}
```

## API Overview

- `OpenAIAdapter`: Main adapter struct wrapping `InferenceEngine`
- `ChatCompletionRequest`: OpenAI-compatible request format
- `ChatCompletionResponse`: OpenAI-compatible response format

See the [API documentation](https://docs.rs/candle-vllm-openai) and the integration guide in [`docs/SDK.md`](../../docs/SDK.md) for full details, including Axum/Tauri/Electron usage patterns.

