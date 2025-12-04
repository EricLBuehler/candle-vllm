# AI Gateway Example

This example demonstrates how to build an AI gateway that routes OpenAI-compatible requests to multiple backend models.

## Features

- **Multi-model routing**: Route requests to different models based on the `model` field
- **Model aliasing**: Map OpenAI model names (e.g., `gpt-3.5-turbo`) to local models
- **Backend management**: Enable/disable backends, set priorities
- **OpenAI-compatible API**: Drop-in replacement for OpenAI API clients

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

The server will start on `http://localhost:8080`.

## API Endpoints

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint.

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### GET /v1/models

List available models and their backends.

```bash
curl http://localhost:8080/v1/models
```

### GET /health

Health check with request statistics.

```bash
curl http://localhost:8080/health
```

## Model Aliases

The gateway maps common model names to available backends:

| Request Model | Backend |
|---------------|---------|
| gpt-3.5-turbo | mistral-7b |
| gpt-4 | llama-3-8b |
| gpt-4-turbo | llama-3-8b |
| claude-3-opus | llama-3-8b |
| claude-3-sonnet | mistral-7b |

## Architecture

```
┌─────────────────────────────────────────┐
│           AI Gateway                     │
│  ┌─────────────────────────────────┐    │
│  │     Request Router              │    │
│  │  - Model resolution             │    │
│  │  - Alias mapping                │    │
│  │  - Load balancing               │    │
│  └─────────────────────────────────┘    │
│              │                          │
│    ┌─────────┼─────────┐                │
│    ▼         ▼         ▼                │
│ ┌──────┐ ┌──────┐ ┌──────┐              │
│ │Mistral│ │Llama │ │Qwen  │  Backends   │
│ │  7B  │ │ 3 8B │ │  7B  │              │
│ └──────┘ └──────┘ └──────┘              │
└─────────────────────────────────────────┘
```

## Extending

To enable real model inference:

1. Load models using `candle-vllm-core`
2. Create `OpenAIAdapter` instances for each backend
3. Replace the mock response with actual inference calls

Example:

```rust
use candle_vllm_openai::OpenAIAdapter;
use candle_vllm_core::InferenceEngineBuilder;

// Load model
let engine = InferenceEngineBuilder::new()
    .model_path("mistralai/Mistral-7B-Instruct-v0.3")
    .build()
    .await?;

// Create adapter
let adapter = OpenAIAdapter::new(engine);

// Use in request handler
let response = adapter.chat_completion(request).await?;
```

