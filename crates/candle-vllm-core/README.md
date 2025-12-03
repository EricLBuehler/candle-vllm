# candle-vllm-core

Core inference engine for candle-vllm. Provides the foundational API for model loading, tokenization, and text generation.

## Features

- Model loading and management
- Tokenization and detokenization
- Text generation with configurable parameters
- Streaming support with incremental tool-call deltas
- KV cache management
- Multi-device support (CPU, CUDA, Metal)

## Usage

```rust
use candle_vllm_core::{InferenceEngine, EngineConfig, GenerationParams};
use candle_core::Device;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Build engine configuration
    let config = EngineConfig::builder()
        .model_path("mistralai/Mistral-7B-Instruct-v0.3")
        .device(Device::Cpu)
        .max_batch_size(16)
        .kv_cache_memory(4 * 1024 * 1024 * 1024) // 4GB
        .build()?;

    // Create engine
    let engine = InferenceEngine::new(config).await?;

    // Tokenize input
    let tokens = engine.tokenize("Hello, world!")?;

    // Generate
    let params = GenerationParams {
        max_tokens: Some(100),
        temperature: Some(0.7),
        ..Default::default()
    };
    let output = engine.generate(tokens, params).await?;

    // Detokenize output
    let text = engine.detokenize(&output.tokens)?;
    println!("Generated: {}", text);

    Ok(())
}
```

## API Overview

- `InferenceEngine`: Main engine struct for inference operations
- `EngineConfig`: Configuration builder for engine setup
- `GenerationParams`: Parameters controlling generation behavior
- `GenerationOutput`: Result containing tokens, finish reason, and stats

See the [API documentation](https://docs.rs/candle-vllm-core) and [`docs/SDK.md`](../../docs/SDK.md) for full integration details, including end-to-end examples across desktop (Tauri/Electron) and Axum web environments.

