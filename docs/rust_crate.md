# Rust Crate Usage

`candle-vllm` can be used as a Rust library for model loading, text generation, embeddings, and serving the OpenAI-compatible API.

## Add the dependency

```toml
[dependencies]
candle-vllm = { path = "../candle-vllm", features = ["cuda"] }
tokio = { version = "1", features = ["full"] }
```

Use the same backend features you would use for the CLI, such as `cuda`, `metal`, `flashattn`, `flashinfer`, or `nccl`.

## Text generation example

```rust
use candle_vllm::api::{EngineBuilder, ModelRepo};
use candle_vllm::openai::requests::{ChatCompletionRequest, Messages};

#[tokio::main]
async fn main() -> candle_core::Result<()> {
    let engine = EngineBuilder::new(ModelRepo::ModelID(("Qwen/Qwen3-0.6B", None)))
        .build_async()
        .await?;

    let request = ChatCompletionRequest {
        model: Some("default".to_string()),
        messages: Messages::Map(vec![std::collections::HashMap::from([
            ("role".to_string(), "user".to_string()),
            ("content".to_string(), "Say hello from the Rust API.".to_string()),
        ])]),
        max_tokens: Some(64),
        ..Default::default()
    };

    let response = engine.generate_request(request).await?;
    println!("{response:?}");
    engine.shutdown();
    Ok(())
}
```

## Embeddings

The Rust API also exposes embeddings through the same engine:

```rust
use candle_vllm::openai::requests::{EmbeddingInput, EmbeddingRequest};

let request = EmbeddingRequest {
    model: Some("default".to_string()),
    input: EmbeddingInput::String("hello world".to_string()),
    encoding_format: Default::default(),
    embedding_type: Default::default(),
};

let response = engine.embed(request)?;
```

## Builder configuration

Useful `EngineBuilder` controls include:

- model source: `ModelRepo::ModelID`, `ModelRepo::ModelPath`, `ModelRepo::ModelFile`
- KV cache sizing: `.with_kvcache_mem_gpu()`, `.with_kvcache_mem_cpu()`
- memory budgeting: `.with_gpu_memory_fraction()`
- sampling defaults: `.with_temperature()`, `.with_top_p()`
- device selection: `.with_device_ids()`
- dtype and quantization: `.with_dtype()`, `.with_isq()`, `.with_fp8_kvcache()`

## Serving HTTP

The current Rust crate API is focused on local generation and embeddings. For the OpenAI-compatible HTTP server, use the `candle-vllm` binary entrypoint from the CLI.

## Running examples

```bash
cargo run --release --features cuda --example simple_gen
```
