# Using `candle-vllm` as a Rust crate

`candle-vllm` can be used as a library in your own Rust projects to perform high-performance LLM inference without running a standalone server.

## Installation

Add `candle-vllm` to your `Cargo.toml`. You must specify the Git repository (or path if local) and enable necessary features.

```toml
[dependencies]
candle-vllm = { git = "https://github.com/EricLBuehler/candle-vllm.git", features = ["cuda"] }
# Or for local development:
# candle-vllm = { path = "../candle-vllm", features = ["cuda"] }
```

### Feature Flags

You must enable at least one backend feature unless you are running on CPU (which is slow for LLMs). common features include:

- **`cuda`**: Helper to enable CUDA support (requires wrapping project to also configure CUDA).
- **`metal`**: For macOS Metal support.
- **`flash-attn`** and **`flash-decoding`**: Enables flash attention (and flash attention for decoding).

## Usage Example

Here is a complete example of how to initialize the engine and generate text.

```rust
use candle_vllm::api::{EngineBuilder, ModelRepo};
use candle_vllm::openai::requests::{ChatCompletionRequest, Messages};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Configure the engine
    // You can specify a model from HF Hub, a local path, or a GGUF file.
    let builder = EngineBuilder::new(ModelRepo::ModelID(("Qwen/Qwen3-0.6B", None)))
        .with_kvcache_mem_gpu(1096) // 1GB KV Cache
        .with_temperature(0.7)
        .with_top_p(0.95);

    // 2. Build the engine
    // This initializes the model, weights, and background processing threads.
    let engine = builder.build_async().await?;

    // 3. Create a request
    let request = ChatCompletionRequest {
        model: "default".to_string(), // Model name is internal reference
        messages: Messages::Map(vec![
            std::collections::HashMap::from([
                ("role".to_string(), "user".to_string()),
                ("content".to_string(), "Tell me a joke about Rust programming.".to_string()),
            ])
        ]),
        max_tokens: Some(100),
        ..Default::default()
    };

    // 4. Generate response
    let response = engine.generate_request(request).await?;
    println!("Response: {:?}", response);

    // 5. Clean shutdown
    engine.shutdown();

    Ok(())
}
```

## Configuration

The `EngineBuilder` provides a fluent API to configure:

- **Model Source**: `ModelRepo::ModelID` (HF Hub), `ModelRepo::ModelPath` (Local), `ModelRepo::ModelFile` (GGUF).
- **Compute Resources**: `.with_device_ids()`, `.with_kvcache_mem_gpu()`, `.with_kvcache_mem_cpu()`.
- **Generation Defaults**: `.with_temperature()`, `.with_top_p()`, etc.
- **Advanced**: `.with_dtype()`, `.with_isq()` (Quantization), `.without_flash_attn()`.

## Standalone Binary Configuration

If you are building a standalone binary that uses `candle-vllm`, ensure your `Cargo.toml` configures the necessary conflicting features correctly.

For example, if you use `cuda`, your `Cargo.toml` might look like:

```toml
[package]
name = "my-inference-app"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-vllm = { git = "https://github.com/EricLBuehler/candle-vllm.git", features = ["cuda"] }
tokio = { version = "1.32", features = ["full"] }
anyhow = "1.0"
```

Then run with:

```bash
cargo run --features candle-vllm/cuda
```

Or just defaults if you enabled it in dependencies.

## Run built-in [example](../examples/simple_gen.rs):

```bash
cd candle-vllm
cargo run --release --features cuda --example simple_gen
```