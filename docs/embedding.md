# Embedding Usage

`candle-vllm` exposes an OpenAI-compatible `POST /v1/embeddings` endpoint for text-capable models that implement embedding forward passes in this repo.

## Start the server

CUDA example:

```bash
cargo run --release --features cuda -- --m Qwen/Qwen3-0.6B --p 8000
```

Metal example:

```bash
cargo run --release --features metal -- --m google/gemma-3-4b-it --p 8000
```

## Request examples

Float embeddings with mean pooling:

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"hello world","model":"default","embedding_type":"mean","encoding_format":"float"}'
```

Base64 embeddings with last-token pooling:

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":["hello","hola"],"embedding_type":"last","encoding_format":"base64"}'
```

## Request fields

- `input`: a string or string array.
- `model`: optional; `default` resolves to the loaded model.
- `embedding_type`: `mean` or `last`.
- `encoding_format`: `float` or `base64`.

## Notes

- Embedding requests use the same tokenizer and context limits as chat requests.
- `embedding_type=mean` averages token hidden states.
- `embedding_type=last` returns the final token hidden state.
- Responses follow the OpenAI schema: `data[].embedding` and `usage.prompt_tokens`.
- Unsupported architectures return an error instead of silently falling back.

## Rust API example

```rust
use candle_vllm::api::{EngineBuilder, ModelRepo};
use candle_vllm::openai::requests::{EmbeddingInput, EmbeddingRequest};

#[tokio::main]
async fn main() -> candle_core::Result<()> {
    let engine = EngineBuilder::new(ModelRepo::ModelID(("Qwen/Qwen3-0.6B", None)))
        .build_async()
        .await?;

    let request = EmbeddingRequest {
        model: Some("default".to_string()),
        input: EmbeddingInput::String("hello world".to_string()),
        encoding_format: Default::default(),
        embedding_type: Default::default(),
    };

    let response = engine.embed(request)?;
    println!("embeddings: {}", response.data.len());
    engine.shutdown();
    Ok(())
}
```
