# Embedding Support

`candle-vllm` supports an OpenAI-compatible `/v1/embeddings` endpoint. This allows you to generate vector embeddings for input text using supported LLM architectures.

## Supported Models

Currently, the following model architectures support embedding generation:
- **Llama** (e.g., Llama 2, Llama 3)
- **Mistral** (e.g., Mistral 7B)
- **Phi-3** (e.g., Phi-3-mini)
- **Gemma** (e.g., Gemma 2B/7B)
- **Qwen** (e.g., Qwen1.5, Qwen2)
- **Qwen3MoE**
- **Quantized (GGUF)**:
    - **Llama**
    - **Phi-3**
    - **Qwen**
    - **Qwen3MoE**
    - **GLM4**

attempting to use other models for embeddings will result in an error.
## Run embedding model

Run supported models as usual
```shell
cargo run --release --features cuda -- --p 8000 --m Qwen/Qwen3-0.6B
```

## Endpoint

**POST** `/v1/embeddings`

### Request Body

```json
{
  "input": "Your text here",
  "model": "model-id",
  "encoding_format":"float"
}
```

- `input`: The text string or array of tokens (currently only single string supported) to embed.
- `model`: (Optional) The ID of the model to use.

### Response

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [ ... vector of floats ... ],
      "index": 0
    }
  ],
  "model": "model-id",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

## Internal Implementation Details

- **Pooling**: Currently, embedding vectors are the hidden states of the **mean of all tokens**. This behavior is typical for many decoder-only embedding models.
- **Inference**: Embedding requests are processed by the same `LLMEngine` as generation requests. They are treated as prefill-only tasks with `max_tokens=0`, returning the hidden states instead of logits.

## Usage Example

You can use standard OpenAI clients or `curl`:

```bash
curl http://localhost:2000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "model": "default",
    "encoding_format":"float"
  }'
```

Or using the Rust API:

```rust
let engine = builder.build()?;

let input = "Hello, world!";
println!("Embedding input: {}", input);

let request = EmbeddingRequest {
    model: Some("default".to_string()),
    input: EmbeddingInput::String(input.to_string()),
    encoding_format: Default::default(),
    embedding_type: Default::default(),
};

let response = engine.embed(request)?;
println!("Response object: {:?}", response.object);
engine.shutdown();
```
