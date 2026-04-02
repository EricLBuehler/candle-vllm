# Multimodal Model Usage

`candle-vllm` supports vision-language requests on the OpenAI-compatible `/v1/chat/completions` endpoint with mixed text and image content.

Currently implemented model families include:

- Qwen3-VL
- Gemma3 vision variants
- Mistral3-VL

## Start the server

Qwen3-VL example:

```bash
cargo run --release --features cuda -- \
  --m Qwen/Qwen3-VL-8B-Instruct \
  --p 8000 \
  --prefix-cache
```

Gemma3 vision example:

```bash
cargo run --release --features cuda -- \
  --m google/gemma-3-4b-it \
  --p 8000 \
  --prefix-cache
```

## Request payloads

Text + image URL:

```json
{
  "model": "default",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": "https://example.com/cat.png"}
      ]
    }
  ]
}
```

Text + base64 image:

```json
{
  "model": "default",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_base64", "image_base64": "data:image/png;base64,..."}
      ]
    }
  ]
}
```

## Notes

- Image content is only supported on multimodal models.
- Direct URL fetching and base64 decoding are both supported.
- Large images and many images increase prefill time and KV usage.
- Prefix cache can still help for repeated multimodal prompts, but image-heavy
  requests remain prefill-dominated.
