# OpenCode + Candle-vLLM

This guide connects OpenCode directly to `candle-vllm` through the built-in OpenAI-compatible `/v1/chat/completions` endpoint.

```
OpenCode -> Candle-vLLM (OpenAI-compatible)
```

## 1) Start candle-vLLM

```bash
cargo run --release --features cuda,nccl,graph,flashinfer,cutlass -- \
  --m Qwen/Qwen3.5-27B-FP8 \
  --d 0 \
  --prefix-cache \
  --p 8000 \
  --gpu-memory-fraction 0.7 \
  --enforce-parser qwen_coder
```

If you prefer FlashAttention, replace `flashinfer` with `flashattn`.

## 2) Discover the served model name

```bash
curl http://localhost:8000/v1/models
```

Use the returned `id` in the OpenCode config.

## 3) Configure OpenCode

Install OpenCode:

```bash
curl -fsSL https://opencode.ai/install | bash
```

Or:

```bash
npm i -g opencode-ai
```

Create `~/.config/opencode/config.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-candle-vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Candle-vLLM Local",
      "options": {
        "baseURL": "http://localhost:8000/v1"
      },
      "models": {
        "qwen3-coder": {
          "name": "Qwen/Qwen3.5-27B-FP8"
        }
      }
    }
  },
  "model": "local-candle-vllm/qwen3-coder"
}
```

## 4) Run OpenCode

```bash
opencode
```

## Notes

- Tool calls follow the normal OpenAI request/response loop.
- For Qwen coder models, `--enforce-parser qwen_coder` is usually the most
  reliable parser setting.
- If OpenCode reports a model mismatch, compare your configured model against
  `GET /v1/models`.

## Troubleshooting

Chat logger:

```bash
export CANDLE_VLLM_CHAT_LOGGER=1
```

Reasoning routing for tool-enabled requests:

```bash
export CANDLE_VLLM_STREAM_AS_REASONING_CONTENT=1
```

Set `CANDLE_VLLM_STREAM_AS_REASONING_CONTENT=0` if the client expects reasoning
to stay inside `content` instead of separate `reasoning_content` fields.
