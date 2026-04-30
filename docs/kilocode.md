# Kilo Code + Candle-vLLM

This guide connects Kilo Code to the built-in OpenAI-compatible `/v1/chat/completions` endpoint exposed by `candle-vllm`.

```
Kilo Code -> Candle-vLLM (OpenAI-compatible)
```

## 1) Start candle-vLLM (at port 8000)

```bash
cargo run --release --features cuda,nccl,graph,flashinfer,cutlass -- \
  --m Qwen/Qwen3.6-27B-FP8 \
  --d 0 \
  --prefix-cache \
  --p 8000 \
  --gpu-memory-fraction 0.5 \
  --enforce-parser qwen_coder
```

If you prefer FlashAttention, replace `flashinfer` with `flashattn`.

## 2) Configure Kilo Code

Install Kilo Code:

```bash
npm install -g @kilocode/cli
```

Create `~/.config/kilo/config.json`:

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
          "name": "Qwen/Qwen3.6-27B-FP8"
        }
      }
    }
  },
  "model": "local-candle-vllm/qwen3-coder"
}
```

Use the exact model id returned by `GET /v1/models` if you are serving a different model.

## 3) Run Kilo Code

```bash
kilo
```

## Notes

- Tool calls follow the standard OpenAI request/response flow.
- For Qwen coder models, `--enforce-parser qwen_coder` is usually the most reliable setting.

## Troubleshooting

Use the built-in chat logger when debugging client/server interaction:

```bash
export CANDLE_VLLM_CHAT_LOGGER=1
```

Logs are written under `./log/`.
