# OpenCode + Candle-vLLM (OpenAI-compatible endpoint)

This guide connects OpenCode directly to Candle-vLLM using the built-in
OpenAI-compatible `/v1/chat/completions` API. No proxy required.

```
OpenCode -> Candle-vLLM (OpenAI-compatible)
```

## 1) Start Candle-vLLM on port 8000

```bash
# Rust build (`flashinfer` is also supported)
cargo build --features cuda,nccl,graph,flashattn,cutlass --release
# Run
./target/release/candle-vllm --m Qwen/Qwen3.5-27B-FP8 --d 0 --prefix-cache --p 8000 --gpu-memory-fraction 0.7 --enforce-parser qwen_coder

# Or
cargo run --features cuda,nccl,graph,flashattn,cutlass --release -- --m Qwen/Qwen3.5-27B-FP8 --d 0 --prefix-cache --p 8000 --gpu-memory-fraction 0.7 --enforce-parser qwen_coder

# Use FlashInfer instead of FlashAttention
cargo run --features cuda,nccl,graph,flashinfer,cutlass --release -- --m Qwen/Qwen3.5-27B-FP8 --d 0 --prefix-cache --p 8000 --gpu-memory-fraction 0.7 --enforce-parser qwen_coder
```

If you are serving a different model, replace `--m` or use `--w` / `--f`.

`--gpu-memory-fraction` means "use this fraction of the GPU memory still free after the model has loaded" for KV/cache budgeting. The default is `0.7`.

## 2) Find the served model name

OpenCode should use the actual model name exposed by the server, validate if accessible.

```bash
curl http://localhost:8000/v1/models
```

Pick the `id` from the response. For a Hugging Face launch with `--m`, this is usually the model id. For a local `--w` or `--f` launch, it is typically derived from the weight folder or filename.

## 3) Configure OpenCode

Install OpenCode

```shell
curl -fsSL https://opencode.ai/install | bash
# or specific version
curl -fsSL https://opencode.ai/install | bash -s -- --version 1.2.19
```

Or install with npm

```shell
npm i -g opencode-ai
```

Create `~/.config/opencode/config.json`

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

Replace the model display name with the exact model id returned by `/v1/models` if you are serving a different model.

## 4) Run OpenCode

```shell
opencode
```

## Notes

1. Candle-vLLM follows the OpenAI tool-calling flow. When the model emits a tool call, OpenCode executes the tool and sends the tool result back in the next request.
2. For Qwen3.5 / Qwen coder models, `--enforce-parser qwen_coder` is often the most reliable setting when testing agent workflows.
3. If OpenCode reports a model selection error, verify that the configured model matches one of the ids returned by `/v1/models`.
