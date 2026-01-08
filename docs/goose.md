# Goose + Candle-vLLM (OpenAI-compatible endpoint)

This guide connects Goose (Rust `AI Agent`) directly to Candle-vLLM using the built-in OpenAI-compatible `/v1/chat/completions` API. No proxy required.

```
Goose -> Candle-vLLM (OpenAI-compatible)
```

## 1) Start Candle-vLLM on port 8000

```bash
# Rust
cargo build --features cuda,nccl,graph,flash-attn,flash-decoding --release
./target/release/candle-vllm --m Qwen/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --server --prefix-cache --p 8000
# Or
cargo run --features cuda,nccl,graph,flash-attn,flash-decoding --release -- --m Qwen/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --ui-server --prefix-cache --p 8000
```

## 2) Configure Goose (e.g., CLI)

### Download and install Goose: https://block.github.io/goose/docs/getting-started/installation/

```shell
# For non-UI system,
export GOOSE_DISABLE_KEYRING=1
```

Export empty API KEY

```shell
export VLLM_API_KEY="empty"
```


### Configure goose with `Custom Providers` and API key `empty`

```shell
goose configure

┌   goose-configure 
│
◇  What would you like to configure?
│  Custom Providers 
│
◇  What would you like to do?
│  Add A Custom Provider 
│
◇  What type of API is this?
│  OpenAI Compatible 
│
◇  What should we call this provider?
│  vllm-rs
│
◇  Provider API URL:
│  http://127.0.0.1:8000/v1/
│
◇  API key:
│  ▪▪▪▪▪
│
◇  Available models (separate with commas):
│  default
│
◇  Does this provider support streaming responses?
│  Yes 
│
◇  Does this provider require custom headers?
│  No 
│
└  Custom provider added: vllm-rs
└  Configuration saved successfully to /root/.config/goose/config.yaml
```

### Run `goose` at any folder

```shell
goose
```