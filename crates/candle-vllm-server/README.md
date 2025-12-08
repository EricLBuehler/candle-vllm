# candle-vllm-server

HTTP server for serving candle-vllm models with OpenAI-compatible API and additional endpoints.

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Model management endpoints
- MCP tool listing endpoint
- Model switching with request queuing
- Streaming support with incremental tool-call deltas
- Automatic MCP tool injection when `mcp.json` (or `CANDLE_VLLM_MCP_CONFIG`) is configured

## Usage

### Command Line

```bash
# Run with a model
cargo run --release --features cuda,nccl -- --m mistralai/Mistral-7B-Instruct-v0.3 --p 2000

# Run with local model path
cargo run --release --features metal -- --w /path/to/model/ --p 2000

# Run with MCP configuration
cargo run --release --features cuda,nccl -- --m mistralai/Mistral-7B-Instruct-v0.3 --mcp-config mcp.json --p 2000
```

### Programmatic Usage

```rust
use candle_vllm_server::run;
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Set up arguments
    let args = vec![
        "--m".to_string(),
        "mistralai/Mistral-7B-Instruct-v0.3".to_string(),
        "--p".to_string(),
        "2000".to_string(),
    ];

    // Run server
    run(args).await?;
    Ok(())
}
```

## API Endpoints

- `POST /v1/chat/completions`: OpenAI-compatible chat completion
- `GET /v1/models`: List available models
- `GET /v1/models/status`: Get model status and queue information
- `POST /v1/models/select`: Switch active model
- `GET /v1/mcp/tools`: List available MCP tools

## Configuration

- `--queue-size`: Maximum queue size per model (default: 10)
- `--request-timeout`: Request timeout in seconds (default: 30)
- `--mcp-config`: Path to MCP configuration JSON file

Environment variables mirror the CLI flags (see `.example.env` / `docs/CONFIGURATION.md`):
- `CANDLE_VLLM_MCP_CONFIG`: Override MCP config path
- `CANDLE_VLLM_MODELS_CONFIG`: Override models registry path
- `RUST_LOG`, `KEEP_ALIVE_INTERVAL`, etc.

See the main [README.md](../../README.md) for full command-line options.

