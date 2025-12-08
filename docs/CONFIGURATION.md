# Configuration Guide

This guide explains how to configure `candle-vllm` using configuration files and environment variables.

## Environment Variables

See `.example.env` in the project root for a complete list of all supported environment variables with examples and descriptions.

### Quick Setup

Copy the example file and customize:
```bash
cp .example.env .env
# Edit .env with your settings
# Note: .env files are not automatically loaded - export variables in your shell
```

### Key Environment Variables

#### MCP Configuration

- **`CANDLE_VLLM_MCP_CONFIG`**: Path to the MCP configuration file (default: `mcp.json` in current directory)
  ```bash
  export CANDLE_VLLM_MCP_CONFIG=/path/to/mcp.json
  ```

- **`MCP_PROXY_PORT`**: Port for MCP HTTP proxy/gateway (default: `3000`)
  - Used when converting command-based MCP servers to HTTP endpoints
  - This is the port where your MCP gateway or proxy is running
  ```bash
  export MCP_PROXY_PORT=3000
  ```

#### Models Configuration

- **`CANDLE_VLLM_MODELS_CONFIG`**: Path to the models registry file (default: `models.yaml` or `models.yml` in current directory)
  ```bash
  export CANDLE_VLLM_MODELS_CONFIG=/path/to/models.yaml
  ```

#### Logging

- **`RUST_LOG`**: Logging level (`error`, `warn`, `info`, `debug`, `trace`)
  ```bash
  export RUST_LOG=info
  ```

#### Testing

- **`CANDLE_VLLM_TEST_MODEL`**: Path to test model for integration tests
- **`CANDLE_VLLM_TEST_DEVICE`**: Device for tests (`cpu`, `cuda`, `metal`)

#### Hugging Face

- **`HF_TOKEN`**: Hugging Face authentication token (for private models)

## MCP Configuration (`mcp.json`)

The MCP configuration file supports two formats:

### Format 1: `servers` Array (Recommended)

```json
{
  "servers": [
    {
      "name": "sequential-thinking",
      "url": "http://localhost:3000/sequential-thinking",
      "auth": null,
      "timeout_secs": 30,
      "instructions": "Optional server instructions"
    }
  ]
}
```

### Format 2: `mcpServers` Object (Claude Desktop Compatible)

```json
{
  "mcpServers": {
    "sequential-thinking": {
      "type": "node",
      "command": "/path/to/mcp-server-sequential-thinking",
      "args": [],
      "env": {},
      "url": "http://localhost:3000/sequential-thinking"
    }
  }
}
```

**Note**: For command-based servers (without `url`), the system will attempt to convert them to HTTP URLs assuming they run on `localhost` with the port specified by the `MCP_PROXY_PORT` environment variable (default: `3000`). For example, if `MCP_PROXY_PORT=3001`, a command-based server named `sequential-thinking` will be converted to `http://localhost:3001/sequential-thinking`. For best results, provide the `url` field explicitly.

### MCP Server Fields

- **`name`**: Unique identifier for the server
- **`url`**: HTTP endpoint URL for the MCP server (required for HTTP servers)
- **`auth`**: Optional authentication token/header
- **`timeout_secs`**: Request timeout in seconds (default: 30)
- **`instructions`**: Optional instructions for the server
- **`command`**: Command to run (for command-based servers, requires conversion to HTTP)
- **`args`**: Command arguments (for command-based servers)
- **`env`**: Environment variables (for command-based servers)

## Models Configuration (`models.yaml`)

See `example.models.yaml` in the project root for a complete example.

### Basic Structure

```yaml
# Optional: Auto-unload idle models after this many seconds
idle_unload_secs: 3600

models:
  - name: mistral-7b
    hf_id: mistralai/Mistral-7B-Instruct-v0.3
    params:
      dtype: f16
      temperature: 0.7
      kvcache_mem_gpu: 8192
```

### Model Fields

- **`name`**: Alias name for the model (used with `--m <name>`)
- **`hf_id`**: Hugging Face model identifier
- **`local_path`**: Local path to model files
- **`weight_file`**: Specific weight file to load
- **`params`**: Model execution parameters (see below)
- **`notes`**: Optional description

### Model Parameters

- **`dtype`**: Data type (`f32`, `f16`, `bf16`)
- **`quantization`**: Quantization method (`q4k`, `q8_0`, etc.)
- **`block_size`**: KV cache block size
- **`max_num_seqs`**: Maximum sequences in batch
- **`kvcache_mem_gpu`**: KV cache memory in MB (GPU)
- **`kvcache_mem_cpu`**: KV cache memory in MB (CPU)
- **`device_ids`**: List of GPU device IDs (empty = CPU)
- **`temperature`**: Sampling temperature (0.0-2.0)
- **`top_p`**: Nucleus sampling parameter
- **`top_k`**: Top-k sampling parameter
- **`frequency_penalty`**: Frequency penalty (-2.0 to 2.0)
- **`presence_penalty`**: Presence penalty (-2.0 to 2.0)
- **`isq`**: In-situ quantization (`q4k`, `q8_0`, etc.)

## Scheduler Pool Configuration

The inference engine uses a resource-aware scheduler based on `prometheus-parking-lot`. The pool configuration is automatically derived from your KV cache settings but can be customized:

### SchedulerPoolConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_units` | `num_gpu_blocks` | Maximum resource units (KV-cache blocks) the pool can use |
| `max_queue_depth` | `1000` | Maximum queued requests before rejecting new ones |
| `default_timeout_secs` | `120` | Timeout for queued requests (seconds) |

### How It Works

1. **Resource Tracking**: Each request's cost is calculated based on prompt length and max_tokens
2. **Capacity Check**: Requests are accepted if `used_units + request_cost <= max_units`
3. **Backpressure**: When `in_flight_requests >= max_queue_depth`, new requests are rejected
4. **Automatic Release**: Resources are released when requests complete

### Tuning Tips

- **High throughput**: Increase `kvcache_mem_gpu` in models.yaml to increase `max_units`
- **Long queues**: The default `max_queue_depth=1000` is suitable for most use cases
- **Short requests**: Lower `max_tokens` reduces resource cost per request

## Auto-Injection of MCP Tools

When MCP servers are configured, their tools are **automatically injected** into chat completion requests unless:

1. The request already specifies `tools` explicitly
2. The request has `tools: []` (empty array)

This means you don't need to manually add tools to every request - they're available by default!

## Tool Call Streaming

Tool calls are now streamed incrementally during generation. As the model generates tool call tokens, they are parsed and sent as `tool_calls` deltas in streaming responses, allowing clients to see tool calls as they're generated rather than waiting for completion.

