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
- **`prompt_cache`**: Prompt cache configuration (see below)

## Prompt Caching Configuration

Prompt caching allows the system to reuse computed KV cache blocks for shared prompt prefixes, significantly reducing latency and computational costs for repeated or similar prompts.

### Overview

When prompt caching is enabled, the system:
- Detects shared prompt prefixes across requests
- Reuses pre-computed KV cache blocks for matching prefixes
- Skips redundant prefill computation for cached tokens
- Reports cache usage in API responses via `usage.prompt_tokens_details.cached_tokens`

### Configuration Methods

Prompt caching can be configured via:
1. **CLI arguments** (highest priority)
2. **Environment variables**
3. **`models.yaml`** (per-model configuration)

### CLI Arguments

```bash
# Enable prompt caching
--prompt-cache

# Select backend (memory, sled, redis)
--prompt-cache-backend sled

# Storage path (for sled backend)
--prompt-cache-path ~/.candle-vllm/cache

# Redis URL (for redis backend)
--prompt-cache-redis-url redis://localhost:6379

# TTL in seconds (for redis backend)
--prompt-cache-ttl 3600

# Maximum cached prefixes (for memory backend)
--prompt-cache-max-prefixes 10000

# Minimum prefix length to cache (tokens)
--prompt-cache-min-length 16
```

### Environment Variables

```bash
# Enable prompt caching
export CANDLE_VLLM_PROMPT_CACHE_ENABLED=true

# Backend selection
export CANDLE_VLLM_PROMPT_CACHE_BACKEND=sled

# Storage path (sled)
export CANDLE_VLLM_PROMPT_CACHE_PATH=~/.candle-vllm/cache

# Redis URL (redis backend)
export CANDLE_VLLM_PROMPT_CACHE_REDIS_URL=redis://localhost:6379

# TTL in seconds (redis backend)
export CANDLE_VLLM_PROMPT_CACHE_TTL=3600

# Maximum prefixes (memory backend)
export CANDLE_VLLM_PROMPT_CACHE_MAX_PREFIXES=10000

# Minimum prefix length (tokens)
export CANDLE_VLLM_PROMPT_CACHE_MIN_LENGTH=16
```

### Models.yaml Configuration

```yaml
models:
  - name: mistral-7b
    hf_id: mistralai/Mistral-7B-Instruct-v0.3
    params:
      prompt_cache:
        enabled: true
        backend: sled
        cache_path: ~/.candle-vllm/cache/mistral
        min_prefix_length: 16
        max_cached_prefixes: 10000
```

### Cache Backends

#### Memory Backend (Default)

- **Fastest** for single-instance deployments
- **Not persistent** - cache is lost on restart
- **LRU eviction** when `max_cached_prefixes` is reached
- **Best for**: Development, testing, single-instance production

```yaml
prompt_cache:
  enabled: true
  backend: memory
  max_cached_prefixes: 10000
```

#### Sled Backend (Recommended for Production)

- **Persistent** - survives process restarts
- **File-based** storage with ACID transactions
- **High performance** for read-heavy workloads
- **Best for**: Single-instance production deployments

```yaml
prompt_cache:
  enabled: true
  backend: sled
  cache_path: ~/.candle-vllm/cache
  min_prefix_length: 16
```

**Note**: Requires `--features prompt-cache-sled` or `--features prompt-cache` build flag.

#### Redis Backend

- **Distributed** - share cache across multiple instances
- **TTL-based expiration** for automatic cleanup
- **Network-based** - requires Redis server
- **Best for**: Multi-instance deployments, horizontal scaling

```yaml
prompt_cache:
  enabled: true
  backend: redis
  redis_url: redis://localhost:6379
  ttl_seconds: 3600
```

**Note**: Requires `--features prompt-cache-redis` or `--features prompt-cache` build flag.

### API Response Fields

When prompt caching is used, responses include cache information:

```json
{
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
    "prompt_tokens_details": {
      "cached_tokens": 75
    }
  },
  "system_fingerprint": "fp_abc123"
}
```

- **`prompt_tokens_details.cached_tokens`**: Number of prompt tokens retrieved from cache
- **`system_fingerprint`**: Model configuration fingerprint for cache validation

### Request-Level Cache Control

Clients can control caching per-request:

```json
{
  "model": "mistral",
  "messages": [...],
  "cache_control": "ephemeral"  // Don't cache this request
}
```

Options:
- **`"ephemeral"`**: Don't cache this request
- **`"ephemeral_user"`**: Don't cache user messages in this request

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Enable prompt caching |
| `backend` | `memory` | Backend type: `memory`, `sled`, `redis` |
| `cache_path` | `~/.candle-vllm/cache` | Storage path (sled backend) |
| `redis_url` | `None` | Redis connection URL (redis backend) |
| `ttl_seconds` | `None` | TTL in seconds (redis backend, no expiration if None) |
| `max_cached_prefixes` | `None` | Maximum cached prefixes (memory backend, unlimited if None) |
| `min_prefix_length` | `16` | Minimum prefix length to cache (in tokens) |
| `model_fingerprint` | Auto | Model config hash for cache invalidation (auto-generated) |

### Best Practices

1. **Use sled backend for production**: Provides persistence and good performance
2. **Set appropriate `min_prefix_length`**: Too low wastes memory, too high reduces hit rate
3. **Monitor cache hit rates**: Use cache statistics to tune configuration
4. **Use model fingerprints**: Ensures cache invalidation when model config changes
5. **Consider TTL for Redis**: Prevents stale cache entries in long-running deployments

### Example: Full Configuration

```yaml
models:
  - name: mistral-7b
    hf_id: mistralai/Mistral-7B-Instruct-v0.3
    params:
      prompt_cache:
        enabled: true
        backend: sled
        cache_path: ~/.candle-vllm/cache/mistral-7b
        min_prefix_length: 32
        max_cached_prefixes: 50000
```

## Scheduler Pool Configuration

The inference engine uses a resource-aware scheduler based on `prometheus-parking-lot`. Pool settings come from the `parking_lot` section in `models.yaml` (global or per-model). If no overrides are provided, the values fall back to defaults derived from your KV cache settings.

### SchedulerPoolConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `worker_threads` | `num_cpus::get()` | Dedicated worker threads for inference execution |
| `max_units` | `num_gpu_blocks` | Maximum resource units (KV-cache blocks) the pool can use |
| `max_queue_depth` | `1000` | Maximum queued requests before rejecting new ones |
| `default_timeout_secs` | `120` | Timeout for queued requests (seconds) |

### Queue Backend Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `queue.backend` | `memory` | Backend type: `memory`, `postgres`, `sqlite`, `surrealdb`, `yaque` |
| `queue.persistence` | `false` | Persist queue when supported |
| `queue.postgres_url` | `None` | PostgreSQL connection string (requires `queue-postgres` feature) |
| `queue.sqlite_path` | `None` | SQLite database path (requires `queue-sqlite` feature) |
| `queue.surreal_path` | `None` | SurrealDB database path (requires `queue-surreal` feature) |
| `queue.yaque_dir` | `None` | Directory for file-backed queue |

### Mailbox Backend Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mailbox.backend` | `memory` | Backend type: `memory`, `postgres`, `sqlite`, `surrealdb` |
| `mailbox.retention_secs` | `3600` | How long to keep completed responses (seconds) |
| `mailbox.postgres_url` | `None` | PostgreSQL connection string (requires `queue-postgres` feature) |
| `mailbox.sqlite_path` | `None` | SQLite database path (requires `queue-sqlite` feature) |
| `mailbox.surreal_path` | `None` | SurrealDB database path (requires `queue-surreal` feature) |

### Backend Comparison

| Backend | Persistence | ACID | Scale | Feature Flag |
|---------|-------------|------|-------|--------------|
| `memory` | No | N/A | Single instance | Default |
| `sqlite` | Yes | Yes | Single instance | `queue-sqlite` |
| `surrealdb` | Yes | Yes | Single instance | `queue-surreal` |
| `postgres` | Yes | Yes | Multi-instance | `queue-postgres` |

### Example Configuration

```yaml
parking_lot:
  pool:
    worker_threads: 4
    max_queue_depth: 100
    timeout_secs: 300

  queue:
    backend: "sqlite"
    sqlite_path: "./data/queue.db"
    persistence: true

  mailbox:
    backend: "sqlite"
    sqlite_path: "./data/mailbox.db"
    retention_secs: 3600
    webhook:
      url: "https://your-app.supabase.co/functions/v1/inference-callback"
      enabled: true
      on_disconnect: true
      auth:
        type: bearer
        token: "${SUPABASE_SERVICE_ROLE_KEY}"
```

## Webhook Configuration

Webhooks allow candle-vllm to notify external services when inference completes. This is especially useful for:
- **Async/fire-and-forget patterns**: Client submits a request and gets notified later
- **Client disconnect recovery**: If a client disconnects before receiving a response, the webhook delivers it
- **Integration with external systems**: Supabase Edge Functions, AWS Lambda, etc.

### Webhook Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `webhook.url` | `None` | Webhook endpoint URL (supports `${ENV_VAR}` interpolation) |
| `webhook.enabled` | `false` | Master switch to enable webhooks |
| `webhook.on_disconnect` | `true` | Fire webhook when client disconnects before response |
| `webhook.on_complete` | `false` | Fire webhook on every completion (in addition to normal response) |
| `webhook.timeout_secs` | `30` | HTTP request timeout |
| `webhook.retry_count` | `3` | Number of retry attempts on failure |
| `webhook.retry_delay_ms` | `1000` | Delay between retries (exponential backoff) |
| `webhook.auth` | `None` | Authentication configuration (see below) |
| `webhook.headers` | `{}` | Additional custom headers |
| `webhook.sign_payload` | `false` | Add HMAC signature even with other auth |
| `webhook.signing_secret` | `None` | Secret for payload signing |

### Authentication Methods

#### Bearer Token (OAuth 2.0 / Supabase Edge Functions)

```yaml
webhook:
  url: "https://your-project.supabase.co/functions/v1/callback"
  enabled: true
  on_disconnect: true
  auth:
    type: bearer
    token: "${SUPABASE_SERVICE_ROLE_KEY}"
```

#### API Key (Custom Header)

```yaml
webhook:
  url: "https://api.example.com/webhooks/inference"
  enabled: true
  auth:
    type: api_key
    header: "X-API-Key"
    key: "${WEBHOOK_API_KEY}"
```

#### HMAC Signature (Highest Security)

```yaml
webhook:
  url: "https://secure.example.com/webhook"
  enabled: true
  auth:
    type: hmac
    secret: "${WEBHOOK_SECRET}"
    algorithm: "sha256"  # or "sha512"
    header: "X-Signature-256"
```

#### Combined Auth + Custom Headers

```yaml
webhook:
  url: "https://your-project.supabase.co/functions/v1/callback"
  enabled: true
  auth:
    type: bearer
    token: "${SUPABASE_ANON_KEY}"
  headers:
    X-Request-Source: "candle-vllm"
    X-Environment: "production"
  sign_payload: true
  signing_secret: "${PAYLOAD_SECRET}"
```

### Environment Variable Interpolation

All webhook configuration values support `${ENV_VAR}` syntax for secure credential management:

```yaml
webhook:
  url: "${WEBHOOK_URL}"
  auth:
    type: bearer
    token: "${WEBHOOK_TOKEN}"
```

### Webhook Payload

The webhook receives the mailbox record as JSON:

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "llama-3.2-1b-instruct",
  "created": 1702000000,
  "status": "completed",
  "response": {
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "choices": [...]
  }
}
```

## REST API: Mailbox & Queue Management

### Mailbox Endpoints

#### List All Mailbox Records

```
GET /v1/mailbox
```

Response:
```json
{
  "object": "list",
  "count": 5,
  "data": [...]
}
```

#### Get a Mailbox Record

```
GET /v1/mailbox/:request_id
GET /v1/mailbox/:request_id?auto_delete=true
```

Query parameters:
- `auto_delete`: If `true`, atomically retrieve and delete the record

Response:
```json
{
  "request_id": "...",
  "model": "...",
  "created": 1702000000,
  "status": "completed",
  "response": {...}
}
```

With `auto_delete=true`:
```json
{
  "object": "mailbox.record",
  "deleted": true,
  "data": {...}
}
```

#### Delete a Mailbox Record

```
DELETE /v1/mailbox/:request_id
```

Returns `204 No Content` on success, `404 Not Found` if not found.

#### Trigger Webhook Manually

```
POST /v1/mailbox/:request_id/webhook
```

Request body (optional):
```json
{
  "url": "https://override-url.example.com/webhook",
  "bearer_token": "optional-override-token",
  "headers": {
    "X-Custom": "value"
  }
}
```

Response:
```json
{
  "status": "delivered",
  "request_id": "...",
  "url": "..."
}
```

### Queue Endpoints

#### List All Queued Requests

```
GET /v1/queues
```

Response:
```json
{
  "object": "list",
  "counts": {"llama-3.2-1b-instruct": 3, "mistral-7b": 1},
  "data": [...]
}
```

#### List Queued Requests by Model

```
GET /v1/queues/:model
```

Response:
```json
{
  "object": "list",
  "model": "llama-3.2-1b-instruct",
  "count": 3,
  "data": [...]
}
```

## Per-Request Webhook Configuration

Clients can override webhook settings per-request using HTTP headers:

| Header | Description |
|--------|-------------|
| `X-Webhook-URL` | Override webhook URL |
| `X-Webhook-Mode` | `on_disconnect`, `always`, or `never` |
| `X-Webhook-Bearer` | Override bearer token |

Example:
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Webhook-URL: https://my-callback.example.com" \
  -H "X-Webhook-Mode: always" \
  -d '{"model": "llama-3.2-1b-instruct", "messages": [...]}'
```

### How It Works

1. **Resource Tracking**: Each request's cost is calculated based on prompt length and max_tokens
2. **Capacity Check**: Requests are accepted if `used_units + request_cost <= max_units`
3. **Backpressure**: When `in_flight_requests >= max_queue_depth`, new requests are rejected
4. **Automatic Release**: Resources are released when requests complete

### Tuning Tips

- **Set worker threads**: Use `parking_lot.pool.worker_threads` in models.yaml to control CPU/GPU-bound worker parallelism.
- **High throughput**: Increase `kvcache_mem_gpu` in models.yaml to increase `max_units`
- **Long queues**: The default `max_queue_depth=1000` is suitable for most use cases
- **Short requests**: Lower `max_tokens` reduces resource cost per request
- **Queue/mailbox backends**: Set `parking_lot.queue` and `parking_lot.mailbox` globally or per-model. Per-model overrides win over global. Supported queue backends: memory (default), postgres (`queue-postgres` feature), sqlite (`queue-sqlite` feature), yaque file queue, surrealdb (`queue-surreal` feature). Mailbox backends: memory or postgres.

## Auto-Injection of MCP Tools

When MCP servers are configured, their tools are **automatically injected** into chat completion requests unless:

1. The request already specifies `tools` explicitly
2. The request has `tools: []` (empty array)

This means you don't need to manually add tools to every request - they're available by default!

## Tool Call Streaming

Tool calls are now streamed incrementally during generation. As the model generates tool call tokens, they are parsed and sent as `tool_calls` deltas in streaming responses, allowing clients to see tool calls as they're generated rather than waiting for completion.
