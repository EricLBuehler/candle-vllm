# candle-vllm SDK Integration Guide

candle-vllm now exposes a **library-first** architecture: the core inference engine (`candle-vllm-core`), the OpenAI compatibility layer (`candle-vllm-openai`), the high-level MCP/session helpers (`candle-vllm-responses`), and the HTTP server (`candle-vllm-server`). This guide shows how to embed those crates inside common apps and how to wire an external UI (Electron/Cherry Studio) to a dedicated inference server.

> **Configuration tip:** copy `.example.env` → `.env` (or export variables manually) and refer to `docs/CONFIGURATION.md` + `example.models.yaml` for every supported knob (`CANDLE_VLLM_MCP_CONFIG`, `CANDLE_VLLM_MODELS_CONFIG`, `RUST_LOG`, `KEEP_ALIVE_INTERVAL`, etc.).

## 1. Common Building Blocks

- **InferenceEngine** – load models (local path or Hugging Face ID), tokenize, generate, stream.
- **OpenAIAdapter** – exposes OpenAI-style `chat_completion` / streaming with incremental tool-call deltas.
- **ResponsesSession** – orchestrates multi-turn conversations plus MCP tool execution (auto-injecting tools defined in `mcp.json`).

Keep one `InferenceEngine` (or `OpenAIAdapter`) per process and clone `Arc`s into your handlers/commands to avoid reloading weights.

---

## 2. Tauri Desktop Integration

### 2.1. Architecture

1. Create a Rust workspace member for your Tauri app and add dependencies:
   ```toml
   [dependencies]
   candle-vllm-core = { path = "../candle-vllm/crates/candle-vllm-core" }
   candle-vllm-openai = { path = "../candle-vllm/crates/candle-vllm-openai" }
   candle-core = "0.5"
   ```
2. Initialize the engine during Tauri setup and store it in `tauri::State`.
3. Expose `#[tauri::command]` functions that call into the engine/adapter.

### 2.2. Example Command

```rust
use candle_vllm_core::{InferenceEngine, EngineConfig};
use candle_vllm_openai::{OpenAIAdapter, chat::ChatCompletionRequest};
use candle_core::Device;
use tauri::State;

pub struct EngineState(pub OpenAIAdapter);

#[tauri::command]
async fn chat_completion(
    adapter: State<'_, EngineState>,
    request: ChatCompletionRequest,
) -> Result<candle_vllm_core::openai::responses::ChatCompletionResponse, String> {
    adapter.0.clone().chat_completion(request).await.map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .manage(EngineState(init_adapter().expect("engine")))
        .invoke_handler(tauri::generate_handler![chat_completion])
        .run(tauri::generate_context!())
        .expect("Tauri failed");
}

fn init_adapter() -> anyhow::Result<OpenAIAdapter> {
    let engine = InferenceEngine::builder()
        .model_path(std::env::var("CANDLE_VLLM_MODEL").unwrap_or_else(|_| "mistralai/Mistral-7B-Instruct-v0.3".into()))
        .device(Device::Cpu) // or Device::Cuda(0)/Device::new_metal(0)
        .build()?;
    Ok(OpenAIAdapter::new(engine))
}
```

### 2.3. Packaging Notes

- Bundle model weights with the app or download them on first launch; store under `AppDir`.
- Ship `.env` plus `mcp.json`/`models.yaml` in `resources/` and load via `CANDLE_VLLM_*`.
- Use streaming (SSE-style) by wiring `chat_completion_stream` into a Tauri async command and emit progress via `window.emit`.

---

## 3. Axum Web Applications / Agents

### 3.1. Minimal Axum Router

```rust
use axum::{routing::post, Json, Router};
use candle_vllm_core::{InferenceEngine, EngineConfig};
use candle_vllm_openai::{OpenAIAdapter, chat::ChatCompletionRequest};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let engine = InferenceEngine::builder()
        .model_path("mistralai/Mistral-7B-Instruct-v0.3")
        .build()
        .await?;
    let adapter = Arc::new(tokio::sync::Mutex::new(OpenAIAdapter::new(engine)));

    let app = Router::new().route(
        "/v1/chat/completions",
        post({
            let adapter = adapter.clone();
            move |Json(req): Json<ChatCompletionRequest>| async move {
                let mut guard = adapter.lock().await;
                let resp = guard.chat_completion(req).await?;
                Ok::<_, anyhow::Error>(Json(resp))
            }
        }),
    );

    axum::Server::bind(&"0.0.0.0:8080".parse()?).serve(app.into_make_service()).await?;
    Ok(())
}
```

### 3.2. Agentic Pipelines

- Use `ResponsesSession` when you need MCP-backed multi-turn conversations. Load the session once (reading `CANDLE_VLLM_MCP_CONFIG` if present) and store in `axum::extract::State`.
- Streaming: wrap `chat_completion_stream` into an `Sse` response (`axum::response::Sse`) and forward chunks directly. The chunks already contain incremental tool-call deltas, so frontends can visualize tool calls as they appear.

### 3.3. Middleware / Observability

- Set `RUST_LOG=info` (or `debug`) to see per-request traces.
- Use `tower_http::trace::TraceLayer` to log API calls and correlate with candle-vllm's scheduler logs.

---

## 4. Electron + Dedicated Inference Server

### 4.1. Recommended Architecture

1. Run `candle-vllm-server` as a background process (local-only or networked).
2. Configure it via `.env` or `CANDLE_VLLM_*`.
3. Electron's main process communicates with the server using the OpenAI-compatible HTTP API; renderer processes call into main via IPC.

### 4.2. Bootstrapping Script

```ts
// main.ts
import { app, BrowserWindow } from 'electron';
import { spawn } from 'child_process';

let server: ReturnType<typeof spawn> | undefined;

app.on('ready', () => {
  server = spawn('./candle-vllm-server', ['--m', 'mistralai/Mistral-7B-Instruct-v0.3', '--p', '2000'], {
    env: { ...process.env, CANDLE_VLLM_MCP_CONFIG: app.getPath('userData') + '/mcp.json' },
  });
  server.stdout?.on('data', data => console.log(data.toString()));
  server.stderr?.on('data', data => console.error(data.toString()));

  const win = new BrowserWindow({ webPreferences: { preload: 'preload.js' } });
  win.loadURL('http://localhost:5173'); // or local file
});

app.on('will-quit', () => server?.kill());
```

### 4.3. Sharing the Server

- Bind to `127.0.0.1` for app-only usage or `0.0.0.0` to expose the API to other clients on the LAN.
- Use the `/v1/mcp/tools` endpoint to provide an in-app UI listing available tools (auto-injected from `mcp.json`).

---

## 5. Cherry Studio Integration

Cherry Studio (`/Users/gqadonis/Projects/cherry-studio`) is an Electron-based IDE for agents. To add candle-vllm support:

1. **Start a managed candle-vllm-server instance**
   - Add a helper to spawn/monitor the server (similar to §4.2) when the user enables the “Local candle-vllm” provider.
   - Configure model choices via `models.yaml` and surface them in the UI; set `CANDLE_VLLM_MODELS_CONFIG` accordingly.
2. **Extend the provider registry**
   - Implement a provider module that translates Cherry Studio prompts to the OpenAI-compatible `/v1/chat/completions` endpoint.
   - Reuse Cherry’s existing OpenAI client path; only the base URL/token fields change (`baseURL = http://127.0.0.1:2000/v1`, token may stay `EMPTY`).
3. **Expose MCP tooling**
   - Fetch `/v1/mcp/tools` and display enabled tools inside the session sidebar.
   - Allow users to edit `mcp.json` via the Cherry settings pane; write the file to `app.getPath('userData')` and restart the server so auto-injection picks it up.
4. **Streaming UI**
   - Subscribe to SSE chunks and render both `delta.content` and `delta.tool_calls`. The tool call deltas arrive incrementally, so you can display function placeholders even before arguments finish streaming.
5. **Packaging**
   - Ship `.example.env` to illustrate required environment variables and document how Cherry populates them at runtime (e.g., `KEEP_ALIVE_INTERVAL`, `HF_TOKEN`).

> For inspiration, search the Cherry Studio repo for existing provider definitions (e.g., `openai.ts`) and mirror their capability flags for streaming/tool calling.

---

## 6. Additional Resources

- [`docs/CONFIGURATION.md`](./CONFIGURATION.md) – detailed env/config reference.
- `.example.env`, `example.models.yaml` – ready-to-edit templates.
- `README.md` – full CLI usage, MCP auto-injection notes, streaming semantics.
- `crates/*/README.md` – crate-level API summaries.

Feel free to extend this guide with more frameworks (FastAPI, Poem, Remix, etc.)—just ensure new examples showcase incremental tool-call streaming and reference the canonical configuration docs.

