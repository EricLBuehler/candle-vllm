# Vision / Image Support Design (Proxy Vision for Ministral-3-3B)

This document specifies how candle-vllm will add **vision (image) support** to the existing OpenAI-compatible service, focusing first on:

- Primary reasoning model: `mistralai/Ministral-3-3B-Reasoning-2512`
- Companion vision model: `Qwen/Qwen2-VL-7B-Instruct` (or a similar small/medium open VLM)

The initial implementation is **proxy-based**:

- The primary model remains a **text-only** decoder.
- A separate **vision model** is used to interpret images and produce textual descriptions.
- Those descriptions are injected into the prompt for the primary model.

This is the design spec to guide implementation by tools/agents (SpecKit, Cursor, etc.).

---

## Goals

1. **Image input support** for the existing OpenAI-compatible `/v1/chat/completions` endpoint:
   - Accept OpenAI-style multimodal message content: `{ role, content: [ {type: "text"}, {type: "image_url"} ] }`.

2. **Self-contained inference**:
   - No external APIs (OpenAI, Anthropic, etc.).
   - All inference done by local models loaded via candle-vllm from HuggingFace weights.

3. **Unified configuration**:
   - `models.yaml` should cover **all engine-related CLI configuration**:
     - `--m`/`--model-id`, `--w`, `--f`
     - `--dtype`
     - `--isq`/quantization
     - `--mem` (KV cache memory)
     - `--prefill-chunk-size`
     - `--max-num-seqs`
     - default sampling params (`temperature`, `top_p`, `top_k`, etc.)
   - CLI flags override `models.yaml` but both map into a single `EngineParams` struct.

4. **Proxy vision for Ministral-3-3B-Reasoning-2512**:
   - If a chat request includes images and targets the Ministral model:
     - Use a configured vision model to generate image captions.
     - Rewrite the user’s message to a **pure text message** that includes those captions.
     - Run the rewritten prompt on `mistralai/Ministral-3-3B-Reasoning-2512`.

5. **Graceful degradation**:
   - Attempt to load both primary and vision models at startup.
   - If the vision model fails (e.g., OOM), log the error, disable vision, but keep the service running with text-only behavior.

6. **Future extensibility**:
   - The design should allow later addition of **native VLMs** (VisionMode::Native) without breaking the proxy logic.

---

## High-Level Architecture

### 1. Configuration Types

#### 1.1. EngineParams

A canonical configuration struct that mirrors engine-related CLI flags and `models.yaml`:

- File: `crates/candle-vllm-core/src/engine_params.rs`

Fields (subset, can be extended):

- Model selection:
  - `gguf_file: Option<String>`     → `--f`
  - `weights_path: Option<String>`  → `--w`
  - `quantization: Option<String>`  → `--isq` / quant flags
- Device / dtype:
  - `device_ids: Vec<i32>`          → `--d`
  - `dtype: Option<String>`         → `--dtype` (`bf16`, `fp16`, `fp32`)
- Memory / KV cache:
  - `mem: Option<u64>`              → `--mem` (total KV cache memory in MB)
  - `kvcache_mem_gpu: Option<u64>`  → optional GPU KV mem split
  - `kvcache_mem_cpu: Option<u64>`  → CPU KV mem split
  - `max_num_seqs: Option<usize>`   → `--max-num-seqs`
- Prefill / performance:
  - `prefill_chunk_size: Option<usize>` → `--prefill-chunk-size`
- Default sampling (used as defaults for API, may be overridden per request):
  - `temperature: Option<f32>`
  - `top_p: Option<f32>`
  - `top_k: Option<i32>`
  - `frequency_penalty: Option<f32>`
  - `presence_penalty: Option<f32>`

EngineParams must implement:

- `Default`
- `Serialize` / `Deserialize` via serde
- `fn to_log_string(&self) -> String` for logging effective config

#### 1.2. ModelCapabilities and Vision Config

- File: `crates/candle-vllm-core/src/config.rs`

Add or ensure:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VisionMode {
    #[serde(alias = "false", alias = "none", alias = "disabled")]
    Disabled,
    #[serde(alias = "proxy")]
    Proxy,
    #[serde(alias = "native")]
    Native,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VisionProxyConfig {
    pub hf_id: String,                   // HF id of vision model
    #[serde(default)]
    pub prompt_template: Option<String>, // optional system/user prompt for captioning
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    #[serde(default)]
    pub vision_mode: VisionMode,
    #[serde(default)]
    pub vision_proxy: Option<VisionProxyConfig>,
    #[serde(default)]
    pub image_token: Option<String>, // reserved for native VLMs later
}
```

#### 1.3. ModelEntry and ModelsFile

- File: `crates/candle-vllm-core/src/models_config.rs`

Structure:

```rust
pub struct ModelEntry {
    pub name: String,
    pub hf_id: Option<String>,
    pub local_path: Option<String>,
    pub weight_file: Option<String>,
    #[serde(default)]
    pub params: EngineParams,
    #[serde(default)]
    pub capabilities: ModelCapabilities,
}

pub struct ModelsFile {
    #[serde(default)]
    pub idle_unload_secs: Option<u64>,
    #[serde(default)]
    pub models: Vec<ModelEntry>,
}

impl ModelsFile {
    pub fn capabilities_map(&self) -> HashMap<String, ModelCapabilities> { ... }
    pub fn params_map(&self) -> HashMap<String, EngineParams> { ... }
}
```

`models.yaml` maps straight into `ModelsFile`.

---

### 2. Mapping EngineParams → EngineConfig / InferenceEngine

The existing core engine configuration lives in `crates/candle-vllm-core/src/api.rs`:

- `EngineConfig`
- `EngineConfigBuilder`
- `InferenceEngine`
- `InferenceEngineBuilder`

We will:

1. Create a helper to build `EngineConfig` from `(model_path, EngineParams)`.
2. Use `InferenceEngine::builder().config(config).build().await` to get a running engine.

#### 2.1. EngineConfigBuilder Overview

From `api.rs`:

- `EngineConfigBuilder::new()`
- `model_path(PathBuf)`
- `device(usize)`              // device ordinal
- `dtype(DType)`
- `max_batch_size(usize)`
- `max_sequence_length(usize)`
- `kv_cache_memory(usize)`     // KV cache in MB → maps from `EngineParams.mem`
- `enable_cuda_graph(bool)`
- `enable_chunked_prefill(bool)`
- `prefill_chunk_size(usize)`
- `build() -> Result<EngineConfig>`

#### 2.2. Helper: build_engine_config_from_params

- File: `crates/candle-vllm-core/src/engine_builder_ext.rs`

Responsibilities:

- Parse `params.dtype` into `DType` and call `.dtype()`.
- Map `params.mem` → `.kv_cache_memory(mem_mb as usize)`.
- Map `params.max_num_seqs` → `.max_batch_size()`.
- If `params.prefill_chunk_size` is set:
  - Call `.enable_chunked_prefill(true)`.
  - Call `.prefill_chunk_size(size)`.
- Select a device ordinal:
  - For now: always `0`.
  - Later: respect `params.device_ids` plus CUDA/Metal features.

Then:

```rust
pub fn build_engine_config_from_params(
    model_path: &str,
    params: &EngineParams,
) -> Result<EngineConfig>;
```

#### 2.3. Helper: build_inference_engine_from_params_async

- Also in `engine_builder_ext.rs`.

Uses:

- `build_engine_config_from_params`
- `InferenceEngine::builder().config(config).build().await`

Logs effective config once per engine:

```rust
tracing::info!(
    "Building InferenceEngine for model `{}` with EngineParams: {}",
    model_path,
    params.to_log_string()
);
```

Signature:

```rust
pub async fn build_inference_engine_from_params_async(
    model_path: &str,
    params: &EngineParams,
) -> Result<InferenceEngine>;
```

---

### 3. Primary + Vision Engines

We will support two engines:

- **Primary**: Ministral-3-3B-Reasoning-2512
- **Vision**: Qwen2-VL-7B-Instruct (or any configured `vision_proxy.hf_id`)

#### 3.1. VisionBackend Enum

- File: `crates/candle-vllm-core/src/openai/vision_backend.rs`

```rust
pub enum VisionBackend {
    Available {
        model_name: String,
        adapter: Arc<OpenAIAdapter>,
    },
    Unavailable {
        reason: String,
    },
}

impl VisionBackend {
    pub fn adapter(&self) -> Option<(String, Arc<OpenAIAdapter>)> { ... }
    pub fn is_available(&self) -> bool { ... }
}
```

#### 3.2. Building engines from ModelsFile

- File: `src/build_engines.rs` (server crate)

Async function:

```rust
pub async fn build_engines(
    models_file: &ModelsFile,
) -> Result<(Arc<OpenAIAdapter>, VisionBackend)>;
```

Logic:

1. Find primary model entry.
   - For now: a fixed name `"ministral-3-3b-reasoning"` (later, configurable via CLI).
2. Use `entry.hf_id` and `entry.params` to call:
   - `build_inference_engine_from_params_async(primary_hf_id, &primary_params)`
   - Wrap result in `OpenAIAdapter`.
3. For vision backend:
   - Find the first model with `capabilities.vision_mode == Proxy` and `vision_proxy.hf_id`.
   - Retrieve the `vision_proxy.hf_id` string.
   - Find a `ModelEntry` whose `hf_id` matches that; read its `params` (or use defaults).
   - Call `build_inference_engine_from_params_async(vision_hf_id, &vision_params)`.
     - On success: `VisionBackend::Available { model_name: vision_hf_id, adapter }`.
     - On failure: `VisionBackend::Unavailable { reason: format!("vision build error: {e}") }`, with a warning log.

---

### 4. Image Processing: MessageContent and Proxy

#### 4.1. MessageContent and ImageUrl

- File: `crates/candle-vllm-core/src/openai/requests.rs`

Replace `ChatMessage.content: Option<String>` with a variant that supports both legacy strings and OpenAI-style parts:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default = "default_detail")]
    pub detail: String, // "auto", "low", "high"
}
```

`ChatMessage`:

```rust
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    // tool_calls, tool_call_id, name unchanged
}
```

Add helper methods:

- `ChatMessage::user_with_images(text: String, image_urls: Vec<String>)`
- `ChatMessage::get_text_content(&self) -> Option<String>`
- `ChatMessage::get_image_urls(&self) -> Vec<String>`
- `ChatMessage::has_images(&self) -> bool`

Add tests to ensure:

- Old string-only JSON still deserializes correctly.
- New `content: [ {type:"text"}, {type:"image_url"}]` deserializes and helpers work.

#### 4.2. ImageDescriptionTool

- File: `crates/candle-vllm-core/src/openai/image_tool.rs`

Define:

```rust
#[allow(async_fn_in_trait)]
pub trait ImageDescriptionTool: Send + Sync {
    async fn describe_image(
        &self,
        image_url: &str,
        user_prompt: Option<&str>,
    ) -> anyhow::Result<String>;
}

pub struct DummyImageDescriptionTool;

#[allow(async_fn_in_trait)]
impl ImageDescriptionTool for DummyImageDescriptionTool {
    async fn describe_image(
        &self,
        image_url: &str,
        _user_prompt: Option<&str>,
    ) -> anyhow::Result<String> {
        Ok(format!("(no vision backend configured; image was at {image_url})"))
    }
}
```

- File: `crates/candle-vllm-core/src/openai/image_tool_local_model.rs`

Define `LocalVisionModelTool` that uses `VisionBackend::Available { model_name, adapter }` to:

- Build a chat request to the vision model with `image_url` and optional system prompt (`vision_proxy.prompt_template`).
- Call `adapter.chat_completion(req).await`.
- Return the caption from `choices[0].message.content`.

#### 4.3. VisionProxyPreprocessor

- File: `crates/candle-vllm-core/src/openai/vision_proxy.rs`

Structure:

```rust
pub struct VisionProxyPreprocessor {
    capabilities: ModelCapabilities,
    image_tool: Arc<dyn ImageDescriptionTool>,
}

impl VisionProxyPreprocessor {
    pub fn new(
        capabilities: ModelCapabilities,
        image_tool: Arc<dyn ImageDescriptionTool>,
    ) -> Self { ... }

    pub async fn preprocess(
        &self,
        request: ChatCompletionRequest,
    ) -> anyhow::Result<ChatCompletionRequest> { ... }
}
```

Behavior:

- If `capabilities.vision_mode != VisionMode::Proxy`, return `request` unchanged.
- Otherwise:
  - Convert `request.messages` into `Vec<ChatMessage>` (using `Messages::to_chat_messages()`).
  - For each `ChatMessage`:
    - If `role != "user"` or !`has_images()` → keep as-is.
    - Else:
      - `text = get_text_content().unwrap_or_default()`.
      - `image_urls = get_image_urls()`.
      - For each `url`:
        - Call `image_tool.describe_image(url, Some(text.as_str()))`.
        - Collect captions like `"Image N: <caption>"`.
      - Build `combined`:

        - If `text` empty: `combined = descs.join("\n")`
        - If `descs` empty: `combined = text`
        - Else: `combined = format!("{text}\n\n{}", descs.join("\n"))`

      - Replace message with `ChatMessage::user(combined)` (text-only).
  - Re-wrap as `ChatCompletionRequest` with `Messages::Chat(new_messages)`.

---

### 5. Server Integration

#### 5.1. EngineState

- File: `candle-vllm/src/engine_state.rs` (or equivalent in server crate)

```rust
pub struct EngineState {
    pub primary: Arc<OpenAIAdapter>,
    pub vision: VisionBackend,
    pub get_capabilities: Arc<dyn Fn(&str) -> ModelCapabilities + Send + Sync>,
    pub image_tool: Arc<dyn ImageDescriptionTool>,
}
```

`get_capabilities` can be built from `models_file.capabilities_map()`.

`image_tool`:

- If `vision.is_available()` → `LocalVisionModelTool::new(vision.clone(), prompt_template)`.
- Else → `DummyImageDescriptionTool`.

#### 5.2. Building EngineState

- File: `candle-vllm/src/main.rs` (or startup module)

Steps:

1. `models_file = load_models_file()` (reads `models.yaml`).
2. `(primary_adapter, vision_backend) = build_engines(&models_file).await`.
3. `capabilities_map = models_file.capabilities_map()`.
4. `get_capabilities = Arc::new(move |name| capabilities_map.get(name).cloned().unwrap_or_default())`.
5. Determine `prompt_template` from first model with `vision_mode=Proxy`.
6. Build `image_tool`:
   - If `vision_backend.Available`:
     - `LocalVisionModelTool::new(vision_backend.clone(), prompt_template)`.
   - Else:
     - `DummyImageDescriptionTool`.
7. Construct `EngineState` and store it in Axum state.

#### 5.3. `/v1/chat/completions` handler

Existing handler likely looks like:

```rust
async fn chat_completions_handler(
    State(adapter): State<OpenAIAdapter>,
    Json(req): Json<ChatCompletionRequest>,
) -> ...
```

New handler:

```rust
async fn chat_completions_handler(
    State(state): State<EngineState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, StatusCode> {
    let capabilities = (state.get_capabilities)(&req.model);

    let preprocessor = VisionProxyPreprocessor::new(
        capabilities.clone(),
        state.image_tool.clone(),
    );

    let req = preprocessor
        .preprocess(req)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let resp = state
        .primary
        .chat_completion(req)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(resp))
}
```

---

## 6. Testing Plan in This Service

1. **Text-only sanity check**
   - Start the server on your M1 (Metal build if enabled).
   - Use `models.yaml` with both models configured.
   - Send a text-only request to `/v1/chat/completions` with `model: "ministral-3-3b-reasoning"`.
   - Confirm Ministral-3-3B responds normally.

2. **Vision backend initialization**
   - Check logs on startup:
     - Effective `EngineParams` for primary and vision.
     - Confirmation that Qwen2-VL backend loads, or a clear warning if it fails.

3. **Multimodal request**
   - Send a request with `content` containing `text` and `image_url` for `model: "ministral-3-3b-reasoning"`.
   - Confirm:
     - `VisionProxyPreprocessor` is invoked.
     - `LocalVisionModelTool.describe_image` is called.
     - The request passed to `primary.chat_completion` has only text content (no `image_url`), with captions appended.

4. **Memory / `--mem` behavior**
   - Adjust `params.mem` in `models.yaml` for one or both models.
   - Confirm logs show `kv_cache_memory` set accordingly when building `EngineConfig`.
   - Optionally, run with different settings to verify stability.

Once this is working reliably inside this service, the same configuration and builder paths can be exposed for programmatic use by external API clients or embedding applications.