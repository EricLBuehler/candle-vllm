# Executive Summary: Library-First Architecture & MCP Support

## What We Accomplished

### 1. Tool Calling Support ✅

We successfully implemented OpenAI-compatible tool calling (function calling) with the following features:

- **OpenAI `tools` format** (not deprecated `functions`)
- **Multiple tool call formats**: Mistral/Ministral, Llama, Qwen, generic JSON
- **Parallel tool calling**: Multiple tools in one response
- **Multi-turn conversations**: Tool responses with `tool_call_id` tracking
- **MCP conversion helpers**: Convert MCP tools to OpenAI format and vice versa

**Key Files:**
- `src/openai/requests.rs` - Request/response types with MCP helpers
- `src/openai/tool_parser.rs` - Parsers for different model formats
- `src/openai/conversation/mod.rs` - Tool result formatting helpers

**Tests:** 29 passing tests covering all tool calling scenarios

### 2. Mistral 3 / Ministral Support ✅

Enhanced support for newer Mistral family models:

- **Nested `rope_parameters`** configuration handling
- **BF16/FP16** model support (FP8 not supported by safetensors loader)
- **Yarn RoPE scaling** for extended context
- **Backward compatible** with older Mistral models

**Supported Models:**
- ✅ `mistralai/Mistral-7B-Instruct-v0.3` (BF16)
- ✅ `mistralai/Ministral-8B-Instruct-2410` (BF16)
- ✅ `mistralai/Ministral-3B-Instruct-2410` (BF16)

### 3. Documentation ✅

Created comprehensive documentation:

1. **ARCHITECTURE.md** - Complete library-first design
2. **LIBRARY_API.md** - Detailed API specification with examples
3. **IMPLEMENTATION_PLAN.md** - Step-by-step migration plan
4. **README.md** - Updated with new features and examples

---

## What's Next: Library-First Restructuring

### Problem Statement

Currently, candle-vllm is structured as a **binary-first** application with a built-in HTTP server. This makes it difficult to:

- Embed in **Tauri desktop apps**
- Use in custom **AI gateways**
- Build **agent frameworks** on top
- Use with **different HTTP servers** (not just Axum)

### Proposed Solution

Restructure into a **multi-crate workspace**:

```
candle-vllm/
├── crates/
│   ├── candle-vllm-core/        # Core inference engine (library)
│   ├── candle-vllm-openai/      # OpenAI compatibility layer
│   ├── candle-vllm-responses/   # Responses API + MCP integration
│   └── candle-vllm-server/      # HTTP server (optional)
└── examples/
    ├── tauri_app/               # Desktop app example
    ├── ai_gateway/              # Custom gateway example
    └── agent_framework/         # Agent system example
```

### Key Benefits

**For End Users:**
- Embed in any Rust application
- No HTTP overhead - direct function calls
- Own your API endpoints
- Build custom agents

**For You:**
- Enable Tauri desktop apps
- Enable AI gateway applications
- Enable agent frameworks
- All using the same inference engine

**Backward Compatible:**
- Existing binary users see no changes
- Can migrate gradually
- No breaking changes

---

## Implementation Phases

### Phase 1: Workspace Setup (Week 1)
- Create multi-crate workspace
- Set up directory structure
- Configure Cargo workspace

### Phase 2: Core Extraction (Week 2)
- Extract `candle-vllm-core` crate
- Move scheduler, models, cache
- Create clean API surface

### Phase 3: OpenAI Layer (Week 3)
- Create `candle-vllm-openai` crate
- Move request/response types
- Implement `OpenAIAdapter`

### Phase 4: Responses API + MCP (Week 4)
- Create `candle-vllm-responses` crate
- Implement `McpClient`
- Build `ResponsesSession` orchestrator

### Phase 5: Update Server (Week 5)
- Refactor server to use libraries
- Maintain backward compatibility
- Test existing workflows

### Phase 6: Examples (Week 6)
- Create Tauri app example
- Create AI gateway example
- Create agent framework example

### Phase 7: Polish (Week 7)
- Complete documentation
- Add integration tests
- Prepare for release

---

## Usage Examples After Restructuring

### Example 1: Tauri Desktop App

```rust
use candle_vllm_core::InferenceEngine;
use candle_vllm_openai::OpenAIAdapter;

#[tauri::command]
async fn chat(message: String) -> Result<String, String> {
    let adapter = state.adapter.lock().await;
    let request = ChatCompletionRequest {
        messages: vec![Message::user(message)],
        ..Default::default()
    };
    let response = adapter.chat_completion(request).await?;
    Ok(response.choices[0].message.content.clone())
}
```

### Example 2: Custom AI Gateway

```rust
use candle_vllm_openai::OpenAIAdapter;

async fn custom_handler(
    Json(req): Json<MyCustomRequest>,
    adapter: Extension<Arc<Mutex<OpenAIAdapter>>>,
) -> Json<MyCustomResponse> {
    // Use your own API format
    let openai_req = convert_to_openai(req);
    let response = adapter.chat_completion(openai_req).await.unwrap();
    Json(convert_from_openai(response))
}
```

### Example 3: Agent with MCP Tools

```rust
use candle_vllm_responses::ResponsesSession;

let mut session = ResponsesSession::builder()
    .model_path("./models/mistral-7b")
    .add_mcp_server("filesystem", "http://localhost:3001/mcp", None)
    .add_mcp_server("github", "http://localhost:3002/mcp", Some("Bearer TOKEN"))
    .build()
    .await?;

let result = session.run_conversation(
    vec![Message::user("Read README and create a GitHub issue")],
    ConversationOptions::default(),
).await?;
```

---

## OpenAI Responses API Support

### What is Responses API?

The Responses API is OpenAI's newer API that:
- **Connects directly to MCP servers**
- **Automatically executes tools**
- **Manages multi-turn conversations**
- **Returns final results** after tool execution

### Our Implementation Plan

**candle-vllm-responses** crate will provide:

```rust
pub struct ResponsesSession {
    adapter: OpenAIAdapter,
    mcp_servers: HashMap<String, McpClient>,
}

impl ResponsesSession {
    // Connect to MCP servers
    pub async fn add_mcp_server(&mut self, name: &str, url: &str, auth: Option<String>);
    
    // Run multi-turn conversation with automatic tool execution
    pub async fn run_conversation(
        &mut self,
        messages: Vec<Message>,
        options: ConversationOptions,
    ) -> Result<ConversationResult>;
}
```

**Features:**
- Direct MCP server connection
- Automatic tool discovery (`list_tools`)
- Automatic tool execution (`call_tool`)
- Multi-turn orchestration
- Tool result formatting
- Error handling and retries

---

## Testing the Current Implementation

### Test Tool Calling

Use the curl examples we added to README.md:

```bash
# Start the server
cargo run --release --features cuda -- --model mistralai/Mistral-7B-Instruct-v0.3

# Test tool calling
curl http://localhost:2000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

### Test Mistral 3 Models

```bash
# Ministral-8B
cargo run --release --features cuda -- \
  --model mistralai/Ministral-8B-Instruct-2410

# With quantization for lower memory
cargo run --release --features cuda -- \
  --model mistralai/Ministral-8B-Instruct-2410 --isq q4k
```

### Run Tests

```bash
# Test tool calling
cargo test --lib openai::requests::tests --no-default-features

# Test conversation helpers
cargo test --lib openai::conversation::tests --no-default-features
```

---

## Decision Points

### Question 1: Start Restructuring Now?

**Option A: Start Now**
- Begin workspace migration
- Incremental changes over 7 weeks
- No breaking changes for existing users

**Option B: Wait**
- Keep current structure
- Add features incrementally
- Restructure later when needed

**Recommendation:** Start now if you want to enable Tauri/gateway use cases soon

### Question 2: Breaking Changes?

**Option A: Maintain 100% Backward Compatibility**
- Keep old binary working exactly as-is
- New library crates are additive
- Deprecation warnings for old APIs

**Option B: Accept Minor Breaking Changes**
- Clean up technical debt
- Simplify APIs
- Provide migration guide

**Recommendation:** Option A - maintain compatibility

### Question 3: Scope of Initial Release?

**Minimal:**
- Core + OpenAI layers only
- No Responses API yet
- Focus on library use case

**Full:**
- All 4 crates including Responses API
- Complete MCP integration
- All examples

**Recommendation:** Start minimal, add Responses API in next release

---

## Next Steps (Your Decision)

### Option 1: Proceed with Restructuring

1. Review IMPLEMENTATION_PLAN.md
2. Start with Phase 1 (workspace setup)
3. Work through phases incrementally
4. I can help with each phase

### Option 2: Continue with Current Structure

1. Keep adding features to current codebase
2. Extract to library later when needed
3. Focus on other priorities first

### Option 3: Hybrid Approach

1. Start workspace setup (Phase 1)
2. Extract core only (Phase 2)
3. Pause and evaluate
4. Continue if beneficial

---

## Resources

- **ARCHITECTURE.md** - Complete design details
- **LIBRARY_API.md** - Full API specification with examples
- **IMPLEMENTATION_PLAN.md** - Step-by-step migration guide
- **README.md** - Updated with tool calling examples

---

## Current Status

✅ **Completed:**
- Tool calling implementation
- MCP helper functions
- Mistral 3/Ministral support
- Comprehensive documentation
- 29 passing tests

🔄 **Ready to Start:**
- Library-first restructuring
- Responses API implementation
- Example applications

⏳ **Waiting on:**
- Your decision on approach
- Priority and timeline
- Resource allocation

---

## Recommendation

**Start with Phase 1-2** (Weeks 1-2):
1. Set up workspace structure
2. Extract core inference engine
3. Evaluate if the approach is working
4. Decide whether to continue with remaining phases

This gives you:
- Early validation of the approach
- A working core library
- Flexibility to pause or continue
- Minimal risk (backward compatible)

**The investment is worth it if you want to:**
- Build Tauri desktop applications
- Create custom AI gateways
- Develop agent frameworks
- Have more architectural flexibility

---

## Questions?

Feel free to ask about:
- Any part of the implementation plan
- Trade-offs between options
- Technical details
- Timeline estimates
- Resource requirements