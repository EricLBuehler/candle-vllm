//! Tauri Application Example for candle-vllm
//!
//! This example demonstrates how to use candle-vllm as a library in an embedded
//! context (like a Tauri desktop application) without requiring an HTTP server.
//!
//! Key features demonstrated:
//! - Direct library usage without HTTP overhead
//! - OpenAI-compatible request/response types
//! - Streaming responses via channels
//! - Tool calling support
//!
//! # Building
//!
//! For CUDA (NVIDIA):
//! ```bash
//! cargo build --release --features cuda
//! ```
//!
//! For Metal (Apple Silicon):
//! ```bash
//! cargo build --release --features metal
//! ```
//!
//! # Usage in Tauri
//!
//! In a real Tauri application, you would:
//! 1. Create commands that wrap the inference functions
//! 2. Use Tauri's event system for streaming responses
//! 3. Manage model lifecycle through Tauri's state management

use anyhow::Result;
use candle_vllm_openai::{
    // Request types
    ChatCompletionRequest, ChatMessage, Messages, Tool, FunctionDefinition,
    // Tool parsing
    get_tool_parser, ParsedOutput,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{info, error};

/// Application state that would be managed by Tauri
pub struct AppState {
    /// Model name/path
    pub model_name: String,
    /// Whether the model is loaded
    pub model_loaded: bool,
    // In a real implementation, this would hold:
    // pub engine: Arc<InferenceEngine>,
}

impl AppState {
    pub fn new(model_name: String) -> Self {
        Self {
            model_name,
            model_loaded: false,
        }
    }
}

/// Chat request from the frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontendChatRequest {
    pub message: String,
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub tools: Option<Vec<Tool>>,
}

/// Chat response to the frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontendChatResponse {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<serde_json::Value>>,
    pub is_complete: bool,
    pub error: Option<String>,
}

/// Convert frontend request to OpenAI-compatible request
fn to_chat_completion_request(
    model: &str,
    req: FrontendChatRequest,
) -> ChatCompletionRequest {
    let mut messages = vec![];
    
    if let Some(system) = req.system_prompt {
        messages.push(ChatMessage::system(system));
    }
    
    messages.push(ChatMessage::user(req.message));
    
    ChatCompletionRequest {
        model: model.to_string(),
        messages: Messages::Chat(messages),
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        tools: req.tools,
        // Set reasonable defaults
        top_p: None,
        min_p: None,
        n: None,
        stop: None,
        stream: Some(false),
        presence_penalty: None,
        repeat_last_n: None,
        frequency_penalty: None,
        logit_bias: None,
        user: None,
        top_k: None,
        best_of: None,
        use_beam_search: None,
        ignore_eos: None,
        skip_special_tokens: None,
        stop_token_ids: None,
        logprobs: None,
        thinking: None,
        tool_choice: None,
        parallel_tool_calls: None,
        conversation_id: None,
        resource_id: None,
    }
}

/// Tauri command: Get a chat completion (non-streaming)
///
/// In a real Tauri app, this would be decorated with `#[tauri::command]`
pub async fn chat_completion(
    _state: &AppState,
    request: FrontendChatRequest,
) -> Result<FrontendChatResponse, String> {
    info!("Processing chat request: {:?}", request.message);
    
    // Convert to OpenAI format
    let _openai_request = to_chat_completion_request("demo-model", request);
    
    // In a real implementation:
    // let response = state.engine.chat_completion(openai_request).await?;
    
    // Demo response
    Ok(FrontendChatResponse {
        content: Some("This is a demo response. In a real Tauri app, this would call the inference engine.".to_string()),
        tool_calls: None,
        is_complete: true,
        error: None,
    })
}

/// Tauri command: Stream chat completion
///
/// Returns a channel receiver that emits response chunks.
/// In Tauri, you'd use events instead: `app.emit_all("chat-chunk", chunk)`
pub async fn chat_completion_stream(
    _state: &AppState,
    request: FrontendChatRequest,
) -> Result<mpsc::Receiver<FrontendChatResponse>, String> {
    let (tx, rx) = mpsc::channel(32);
    
    // Convert to OpenAI format
    let _openai_request = to_chat_completion_request("demo-model", request);
    
    // Spawn a task to send chunks
    tokio::spawn(async move {
        // In a real implementation, you'd iterate over the stream from the engine
        let chunks = vec![
            "This ",
            "is ",
            "a ",
            "streaming ",
            "response ",
            "demo.",
        ];
        
        for (i, chunk) in chunks.iter().enumerate() {
            let response = FrontendChatResponse {
                content: Some(chunk.to_string()),
                tool_calls: None,
                is_complete: i == chunks.len() - 1,
                error: None,
            };
            
            if tx.send(response).await.is_err() {
                break; // Receiver dropped
            }
            
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    });
    
    Ok(rx)
}

/// Demonstrate tool calling with the library
fn demonstrate_tool_parsing() {
    info!("Demonstrating tool parsing capabilities...");
    
    // Create a tool definition
    let weather_tool = Tool::function(
        FunctionDefinition::new("get_weather")
            .with_description("Get the current weather in a location")
            .with_parameters(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state"
                    }
                },
                "required": ["location"]
            }))
    );
    
    info!("Created tool: {:?}", weather_tool.name());
    
    // Demonstrate parsing model output with tool calls
    let parser = get_tool_parser("mistral");
    
    // Simulate Mistral-style tool call output
    let model_output = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "San Francisco, CA"}}]"#;
    
    let parsed = parser.parse(model_output);
    
    match parsed {
        ParsedOutput::ToolCalls(calls) => {
            info!("Parsed {} tool call(s):", calls.len());
            for call in calls {
                info!("  - Function: {}", call.name);
                info!("    Arguments: {}", call.arguments);
            }
        }
        ParsedOutput::Text(text) => {
            info!("Parsed text: {}", text);
        }
        ParsedOutput::Mixed { text, tool_calls } => {
            info!("Parsed mixed output:");
            info!("  Text: {}", text);
            info!("  Tool calls: {:?}", tool_calls.len());
        }
    }
}

/// Demonstrate conversation building
fn demonstrate_conversation() {
    use candle_vllm_openai::conversation::ToolConversationBuilder;
    use candle_vllm_openai::types::requests::{ToolCall, FunctionCall};
    
    info!("Demonstrating conversation building...");
    
    let mut builder = ToolConversationBuilder::new("mistralai/Mistral-7B");
    
    // Add a user message
    builder.add_user_message("What's the weather in Paris?");
    
    // Add assistant response with tool call
    builder.add_assistant_tool_calls(vec![
        ToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location": "Paris, France"}"#.to_string(),
            },
        }
    ]);
    
    // Add tool result
    builder.add_tool_result(
        "call_123",
        r#"{"temperature": 18, "condition": "Partly cloudy"}"#,
        Some("get_weather".to_string()),
    );
    
    // Add final assistant response
    builder.add_assistant_response("The weather in Paris is 18°C and partly cloudy.");
    
    info!("Built conversation with {} messages", builder.messages().len());
    info!("Model family: {:?}", builder.model_family());
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("candle_vllm_tauri_example=info".parse().unwrap())
        )
        .init();
    
    info!("candle-vllm Tauri Example");
    info!("=========================");
    info!("");
    info!("This example demonstrates library-first usage of candle-vllm");
    info!("suitable for embedding in desktop applications like Tauri.");
    info!("");
    
    // Create application state
    let state = AppState::new("demo-model".to_string());
    
    // Demonstrate non-streaming chat
    info!("--- Non-streaming Chat Demo ---");
    let request = FrontendChatRequest {
        message: "Hello, how are you?".to_string(),
        system_prompt: Some("You are a helpful assistant.".to_string()),
        temperature: Some(0.7),
        max_tokens: Some(100),
        tools: None,
    };
    
    match chat_completion(&state, request).await {
        Ok(response) => {
            info!("Response: {:?}", response.content);
        }
        Err(e) => {
            error!("Error: {}", e);
        }
    }
    
    info!("");
    
    // Demonstrate streaming chat
    info!("--- Streaming Chat Demo ---");
    let stream_request = FrontendChatRequest {
        message: "Tell me a story.".to_string(),
        system_prompt: None,
        temperature: Some(0.8),
        max_tokens: Some(200),
        tools: None,
    };
    
    match chat_completion_stream(&state, stream_request).await {
        Ok(mut rx) => {
            info!("Streaming response:");
            let mut full_response = String::new();
            while let Some(chunk) = rx.recv().await {
                if let Some(content) = chunk.content {
                    full_response.push_str(&content);
                    print!("{}", content);
                }
                if chunk.is_complete {
                    println!();
                    info!("Stream complete. Full response: {}", full_response);
                }
            }
        }
        Err(e) => {
            error!("Error: {}", e);
        }
    }
    
    info!("");
    
    // Demonstrate tool parsing
    demonstrate_tool_parsing();
    
    info!("");
    
    // Demonstrate conversation building
    demonstrate_conversation();
    
    info!("");
    info!("=========================");
    info!("Example completed!");
    info!("");
    info!("To use this in a real Tauri app:");
    info!("1. Add tauri as a dependency");
    info!("2. Decorate functions with #[tauri::command]");
    info!("3. Register commands in tauri::Builder");
    info!("4. Use Tauri events for streaming responses");
    info!("");
    info!("For model loading, you would initialize the InferenceEngine");
    info!("from candle-vllm-core and store it in Tauri's managed state.");
    
    Ok(())
}

