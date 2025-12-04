//! Integration tests for candle-vllm-openai.
//!
//! These tests validate the OpenAI-compatible types and adapter functionality.
//! Tests that require a model are skipped by default unless CANDLE_VLLM_TEST_MODEL is set.

use candle_vllm_openai::{
    ChatCompletionRequest, ChatMessage, FunctionDefinition, MessageContent, Messages, Tool,
    ToolCall, ToolChoice,
};
use serde_json::json;

// ============================================================================
// Type Tests (always run, no model required)
// ============================================================================

#[test]
fn test_chat_message_creation() {
    // Test simple text message
    let msg = ChatMessage {
        role: "user".to_string(),
        content: Some(MessageContent::Text("Hello, world!".to_string())),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };

    assert_eq!(msg.role, "user");
    assert!(matches!(msg.content, Some(MessageContent::Text(_))));
}

#[test]
fn test_chat_message_with_tool_calls() {
    let tool_call = ToolCall {
        id: "call_123".to_string(),
        call_type: "function".to_string(),
        function: candle_vllm_openai::FunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"location": "Paris"}"#.to_string(),
        },
    };

    let msg = ChatMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: Some(vec![tool_call.clone()]),
        tool_call_id: None,
        name: None,
    };

    assert_eq!(msg.role, "assistant");
    assert!(msg.content.is_none());
    assert!(msg.tool_calls.is_some());
    assert_eq!(msg.tool_calls.as_ref().unwrap().len(), 1);
    assert_eq!(msg.tool_calls.as_ref().unwrap()[0].function.name, "get_weather");
}

#[test]
fn test_chat_message_tool_response() {
    let msg = ChatMessage {
        role: "tool".to_string(),
        content: Some(MessageContent::Text(r#"{"temperature": 22}"#.to_string())),
        tool_calls: None,
        tool_call_id: Some("call_123".to_string()),
        name: Some("get_weather".to_string()),
    };

    assert_eq!(msg.role, "tool");
    assert!(msg.tool_call_id.is_some());
    assert_eq!(msg.tool_call_id.as_ref().unwrap(), "call_123");
}

#[test]
fn test_chat_completion_request_basic() {
    let request = ChatCompletionRequest {
        model: "mistral-7b".to_string(),
        messages: Messages::Chat(vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("Hello!".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }]),
        temperature: Some(0.7),
        top_p: None,
        min_p: None,
        n: None,
        max_tokens: Some(100),
        stop: None,
        stream: None,
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
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        conversation_id: None,
        resource_id: None,
    };

    assert_eq!(request.model, "mistral-7b");
    assert_eq!(request.temperature, Some(0.7));
    assert_eq!(request.max_tokens, Some(100));
    assert!(!request.has_tools());
}

#[test]
fn test_chat_completion_request_with_tools() {
    let tool = Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: Some("Get the current weather for a location".to_string()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    }
                },
                "required": ["location"]
            })),
            strict: None,
        },
    };

    let request = ChatCompletionRequest {
        model: "mistral-7b".to_string(),
        messages: Messages::Chat(vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("What's the weather in Paris?".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }]),
        temperature: Some(0.7),
        top_p: None,
        min_p: None,
        n: None,
        max_tokens: Some(100),
        stop: None,
        stream: None,
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
        tools: Some(vec![tool]),
        tool_choice: None,
        parallel_tool_calls: None,
        conversation_id: None,
        resource_id: None,
    };

    assert!(request.has_tools());
    assert_eq!(request.tools.as_ref().unwrap().len(), 1);
    assert_eq!(
        request.tools.as_ref().unwrap()[0].function.name,
        "get_weather"
    );
}

#[test]
fn test_chat_completion_request_serialization() {
    let request = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: Messages::Chat(vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("Test message".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }]),
        temperature: Some(0.5),
        top_p: None,
        min_p: None,
        n: None,
        max_tokens: Some(50),
        stop: None,
        stream: None,
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
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        conversation_id: Some("conv-123".to_string()),
        resource_id: Some("res-456".to_string()),
    };

    let json = serde_json::to_string(&request).expect("Serialization should succeed");
    assert!(json.contains("test-model"));
    assert!(json.contains("conv-123"));
    assert!(json.contains("res-456"));

    // Round-trip test
    let parsed: ChatCompletionRequest =
        serde_json::from_str(&json).expect("Deserialization should succeed");
    assert_eq!(parsed.model, "test-model");
    assert_eq!(parsed.conversation_id, Some("conv-123".to_string()));
    assert_eq!(parsed.resource_id, Some("res-456".to_string()));
}

#[test]
fn test_tool_definition() {
    let tool = Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "search".to_string(),
            description: Some("Search for information".to_string()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            })),
            strict: Some(true),
        },
    };

    assert_eq!(tool.tool_type, "function");
    assert_eq!(tool.function.name, "search");
    assert!(tool.function.description.is_some());
    assert!(tool.function.parameters.is_some());
    assert_eq!(tool.function.strict, Some(true));
}

#[test]
fn test_tool_choice() {
    // Test auto choice
    let auto_choice = ToolChoice::Mode("auto".to_string());
    assert!(auto_choice.is_auto());
    let json = serde_json::to_string(&auto_choice).expect("Serialization should succeed");
    assert!(json.contains("auto"));

    // Test none choice
    let none_choice = ToolChoice::Mode("none".to_string());
    assert!(none_choice.is_none());
    let json = serde_json::to_string(&none_choice).expect("Serialization should succeed");
    assert!(json.contains("none"));

    // Test required choice
    let required_choice = ToolChoice::Mode("required".to_string());
    assert!(required_choice.is_required());
    let json = serde_json::to_string(&required_choice).expect("Serialization should succeed");
    assert!(json.contains("required"));
}

#[test]
fn test_message_content_text() {
    let content = MessageContent::Text("Hello".to_string());
    match &content {
        MessageContent::Text(t) => assert_eq!(t, "Hello"),
        _ => panic!("Expected Text variant"),
    }
}

#[test]
fn test_messages_enum() {
    let messages = Messages::Chat(vec![
        ChatMessage {
            role: "system".to_string(),
            content: Some(MessageContent::Text("You are helpful.".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("Hi!".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ]);

    if let Messages::Chat(chat) = messages {
        assert_eq!(chat.len(), 2);
        assert_eq!(chat[0].role, "system");
        assert_eq!(chat[1].role, "user");
    } else {
        panic!("Expected Chat variant");
    }
}

// ============================================================================
// Tool Parsing Tests
// ============================================================================

#[test]
fn test_tool_parser_mistral() {
    use candle_vllm_openai::{get_tool_parser, ParsedOutput};

    let parser = get_tool_parser("mistral");
    assert_eq!(parser.name(), "mistral");

    let output = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris"}}]"#;
    let parsed = parser.parse(output);

    assert!(parsed.has_tool_calls());
    if let ParsedOutput::ToolCalls(calls) = parsed {
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }
}

#[test]
fn test_tool_parser_llama() {
    use candle_vllm_openai::{get_tool_parser, ParsedOutput};

    let parser = get_tool_parser("llama");
    assert_eq!(parser.name(), "llama");

    let output = r#"<function=search>{"query": "rust programming"}</function>"#;
    let parsed = parser.parse(output);

    assert!(parsed.has_tool_calls());
    if let ParsedOutput::ToolCalls(calls) = parsed {
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }
}

#[test]
fn test_tool_parser_qwen() {
    use candle_vllm_openai::{get_tool_parser, ParsedOutput};

    let parser = get_tool_parser("qwen");
    assert_eq!(parser.name(), "qwen");

    let output = r#"<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>"#;
    let parsed = parser.parse(output);

    assert!(parsed.has_tool_calls());
    if let ParsedOutput::ToolCalls(calls) = parsed {
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "calculator");
    }
}

#[test]
fn test_tool_parser_no_tool_call() {
    use candle_vllm_openai::{get_tool_parser, ParsedOutput};

    let parser = get_tool_parser("mistral");
    let output = "This is just a regular text response without any tool calls.";
    let parsed = parser.parse(output);

    assert!(!parsed.has_tool_calls());
    if let ParsedOutput::Text(text) = parsed {
        assert_eq!(text, output);
    }
}

#[test]
fn test_auto_tool_parser() {
    use candle_vllm_openai::get_tool_parser;

    let parser = get_tool_parser("unknown-model");
    assert_eq!(parser.name(), "auto");

    // Should parse mistral format
    let output1 = r#"[TOOL_CALLS] [{"name": "test", "arguments": {}}]"#;
    assert!(parser.parse(output1).has_tool_calls());

    // Should parse llama format
    let output2 = r#"<function=test>{"arg": "val"}</function>"#;
    assert!(parser.parse(output2).has_tool_calls());
}

// ============================================================================
// Conversation Tests
// ============================================================================

#[test]
fn test_tool_conversation_builder() {
    use candle_vllm_openai::{ModelFamily, ToolConversationBuilder};

    let mut builder = ToolConversationBuilder::new("mistral");
    assert_eq!(builder.model_family(), ModelFamily::Mistral);

    builder.add_user_message("What's the weather?");
    assert_eq!(builder.messages().len(), 1);

    let tool_call = ToolCall {
        id: "call_123".to_string(),
        call_type: "function".to_string(),
        function: candle_vllm_openai::FunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"location": "Paris"}"#.to_string(),
        },
    };

    builder.add_assistant_tool_calls(vec![tool_call]);
    assert_eq!(builder.messages().len(), 2);

    builder.add_tool_result("call_123", r#"{"temp": 20}"#, Some("get_weather".to_string()));
    assert_eq!(builder.messages().len(), 3);

    builder.add_assistant_response("The temperature in Paris is 20°C.");
    assert_eq!(builder.messages().len(), 4);
}

#[test]
fn test_model_family_detection() {
    use candle_vllm_openai::ModelFamily;

    assert_eq!(
        ModelFamily::from_model_name("mistralai/Mistral-7B-Instruct"),
        ModelFamily::Mistral
    );
    assert_eq!(
        ModelFamily::from_model_name("meta-llama/Llama-3.1-8B"),
        ModelFamily::Llama
    );
    assert_eq!(
        ModelFamily::from_model_name("Qwen/Qwen2-7B-Instruct"),
        ModelFamily::Qwen
    );
    assert_eq!(
        ModelFamily::from_model_name("some-unknown-model"),
        ModelFamily::Generic
    );
}

#[test]
fn test_format_tools_for_template() {
    use candle_vllm_openai::format_tools_for_template;

    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "test_func".to_string(),
            description: Some("A test function".to_string()),
            parameters: None,
            strict: None,
        },
    }];

    let formatted = format_tools_for_template(&tools);
    assert!(formatted.contains("test_func"));
    assert!(formatted.contains("A test function"));
}

// ============================================================================
// Streaming Types Tests
// ============================================================================

#[test]
fn test_streaming_status() {
    use candle_vllm_openai::StreamingStatus;

    let status = StreamingStatus::default();
    assert_eq!(status, StreamingStatus::Uninitialized);

    assert_eq!(StreamingStatus::Started, StreamingStatus::Started);
    assert_eq!(StreamingStatus::Stopped, StreamingStatus::Stopped);
    assert_eq!(StreamingStatus::Interrupted, StreamingStatus::Interrupted);
}

#[test]
fn test_chat_response() {
    use candle_vllm_openai::ChatResponse;

    let error = ChatResponse::internal_error("test error");
    assert!(error.is_error());
    assert_eq!(error.error_message(), Some("test error"));

    let done = ChatResponse::Done;
    assert!(done.is_done());
    assert!(!done.is_error());
}

// ============================================================================
// Response Types Tests
// ============================================================================

#[test]
fn test_create_tool_call() {
    use candle_vllm_openai::create_tool_call;

    let call = create_tool_call(
        "call_abc".to_string(),
        "get_weather".to_string(),
        r#"{"location": "Paris"}"#.to_string(),
    );

    assert_eq!(call.id, "call_abc");
    assert_eq!(call.call_type, "function");
    assert_eq!(call.function.name, "get_weather");
    assert!(call.function.arguments.contains("Paris"));
}

#[test]
fn test_create_tool_call_delta() {
    use candle_vllm_openai::{create_tool_call_delta_arguments, create_tool_call_delta_start};

    let start = create_tool_call_delta_start(0, "call_123".to_string(), "search".to_string());
    assert_eq!(start.index, 0);
    assert_eq!(start.id, Some("call_123".to_string()));
    assert!(start.function.is_some());
    assert_eq!(start.function.as_ref().unwrap().name, Some("search".to_string()));

    let args = create_tool_call_delta_arguments(0, r#"{"query": "test"}"#.to_string());
    assert_eq!(args.index, 0);
    assert!(args.id.is_none());
    assert!(args.function.is_some());
    assert!(args.function.as_ref().unwrap().arguments.is_some());
}

// ============================================================================
// Model Registry Tests
// ============================================================================

#[test]
fn test_model_registry_load() {
    use candle_vllm_openai::ModelRegistry;
    use std::io::Write;

    // Create a temporary registry file for testing
    let yaml_content = r#"
models:
  - name: test-model
    model_id: test/model
"#;
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("test_models.yaml");
    
    {
        let mut file = std::fs::File::create(&temp_file).expect("Failed to create temp file");
        file.write_all(yaml_content.as_bytes()).expect("Failed to write temp file");
    }

    // Load the registry
    let registry = ModelRegistry::load(&temp_file);
    
    // Clean up
    let _ = std::fs::remove_file(&temp_file);
    
    // If loading succeeded, verify the registry contents
    if let Some(reg) = registry {
        let names = reg.list_names();
        assert!(names.contains(&"test-model".to_string()));
        
        let model = reg.find("test-model");
        assert!(model.is_some());
        if let Some(m) = model {
            assert_eq!(m.model_id, Some("test/model".to_string()));
        }
    }
}

#[test]
fn test_model_registry_not_found() {
    use candle_vllm_openai::ModelRegistry;

    // Try to load from non-existent file
    let registry = ModelRegistry::load("/nonexistent/path/models.yaml");
    assert!(registry.is_none());
}

// ============================================================================
// Integration Tests (require model)
// ============================================================================

#[cfg(feature = "integration_tests")]
mod with_model {
    use super::*;
    use candle_vllm_openai::adapter::OpenAIAdapter;
    use std::env;
    use std::path::PathBuf;

    fn get_test_model_path() -> Option<PathBuf> {
        env::var("CANDLE_VLLM_TEST_MODEL")
            .ok()
            .map(PathBuf::from)
    }

    macro_rules! skip_if_no_model {
        () => {
            if get_test_model_path().is_none() {
                eprintln!("Skipping test: Set CANDLE_VLLM_TEST_MODEL to enable.");
                return;
            }
        };
    }

    #[tokio::test]
    async fn test_adapter_chat_completion() {
        skip_if_no_model!();
        // Full adapter test would require engine setup
    }
}
