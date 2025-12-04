//! Integration tests for candle-vllm-core.
//!
//! These tests validate the core types and API. Tests that require a model
//! are gated behind the integration_tests feature and require CANDLE_VLLM_TEST_MODEL.

use candle_vllm_core::api::{EngineConfig, Error, GenerationParams, FinishReason};
use candle_vllm_core::openai::requests::{
    ChatCompletionRequest, ChatMessage, FunctionDefinition, MessageContent, Messages, Tool,
    ToolCall,
};
use candle_vllm_core::openai::responses::{
    APIError, ChatChoice, ChatChoiceData, ChatCompletionResponse, ChatCompletionUsageResponse,
    Choice, ChoiceData,
};
use candle_vllm_core::openai::sampling_params::SamplingParams;
use serde_json::json;
use std::path::PathBuf;

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_engine_config_from_path() {
    let config = EngineConfig::from_model_path("/fake/path");
    assert_eq!(config.model_path, PathBuf::from("/fake/path"));
    assert!(config.device.is_none());
    assert!(config.dtype.is_none());
    assert!(!config.enable_cuda_graph);
}

#[test]
fn test_engine_config_builder() {
    let config = EngineConfig::builder()
        .model_path("/test/model")
        .device(0)
        .max_batch_size(8)
        .kv_cache_memory(1024 * 1024 * 1024)
        .enable_cuda_graph(true)
        .build()
        .expect("Config build should succeed");

    assert_eq!(config.model_path, PathBuf::from("/test/model"));
    assert_eq!(config.device, Some(0));
    assert_eq!(config.max_batch_size, Some(8));
    assert_eq!(config.kv_cache_memory, Some(1024 * 1024 * 1024));
    assert!(config.enable_cuda_graph);
}

#[test]
fn test_engine_config_builder_missing_path() {
    let result = EngineConfig::builder()
        .device(0)
        .build();

    assert!(result.is_err());
    if let Err(Error::Config(msg)) = result {
        assert!(msg.contains("model_path"));
    }
}

// ============================================================================
// Generation Params Tests
// ============================================================================

#[test]
fn test_generation_params_default() {
    let params = GenerationParams::default();

    assert_eq!(params.max_tokens, Some(128));
    assert_eq!(params.temperature, Some(0.7));
    assert_eq!(params.top_p, Some(1.0));
    assert_eq!(params.top_k, Some(-1));
    assert!(params.stop_sequences.is_none());
}

#[test]
fn test_generation_params_custom() {
    let params = GenerationParams {
        max_tokens: Some(50),
        temperature: Some(0.5),
        top_p: Some(0.9),
        top_k: Some(40),
        repetition_penalty: Some(1.1),
        frequency_penalty: Some(0.5),
        presence_penalty: Some(0.5),
        stop_sequences: Some(vec!["END".to_string()]),
        logprobs: Some(5),
        seed: Some(42),
    };

    assert_eq!(params.max_tokens, Some(50));
    assert_eq!(params.temperature, Some(0.5));
    assert_eq!(params.stop_sequences, Some(vec!["END".to_string()]));
    assert_eq!(params.seed, Some(42));
}

// ============================================================================
// Finish Reason Tests
// ============================================================================

#[test]
fn test_finish_reason_equality() {
    assert_eq!(FinishReason::Stop, FinishReason::Stop);
    assert_eq!(FinishReason::Length, FinishReason::Length);
    assert_eq!(FinishReason::StopSequence, FinishReason::StopSequence);
    assert_eq!(FinishReason::Cancelled, FinishReason::Cancelled);

    assert_ne!(FinishReason::Stop, FinishReason::Length);
    assert_ne!(FinishReason::Stop, FinishReason::Error("test".to_string()));
}

#[test]
fn test_finish_reason_error() {
    let error = FinishReason::Error("Test error".to_string());
    if let FinishReason::Error(msg) = error {
        assert_eq!(msg, "Test error");
    } else {
        panic!("Expected Error variant");
    }
}

// ============================================================================
// OpenAI Request Types Tests
// ============================================================================

#[test]
fn test_chat_message_basic() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: Some(MessageContent::Text("Hello".to_string())),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };

    assert_eq!(msg.role, "user");
    assert!(matches!(msg.content, Some(MessageContent::Text(_))));
}

#[test]
fn test_chat_message_with_tool_call() {
    let tool_call = ToolCall {
        id: "call_123".to_string(),
        call_type: "function".to_string(),
        function: candle_vllm_core::openai::requests::FunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"location": "Paris"}"#.to_string(),
        },
    };

    let msg = ChatMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: Some(vec![tool_call]),
        tool_call_id: None,
        name: None,
    };

    assert!(msg.tool_calls.is_some());
    assert_eq!(msg.tool_calls.as_ref().unwrap().len(), 1);
}

#[test]
fn test_chat_completion_request_basic() {
    let request = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: Messages::Chat(vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("Test".to_string())),
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

    assert_eq!(request.model, "test-model");
    assert_eq!(request.max_tokens, Some(100));
    assert!(!request.has_tools());
}

#[test]
fn test_chat_completion_request_with_tools() {
    let tool = Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "search".to_string(),
            description: Some("Search for info".to_string()),
            parameters: Some(json!({"type": "object"})),
            strict: None,
        },
    };

    let request = ChatCompletionRequest {
        model: "test".to_string(),
        messages: Messages::Chat(vec![]),
        temperature: None,
        top_p: None,
        min_p: None,
        n: None,
        max_tokens: None,
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
}

#[test]
fn test_chat_completion_request_serialization() {
    let request = ChatCompletionRequest {
        model: "test".to_string(),
        messages: Messages::Chat(vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("Hello".to_string())),
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
    assert!(json.contains("test"));
    assert!(json.contains("conv-123"));

    let parsed: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.model, "test");
    assert_eq!(parsed.conversation_id, Some("conv-123".to_string()));
}

// ============================================================================
// OpenAI Response Types Tests
// ============================================================================

#[test]
fn test_api_error() {
    let error = APIError::new("Test error".to_string());
    assert!(error.to_string().contains("Test error"));

    let error2 = APIError::new_str("Another error");
    assert!(error2.to_string().contains("Another error"));

    let error3 = APIError::from("From string");
    assert!(error3.to_string().contains("From string"));
}

#[test]
fn test_chat_choice_data() {
    let data = ChatChoiceData::text("Hello world".to_string());
    assert_eq!(data.content, Some("Hello world".to_string()));
    assert_eq!(data.role, "assistant");
    assert!(!data.has_tool_calls());

    let tool_call = ToolCall {
        id: "call_1".to_string(),
        call_type: "function".to_string(),
        function: candle_vllm_core::openai::requests::FunctionCall {
            name: "test".to_string(),
            arguments: "{}".to_string(),
        },
    };

    let data_with_tools = ChatChoiceData::with_tool_calls(vec![tool_call]);
    assert!(data_with_tools.content.is_none());
    assert!(data_with_tools.has_tool_calls());
}

#[test]
fn test_chat_choice() {
    let choice = ChatChoice {
        message: ChatChoiceData::text("Response".to_string()),
        finish_reason: Some("stop".to_string()),
        index: 0,
        logprobs: None,
    };

    assert_eq!(choice.determine_finish_reason(), "stop");

    let tool_call = ToolCall {
        id: "call_1".to_string(),
        call_type: "function".to_string(),
        function: candle_vllm_core::openai::requests::FunctionCall {
            name: "test".to_string(),
            arguments: "{}".to_string(),
        },
    };

    let choice_with_tools = ChatChoice {
        message: ChatChoiceData::with_tool_calls(vec![tool_call]),
        finish_reason: Some("stop".to_string()),
        index: 0,
        logprobs: None,
    };

    assert_eq!(choice_with_tools.determine_finish_reason(), "tool_calls");
}

#[test]
fn test_chat_completion_response() {
    let response = ChatCompletionResponse {
        id: "cmpl-123".to_string(),
        choices: vec![ChatChoice {
            message: ChatChoiceData::text("Hello".to_string()),
            finish_reason: Some("stop".to_string()),
            index: 0,
            logprobs: None,
        }],
        created: 1234567890,
        model: "test-model".to_string(),
        object: "chat.completion",
        usage: ChatCompletionUsageResponse {
            request_id: "req-123".to_string(),
            created: 1234567890,
            completion_tokens: 5,
            prompt_tokens: 10,
            total_tokens: 15,
            prompt_time_costs: 100,
            completion_time_costs: 200,
        },
        conversation_id: Some("conv-123".to_string()),
        resource_id: None,
    };

    assert_eq!(response.id, "cmpl-123");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.usage.total_tokens, 15);
    assert_eq!(response.conversation_id, Some("conv-123".to_string()));
}

#[test]
fn test_streaming_choice_data() {
    let content_delta = ChoiceData::content("chunk".to_string());
    assert_eq!(content_delta.content, Some("chunk".to_string()));
    assert!(!content_delta.is_empty());

    let role_delta = ChoiceData::role("assistant".to_string());
    assert_eq!(role_delta.role, Some("assistant".to_string()));
    assert!(!role_delta.is_empty());

    let empty_delta = ChoiceData::empty();
    assert!(empty_delta.is_empty());
}

#[test]
fn test_streaming_choice() {
    let content_chunk = Choice::content_chunk(0, "hello".to_string());
    assert_eq!(content_chunk.index, 0);
    assert!(content_chunk.finish_reason.is_none());
    assert_eq!(content_chunk.delta.content, Some("hello".to_string()));

    let finish_chunk = Choice::finish_chunk(0, "stop");
    assert_eq!(finish_chunk.finish_reason, Some("stop".to_string()));
    assert!(finish_chunk.delta.is_empty());
}

// ============================================================================
// Sampling Params Tests
// ============================================================================

#[test]
fn test_sampling_params_new() {
    use candle_vllm_core::openai::sampling_params::EarlyStoppingCondition;
    
    let params = SamplingParams::new(
        1,                                              // n
        None,                                           // best_of
        0.0,                                            // presence_penalty
        0.0,                                            // frequency_penalty
        None,                                           // repeat_last_n
        Some(0.7),                                      // temperature
        None,                                           // top_p
        None,                                           // min_p
        None,                                           // top_k
        false,                                          // use_beam_search
        1.0,                                            // length_penalty
        EarlyStoppingCondition::UnlikelyBetterCandidates, // early_stopping
        None,                                           // stop
        vec![],                                         // stop_token_ids
        false,                                          // ignore_eos
        256,                                            // max_tokens
        None,                                           // logprobs
        None,                                           // prompt_logprobs
        true,                                           // skip_special_tokens
        None,                                           // thinking
    ).expect("SamplingParams should be created");

    assert_eq!(params.temperature, Some(0.7));
    assert_eq!(params.max_tokens, 256);
    assert_eq!(params.n, 1);
}

#[test]
fn test_sampling_params_temperature() {
    use candle_vllm_core::openai::sampling_params::EarlyStoppingCondition;
    
    let params = SamplingParams::new(
        1,
        None,
        0.0,
        0.0,
        None,
        Some(0.5), // Different temperature
        None,
        None,
        None,
        false,
        1.0,
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        None,
        vec![],
        false,
        100,
        None,
        None,
        true,
        None,
    ).expect("SamplingParams should be created");

    assert_eq!(params.temperature, Some(0.5));
}

// ============================================================================
// Tool Parser Tests (using core's tool_parser module)
// ============================================================================

#[test]
fn test_tool_parser_mistral_format() {
    use candle_vllm_core::openai::tool_parser::{get_tool_parser, ParsedOutput};

    let parser = get_tool_parser("mistral");
    let output = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris"}}]"#;
    let parsed = parser.parse(output);

    assert!(parsed.has_tool_calls());
    if let ParsedOutput::ToolCalls(calls) = parsed {
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    } else {
        panic!("Expected ToolCalls variant");
    }
}

#[test]
fn test_tool_parser_no_tool_call() {
    use candle_vllm_core::openai::tool_parser::{get_tool_parser, ParsedOutput};

    let parser = get_tool_parser("mistral");
    let output = "Just a regular text response without any tool calls.";
    let parsed = parser.parse(output);

    assert!(!parsed.has_tool_calls());
    if let ParsedOutput::Text(text) = parsed {
        assert_eq!(text, output);
    } else {
        panic!("Expected Text variant");
    }
}

// ============================================================================
// Error Type Tests
// ============================================================================

#[test]
fn test_error_types() {
    let model_err = Error::ModelLoad("Failed to load".to_string());
    assert!(model_err.to_string().contains("model load error"));

    let tok_err = Error::Tokenization("Bad token".to_string());
    assert!(tok_err.to_string().contains("tokenization error"));

    let gen_err = Error::Generation("Gen failed".to_string());
    assert!(gen_err.to_string().contains("generation error"));

    let cancelled = Error::Cancelled;
    assert!(cancelled.to_string().contains("cancelled"));
}
