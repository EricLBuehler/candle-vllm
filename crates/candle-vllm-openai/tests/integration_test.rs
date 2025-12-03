//! Integration tests for candle-vllm-openai.
//!
//! These tests require a model to be available. Set CANDLE_VLLM_TEST_MODEL
//! environment variable to point to a model directory.

use candle_vllm_openai::adapter::OpenAIAdapter;
use candle_vllm_core::openai::requests::{ChatCompletionRequest, ChatMessage, Messages};
use std::env;
use std::path::PathBuf;
use candle_vllm_core::api::{EngineConfig, InferenceEngine};

mod test_utils {
    use super::*;

    pub fn get_test_model_path() -> Option<PathBuf> {
        env::var("CANDLE_VLLM_TEST_MODEL")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                let candidates = vec![
                    "./test-models/mistral-7b",
                    "./models/mistral-7b",
                    "../test-models/mistral-7b",
                ];
                candidates
                    .into_iter()
                    .map(PathBuf::from)
                    .find(|p| p.exists())
            })
    }

    pub fn get_test_device_ordinal() -> usize {
        if let Ok(device_str) = env::var("CANDLE_VLLM_TEST_DEVICE") {
            match device_str.as_str() {
                "cuda" | "metal" => 0, // Use device 0 for GPU
                _ => 0,
            }
        } else {
            0 // Default to device 0
        }
    }

    pub async fn create_test_engine() -> Option<InferenceEngine> {
        let model_path = get_test_model_path()?;
        let config = EngineConfig::builder()
            .model_path(model_path)
            .device(get_test_device_ordinal())
            .max_batch_size(1)
            .kv_cache_memory(512 * 1024 * 1024)
            .build()
            .ok()?;
        InferenceEngine::new(config).await.ok()
    }

}

use test_utils::*;

macro_rules! skip_if_no_model {
    () => {
        if test_utils::get_test_model_path().is_none() {
            eprintln!("Skipping test: No test model available. Set CANDLE_VLLM_TEST_MODEL environment variable.");
            return;
        }
    };
}

#[tokio::test]
async fn test_adapter_creation() {
    skip_if_no_model!();
    
    let engine = test_utils::create_test_engine().await.unwrap();
    let _adapter = OpenAIAdapter::new(engine);
    
    // Verify adapter can be created
    // (Adapter doesn't expose engine directly, but we can test chat_completion)
}

#[tokio::test]
async fn test_chat_completion_basic() {
    skip_if_no_model!();
    
    let engine = test_utils::create_test_engine().await.unwrap();
    let mut adapter = OpenAIAdapter::new(engine);
    
    let request = ChatCompletionRequest {
        model: "local".to_string(),
        messages: Messages::Chat(vec![ChatMessage {
            role: "user".to_string(),
            content: Some("Say hello in one word.".to_string()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }]),
        max_tokens: Some(10),
        temperature: Some(0.7),
        top_p: None,
        min_p: None,
        n: None,
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
    };
    
    let response = adapter.chat_completion(request).await.expect("Chat completion should succeed");
    
    assert_eq!(response.model, "local");
    assert!(!response.choices.is_empty(), "Should have choices");
    
    let choice = &response.choices[0];
    assert_eq!(choice.message.role, "assistant");
    assert!(choice.message.content.is_some(), "Should have content");
    
    let content = choice.message.content.as_ref().unwrap();
    assert!(!content.is_empty(), "Content should not be empty");
}

#[tokio::test]
async fn test_chat_completion_multi_turn() {
    skip_if_no_model!();
    
    let engine = test_utils::create_test_engine().await.unwrap();
    let mut adapter = OpenAIAdapter::new(engine);
    
    // First turn
    let request1 = ChatCompletionRequest {
        model: "local".to_string(),
        messages: Messages::Chat(vec![ChatMessage {
            role: "user".to_string(),
            content: Some("My name is Alice.".to_string()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }]),
        max_tokens: Some(5),
        temperature: Some(0.7),
        top_p: None,
        min_p: None,
        n: None,
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
    };
    
    let response1 = adapter.chat_completion(request1).await.expect("First turn should succeed");
    let assistant_data = response1.choices[0].message.clone();
    
    // Convert ChatChoiceData to ChatMessage
    let assistant_msg1 = ChatMessage {
        role: assistant_data.role.clone(),
        content: assistant_data.content.clone(),
        tool_calls: assistant_data.tool_calls.clone(),
        tool_call_id: None,
        name: None,
    };
    
    // Second turn with conversation history
    let request2 = ChatCompletionRequest {
        model: "local".to_string(),
        messages: Messages::Chat(vec![
            ChatMessage {
                role: "user".to_string(),
                content: Some("My name is Alice.".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            assistant_msg1,
            ChatMessage {
                role: "user".to_string(),
                content: Some("What is my name?".to_string()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ]),
        max_tokens: Some(10),
        temperature: Some(0.7),
        top_p: None,
        min_p: None,
        n: None,
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
    };
    
    let response2 = adapter.chat_completion(request2).await.expect("Second turn should succeed");
    let content = response2.choices[0].message.content.as_ref().unwrap();
    
    // The model should remember the name (though not guaranteed)
    assert!(!content.is_empty(), "Should have response");
}

#[tokio::test]
async fn test_chat_completion_with_tools() {
    skip_if_no_model!();
    
    let engine = test_utils::create_test_engine().await.unwrap();
    let mut adapter = OpenAIAdapter::new(engine);
    
    let request = ChatCompletionRequest {
        model: "local".to_string(),
        messages: Messages::Chat(vec![ChatMessage {
            role: "user".to_string(),
            content: Some("What is the weather in San Francisco?".to_string()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }]),
        tools: Some(vec![candle_vllm_core::openai::requests::Tool {
            tool_type: "function".to_string(),
            function: candle_vllm_core::openai::requests::FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get the weather for a location".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                })),
                strict: None,
            },
        }]),
        tool_choice: None,
        max_tokens: Some(50),
        temperature: Some(0.7),
        top_p: None,
        min_p: None,
        n: None,
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
        parallel_tool_calls: None,
    };
    
    let response = adapter.chat_completion(request).await.expect("Chat completion with tools should succeed");
    
    assert!(!response.choices.is_empty());
    let choice = &response.choices[0];
    
    // The model may or may not call the tool depending on its training
    // This test verifies the API works with tools, not that tools are called
    if let Some(ref tool_calls) = choice.message.tool_calls {
        assert!(!tool_calls.is_empty(), "If tool calls are present, they should not be empty");
        let tool_call = &tool_calls[0];
        assert_eq!(tool_call.call_type, "function");
        assert!(!tool_call.function.name.is_empty());
    }
}

#[tokio::test]
async fn test_chat_completion_parameters() {
    skip_if_no_model!();
    
    let engine = test_utils::create_test_engine().await.unwrap();
    let mut adapter = OpenAIAdapter::new(engine);
    
    // Test with various parameters
    let request = ChatCompletionRequest {
        model: "local".to_string(),
        messages: Messages::Chat(vec![ChatMessage {
            role: "user".to_string(),
            content: Some("Count to three:".to_string()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }]),
        max_tokens: Some(10),
        temperature: Some(0.1), // Low temperature for deterministic output
        top_p: Some(0.9),
        top_k: Some(40),
        frequency_penalty: Some(0.5),
        presence_penalty: Some(0.5),
        min_p: None,
        n: None,
        stop: None,
        stream: None,
        repeat_last_n: None,
        logit_bias: None,
        user: None,
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
    };
    
    let response = adapter.chat_completion(request).await.expect("Chat completion with parameters should succeed");
    assert!(!response.choices.is_empty());
    assert!(response.choices[0].message.content.is_some());
}

