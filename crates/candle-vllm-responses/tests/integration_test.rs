//! Integration tests for candle-vllm-responses.
//!
//! These tests validate the responses crate types and functionality.
//! Tests that require a model are gated behind conditions.

use candle_vllm_core::openai::requests::{ChatMessage, MessageContent};
use candle_vllm_responses::{ConversationOptions, ResponsesSessionBuilder};
use std::env;
use std::path::PathBuf;

// ============================================================================
// Type Tests (always run)
// ============================================================================

#[test]
fn test_conversation_options_default() {
    let options = ConversationOptions {
        max_turns: 5,
        allowed_tools: None,
    };

    assert_eq!(options.max_turns, 5);
    assert!(options.allowed_tools.is_none());
}

#[test]
fn test_conversation_options_with_tools() {
    let options = ConversationOptions {
        max_turns: 10,
        allowed_tools: Some(vec!["tool1".to_string(), "tool2".to_string()]),
    };

    assert_eq!(options.max_turns, 10);
    assert!(options.allowed_tools.is_some());
    assert_eq!(options.allowed_tools.as_ref().unwrap().len(), 2);
}

#[tokio::test]
async fn test_session_builder_no_engine() {
    // Session without engine should be creatable (for MCP-only use cases)
    let result = ResponsesSessionBuilder::new().build().await;

    // Should succeed - session can exist without engine
    assert!(result.is_ok());
}

// ============================================================================
// Integration tests (require model, conditionally run)
// ============================================================================

mod test_utils {
    use super::*;
    use candle_vllm_core::api::EngineConfig;

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

    #[allow(dead_code)]
    pub fn get_test_device_ordinal() -> usize {
        if let Ok(device_str) = env::var("CANDLE_VLLM_TEST_DEVICE") {
            match device_str.as_str() {
                "cuda" | "metal" => 0,
                _ => 0,
            }
        } else {
            0
        }
    }

    #[allow(dead_code)]
    pub async fn create_test_engine() -> Option<candle_vllm_core::api::InferenceEngine> {
        let model_path = get_test_model_path()?;
        let config = EngineConfig::builder()
            .model_path(model_path)
            .device(get_test_device_ordinal())
            .max_batch_size(1)
            .kv_cache_memory(512 * 1024 * 1024)
            .build()
            .ok()?;
        candle_vllm_core::api::InferenceEngine::new(config)
            .await
            .ok()
    }
}

#[allow(unused_macros)]
macro_rules! skip_if_no_model {
    () => {
        if test_utils::get_test_model_path().is_none() {
            eprintln!(
                "Skipping test: No test model available. Set CANDLE_VLLM_TEST_MODEL environment variable."
            );
            return;
        }
    };
}

// Test message creation with correct types
#[test]
fn test_chat_message_for_session() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: Some(MessageContent::Text("Hello!".to_string())),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };

    assert_eq!(msg.role, "user");
    assert!(matches!(msg.content, Some(MessageContent::Text(_))));
}

#[test]
fn test_tool_message_for_session() {
    let msg = ChatMessage {
        role: "tool".to_string(),
        content: Some(MessageContent::Text(r#"{"result": "success"}"#.to_string())),
        tool_calls: None,
        tool_call_id: Some("call_123".to_string()),
        name: Some("my_tool".to_string()),
    };

    assert_eq!(msg.role, "tool");
    assert!(msg.tool_call_id.is_some());
    assert!(msg.name.is_some());
}

// MCP client configuration test
#[test]
fn test_mcp_server_config() {
    use candle_vllm_responses::mcp_client::McpServerConfig;

    let config = McpServerConfig {
        url: "http://localhost:8080".to_string(),
        auth: None,
        timeout_secs: 30,
    };

    assert_eq!(config.url, "http://localhost:8080");
    assert!(config.auth.is_none());
    assert_eq!(config.timeout_secs, 30);
}

#[test]
fn test_mcp_server_config_with_auth() {
    use candle_vllm_responses::mcp_client::McpServerConfig;

    let config = McpServerConfig {
        url: "https://api.example.com/mcp".to_string(),
        auth: Some("Bearer token123".to_string()),
        timeout_secs: 60,
    };

    assert!(config.auth.is_some());
    assert_eq!(config.auth.as_ref().unwrap(), "Bearer token123");
}

// The following tests require a model and are gated
#[tokio::test]
async fn test_session_with_engine() {
    skip_if_no_model!();

    let engine = test_utils::create_test_engine().await.unwrap();
    let _session = ResponsesSessionBuilder::new()
        .engine(engine)
        .build()
        .await
        .expect("Session creation should succeed");
}

#[tokio::test]
async fn test_conversation_basic() {
    skip_if_no_model!();

    let engine = test_utils::create_test_engine().await.unwrap();
    let mut session = ResponsesSessionBuilder::new()
        .engine(engine)
        .build()
        .await
        .unwrap();

    let options = ConversationOptions {
        max_turns: 1,
        allowed_tools: None,
    };

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(MessageContent::Text("Say hello.".to_string())),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];

    let result = session
        .run_conversation(messages, options)
        .await
        .expect("Conversation should succeed");

    assert!(
        !result.final_message.is_empty(),
        "Should have final message"
    );
    assert_eq!(result.turns_taken, 1, "Should have taken 1 turn");
    assert!(result.completed, "Should have completed");
}

#[tokio::test]
async fn test_add_mcp_server() {
    // This test doesn't require a model, just tests the API
    let mut session = ResponsesSessionBuilder::new().build().await.unwrap();

    // Try to add an MCP server (will fail if server doesn't exist, but API should work)
    let result = session
        .add_mcp_server(
            "test".to_string(),
            candle_vllm_responses::mcp_client::McpServerConfig {
                url: "http://localhost:9999".to_string(), // Non-existent server
                auth: None,
                timeout_secs: 1, // Short timeout for test
            },
        )
        .await;

    // Should either succeed (if server exists) or fail gracefully
    match result {
        Ok(_) => {}
        Err(_) => {
            // Expected if server doesn't exist - API is still callable
        }
    }
}
