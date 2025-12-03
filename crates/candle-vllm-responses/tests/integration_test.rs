//! Integration tests for candle-vllm-responses.
//!
//! These tests require a model and optionally MCP servers to be available.

use candle_vllm_responses::{ResponsesSession, ResponsesSessionBuilder, ConversationOptions};
use candle_vllm_core::openai::requests::ChatMessage;
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
async fn test_session_creation() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await.unwrap();
    let session = ResponsesSessionBuilder::new()
        .engine(engine)
        .build()
        .await
        .expect("Session creation should succeed");
    
    // Session should be created successfully
    // (We can't easily test internal state without public accessors)
}

#[tokio::test]
async fn test_session_without_engine() {
    let _session = ResponsesSessionBuilder::new()
        .build()
        .await
        .expect("Session without engine should succeed");
    
    // Session without engine should still be creatable
    // (for MCP-only use cases)
}

#[tokio::test]
async fn test_conversation_basic() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await.unwrap();
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
        content: Some("Say hello.".to_string()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];
    
    let result = session.run_conversation(messages, options).await
        .expect("Conversation should succeed");
    
    assert!(!result.final_message.is_empty(), "Should have final message");
    assert_eq!(result.turns_taken, 1, "Should have taken 1 turn");
    assert!(result.completed, "Should have completed");
    assert!(result.tool_calls.is_empty(), "Should have no tool calls for simple request");
}

#[tokio::test]
async fn test_conversation_max_turns() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await.unwrap();
    let mut session = ResponsesSessionBuilder::new()
        .engine(engine)
        .build()
        .await
        .unwrap();
    
    let options = ConversationOptions {
        max_turns: 2,
        allowed_tools: None,
    };
    
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some("Hello.".to_string()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];
    
    let result = session.run_conversation(messages, options).await
        .expect("Conversation should succeed");
    
    assert!(result.turns_taken <= 2, "Should respect max_turns");
}

#[tokio::test]
async fn test_list_openai_tools() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await.unwrap();
    let _session = ResponsesSessionBuilder::new()
        .engine(engine)
        .build()
        .await
        .unwrap();
    
    // Test listing tools when no MCP servers are configured
    // Note: This test verifies session creation, actual tool listing
    // would require MCP servers to be configured
}

#[tokio::test]
async fn test_add_mcp_server() {
    skip_if_no_model!();
    
    let engine = create_test_engine().await.unwrap();
    let mut session = ResponsesSessionBuilder::new()
        .engine(engine)
        .build()
        .await
        .unwrap();
    
    // Try to add an MCP server (will fail if server doesn't exist, but API should work)
    let result = session.add_mcp_server(
        "test".to_string(),
        candle_vllm_responses::mcp_client::McpServerConfig {
            url: "http://localhost:9999".to_string(), // Non-existent server
            auth: None,
            timeout_secs: 1, // Short timeout for test
        },
    ).await;
    
    // Should either succeed (if server exists) or fail gracefully
    // The important thing is the API works
    match result {
        Ok(_) => {},
        Err(_) => {
            // Expected if server doesn't exist
            // This test verifies the API is callable
        }
    }
}

