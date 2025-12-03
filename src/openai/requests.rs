use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ============================================================================
// Tool Calling Types (OpenAI Compatible)
// ============================================================================

/// Definition of a function that can be called by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// The name of the function to be called
    pub name: String,
    /// A description of what the function does
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The parameters the function accepts, described as a JSON Schema object
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    /// Whether to enforce strict parameter validation (OpenAI extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A tool that the model can use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// The type of tool (currently only "function" is supported)
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition
    pub function: FunctionDefinition,
}

/// Specifies a specific function to call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    /// The name of the function to call
    pub name: String,
}

/// Specifies a specific tool to use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceSpecific {
    /// The type of tool (currently only "function" is supported)
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function to call
    pub function: ToolChoiceFunction,
}

/// Controls which (if any) tool the model should use
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// "none", "auto", or "required"
    Mode(String),
    /// Specifies a specific tool to use
    Specific(ToolChoiceSpecific),
}

impl ToolChoice {
    pub fn is_none(&self) -> bool {
        matches!(self, ToolChoice::Mode(s) if s == "none")
    }

    pub fn is_auto(&self) -> bool {
        matches!(self, ToolChoice::Mode(s) if s == "auto")
    }

    pub fn is_required(&self) -> bool {
        matches!(self, ToolChoice::Mode(s) if s == "required")
    }
}

/// A function call made by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The name of the function to call
    pub name: String,
    /// The arguments to call the function with, as a JSON string
    pub arguments: String,
}

/// A tool call made by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// The ID of the tool call
    pub id: String,
    /// The type of tool call (currently only "function" is supported)
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function call details
    pub function: FunctionCall,
}

/// Delta for streaming tool calls - function part
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FunctionCallDelta {
    /// The name of the function (only present in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The arguments fragment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Delta for streaming tool calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// The index of the tool call in the tool_calls array
    pub index: usize,
    /// The ID of the tool call (only present in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The type of tool call (only present in first chunk)
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    /// The function call delta
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

// ============================================================================
// Message Types
// ============================================================================

/// A chat message with full support for tool calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of the message author (system, user, assistant, tool)
    pub role: String,
    /// The content of the message (can be null for assistant messages with tool_calls)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls made by the assistant (only for role="assistant")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The ID of the tool call this message is responding to (only for role="tool")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// The name of the function (for tool responses)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: None,
        }
    }
}

/// Messages can be either a list of chat messages or a raw string prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Messages {
    /// Structured chat messages (supports tool calls)
    Chat(Vec<ChatMessage>),
    /// Legacy format: simple map of role -> content
    Map(Vec<HashMap<String, String>>),
    /// Raw string prompt
    Literal(String),
}

impl Messages {
    /// Convert messages to ChatMessage format for unified processing
    pub fn to_chat_messages(&self) -> Vec<ChatMessage> {
        match self {
            Messages::Chat(messages) => messages.clone(),
            Messages::Map(maps) => maps
                .iter()
                .map(|m| ChatMessage {
                    role: m.get("role").cloned().unwrap_or_default(),
                    content: m.get("content").cloned(),
                    tool_calls: None,
                    tool_call_id: None,
                    name: m.get("name").cloned(),
                })
                .collect(),
            Messages::Literal(s) => vec![ChatMessage::user(s.clone())],
        }
    }

    /// Check if any message contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        match self {
            Messages::Chat(messages) => messages.iter().any(|m| m.tool_calls.is_some()),
            _ => false,
        }
    }

    /// Check if any message is a tool response
    pub fn has_tool_responses(&self) -> bool {
        match self {
            Messages::Chat(messages) => messages.iter().any(|m| m.role == "tool"),
            _ => false,
        }
    }
}

// ============================================================================
// Stop Tokens
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopTokens {
    Multi(Vec<String>),
    Single(String),
}

impl StopTokens {
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            StopTokens::Multi(v) => v.clone(),
            StopTokens::Single(s) => vec![s.clone()],
        }
    }
}

// ============================================================================
// Chat Completion Request
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// ID of the model to use
    pub model: String,
    /// The messages to generate completions for
    pub messages: Messages,
    /// Sampling temperature (0.0 to 2.0)
    pub temperature: Option<f32>,
    /// Nucleus sampling parameter
    pub top_p: Option<f32>,
    /// Minimum probability threshold
    pub min_p: Option<f32>,
    /// Number of completions to generate
    #[serde(default)]
    pub n: Option<usize>,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<usize>,
    /// Stop sequences
    #[serde(default)]
    pub stop: Option<StopTokens>,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: Option<bool>,
    /// Presence penalty (-2.0 to 2.0)
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Number of tokens to consider for repeat penalty
    pub repeat_last_n: Option<usize>,
    /// Frequency penalty (-2.0 to 2.0)
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Token bias adjustments
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,
    /// User identifier
    #[serde(default)]
    pub user: Option<String>,
    /// Top-k sampling parameter
    pub top_k: Option<isize>,
    /// Number of best completions to consider
    #[serde(default)]
    pub best_of: Option<usize>,
    /// Whether to use beam search
    #[serde(default)]
    pub use_beam_search: Option<bool>,
    /// Whether to ignore end-of-sequence token
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    /// Whether to skip special tokens in output
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
    /// Stop token IDs
    #[serde(default)]
    pub stop_token_ids: Option<Vec<usize>>,
    /// Whether to return log probabilities
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Enable thinking/reasoning mode
    pub thinking: Option<bool>,

    // ========================================================================
    // Tool Calling Parameters
    // ========================================================================
    /// A list of tools the model may call
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Controls which tool (if any) the model should use
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Whether to enable parallel tool calling (default: true)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

impl ChatCompletionRequest {
    /// Check if this request has tools defined
    pub fn has_tools(&self) -> bool {
        self.tools.as_ref().map_or(false, |t| !t.is_empty())
    }

    /// Check if tool calling is disabled
    pub fn tools_disabled(&self) -> bool {
        self.tool_choice.as_ref().map_or(false, |tc| tc.is_none())
    }

    /// Get the tool choice mode
    pub fn get_tool_choice_mode(&self) -> &str {
        match &self.tool_choice {
            Some(ToolChoice::Mode(m)) => m.as_str(),
            Some(ToolChoice::Specific(_)) => "specific",
            None => "auto",
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_tool() {
        let json = r#"{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }"#;

        let tool: Tool = serde_json::from_str(json).unwrap();
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "get_weather");
    }

    #[test]
    fn test_deserialize_tool_choice_auto() {
        let json = r#""auto""#;
        let choice: ToolChoice = serde_json::from_str(json).unwrap();
        assert!(choice.is_auto());
    }

    #[test]
    fn test_deserialize_tool_choice_specific() {
        let json = r#"{"type": "function", "function": {"name": "get_weather"}}"#;
        let choice: ToolChoice = serde_json::from_str(json).unwrap();
        match choice {
            ToolChoice::Specific(s) => {
                assert_eq!(s.function.name, "get_weather");
            }
            _ => panic!("Expected specific tool choice"),
        }
    }

    #[test]
    fn test_deserialize_chat_message_with_tool_calls() {
        let json = r#"{
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"Paris\"}"
                }
            }]
        }"#;

        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert!(msg.content.is_none());
        assert!(msg.tool_calls.is_some());
        let tool_calls = msg.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_deserialize_tool_response_message() {
        let json = r#"{
            "role": "tool",
            "content": "{\"temperature\": 22}",
            "tool_call_id": "call_123"
        }"#;

        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_messages_backward_compatibility() {
        // Test that the old HashMap format still works
        let json = r#"[{"role": "user", "content": "Hello"}]"#;
        let messages: Messages = serde_json::from_str(json).unwrap();

        let chat_messages = messages.to_chat_messages();
        assert_eq!(chat_messages.len(), 1);
        assert_eq!(chat_messages[0].role, "user");
        assert_eq!(chat_messages[0].content, Some("Hello".to_string()));
    }

    #[test]
    fn test_request_with_tools() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": "auto"
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(request.has_tools());
        assert!(!request.tools_disabled());
        assert_eq!(request.get_tool_choice_mode(), "auto");
    }
}
