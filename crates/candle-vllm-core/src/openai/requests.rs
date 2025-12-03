use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ============================================================================
// Tool Calling Types (OpenAI Compatible)
// ============================================================================
//
// NOTE: We use the `tools` parameter (NOT the deprecated `functions` parameter).
// The `functions` parameter was deprecated by OpenAI in late 2023.
// All tool definitions use the `{"type": "function", "function": {...}}` wrapper.
//
// For MCP (Model Context Protocol) integration:
// - MCP tools should be converted to this format with `Tool::from_mcp()`
// - Use `tool_call_id` tracking for multi-turn conversations
// - Parallel tool calling is supported (multiple tool_calls in one response)
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
    /// When true, the model will strictly follow the JSON schema for parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl FunctionDefinition {
    /// Create a new function definition
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: None,
            strict: None,
        }
    }

    /// Set the description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the parameters schema
    pub fn with_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.parameters = Some(parameters);
        self
    }

    /// Enable strict mode for parameter validation
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }
}

/// A tool that the model can use
///
/// This uses the OpenAI `tools` format (NOT the deprecated `functions` format).
/// For MCP integration, use `Tool::from_mcp()` to convert MCP tool definitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// The type of tool (currently only "function" is supported)
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition
    pub function: FunctionDefinition,
}

impl Tool {
    /// Create a new function tool
    pub fn function(definition: FunctionDefinition) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: definition,
        }
    }

    /// Create a tool from MCP tool format
    ///
    /// MCP tools have a slightly different format, this converts them to OpenAI format.
    /// MCP format: `{"name": "...", "description": "...", "inputSchema": {...}}`
    /// OpenAI format: `{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}`
    pub fn from_mcp(mcp_tool: &serde_json::Value) -> Option<Self> {
        let name = mcp_tool.get("name")?.as_str()?.to_string();
        let description = mcp_tool
            .get("description")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string());

        // MCP uses "inputSchema", OpenAI uses "parameters"
        let parameters = mcp_tool
            .get("inputSchema")
            .or_else(|| mcp_tool.get("parameters"))
            .cloned();

        Some(Self {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name,
                description,
                parameters,
                strict: Some(true), // MCP tools should use strict mode
            },
        })
    }

    /// Convert multiple MCP tools to OpenAI format
    pub fn from_mcp_list(mcp_tools: &[serde_json::Value]) -> Vec<Self> {
        mcp_tools.iter().filter_map(Self::from_mcp).collect()
    }

    /// Filter tools by allowed names (for MCP `allowed_tools` support)
    ///
    /// This is useful for reducing token usage when MCP servers expose many tools.
    pub fn filter_by_names(tools: &[Tool], allowed_names: &[&str]) -> Vec<Tool> {
        tools
            .iter()
            .filter(|t| allowed_names.contains(&t.function.name.as_str()))
            .cloned()
            .collect()
    }

    /// Get the function name
    pub fn name(&self) -> &str {
        &self.function.name
    }

    /// Check if this is a function tool
    pub fn is_function(&self) -> bool {
        self.tool_type == "function"
    }
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
///
/// The `id` field is critical for MCP multi-turn conversations.
/// When sending tool results back, the `tool_call_id` in the tool message
/// must match the `id` from the original tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// The ID of the tool call - MUST be referenced in tool response messages
    pub id: String,
    /// The type of tool call (currently only "function" is supported)
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function call details
    pub function: FunctionCall,
}

impl ToolCall {
    /// Create a new function tool call
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    /// Generate a new tool call with a random ID
    pub fn with_random_id(name: impl Into<String>, arguments: impl Into<String>) -> Self {
        let id = format!(
            "call_{}",
            uuid::Uuid::new_v4().to_string().replace("-", "")[..24].to_string()
        );
        Self::new(id, name, arguments)
    }

    /// Get the function name
    pub fn name(&self) -> &str {
        &self.function.name
    }

    /// Get the arguments as a string
    pub fn arguments(&self) -> &str {
        &self.function.arguments
    }

    /// Parse the arguments as JSON
    pub fn parse_arguments(&self) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::from_str(&self.function.arguments)
    }

    /// Convert to MCP tool call format for execution
    ///
    /// Returns a JSON value suitable for MCP server execution
    pub fn to_mcp_call(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.function.name,
            "arguments": self.parse_arguments().unwrap_or(serde_json::json!({}))
        })
    }
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

    /// Create a tool response message with function name
    ///
    /// This is the preferred format for MCP tool responses, including the function name
    /// for better context in multi-turn conversations.
    pub fn tool_with_name(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: Some(name.into()),
        }
    }

    /// Create a tool response from MCP execution result
    ///
    /// Converts an MCP tool result to the OpenAI message format.
    /// MCP results typically come as `{"content": [...], "isError": bool}`
    pub fn from_mcp_result(
        tool_call_id: impl Into<String>,
        mcp_result: &serde_json::Value,
        name: Option<String>,
    ) -> Self {
        // MCP results can have complex content arrays, flatten to string
        let content =
            if let Some(content_array) = mcp_result.get("content").and_then(|c| c.as_array()) {
                content_array
                    .iter()
                    .filter_map(|item| {
                        if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                            Some(text.to_string())
                        } else {
                            // For non-text content, serialize it
                            serde_json::to_string(item).ok()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            } else if let Some(text) = mcp_result.get("content").and_then(|c| c.as_str()) {
                text.to_string()
            } else {
                // Fallback: serialize the whole result
                serde_json::to_string(mcp_result).unwrap_or_else(|_| "{}".to_string())
            };

        Self {
            role: "tool".to_string(),
            content: Some(content),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name,
        }
    }

    /// Check if this is a tool response message
    pub fn is_tool_response(&self) -> bool {
        self.role == "tool" && self.tool_call_id.is_some()
    }

    /// Check if this is an assistant message with tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.role == "assistant" && self.tool_calls.as_ref().map_or(false, |tc| !tc.is_empty())
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

    // ========================================================================
    // MCP Integration Tests
    // ========================================================================

    #[test]
    fn test_tool_from_mcp_format() {
        // MCP tool format uses "inputSchema" instead of "parameters"
        let mcp_tool = serde_json::json!({
            "name": "read_file",
            "description": "Read contents of a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    }
                },
                "required": ["path"]
            }
        });

        let tool = Tool::from_mcp(&mcp_tool).unwrap();
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "read_file");
        assert_eq!(
            tool.function.description,
            Some("Read contents of a file".to_string())
        );
        assert!(tool.function.parameters.is_some());
        assert_eq!(tool.function.strict, Some(true)); // MCP tools use strict mode
    }

    #[test]
    fn test_tool_from_mcp_list() {
        let mcp_tools = vec![
            serde_json::json!({
                "name": "tool1",
                "description": "First tool",
                "inputSchema": {"type": "object", "properties": {}}
            }),
            serde_json::json!({
                "name": "tool2",
                "description": "Second tool",
                "inputSchema": {"type": "object", "properties": {}}
            }),
        ];

        let tools = Tool::from_mcp_list(&mcp_tools);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name(), "tool1");
        assert_eq!(tools[1].name(), "tool2");
    }

    #[test]
    fn test_tool_filter_by_names() {
        let tools = vec![
            Tool::function(FunctionDefinition::new("tool1")),
            Tool::function(FunctionDefinition::new("tool2")),
            Tool::function(FunctionDefinition::new("tool3")),
        ];

        let filtered = Tool::filter_by_names(&tools, &["tool1", "tool3"]);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].name(), "tool1");
        assert_eq!(filtered[1].name(), "tool3");
    }

    #[test]
    fn test_tool_call_to_mcp() {
        let tool_call = ToolCall::new(
            "call_123",
            "get_weather",
            r#"{"location": "San Francisco"}"#,
        );

        let mcp_call = tool_call.to_mcp_call();
        assert_eq!(mcp_call["name"], "get_weather");
        assert_eq!(mcp_call["arguments"]["location"], "San Francisco");
    }

    #[test]
    fn test_chat_message_from_mcp_result() {
        // MCP result format with content array
        let mcp_result = serde_json::json!({
            "content": [
                {"type": "text", "text": "The weather is sunny"},
                {"type": "text", "text": "Temperature: 72°F"}
            ],
            "isError": false
        });

        let msg =
            ChatMessage::from_mcp_result("call_123", &mcp_result, Some("get_weather".to_string()));
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id, Some("call_123".to_string()));
        assert_eq!(msg.name, Some("get_weather".to_string()));
        assert!(msg.content.as_ref().unwrap().contains("sunny"));
        assert!(msg.content.as_ref().unwrap().contains("72°F"));
    }

    #[test]
    fn test_function_definition_builder() {
        let func = FunctionDefinition::new("get_weather")
            .with_description("Get weather for a location")
            .with_parameters(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }))
            .with_strict(true);

        assert_eq!(func.name, "get_weather");
        assert_eq!(
            func.description,
            Some("Get weather for a location".to_string())
        );
        assert!(func.parameters.is_some());
        assert_eq!(func.strict, Some(true));
    }

    #[test]
    fn test_tool_call_with_random_id() {
        let call = ToolCall::with_random_id("test_func", "{}");
        assert!(call.id.starts_with("call_"));
        assert_eq!(call.id.len(), 29); // "call_" + 24 chars
        assert_eq!(call.name(), "test_func");
    }

    #[test]
    fn test_tool_message_deserialization() {
        // Test that tool role messages deserialize correctly (for MCP support)
        let json = r#"{
            "role": "tool",
            "content": "{\"temperature\": 22, \"unit\": \"celsius\"}",
            "tool_call_id": "call_abc123",
            "name": "get_weather"
        }"#;

        let message: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.role, "tool");
        assert_eq!(message.tool_call_id, Some("call_abc123".to_string()));
        assert_eq!(message.name, Some("get_weather".to_string()));
        assert!(message.content.is_some());
    }

    #[test]
    fn test_assistant_message_with_tool_calls() {
        // Test assistant message with tool_calls (response from model)
        let json = r#"{
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"San Francisco\"}"
                }
            }]
        }"#;

        let message: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.role, "assistant");
        assert!(message.content.is_none());
        assert!(message.tool_calls.is_some());
        let tool_calls = message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc123");
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_mcp_multi_turn_conversation() {
        // Test complete MCP-style multi-turn tool calling flow:
        // 1. User asks question
        // 2. Assistant responds with tool_calls
        // 3. Tool response with results
        // 4. User can continue conversation
        let json = r#"{
            "model": "mistral",
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in San Francisco?"
                },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": "{\"location\": \"San Francisco, CA\"}"
                        }
                    }]
                },
                {
                    "role": "tool",
                    "content": "{\"temperature\": 72, \"unit\": \"fahrenheit\", \"description\": \"Sunny\"}",
                    "tool_call_id": "call_abc123",
                    "name": "get_current_weather"
                }
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(request.has_tools());

        // Verify all messages parsed correctly
        if let Messages::Chat(messages) = &request.messages {
            assert_eq!(messages.len(), 3);

            // First message: user
            assert_eq!(messages[0].role, "user");
            assert!(messages[0].content.is_some());

            // Second message: assistant with tool_calls
            assert_eq!(messages[1].role, "assistant");
            assert!(messages[1].content.is_none());
            assert!(messages[1].tool_calls.is_some());
            let tool_calls = messages[1].tool_calls.as_ref().unwrap();
            assert_eq!(tool_calls[0].id, "call_abc123");

            // Third message: tool response
            assert_eq!(messages[2].role, "tool");
            assert_eq!(messages[2].tool_call_id, Some("call_abc123".to_string()));
            assert_eq!(messages[2].name, Some("get_current_weather".to_string()));
        } else {
            panic!("Expected Chat messages format");
        }
    }

    #[test]
    fn test_multiple_tool_calls_and_responses() {
        // Test parallel tool calls (multiple tools called at once)
        let json = r#"{
            "model": "mistral",
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in SF and NYC?"
                },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_sf",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"San Francisco\"}"
                            }
                        },
                        {
                            "id": "call_nyc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"New York\"}"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "content": "{\"temp\": 72}",
                    "tool_call_id": "call_sf",
                    "name": "get_weather"
                },
                {
                    "role": "tool",
                    "content": "{\"temp\": 65}",
                    "tool_call_id": "call_nyc",
                    "name": "get_weather"
                }
            ]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        if let Messages::Chat(messages) = &request.messages {
            assert_eq!(messages.len(), 4);

            // Check assistant has multiple tool calls
            let tool_calls = messages[1].tool_calls.as_ref().unwrap();
            assert_eq!(tool_calls.len(), 2);
            assert_eq!(tool_calls[0].id, "call_sf");
            assert_eq!(tool_calls[1].id, "call_nyc");

            // Check both tool responses
            assert_eq!(messages[2].role, "tool");
            assert_eq!(messages[2].tool_call_id, Some("call_sf".to_string()));

            assert_eq!(messages[3].role, "tool");
            assert_eq!(messages[3].tool_call_id, Some("call_nyc".to_string()));
        } else {
            panic!("Expected Chat messages format");
        }
    }

    #[test]
    fn test_tool_choice_required() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "greet",
                    "description": "Greet someone",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": "required"
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.get_tool_choice_mode(), "required");
    }

    #[test]
    fn test_tool_choice_none() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "greet",
                    "description": "Greet someone",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": "none"
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(request.tools_disabled());
    }
}
