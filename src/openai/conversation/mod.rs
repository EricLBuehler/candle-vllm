pub mod default_conversation;
use serde::Serialize;

use crate::openai::requests::{Tool, ToolCall};

/// A trait for using conversation managers with a `ModulePipeline`.
pub trait Conversation {
    fn set_system_message(&mut self, system_message: Option<String>);

    fn append_message(&mut self, role: String, message: String);

    /// Append a message with full tool support
    fn append_message_ext(
        &mut self,
        role: String,
        content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        tool_call_id: Option<String>,
        name: Option<String>,
    );

    fn get_roles(&self) -> &(String, String);

    fn apply_chat_template(
        &self,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, ApplyChatTemplateError>;

    fn get_prompt(&mut self, thinking: bool) -> String;

    fn clear_message(&mut self);

    /// Set the available tools for this conversation
    fn set_tools(&mut self, tools: Option<Vec<Tool>>);

    /// Get the available tools
    fn get_tools(&self) -> Option<&Vec<Tool>>;

    /// Check if tools are available
    fn has_tools(&self) -> bool {
        self.get_tools().map_or(false, |t| !t.is_empty())
    }
}

/// A message in the conversation with full tool support
#[derive(Serialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Create a simple text message
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::text("user", content)
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text("assistant", content)
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::text("system", content)
    }

    /// Create an assistant message with tool calls
    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
        }
    }

    /// Create an assistant message with both content and tool calls
    pub fn assistant_with_content_and_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.into()),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
        }
    }

    /// Create a tool response message
    pub fn tool_response(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        name: Option<String>,
    ) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name,
        }
    }

    /// Check if this message has tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().map_or(false, |tc| !tc.is_empty())
    }

    /// Check if this is a tool response
    pub fn is_tool_response(&self) -> bool {
        self.role == "tool" && self.tool_call_id.is_some()
    }

    /// Get content as string, returning empty string if None
    pub fn content_str(&self) -> &str {
        self.content.as_deref().unwrap_or("")
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ApplyChatTemplateError {
    #[error("failed to add template")]
    AddTemplateError(#[source] minijinja::Error),
    #[error("failed to get template")]
    GetTemplateError(#[source] minijinja::Error),
    #[error("failed to render")]
    RenderTemplateError(#[source] minijinja::Error),
}

/// Helper to format tools for chat templates
pub fn format_tools_for_template(tools: &[Tool]) -> String {
    serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".to_string())
}

/// Helper to format a single tool for Mistral-style templates
pub fn format_tool_mistral(tool: &Tool) -> String {
    serde_json::json!({
        "type": tool.tool_type,
        "function": {
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": tool.function.parameters
        }
    })
    .to_string()
}

/// Helper to format tools for Mistral-style templates
pub fn format_tools_mistral(tools: &[Tool]) -> String {
    let tool_strings: Vec<String> = tools.iter().map(format_tool_mistral).collect();
    format!("[{}]", tool_strings.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::requests::{FunctionCall, FunctionDefinition};

    #[test]
    fn test_message_text() {
        let msg = Message::text("user", "Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, Some("Hello".to_string()));
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn test_message_with_tool_calls() {
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location": "Paris"}"#.to_string(),
            },
        };
        let msg = Message::assistant_with_tool_calls(vec![tool_call]);

        assert_eq!(msg.role, "assistant");
        assert!(msg.content.is_none());
        assert!(msg.has_tool_calls());
    }

    #[test]
    fn test_tool_response_message() {
        let msg = Message::tool_response(
            "call_123",
            r#"{"temp": 22}"#,
            Some("get_weather".to_string()),
        );

        assert_eq!(msg.role, "tool");
        assert!(msg.is_tool_response());
        assert_eq!(msg.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_format_tools_mistral() {
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get weather info".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                })),
                strict: None,
            },
        }];

        let formatted = format_tools_mistral(&tools);
        assert!(formatted.contains("get_weather"));
        assert!(formatted.contains("function"));
    }
}
