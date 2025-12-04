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

// ============================================================================
// Tool Result Formatting Helpers (for MCP/multi-turn tool calling)
// ============================================================================

/// Format a tool result for Mistral-style models
/// Mistral expects: [TOOL_RESULTS] {"call_id": "...", "content": "..."} [/TOOL_RESULTS]
pub fn format_tool_result_mistral(
    tool_call_id: &str,
    content: &str,
    _name: Option<&str>,
) -> String {
    serde_json::json!({
        "call_id": tool_call_id,
        "content": content
    })
    .to_string()
}

/// Format multiple tool results for Mistral-style models
pub fn format_tool_results_mistral(results: &[(String, String, Option<String>)]) -> String {
    let formatted: Vec<String> = results
        .iter()
        .map(|(id, content, name)| format_tool_result_mistral(id, content, name.as_deref()))
        .collect();
    format!("[TOOL_RESULTS] {} [/TOOL_RESULTS]", formatted.join(", "))
}

/// Format a tool result for Llama-style models
/// Llama 3.1+ expects: <|python_tag|>{"output": "..."}<|eom_id|>
/// Or for ipython style: <|start_header_id|>ipython<|end_header_id|>\n\n{result}
pub fn format_tool_result_llama(content: &str, _name: Option<&str>) -> String {
    format!("<|start_header_id|>ipython<|end_header_id|>\n\n{}", content)
}

/// Format a tool result for Qwen-style models
/// Qwen expects: <tool_response>{"name": "...", "content": "..."}</tool_response>
pub fn format_tool_result_qwen(content: &str, name: Option<&str>) -> String {
    let result = if let Some(func_name) = name {
        serde_json::json!({
            "name": func_name,
            "content": content
        })
    } else {
        serde_json::json!({
            "content": content
        })
    };
    format!("<tool_response>{}</tool_response>", result)
}

/// Format a tool result for generic JSON format
pub fn format_tool_result_json(tool_call_id: &str, content: &str, name: Option<&str>) -> String {
    let mut result = serde_json::json!({
        "tool_call_id": tool_call_id,
        "content": content
    });
    if let Some(func_name) = name {
        result["name"] = serde_json::Value::String(func_name.to_string());
    }
    result.to_string()
}

/// Detect model family from model name and format tool result appropriately
pub fn format_tool_result_for_model(
    model_name: &str,
    tool_call_id: &str,
    content: &str,
    name: Option<&str>,
) -> String {
    let model_lower = model_name.to_lowercase();

    if model_lower.contains("mistral") || model_lower.contains("ministral") {
        format_tool_result_mistral(tool_call_id, content, name)
    } else if model_lower.contains("llama") {
        format_tool_result_llama(content, name)
    } else if model_lower.contains("qwen") {
        format_tool_result_qwen(content, name)
    } else {
        // Default to JSON format
        format_tool_result_json(tool_call_id, content, name)
    }
}

/// Helper struct for building multi-turn tool conversations
#[derive(Debug, Clone)]
pub struct ToolConversationBuilder {
    messages: Vec<Message>,
    model_family: ModelFamily,
}

/// Model family for tool formatting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Mistral,
    Llama,
    Qwen,
    Generic,
}

impl ModelFamily {
    /// Detect model family from model name
    pub fn from_model_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("mistral") || lower.contains("ministral") {
            ModelFamily::Mistral
        } else if lower.contains("llama") {
            ModelFamily::Llama
        } else if lower.contains("qwen") {
            ModelFamily::Qwen
        } else {
            ModelFamily::Generic
        }
    }
}

impl ToolConversationBuilder {
    /// Create a new builder for a specific model
    pub fn new(model_name: &str) -> Self {
        Self {
            messages: Vec::new(),
            model_family: ModelFamily::from_model_name(model_name),
        }
    }

    /// Add a user message
    pub fn add_user_message(&mut self, content: impl Into<String>) -> &mut Self {
        self.messages.push(Message::user(content));
        self
    }

    /// Add an assistant message with tool calls
    pub fn add_assistant_tool_calls(&mut self, tool_calls: Vec<ToolCall>) -> &mut Self {
        self.messages
            .push(Message::assistant_with_tool_calls(tool_calls));
        self
    }

    /// Add a tool result message
    pub fn add_tool_result(
        &mut self,
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        name: Option<String>,
    ) -> &mut Self {
        self.messages
            .push(Message::tool_response(tool_call_id, content, name));
        self
    }

    /// Add an assistant message (final response after tool execution)
    pub fn add_assistant_response(&mut self, content: impl Into<String>) -> &mut Self {
        self.messages.push(Message::assistant(content));
        self
    }

    /// Get the messages
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Take ownership of the messages
    pub fn into_messages(self) -> Vec<Message> {
        self.messages
    }

    /// Get the model family
    pub fn model_family(&self) -> ModelFamily {
        self.model_family
    }
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
    fn test_format_tool_result_mistral() {
        let result =
            format_tool_result_mistral("call_123", r#"{"temperature": 22}"#, Some("get_weather"));
        assert!(result.contains("call_123"));
        assert!(result.contains("temperature"));
    }

    #[test]
    fn test_format_tool_result_llama() {
        let result = format_tool_result_llama(r#"{"temperature": 22}"#, Some("get_weather"));
        assert!(result.contains("ipython"));
        assert!(result.contains("temperature"));
    }

    #[test]
    fn test_format_tool_result_qwen() {
        let result = format_tool_result_qwen(r#"{"temperature": 22}"#, Some("get_weather"));
        assert!(result.contains("<tool_response>"));
        assert!(result.contains("get_weather"));
    }

    #[test]
    fn test_model_family_detection() {
        assert_eq!(
            ModelFamily::from_model_name("mistralai/Mistral-7B-Instruct-v0.3"),
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
            ModelFamily::from_model_name("some-other-model"),
            ModelFamily::Generic
        );
    }

    #[test]
    fn test_tool_conversation_builder() {
        let mut builder = ToolConversationBuilder::new("mistralai/Mistral-7B");
        builder
            .add_user_message("What's the weather?")
            .add_assistant_tool_calls(vec![ToolCall {
                id: "call_123".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "get_weather".to_string(),
                    arguments: r#"{"location": "Paris"}"#.to_string(),
                },
            }])
            .add_tool_result(
                "call_123",
                r#"{"temp": 20}"#,
                Some("get_weather".to_string()),
            )
            .add_assistant_response("The temperature in Paris is 20°C.");

        assert_eq!(builder.messages().len(), 4);
        assert_eq!(builder.model_family(), ModelFamily::Mistral);
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
