// src/tools/mod.rs
//! Tool calling support for vLLM.rs
//!
//! This module provides OpenAI-compatible tool calling functionality,
//! allowing LLMs to invoke external functions and tools.

pub mod parser;
pub mod stream_parser;
pub mod schema;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A tool definition following OpenAI's function calling format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Type of the tool, always "function" for now
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition
    pub function: FunctionDefinition,
}

impl Tool {
    /// Create a new function tool
    pub fn function(name: impl Into<String>, description: impl Into<String>) -> ToolBuilder {
        ToolBuilder::new(name.into(), description.into())
    }
}

/// Definition of a callable function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Name of the function
    pub name: String,
    /// Description of what the function does
    pub description: String,
    /// JSON Schema for the function parameters
    pub parameters: Value,
    /// Whether to enable strict schema adherence
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Builder for creating Tool definitions
pub struct ToolBuilder {
    name: String,
    description: String,
    parameters: Value,
    strict: Option<bool>,
}

impl ToolBuilder {
    fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            strict: None,
        }
    }

    /// Add a parameter to the function
    pub fn param(
        mut self,
        name: impl Into<String>,
        param_type: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        if let Some(props) = self.parameters.get_mut("properties") {
            props[&name] = serde_json::json!({
                "type": param_type.into(),
                "description": description.into()
            });
        }
        if required {
            if let Some(req) = self.parameters.get_mut("required") {
                if let Some(arr) = req.as_array_mut() {
                    arr.push(Value::String(name));
                }
            }
        }
        self
    }

    /// Set custom parameters schema
    pub fn parameters_schema(mut self, schema: Value) -> Self {
        self.parameters = schema;
        self
    }

    /// Enable strict mode
    pub fn strict(mut self, value: bool) -> Self {
        self.strict = Some(value);
        self
    }

    /// Build the final Tool
    pub fn build(self) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: self.name,
                description: self.description,
                parameters: self.parameters,
                strict: self.strict,
            },
        }
    }
}

/// Tool choice configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Let the model decide
    Auto(String),
    /// Force no tool usage
    None(String),
    /// Force a specific tool
    Function {
        #[serde(rename = "type")]
        choice_type: String,
        function: ToolChoiceFunction,
    },
}

impl ToolChoice {
    pub fn auto() -> Self {
        ToolChoice::Auto("auto".to_string())
    }

    pub fn none() -> Self {
        ToolChoice::None("none".to_string())
    }

    pub fn function(name: impl Into<String>) -> Self {
        ToolChoice::Function {
            choice_type: "function".to_string(),
            function: ToolChoiceFunction { name: name.into() },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

/// A tool call made by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Index of this tool call in the tool_calls array (streaming only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    /// Unique identifier for this tool call
    pub id: String,
    /// Type of tool call (always "function")
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function call details
    pub function: FunctionCall,
}

impl ToolCall {
    /// Create a new tool call
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            index: None,
            id: id.into(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    pub fn with_index(mut self, index: usize) -> Self {
        self.index = Some(index);
        self
    }
}

/// Details of a function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function to call
    pub name: String,
    /// JSON string of arguments
    pub arguments: String,
}

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The tool call ID this result corresponds to
    pub tool_call_id: String,
    /// The result content (typically JSON or text)
    pub content: String,
    /// Whether the tool execution was successful
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl ToolResult {
    /// Create a successful tool result
    pub fn success(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            is_error: None,
        }
    }

    /// Create an error tool result
    pub fn error(tool_call_id: impl Into<String>, error_message: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: error_message.into(),
            is_error: Some(true),
        }
    }
}

/// Format tool definitions for injection into the prompt
#[derive(Debug, Clone)]
pub struct ToolFormat {}

impl ToolFormat {
    /// Format tools for inclusion in the system prompt
    /// Uses explicit instructions to ensure models output tool calls in the expected format
    pub fn format_tools(tools: &[Tool]) -> String {
        let rule = String::from(
            "IMPORTANT: For each function call, you MUST wrapped function name and arguments in <tool_call></tool_call> tags.\n\n\
            Do NOT USE ANY code blocks. Required format:\n\
            <tool_call>\n\
            {\"name\": \"<function-name>\", \"arguments\": <args-json-object>}\n\
            </tool_call>\n\n\
            Rules:\n\
            - Wrapper function name and arguments with <tool_call> and </tool_call> tags\n\
            - Always use the exact <tool_call></tool_call> format shown above\n\
            - Do NOT USE ANY code blocks\n\
            - The \"name\" and \"arguments\" are necessary fields\n",
        );

        let mut output = String::from(
            "# Tools\n\n\
            You may call one or more functions to assist with the user query.\n\n\
            You are provided with function signatures within <tools></tools> XML tags:\n\
            <tools>\n",
        );
        output.push_str(&rule);

        for tool in tools {
            output.push_str(&serde_json::to_string(&tool.function).unwrap_or_default());
            output.push('\n');
        }

        output.push_str("</tools>\n\n");
        output.push_str(&rule);

        output
    }

    /// Get tool prompt for a specific tool config (model-aware tags).
    pub fn get_tool_prompt(tool_config: &crate::tools::stream_parser::ToolConfig) -> String {
        let start_tag = &tool_config.start_token_str;
        let end_tag = &tool_config.end_token_str;
        format!(
            "MOST IMPORTANT INSTRUCTION, **MUST** FOLLOW: For each function call, you MUST wrap function name and arguments in {start_tag}{end_tag} tags.\n\n\
            Do NOT USE ANY code blocks. Required format:\n\
            {start_tag}\n\
            {{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}}\n\
            {end_tag}\n\n\
            Rules:\n\
            - Wrap function name and arguments with {start_tag} and {end_tag} tags\n\
            - Always use the exact {start_tag}{end_tag} format shown above\n\
            - Do NOT USE ANY code blocks\n\
            - Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.\n\
            - Always adhere to this format for the tool use to ensure proper parsing and execution.\n\
            - The \"name\" and \"arguments\" are necessary fields\n\
            - MUST FOLLOW the above instruction when using tool call!"
        )
    }
}
