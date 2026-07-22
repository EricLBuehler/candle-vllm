// src/tools/mod.rs
//! Tool calling support for candle-vLLM
//!
//! This module provides OpenAI-compatible tool calling functionality,
//! allowing LLMs to invoke external functions and tools.

pub mod helpers;
pub mod parser;
pub mod schema;
pub mod stream_parser;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub use openai_protocol::common::{Function, Tool};
pub type FunctionDefinition = Function;

/// Adapter from xInfer's OpenAI tool-call representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCall,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            index: None,
            id: id.into(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.into(),
                arguments: Some(arguments.into()),
            },
        }
    }

    pub fn with_index(mut self, index: usize) -> Self {
        self.index = Some(index);
        self
    }
}

/// Builder for creating Tool definitions
pub struct ToolBuilder {
    name: String,
    description: String,
    parameters: Value,
    strict: Option<bool>,
}

impl ToolBuilder {
    pub fn new(name: String, description: String) -> Self {
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
            function: Function {
                name: self.name,
                description: Some(self.description),
                parameters: self.parameters,
                strict: self.strict,
            },
        }
    }
}

/// Create a new function tool builder (replacement for Tool::function).
pub fn function_tool(name: impl Into<String>, description: impl Into<String>) -> ToolBuilder {
    ToolBuilder::new(name.into(), description.into())
}

/// Tool choice configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String modes: "auto" | "none" | "required"
    Mode(ToolChoiceMode),
    /// Force a specific tool
    Function {
        #[serde(rename = "type")]
        choice_type: ToolChoiceType,
        function: ToolChoiceFunction,
    },
}

impl ToolChoice {
    pub fn auto() -> Self {
        ToolChoice::Mode(ToolChoiceMode::Auto)
    }

    pub fn none() -> Self {
        ToolChoice::Mode(ToolChoiceMode::None)
    }

    pub fn required() -> Self {
        ToolChoice::Mode(ToolChoiceMode::Required)
    }

    pub fn function(name: impl Into<String>) -> Self {
        ToolChoice::Function {
            choice_type: ToolChoiceType::Function,
            function: ToolChoiceFunction { name: name.into() },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    Auto,
    None,
    Required,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceType {
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

/// Build a ToolCall from name/arguments with a provided ID.
pub fn new_tool_call(
    id: impl Into<String>,
    name: impl Into<String>,
    arguments: impl Into<String>,
) -> ToolCall {
    ToolCall {
        index: None,
        id: id.into(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: name.into(),
            arguments: Some(arguments.into()),
        },
    }
}

/// Generate a compact tool call ID with required `call_` prefix.
/// Uses 16 hex chars (64 bits) from UUIDv4 for low collision risk and shorter payloads.
pub fn generate_tool_call_id() -> String {
    let raw = Uuid::new_v4().simple().to_string();
    format!("call_{}", &raw[..16])
}

/// Convert a parsed tool call into an OpenAI-compatible ToolCall.
pub fn tool_call_from_parser(parsed: tool_parser::ToolCall) -> ToolCall {
    ToolCall {
        index: None,
        id: generate_tool_call_id(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: parsed.function.name,
            arguments: Some(parsed.function.arguments),
        },
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_choice_deserializes_string_modes() {
        let auto: ToolChoice = serde_json::from_str(r#""auto""#).unwrap();
        let none: ToolChoice = serde_json::from_str(r#""none""#).unwrap();
        let required: ToolChoice = serde_json::from_str(r#""required""#).unwrap();

        assert!(matches!(auto, ToolChoice::Mode(ToolChoiceMode::Auto)));
        assert!(matches!(none, ToolChoice::Mode(ToolChoiceMode::None)));
        assert!(matches!(
            required,
            ToolChoice::Mode(ToolChoiceMode::Required)
        ));
    }

    #[test]
    fn tool_choice_deserializes_function_mode() {
        let choice: ToolChoice =
            serde_json::from_str(r#"{"type":"function","function":{"name":"read_file"}}"#).unwrap();
        match choice {
            ToolChoice::Function {
                choice_type,
                function,
            } => {
                assert_eq!(choice_type, ToolChoiceType::Function);
                assert_eq!(function.name, "read_file");
            }
            _ => panic!("expected function tool choice"),
        }
    }
}
