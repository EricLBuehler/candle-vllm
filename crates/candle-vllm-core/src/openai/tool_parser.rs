//! Tool Call Parser Module
//!
//! This module provides parsing functionality to extract tool calls from model outputs.
//! Different models have different formats for tool calls, so we provide model-specific
//! parsers that implement a common trait.

use crate::openai::requests::{FunctionCall, ToolCall};
use regex::Regex;
use serde::{Deserialize, Serialize};

// ============================================================================
// Parsed Output Types
// ============================================================================

/// Represents a parsed tool call extracted from model output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedToolCall {
    /// Unique identifier for the tool call
    pub id: String,
    /// Name of the function to call
    pub name: String,
    /// JSON string of arguments
    pub arguments: String,
}

impl ParsedToolCall {
    /// Create a new ParsedToolCall with a generated ID
    pub fn new(name: String, arguments: String) -> Self {
        Self {
            id: format!(
                "call_{}",
                uuid::Uuid::new_v4().to_string().replace("-", "")[..24].to_string()
            ),
            name,
            arguments,
        }
    }

    /// Create a new ParsedToolCall with a specific ID
    pub fn with_id(id: String, name: String, arguments: String) -> Self {
        Self {
            id,
            name,
            arguments,
        }
    }

    /// Convert to the API ToolCall type
    pub fn to_tool_call(self) -> ToolCall {
        ToolCall {
            id: self.id,
            call_type: "function".to_string(),
            function: FunctionCall {
                name: self.name,
                arguments: self.arguments,
            },
        }
    }
}

/// Result of parsing model output for tool calls
#[derive(Debug, Clone)]
pub enum ParsedOutput {
    /// The output is plain text with no tool calls
    Text(String),
    /// The output contains only tool calls
    ToolCalls(Vec<ParsedToolCall>),
    /// The output contains both text and tool calls
    Mixed {
        text: String,
        tool_calls: Vec<ParsedToolCall>,
    },
}

impl ParsedOutput {
    /// Check if the output contains any tool calls
    pub fn has_tool_calls(&self) -> bool {
        matches!(
            self,
            ParsedOutput::ToolCalls(_) | ParsedOutput::Mixed { .. }
        )
    }

    /// Get the text content, if any
    pub fn text(&self) -> Option<&str> {
        match self {
            ParsedOutput::Text(t) => Some(t),
            ParsedOutput::Mixed { text, .. } => Some(text),
            ParsedOutput::ToolCalls(_) => None,
        }
    }

    /// Get the tool calls, if any
    pub fn tool_calls(&self) -> Option<&[ParsedToolCall]> {
        match self {
            ParsedOutput::ToolCalls(tc) => Some(tc),
            ParsedOutput::Mixed { tool_calls, .. } => Some(tool_calls),
            ParsedOutput::Text(_) => None,
        }
    }

    /// Convert tool calls to API format
    pub fn into_api_tool_calls(self) -> Option<Vec<ToolCall>> {
        match self {
            ParsedOutput::ToolCalls(tc) => Some(tc.into_iter().map(|t| t.to_tool_call()).collect()),
            ParsedOutput::Mixed { tool_calls, .. } => {
                Some(tool_calls.into_iter().map(|t| t.to_tool_call()).collect())
            }
            ParsedOutput::Text(_) => None,
        }
    }
}

// ============================================================================
// Incremental Parsing Types
// ============================================================================

/// State of incremental tool call parsing
#[derive(Debug, Clone)]
pub enum ToolParseState {
    /// Not a tool call (regular content)
    NotToolCall,
    /// Tool call in progress (partial data)
    InProgress(PartialToolCall),
    /// Tool call complete
    Complete(ParsedToolCall),
}

/// Partial tool call being built incrementally
#[derive(Debug, Clone)]
pub struct PartialToolCall {
    /// Tool name (if detected)
    pub name: Option<String>,
    /// Partial arguments accumulated so far
    pub arguments: String,
    /// Format detected
    pub format: ToolCallFormat,
}

/// Format of tool call detected
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallFormat {
    Mistral,
    Llama,
    Qwen,
    Json,
    Unknown,
}

/// Trait for incremental tool call parsing
pub trait IncrementalToolParser {
    /// Parse incrementally as tokens arrive
    fn parse_incremental(&self, buffer: &str) -> ToolParseState;
}

// ============================================================================
// Tool Call Parser Trait
// ============================================================================

/// Trait for parsing tool calls from model output
pub trait ToolCallParser: Send + Sync {
    /// Parse the model output and extract any tool calls
    fn parse(&self, output: &str) -> ParsedOutput;

    /// Get the name of this parser (for debugging)
    fn name(&self) -> &'static str;

    /// Check if the output might contain a tool call (quick check)
    fn might_contain_tool_call(&self, output: &str) -> bool;

    /// Get stop sequences that should be added when tools are enabled
    fn tool_stop_sequences(&self) -> Vec<String> {
        vec![]
    }
}

// ============================================================================
// Mistral Tool Parser
// ============================================================================

/// Parser for Mistral/Ministral style tool calls
/// Format: [TOOL_CALLS] [{"name": "func_name", "arguments": {...}}, ...]
pub struct MistralToolParser {
    tool_call_pattern: Regex,
}

impl Default for MistralToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MistralToolParser {
    pub fn new() -> Self {
        Self {
            // Match [TOOL_CALLS] followed by a JSON array
            tool_call_pattern: Regex::new(
                r"(?s)\[TOOL_CALLS\]\s*(\[[\s\S]*?\])(?:\s*$|\s*\[/TOOL_CALLS\])",
            )
            .expect("Invalid regex pattern"),
        }
    }

    fn parse_tool_calls_json(&self, json_str: &str) -> Option<Vec<ParsedToolCall>> {
        // Try to parse as a JSON array of tool calls
        let parsed: Result<Vec<serde_json::Value>, _> = serde_json::from_str(json_str);

        match parsed {
            Ok(calls) => {
                let tool_calls: Vec<ParsedToolCall> = calls
                    .into_iter()
                    .filter_map(|call| {
                        let name = call.get("name")?.as_str()?.to_string();
                        let arguments = if let Some(args) = call.get("arguments") {
                            if args.is_string() {
                                args.as_str()?.to_string()
                            } else {
                                serde_json::to_string(args).ok()?
                            }
                        } else {
                            "{}".to_string()
                        };
                        Some(ParsedToolCall::new(name, arguments))
                    })
                    .collect();

                if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                }
            }
            Err(_) => None,
        }
    }
}

impl ToolCallParser for MistralToolParser {
    fn name(&self) -> &'static str {
        "mistral"
    }

    fn might_contain_tool_call(&self, output: &str) -> bool {
        output.contains("[TOOL_CALLS]")
    }

    fn parse(&self, output: &str) -> ParsedOutput {
        if !self.might_contain_tool_call(output) {
            return ParsedOutput::Text(output.to_string());
        }

        if let Some(captures) = self.tool_call_pattern.captures(output) {
            if let Some(json_match) = captures.get(1) {
                if let Some(tool_calls) = self.parse_tool_calls_json(json_match.as_str()) {
                    // Check for text before [TOOL_CALLS]
                    let text_before = output
                        .split("[TOOL_CALLS]")
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();

                    if text_before.is_empty() {
                        return ParsedOutput::ToolCalls(tool_calls);
                    } else {
                        return ParsedOutput::Mixed {
                            text: text_before,
                            tool_calls,
                        };
                    }
                }
            }
        }

        ParsedOutput::Text(output.to_string())
    }

    fn tool_stop_sequences(&self) -> Vec<String> {
        vec!["[/TOOL_CALLS]".to_string()]
    }
}

// ============================================================================
// Llama 3.1+ Tool Parser
// ============================================================================

/// Parser for Llama 3.1+ style tool calls
/// Format: <|python_tag|>{"name": "func", "parameters": {...}}<|eom_id|>
/// Or: <function=func_name>{"arg": "value"}</function>
pub struct LlamaToolParser {
    python_tag_pattern: Regex,
    function_tag_pattern: Regex,
}

impl Default for LlamaToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaToolParser {
    pub fn new() -> Self {
        Self {
            python_tag_pattern: Regex::new(
                r"<\|python_tag\|>\s*(\{[\s\S]*?\})\s*(?:<\|eom_id\|>|$)",
            )
            .expect("Invalid regex pattern"),
            function_tag_pattern: Regex::new(r"<function=([^>]+)>\s*(\{[\s\S]*?\})\s*</function>")
                .expect("Invalid regex pattern"),
        }
    }
}

impl ToolCallParser for LlamaToolParser {
    fn name(&self) -> &'static str {
        "llama"
    }

    fn might_contain_tool_call(&self, output: &str) -> bool {
        output.contains("<|python_tag|>") || output.contains("<function=")
    }

    fn parse(&self, output: &str) -> ParsedOutput {
        if !self.might_contain_tool_call(output) {
            return ParsedOutput::Text(output.to_string());
        }

        let mut tool_calls = Vec::new();
        let mut remaining_text = output.to_string();

        // Try python_tag format first
        for captures in self.python_tag_pattern.captures_iter(output) {
            if let Some(json_match) = captures.get(1) {
                if let Ok(call) = serde_json::from_str::<serde_json::Value>(json_match.as_str()) {
                    if let Some(name) = call.get("name").and_then(|n| n.as_str()) {
                        let arguments = call
                            .get("parameters")
                            .or_else(|| call.get("arguments"))
                            .map(|a| {
                                if a.is_string() {
                                    a.as_str().unwrap_or("{}").to_string()
                                } else {
                                    serde_json::to_string(a).unwrap_or_else(|_| "{}".to_string())
                                }
                            })
                            .unwrap_or_else(|| "{}".to_string());

                        tool_calls.push(ParsedToolCall::new(name.to_string(), arguments));
                        remaining_text = remaining_text
                            .replace(captures.get(0).unwrap().as_str(), "")
                            .trim()
                            .to_string();
                    }
                }
            }
        }

        // Try function tag format
        for captures in self.function_tag_pattern.captures_iter(output) {
            if let (Some(name_match), Some(args_match)) = (captures.get(1), captures.get(2)) {
                let name = name_match.as_str().to_string();
                let arguments = args_match.as_str().to_string();
                tool_calls.push(ParsedToolCall::new(name, arguments));
                remaining_text = remaining_text
                    .replace(captures.get(0).unwrap().as_str(), "")
                    .trim()
                    .to_string();
            }
        }

        if tool_calls.is_empty() {
            ParsedOutput::Text(output.to_string())
        } else if remaining_text.is_empty() {
            ParsedOutput::ToolCalls(tool_calls)
        } else {
            ParsedOutput::Mixed {
                text: remaining_text,
                tool_calls,
            }
        }
    }

    fn tool_stop_sequences(&self) -> Vec<String> {
        vec!["<|eom_id|>".to_string(), "</function>".to_string()]
    }
}

// ============================================================================
// Generic JSON Tool Parser
// ============================================================================

/// Parser for generic JSON tool calls
/// Attempts to parse tool calls from JSON objects in the output
pub struct JsonToolParser {
    json_pattern: Regex,
}

impl Default for JsonToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonToolParser {
    pub fn new() -> Self {
        Self {
            // Simple pattern to detect potential tool call JSON
            json_pattern: Regex::new(
                r#"\{[^{}]*"(?:name|function)"[^{}]*"(?:arguments|parameters)"[^{}]*\}"#,
            )
            .expect("Invalid regex pattern"),
        }
    }

    fn try_parse_tool_call(&self, json_str: &str) -> Option<ParsedToolCall> {
        let value: serde_json::Value = serde_json::from_str(json_str).ok()?;

        let name = value
            .get("name")
            .or_else(|| value.get("function"))
            .and_then(|n| n.as_str())?
            .to_string();

        let arguments = value
            .get("arguments")
            .or_else(|| value.get("parameters"))
            .map(|a| {
                if a.is_string() {
                    a.as_str().unwrap_or("{}").to_string()
                } else {
                    serde_json::to_string(a).unwrap_or_else(|_| "{}".to_string())
                }
            })
            .unwrap_or_else(|| "{}".to_string());

        Some(ParsedToolCall::new(name, arguments))
    }
}

impl ToolCallParser for JsonToolParser {
    fn name(&self) -> &'static str {
        "json"
    }

    fn might_contain_tool_call(&self, output: &str) -> bool {
        output.contains("\"name\"") || output.contains("\"function\"")
    }

    fn parse(&self, output: &str) -> ParsedOutput {
        // Try to parse the entire output as a tool call
        if let Some(tool_call) = self.try_parse_tool_call(output.trim()) {
            return ParsedOutput::ToolCalls(vec![tool_call]);
        }

        // Try to find JSON objects in the output
        let mut tool_calls = Vec::new();
        for captures in self.json_pattern.find_iter(output) {
            if let Some(tool_call) = self.try_parse_tool_call(captures.as_str()) {
                tool_calls.push(tool_call);
            }
        }

        if tool_calls.is_empty() {
            ParsedOutput::Text(output.to_string())
        } else {
            // Try to extract text that's not part of tool calls
            let mut text = output.to_string();
            for captures in self.json_pattern.find_iter(output) {
                text = text.replace(captures.as_str(), "");
            }
            let text = text.trim().to_string();

            if text.is_empty() {
                ParsedOutput::ToolCalls(tool_calls)
            } else {
                ParsedOutput::Mixed { text, tool_calls }
            }
        }
    }
}

// ============================================================================
// Qwen Tool Parser
// ============================================================================

/// Parser for Qwen style tool calls
/// Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
/// Or: ✿FUNCTION✿: func_name\n✿ARGS✿: {...}
pub struct QwenToolParser {
    tool_call_tag_pattern: Regex,
    function_pattern: Regex,
}

impl Default for QwenToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl QwenToolParser {
    pub fn new() -> Self {
        Self {
            tool_call_tag_pattern: Regex::new(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>")
                .expect("Invalid regex pattern"),
            function_pattern: Regex::new(r"✿FUNCTION✿:\s*(\w+)\s*\n✿ARGS✿:\s*(\{[\s\S]*?\})")
                .expect("Invalid regex pattern"),
        }
    }
}

impl ToolCallParser for QwenToolParser {
    fn name(&self) -> &'static str {
        "qwen"
    }

    fn might_contain_tool_call(&self, output: &str) -> bool {
        output.contains("<tool_call>") || output.contains("✿FUNCTION✿")
    }

    fn parse(&self, output: &str) -> ParsedOutput {
        if !self.might_contain_tool_call(output) {
            return ParsedOutput::Text(output.to_string());
        }

        let mut tool_calls = Vec::new();
        let mut remaining_text = output.to_string();

        // Try <tool_call> format
        for captures in self.tool_call_tag_pattern.captures_iter(output) {
            if let Some(json_match) = captures.get(1) {
                if let Ok(call) = serde_json::from_str::<serde_json::Value>(json_match.as_str()) {
                    if let Some(name) = call.get("name").and_then(|n| n.as_str()) {
                        let arguments = call
                            .get("arguments")
                            .map(|a| {
                                if a.is_string() {
                                    a.as_str().unwrap_or("{}").to_string()
                                } else {
                                    serde_json::to_string(a).unwrap_or_else(|_| "{}".to_string())
                                }
                            })
                            .unwrap_or_else(|| "{}".to_string());

                        tool_calls.push(ParsedToolCall::new(name.to_string(), arguments));
                        remaining_text = remaining_text
                            .replace(captures.get(0).unwrap().as_str(), "")
                            .trim()
                            .to_string();
                    }
                }
            }
        }

        // Try ✿FUNCTION✿ format
        for captures in self.function_pattern.captures_iter(output) {
            if let (Some(name_match), Some(args_match)) = (captures.get(1), captures.get(2)) {
                let name = name_match.as_str().to_string();
                let arguments = args_match.as_str().to_string();
                tool_calls.push(ParsedToolCall::new(name, arguments));
                remaining_text = remaining_text
                    .replace(captures.get(0).unwrap().as_str(), "")
                    .trim()
                    .to_string();
            }
        }

        if tool_calls.is_empty() {
            ParsedOutput::Text(output.to_string())
        } else if remaining_text.is_empty() {
            ParsedOutput::ToolCalls(tool_calls)
        } else {
            ParsedOutput::Mixed {
                text: remaining_text,
                tool_calls,
            }
        }
    }

    fn tool_stop_sequences(&self) -> Vec<String> {
        vec!["</tool_call>".to_string()]
    }
}

// ============================================================================
// Composite/Auto Parser
// ============================================================================

/// A composite parser that tries multiple parsers in order
pub struct AutoToolParser {
    parsers: Vec<Box<dyn ToolCallParser>>,
}

impl Default for AutoToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoToolParser {
    pub fn new() -> Self {
        Self {
            parsers: vec![
                Box::new(MistralToolParser::new()),
                Box::new(LlamaToolParser::new()),
                Box::new(QwenToolParser::new()),
                Box::new(JsonToolParser::new()),
            ],
        }
    }

    pub fn with_parsers(parsers: Vec<Box<dyn ToolCallParser>>) -> Self {
        Self { parsers }
    }
}

impl ToolCallParser for AutoToolParser {
    fn name(&self) -> &'static str {
        "auto"
    }

    fn might_contain_tool_call(&self, output: &str) -> bool {
        self.parsers
            .iter()
            .any(|p| p.might_contain_tool_call(output))
    }

    fn parse(&self, output: &str) -> ParsedOutput {
        // Try each parser that might match
        for parser in &self.parsers {
            if parser.might_contain_tool_call(output) {
                let result = parser.parse(output);
                if result.has_tool_calls() {
                    return result;
                }
            }
        }

        // Fall back to text
        ParsedOutput::Text(output.to_string())
    }

    fn tool_stop_sequences(&self) -> Vec<String> {
        self.parsers
            .iter()
            .flat_map(|p| p.tool_stop_sequences())
            .collect()
    }
}

// ============================================================================
// Parser Factory
// ============================================================================

/// Get the appropriate tool parser for a given model
pub fn get_tool_parser(model_name: &str) -> Box<dyn ToolCallParser> {
    let model_lower = model_name.to_lowercase();

    if model_lower.contains("mistral") || model_lower.contains("ministral") {
        Box::new(MistralToolParser::new())
    } else if model_lower.contains("llama") {
        Box::new(LlamaToolParser::new())
    } else if model_lower.contains("qwen") {
        Box::new(QwenToolParser::new())
    } else {
        // Default to auto parser which tries all formats
        Box::new(AutoToolParser::new())
    }
}

/// Get a parser by name
pub fn get_tool_parser_by_name(name: &str) -> Option<Box<dyn ToolCallParser>> {
    match name.to_lowercase().as_str() {
        "mistral" | "ministral" => Some(Box::new(MistralToolParser::new())),
        "llama" => Some(Box::new(LlamaToolParser::new())),
        "qwen" => Some(Box::new(QwenToolParser::new())),
        "json" => Some(Box::new(JsonToolParser::new())),
        "auto" => Some(Box::new(AutoToolParser::new())),
        _ => None,
    }
}

// ============================================================================
// Incremental Parsing Implementations
// ============================================================================

impl IncrementalToolParser for MistralToolParser {
    fn parse_incremental(&self, buffer: &str) -> ToolParseState {
        // Look for [TOOL_CALLS] marker
        if !buffer.contains("[TOOL_CALLS]") {
            return ToolParseState::NotToolCall;
        }

        // Find the start of the JSON array
        if let Some(marker_pos) = buffer.find("[TOOL_CALLS]") {
            let after_marker = &buffer[marker_pos + 12..]; // Skip "[TOOL_CALLS]"

            // Look for opening bracket of JSON array
            if let Some(json_start) = after_marker.trim_start().find('[') {
                let json_part = &after_marker.trim_start()[json_start..];

                // Try to parse complete JSON
                if let Ok(calls) = serde_json::from_str::<Vec<serde_json::Value>>(json_part) {
                    // Complete! Extract first tool call
                    if let Some(call) = calls.first() {
                        if let (Some(name), Some(args)) = (
                            call.get("name").and_then(|n| n.as_str()),
                            call.get("arguments"),
                        ) {
                            let arguments = if args.is_string() {
                                args.as_str().unwrap_or("{}").to_string()
                            } else {
                                serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string())
                            };

                            return ToolParseState::Complete(ParsedToolCall::new(
                                name.to_string(),
                                arguments,
                            ));
                        }
                    }
                }

                // Partial JSON, try to extract what we can
                // Look for "name" field
                let name = if let Some(name_start) = json_part.find(r#""name""#) {
                    let after_name = &json_part[name_start + 6..];
                    if let Some(colon) = after_name.find(':') {
                        let value_part = &after_name[colon + 1..].trim_start();
                        if value_part.starts_with('"') {
                            // Extract quoted string
                            if let Some(end_quote) = value_part[1..].find('"') {
                                Some(value_part[1..end_quote + 1].to_string())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Accumulate arguments
                let arguments = if let Some(args_start) = json_part.find(r#""arguments""#) {
                    let after_args = &json_part[args_start + 11..];
                    if let Some(colon) = after_args.find(':') {
                        after_args[colon + 1..].trim_start().to_string()
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

                return ToolParseState::InProgress(PartialToolCall {
                    name,
                    arguments,
                    format: ToolCallFormat::Mistral,
                });
            }
        }

        ToolParseState::NotToolCall
    }
}

impl IncrementalToolParser for LlamaToolParser {
    fn parse_incremental(&self, buffer: &str) -> ToolParseState {
        // Look for <function= pattern
        if !buffer.contains("<function=") {
            return ToolParseState::NotToolCall;
        }

        if let Some(start) = buffer.find("<function=") {
            let after_start = &buffer[start + 10..];

            // Extract function name
            if let Some(close) = after_start.find('>') {
                let name = after_start[..close].to_string();
                let json_part = &after_start[close + 1..];

                // Check for closing tag
                if json_part.contains("</function>") {
                    // Complete!
                    if let Some(end) = json_part.find("</function>") {
                        let arguments = json_part[..end].to_string();
                        return ToolParseState::Complete(ParsedToolCall::new(name, arguments));
                    }
                }

                // In progress
                return ToolParseState::InProgress(PartialToolCall {
                    name: Some(name),
                    arguments: json_part.to_string(),
                    format: ToolCallFormat::Llama,
                });
            }
        }

        ToolParseState::NotToolCall
    }
}

impl IncrementalToolParser for QwenToolParser {
    fn parse_incremental(&self, buffer: &str) -> ToolParseState {
        // Look for <tool_call> pattern
        if !buffer.contains("<tool_call>") {
            return ToolParseState::NotToolCall;
        }

        if let Some(start) = buffer.find("<tool_call>") {
            let json_part = &buffer[start + 11..];

            // Check for closing tag
            if json_part.contains("</tool_call>") {
                // Complete!
                if let Some(end) = json_part.find("</tool_call>") {
                    let json_str = &json_part[..end];
                    if let Ok(call) = serde_json::from_str::<serde_json::Value>(json_str) {
                        if let (Some(name), Some(args)) = (
                            call.get("name").and_then(|n| n.as_str()),
                            call.get("arguments"),
                        ) {
                            let arguments = if args.is_string() {
                                args.as_str().unwrap_or("{}").to_string()
                            } else {
                                serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string())
                            };

                            return ToolParseState::Complete(ParsedToolCall::new(
                                name.to_string(),
                                arguments,
                            ));
                        }
                    }
                }
            }

            // In progress - try to extract partial info
            let name = if let Some(name_start) = json_part.find(r#""name""#) {
                let after_name = &json_part[name_start + 6..];
                if let Some(colon) = after_name.find(':') {
                    let value_part = &after_name[colon + 1..].trim_start();
                    if value_part.starts_with('"') {
                        if let Some(end_quote) = value_part[1..].find('"') {
                            Some(value_part[1..end_quote + 1].to_string())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            return ToolParseState::InProgress(PartialToolCall {
                name,
                arguments: json_part.to_string(),
                format: ToolCallFormat::Qwen,
            });
        }

        ToolParseState::NotToolCall
    }
}

impl IncrementalToolParser for JsonToolParser {
    fn parse_incremental(&self, buffer: &str) -> ToolParseState {
        // Simple JSON tool call detection
        let trimmed = buffer.trim();
        if !trimmed.starts_with('{') {
            return ToolParseState::NotToolCall;
        }

        // Try to parse as complete JSON
        if let Ok(call) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let (Some(name), Some(args)) = (
                call.get("name").and_then(|n| n.as_str()),
                call.get("arguments"),
            ) {
                let arguments = if args.is_string() {
                    args.as_str().unwrap_or("{}").to_string()
                } else {
                    serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string())
                };

                return ToolParseState::Complete(ParsedToolCall::new(name.to_string(), arguments));
            }
        }

        // In progress
        ToolParseState::InProgress(PartialToolCall {
            name: None,
            arguments: trimmed.to_string(),
            format: ToolCallFormat::Json,
        })
    }
}

impl IncrementalToolParser for AutoToolParser {
    fn parse_incremental(&self, buffer: &str) -> ToolParseState {
        // Try each parser in order
        for parser in &self.parsers {
            let result = match parser.name() {
                "mistral" => MistralToolParser::new().parse_incremental(buffer),
                "llama" => LlamaToolParser::new().parse_incremental(buffer),
                "qwen" => QwenToolParser::new().parse_incremental(buffer),
                "json" => JsonToolParser::new().parse_incremental(buffer),
                _ => ToolParseState::NotToolCall,
            };

            match result {
                ToolParseState::NotToolCall => continue,
                other => return other,
            }
        }

        ToolParseState::NotToolCall
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_parser_basic() {
        let parser = MistralToolParser::new();
        let output =
            r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris"}}]"#;

        let result = parser.parse(output);
        assert!(result.has_tool_calls());

        if let ParsedOutput::ToolCalls(calls) = result {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "get_weather");
            assert!(calls[0].arguments.contains("Paris"));
        } else {
            panic!("Expected ToolCalls");
        }
    }

    #[test]
    fn test_mistral_parser_with_text() {
        let parser = MistralToolParser::new();
        let output = r#"Let me check the weather for you.
[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris"}}]"#;

        let result = parser.parse(output);

        if let ParsedOutput::Mixed { text, tool_calls } = result {
            assert!(text.contains("Let me check"));
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].name, "get_weather");
        } else {
            panic!("Expected Mixed");
        }
    }

    #[test]
    fn test_mistral_parser_multiple_calls() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris"}}, {"name": "get_time", "arguments": {"timezone": "CET"}}]"#;

        let result = parser.parse(output);

        if let ParsedOutput::ToolCalls(calls) = result {
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].name, "get_weather");
            assert_eq!(calls[1].name, "get_time");
        } else {
            panic!("Expected ToolCalls");
        }
    }

    #[test]
    fn test_mistral_parser_no_tool_call() {
        let parser = MistralToolParser::new();
        let output = "Just a regular response without any tool calls.";

        let result = parser.parse(output);

        if let ParsedOutput::Text(text) = result {
            assert_eq!(text, output);
        } else {
            panic!("Expected Text");
        }
    }

    #[test]
    fn test_llama_parser_python_tag() {
        let parser = LlamaToolParser::new();
        let output = r#"<|python_tag|>{"name": "search", "parameters": {"query": "rust programming"}}<|eom_id|>"#;

        let result = parser.parse(output);
        assert!(result.has_tool_calls());

        if let ParsedOutput::ToolCalls(calls) = result {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "search");
        } else {
            panic!("Expected ToolCalls");
        }
    }

    #[test]
    fn test_llama_parser_function_tag() {
        let parser = LlamaToolParser::new();
        let output = r#"<function=get_weather>{"location": "Tokyo"}</function>"#;

        let result = parser.parse(output);
        assert!(result.has_tool_calls());

        if let ParsedOutput::ToolCalls(calls) = result {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "get_weather");
        } else {
            panic!("Expected ToolCalls");
        }
    }

    #[test]
    fn test_qwen_parser_tool_call_tag() {
        let parser = QwenToolParser::new();
        let output =
            r#"<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>"#;

        let result = parser.parse(output);
        assert!(result.has_tool_calls());

        if let ParsedOutput::ToolCalls(calls) = result {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "calculator");
        } else {
            panic!("Expected ToolCalls");
        }
    }

    #[test]
    fn test_json_parser_basic() {
        let parser = JsonToolParser::new();
        let output = r#"{"name": "search", "arguments": {"query": "test"}}"#;

        let result = parser.parse(output);
        assert!(result.has_tool_calls());

        if let ParsedOutput::ToolCalls(calls) = result {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "search");
        } else {
            panic!("Expected ToolCalls");
        }
    }

    #[test]
    fn test_auto_parser() {
        let parser = AutoToolParser::new();

        // Test Mistral format
        let output1 = r#"[TOOL_CALLS] [{"name": "test1", "arguments": {}}]"#;
        assert!(parser.parse(output1).has_tool_calls());

        // Test Llama format
        let output2 = r#"<function=test2>{"arg": "val"}</function>"#;
        assert!(parser.parse(output2).has_tool_calls());

        // Test plain text
        let output3 = "Just regular text";
        assert!(!parser.parse(output3).has_tool_calls());
    }

    #[test]
    fn test_get_tool_parser() {
        let mistral_parser = get_tool_parser("mistralai/Ministral-3B");
        assert_eq!(mistral_parser.name(), "mistral");

        let llama_parser = get_tool_parser("meta-llama/Llama-3.1-8B");
        assert_eq!(llama_parser.name(), "llama");

        let qwen_parser = get_tool_parser("Qwen/Qwen2-7B");
        assert_eq!(qwen_parser.name(), "qwen");

        let auto_parser = get_tool_parser("unknown-model");
        assert_eq!(auto_parser.name(), "auto");
    }

    #[test]
    fn test_parsed_tool_call_to_api() {
        let parsed = ParsedToolCall::new(
            "get_weather".to_string(),
            r#"{"location": "Paris"}"#.to_string(),
        );

        let api_call = parsed.to_tool_call();
        assert_eq!(api_call.call_type, "function");
        assert_eq!(api_call.function.name, "get_weather");
        assert!(api_call.id.starts_with("call_"));
    }

    #[test]
    fn test_parsed_output_into_api_tool_calls() {
        let parsed = ParsedOutput::ToolCalls(vec![
            ParsedToolCall::new("func1".to_string(), "{}".to_string()),
            ParsedToolCall::new("func2".to_string(), "{}".to_string()),
        ]);

        let api_calls = parsed.into_api_tool_calls().unwrap();
        assert_eq!(api_calls.len(), 2);
        assert_eq!(api_calls[0].function.name, "func1");
        assert_eq!(api_calls[1].function.name, "func2");
    }
}
