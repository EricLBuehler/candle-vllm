// src/tools/stream_parser.rs
//! Streaming tool call parser — detects and buffers tool calls during streaming.
//! Ported from xInfer (vllm.rs) server/parser.rs with tool-parser crate integration.

use crate::tools::{Tool, ToolCall};
use serde_json::{Map, Value};
use std::collections::HashSet;
use tokenizers::Tokenizer;
use tool_parser::{
    types::{StreamingParseResult, ToolCallItem},
    ParserFactory, ToolParser as ExternalToolParser,
};

#[derive(Debug, Clone, PartialEq)]
pub enum ToolModelType {
    Qwen3,
    Qwen3MoE,
    Qwen3_5,
    Qwen3_5MoE,
    LLaMa,
    LLaMa4,
    Phi,
    Phi4,
    Mistral,
    Mistral3VL,
    GLM4,
    GLM4MoE,
    GLM4MoeLite,
    Yi,
    StableLM,
    DeepSeek,
    GLM5,
    Gemma,
    Gemma3,
    Gemma4,
    Qwen3VL,
    MiniMax,
}

/// Look up the JSON schema for a parameter from a tool's properties definition.
/// Supports `anyOf`, `oneOf`, `allOf`, direct `type`, and `enum` fields.
fn extract_schema_types(schema: &Value) -> Vec<String> {
    let Some(obj) = schema.as_object() else {
        return vec!["string".to_string()];
    };
    let mut types = Vec::new();

    if let Some(t) = obj.get("type") {
        match t {
            Value::String(s) => types.push(s.clone()),
            Value::Array(arr) => {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        types.push(s.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(choices)) = obj.get(key) {
            for choice in choices {
                types.extend(extract_schema_types(choice));
            }
        }
    }

    if let Some(Value::Array(enum_vals)) = obj.get("enum") {
        for val in enum_vals {
            match val {
                Value::Null => types.push("null".to_string()),
                Value::Bool(_) => types.push("boolean".to_string()),
                Value::Number(n) => {
                    if n.is_i64() || n.is_u64() {
                        types.push("integer".to_string());
                    } else {
                        types.push("number".to_string());
                    }
                }
                Value::String(_) => types.push("string".to_string()),
                Value::Array(_) => types.push("array".to_string()),
                Value::Object(_) => types.push("object".to_string()),
            }
        }
    }

    if types.is_empty() {
        types.push("string".to_string());
    }
    types.sort();
    types.dedup();
    types
}

/// Convert a raw string parameter value to the correct JSON type based on
/// the tool schema, following vLLM's `MinimaxM2ToolParser` approach.
/// When only "string" is in `schema_types` (the default when no schema is
/// found), JSON parsing is still attempted first so that arrays/objects
/// passed as parameter values are preserved.
fn coerce_param_value(raw: &str, schema_types: &[String]) -> Value {
    let lower = raw.to_ascii_lowercase();
    if matches!(lower.as_str(), "null" | "none" | "nil") {
        return Value::Null;
    }

    // When the schema explicitly provides non-string types, use priority-based coercion.
    let has_explicit_types = schema_types
        .iter()
        .any(|t| !matches!(t.as_str(), "string" | "str" | "text"));

    if has_explicit_types {
        static TYPE_PRIORITY: &[&str] =
            &["integer", "number", "boolean", "object", "array", "string"];

        for ptype in TYPE_PRIORITY {
            if !schema_types.iter().any(|t| t == ptype) {
                continue;
            }
            match *ptype {
                "integer" => {
                    if let Ok(n) = raw.parse::<i64>() {
                        return Value::Number(n.into());
                    }
                }
                "number" => {
                    if let Ok(f) = raw.parse::<f64>() {
                        if f == (f as i64) as f64 {
                            return Value::Number((f as i64).into());
                        }
                        if let Some(n) = serde_json::Number::from_f64(f) {
                            return Value::Number(n);
                        }
                    }
                }
                "boolean" => {
                    let l = raw.trim().to_ascii_lowercase();
                    if matches!(l.as_str(), "true" | "1" | "yes" | "on") {
                        return Value::Bool(true);
                    }
                    if matches!(l.as_str(), "false" | "0" | "no" | "off") {
                        return Value::Bool(false);
                    }
                }
                "object" | "array" => {
                    if let Ok(v) = serde_json::from_str::<Value>(raw) {
                        return v;
                    }
                }
                "string" => return Value::String(raw.to_string()),
                _ => {}
            }
        }
    }

    // Fallback: try JSON, then string
    serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Resolve the parameter properties for a function from the available tools.
fn resolve_param_properties<'a>(
    function_name: &str,
    tools: &'a [Tool],
) -> Option<&'a serde_json::Map<String, Value>> {
    for tool in tools {
        if tool.function.name == function_name {
            return tool
                .function
                .parameters
                .get("properties")
                .and_then(|v| v.as_object());
        }
    }
    None
}

/// Manually parse MiniMax XML tool call format.
/// Format: `<minimax:tool_call><invoke name="..."><parameter name="...">...</parameter></invoke></minimax:tool_call>`
fn parse_minimax_xml_tool_calls(text: &str, tools: &[Tool]) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut search_from = 0;

    while let Some(invoke_start) = text[search_from..].find("<invoke name=") {
        let abs_invoke_start = search_from + invoke_start;
        let invoke_section = &text[abs_invoke_start..];

        let name_start = "<invoke name=".len();
        let quote_char = invoke_section.chars().nth(name_start);
        let Some(quote) = quote_char else {
            search_from = abs_invoke_start + 1;
            continue;
        };
        if quote != '"' && quote != '\'' {
            search_from = abs_invoke_start + 1;
            continue;
        }

        let name_content_start = name_start + 1;
        let Some(name_end_rel) = invoke_section[name_content_start..].find(quote) else {
            search_from = abs_invoke_start + 1;
            continue;
        };
        let function_name = &invoke_section[name_content_start..name_content_start + name_end_rel];

        let invoke_end = if let Some(end_rel) = invoke_section.find("</invoke>") {
            abs_invoke_start + end_rel + "</invoke>".len()
        } else {
            text.len()
        };

        let invoke_block = &text[abs_invoke_start..invoke_end];
        let param_props = resolve_param_properties(function_name, tools);

        let mut args = Map::new();
        let mut param_search = 0;
        while let Some(param_start) = invoke_block[param_search..].find("<parameter name=") {
            let abs_param_start = param_search + param_start;
            let param_section = &invoke_block[abs_param_start..];

            let pname_start = "<parameter name=".len();
            let pquote_char = param_section.chars().nth(pname_start);
            let Some(pquote) = pquote_char else {
                param_search = abs_param_start + 1;
                continue;
            };
            if pquote != '"' && pquote != '\'' {
                param_search = abs_param_start + 1;
                continue;
            }

            let pname_content_start = pname_start + 1;
            let Some(pname_end_rel) = param_section[pname_content_start..].find(pquote) else {
                param_search = abs_param_start + 1;
                continue;
            };
            let param_name =
                &param_section[pname_content_start..pname_content_start + pname_end_rel];

            let Some(value_start_rel) = param_section[pname_content_start + pname_end_rel..]
                .find('>')
                .map(|p| pname_content_start + pname_end_rel + p + 1)
            else {
                param_search = abs_param_start + 1;
                continue;
            };

            let value_section = &param_section[value_start_rel..];
            let value_end = value_section
                .find("</parameter>")
                .unwrap_or(value_section.len());
            let param_value = value_section[..value_end].trim();

            let schema_types = param_props
                .and_then(|props| props.get(param_name))
                .map(extract_schema_types)
                .unwrap_or_else(|| vec!["string".to_string()]);

            let json_value = coerce_param_value(param_value, &schema_types);
            args.insert(param_name.to_string(), json_value);

            param_search = abs_param_start + value_start_rel + value_end;
        }

        if !function_name.is_empty() {
            let args_str =
                serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".to_string());
            calls.push(crate::tools::new_tool_call(
                crate::tools::generate_tool_call_id(),
                function_name.to_string(),
                args_str,
            ));
        }

        search_from = invoke_end;
    }

    calls
}

/// Parser state for streaming tool call detection
#[derive(Debug, Clone, PartialEq)]
pub enum ParserState {
    /// Normal streaming mode - tokens pass through
    Normal,
    /// Potential start tag detected (partial match)
    // MaybeStart,
    /// Buffering mode - accumulating confirmed tool call content
    Buffering,
}

/// Result of processing a token in the stream
#[derive(Debug, Clone)]
pub enum StreamResult {
    /// Normal content - send to client
    Content(String),
    /// Buffering - don't send anything yet
    Buffering,
    /// Tool calls parsed - return tool calls for deferred emission
    ToolCalls(Vec<ToolCall>),
    /// False positive - flush accumulated buffer as content
    FlushBuffer(String),
}

/// Result of finalizing a buffered tool call at end-of-stream.
#[derive(Debug, Clone)]
pub enum BufferedFinalizeResult {
    ToolCalls(Vec<ToolCall>),
    FlushBuffer(String),
}

/// Configuration for model-specific tool call detection
#[derive(Clone, Debug)]
pub struct ToolConfig {
    pub start_token_ids: HashSet<u32>,
    pub end_token_ids: HashSet<u32>,
    pub start_token_str: String,
    pub end_token_str: String,
    pub start_is_special: bool,
    pub end_is_special: bool,
}

impl ToolConfig {
    /// Create tool config for a specific model type
    pub fn for_model_type(model_type: &ToolModelType) -> Self {
        let mut start_ids = HashSet::new();
        let mut end_ids = HashSet::new();

        match model_type {
            ToolModelType::LLaMa => {
                // Llama 3/3.1
                start_ids.insert(128010); // <|python_tag|>
                end_ids.insert(128008); // <|eom_id|>
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<|python_tag|>".to_string(),
                    end_token_str: "<|eom_id|>".to_string(),
                    start_is_special: false,
                    end_is_special: false,
                }
            }
            ToolModelType::LLaMa4 => {
                // Llama 4 uses pythonic tool call format: [func_name(param=value)]
                start_ids.insert(200016); // <|python_start|>
                end_ids.insert(200007); // <|eom|>
                end_ids.insert(200008); // <|eot|>
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<|python_start|>".to_string(),
                    end_token_str: "<|eom|>".to_string(),
                    start_is_special: true,
                    end_is_special: true,
                }
            }
            ToolModelType::Qwen3
            | ToolModelType::Qwen3MoE
            | ToolModelType::Qwen3_5
            | ToolModelType::Qwen3_5MoE
            | ToolModelType::Qwen3VL => {
                // Qwen 2.5 / 3
                start_ids.insert(151657); // <tool_call>
                end_ids.insert(151658); // </tool_call>
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<tool_call>".to_string(),
                    end_token_str: "</tool_call>".to_string(),
                    start_is_special: false,
                    end_is_special: false,
                }
            }
            ToolModelType::Mistral | ToolModelType::Mistral3VL => {
                // Mistral v3
                start_ids.insert(9); // [TOOL_CALLS]
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "[TOOL_CALLS]".to_string(),
                    end_token_str: "]".to_string(),
                    start_is_special: false,
                    end_is_special: false,
                }
            }
            ToolModelType::Gemma | ToolModelType::Gemma3 => {
                // Gemma 2/3 - uses text-only matching
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<start_function_call>".to_string(),
                    end_token_str: "<end_function_call>".to_string(),
                    start_is_special: false,
                    end_is_special: false,
                }
            }
            ToolModelType::Gemma4 => {
                start_ids.insert(48); // <|tool_call>
                end_ids.insert(49); // <tool_call|>
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<|tool_call>".to_string(),
                    end_token_str: "<tool_call|>".to_string(),
                    start_is_special: true,
                    end_is_special: true,
                }
            }
            // Phi, GLM, Yi, StableLM, DeepSeek - use Qwen format (text-only)
            ToolModelType::Phi
            | ToolModelType::Phi4
            | ToolModelType::GLM4
            | ToolModelType::GLM4MoE
            | ToolModelType::GLM4MoeLite
            | ToolModelType::GLM5
            | ToolModelType::Yi
            | ToolModelType::StableLM
            | ToolModelType::DeepSeek => ToolConfig {
                start_token_ids: HashSet::new(),
                end_token_ids: HashSet::new(),
                start_token_str: "<tool_call>".to_string(),
                end_token_str: "</tool_call>".to_string(),
                start_is_special: false,
                end_is_special: false,
            },
            ToolModelType::MiniMax => ToolConfig {
                // MiniMax tokenizer ships dedicated tool envelope tokens:
                //   200052 => <minimax:tool_call>
                //   200053 => </minimax:tool_call>
                // Keep them here so streaming detection can prefer token IDs.
                start_token_ids: HashSet::from([200052]),
                end_token_ids: HashSet::from([200053]),
                start_token_str: "<minimax:tool_call>".to_string(),
                end_token_str: "</minimax:tool_call>".to_string(),
                start_is_special: false,
                end_is_special: false,
            },
        }
    }

    /// Returns true if this config has special token IDs for detection
    pub fn has_special_tokens(&self) -> bool {
        self.has_start_tokens()
    }

    /// Returns true if start token IDs are available
    pub fn has_start_tokens(&self) -> bool {
        !self.start_token_ids.is_empty()
    }

    /// Returns true if end token IDs are available
    pub fn has_end_tokens(&self) -> bool {
        !self.end_token_ids.is_empty()
    }

    /// Validate special token IDs against the tokenizer, falling back to text-only matching if needed.
    /// Also auto-populates token IDs from the tokenizer when the config starts with empty sets.
    pub fn validate_with_tokenizer(&mut self, tokenizer: &Tokenizer, model_type: &ToolModelType) {
        if self.has_start_tokens() {
            if !Self::matches_single_token(tokenizer, &self.start_token_str, &self.start_token_ids)
            {
                if Self::try_rebind_single_token_id(
                    tokenizer,
                    &self.start_token_str,
                    &mut self.start_token_ids,
                ) {
                    tracing::warn!(
                        "Tool start token IDs corrected from tokenizer for model {:?}: {:?}",
                        model_type,
                        self.start_token_ids
                    );
                } else {
                    tracing::warn!(
                        "Tool start token IDs not supported by tokenizer for model {:?}, falling back to text matching",
                        model_type
                    );
                    self.start_token_ids.clear();
                }
            }
        } else if !self.start_token_str.is_empty() {
            if Self::try_rebind_single_token_id(
                tokenizer,
                &self.start_token_str,
                &mut self.start_token_ids,
            ) {
                self.start_is_special = true;
                tracing::info!(
                    "Tool start token IDs auto-populated from tokenizer for model {:?}: {:?}",
                    model_type,
                    self.start_token_ids
                );
            }
        }

        if self.has_end_tokens() {
            if !Self::matches_single_token(tokenizer, &self.end_token_str, &self.end_token_ids) {
                if Self::try_rebind_single_token_id(
                    tokenizer,
                    &self.end_token_str,
                    &mut self.end_token_ids,
                ) {
                    tracing::warn!(
                        "Tool end token IDs corrected from tokenizer for model {:?}: {:?}",
                        model_type,
                        self.end_token_ids
                    );
                } else {
                    tracing::warn!(
                        "Tool end token IDs not supported by tokenizer for model {:?}, falling back to text matching",
                        model_type
                    );
                    self.end_token_ids.clear();
                }
            }
        } else if !self.end_token_str.is_empty() {
            if Self::try_rebind_single_token_id(
                tokenizer,
                &self.end_token_str,
                &mut self.end_token_ids,
            ) {
                self.end_is_special = true;
                tracing::info!(
                    "Tool end token IDs auto-populated from tokenizer for model {:?}: {:?}",
                    model_type,
                    self.end_token_ids
                );
            }
        }
    }

    /// Create tool config from tokenizer, dynamically extracting tool call token IDs.
    /// This method first creates a config for the model type, then validates and overrides
    /// the token IDs using the actual tokenizer.
    pub fn from_tokenizer(tokenizer: &Tokenizer, model_type: &ToolModelType) -> Self {
        let mut config = Self::for_model_type(model_type);
        config.validate_with_tokenizer(tokenizer, model_type);
        config
    }

    /// Resolve tool call end token IDs using tokenizer and the validated config.
    pub fn tool_call_end_ids(&self, tokenizer: &Tokenizer) -> Vec<u32> {
        let mut tool_call_end_ids: Vec<u32> = Vec::new();

        let mut used_special = false;
        if self.has_end_tokens() {
            let mut use_special = true;
            if !self.end_token_str.is_empty() {
                if let Ok(encoded) = tokenizer.encode(self.end_token_str.as_str(), false) {
                    let ids = encoded.get_ids();
                    if ids.len() != 1 || !self.end_token_ids.contains(&ids[0]) {
                        use_special = false;
                    }
                } else {
                    use_special = false;
                }
            }
            if use_special {
                tool_call_end_ids.extend(self.end_token_ids.iter().copied());
                used_special = true;
            }
        }

        if !used_special && !self.end_token_str.is_empty() && self.end_token_str.starts_with('<') {
            // Only use text tags that look like explicit tool markers to avoid false positives.
            if let Ok(encoded) = tokenizer.encode(self.end_token_str.as_str(), false) {
                let ids = encoded.get_ids();
                if ids.len() == 1 {
                    tool_call_end_ids.push(ids[0]);
                }
            }
        }

        tool_call_end_ids
    }

    /// Resolve tool call start token IDs using tokenizer and the validated config.
    pub fn tool_call_start_ids(&self, tokenizer: &Tokenizer) -> Vec<u32> {
        let mut tool_call_start_ids: Vec<u32> = Vec::new();

        let mut used_special = false;
        if self.has_start_tokens() {
            let mut use_special = true;
            if !self.start_token_str.is_empty() {
                if let Ok(encoded) = tokenizer.encode(self.start_token_str.as_str(), false) {
                    let ids = encoded.get_ids();
                    if ids.len() != 1 || !self.start_token_ids.contains(&ids[0]) {
                        use_special = false;
                    }
                } else {
                    use_special = false;
                }
            }
            if use_special {
                tool_call_start_ids.extend(self.start_token_ids.iter().copied());
                used_special = true;
            }
        }

        if !used_special
            && !self.start_token_str.is_empty()
            && self.start_token_str.starts_with('<')
        {
            // Only use text tags that look like explicit tool markers to avoid false positives.
            if let Ok(encoded) = tokenizer.encode(self.start_token_str.as_str(), false) {
                let ids = encoded.get_ids();
                if ids.len() == 1 {
                    tool_call_start_ids.push(ids[0]);
                }
            }
        }

        tool_call_start_ids
    }

    fn matches_single_token(tokenizer: &Tokenizer, text: &str, token_ids: &HashSet<u32>) -> bool {
        if text.is_empty() {
            return false;
        }
        match tokenizer.encode(text, false) {
            Ok(encoded) => {
                let ids = encoded.get_ids();
                ids.len() == 1 && token_ids.contains(&ids[0])
            }
            Err(_) => false,
        }
    }

    fn try_rebind_single_token_id(
        tokenizer: &Tokenizer,
        text: &str,
        token_ids: &mut HashSet<u32>,
    ) -> bool {
        if text.is_empty() {
            return false;
        }

        if let Ok(encoded) = tokenizer.encode(text, false) {
            let ids = encoded.get_ids();
            if ids.len() == 1 {
                token_ids.clear();
                token_ids.insert(ids[0]);
                return true;
            }
        }

        if let Some(id) = tokenizer.get_vocab(true).get(text).copied() {
            token_ids.clear();
            token_ids.insert(id);
            return true;
        }

        false
    }
}

/// Streaming tool parser that handles tool call detection and buffering
pub struct StreamToolParser {
    config: ToolConfig,
    state: ParserState,
    buffer: String,
    model_id: String,
    parse_strategy: String,
    parser: Box<dyn ExternalToolParser>,
    tools: Vec<Tool>,
    streaming_calls: Vec<StreamingToolCallState>,
    // Accumulated output for final parsing
    accumulated_output: String,
    // Reasoning block tracking
    active_reasoning_end: Option<String>,
    // Code block tracking
    in_code_block: bool,
    // Set when incremental parsing found ToolCallItem(s) for the latest processed token.
    saw_buffer_parse_activity: bool,
    // Set when any parsing activity occurs during the current buffering window.
    buffer_had_parse_activity: bool,
    // Candidate end marker seen while buffering; used to avoid false end hits inside content.
    pending_end_marker_candidate: bool,
    // True when the current buffering window started from a dedicated special start token ID.
    buffer_started_from_special_token: bool,
    // True when non-marker content arrived after the start marker in the current buffering window.
    buffer_saw_non_marker_content: bool,
    // When true, tool call detection is active even inside reasoning blocks.
    // Used when reasoning content is streamed separately (STREAM_AS_REASONING_CONTENT).
    detect_tools_in_reasoning: bool,
}

/// Reasoning marker pairs: (start, end)
const REASONING_MARKERS: &[(&str, &str)] = &[
    ("<think>", "</think>"),
    ("<|think|>", "<|/think|>"),
    ("[THINK]", "[/THINK]"),
    ("<thought>", "</thought>"),
    ("<|channel>", "<channel|>"),
];

pub fn reasoning_markers() -> &'static [(&'static str, &'static str)] {
    REASONING_MARKERS
}

pub use crate::openai::conversation::default_conversation::extract_reasoning_content;

/// xInfer's response router removes individual reasoning markers before
/// placing visible text in an OpenAI delta field.
pub fn strip_reasoning_markers(text: &str) -> String {
    let mut result = text.to_string();
    for &(open, close) in reasoning_markers() {
        result = result.replace(open, "");
        result = result.replace(close, "");
    }
    result
}

/// Strip all reasoning blocks (matched start/end pairs) from text.
/// Unmatched opening markers are also removed up to the end of the string.
pub fn strip_reasoning_blocks(text: &str) -> String {
    let mut result = text.to_string();
    for &(start, end) in REASONING_MARKERS {
        loop {
            let Some(start_idx) = result.find(start) else {
                break;
            };
            let inner_start = start_idx + start.len();
            if let Some(end_rel) = result[inner_start..].find(end) {
                let end_idx = inner_start + end_rel + end.len();
                result.replace_range(start_idx..end_idx, "");
            } else {
                // Unmatched opening marker: remove from start marker to end of string
                // to avoid leaving dangling reasoning content.
                result.truncate(start_idx);
                break;
            }
        }
    }
    result
}

/// Detect whether a rendered prompt already ends inside a reasoning block.
///
/// This happens for templates that prefill `<think>` in `add_generation_prompt`.
/// Returns the corresponding end marker if matched.
pub fn detect_prefilled_reasoning_end_marker(prompt: &str) -> Option<String> {
    let trimmed = prompt.trim_end();
    for &(start, end) in REASONING_MARKERS {
        if trimmed.ends_with(start) {
            return Some(end.to_string());
        }
    }
    None
}

impl StreamToolParser {
    /// Create a new parser for the given model type
    pub fn new(model_type: ToolModelType, model_id: String) -> Self {
        let config = ToolConfig::for_model_type(&model_type);
        Self::new_with_config(&model_type, model_id, config, Vec::new(), None)
    }

    /// Create a new parser with a pre-validated tool config
    pub fn new_with_config(
        model_type: &ToolModelType,
        model_id: String,
        config: ToolConfig,
        tools: Vec<Tool>,
        enforce_parser: Option<String>,
    ) -> Self {
        let parse_strategy = match model_type {
            ToolModelType::Mistral | ToolModelType::Mistral3VL => "mistral_list",
            ToolModelType::Gemma4 => "gemma4",
            ToolModelType::LLaMa4 => "pythonic",
            _ => "json",
        }
        .to_string();

        let factory = ParserFactory::new();
        let parser_name = if let Some(name) = enforce_parser.as_ref().and_then(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        }) {
            if !factory.registry().has_parser(name) {
                let valid = factory.list_parsers().join(", ");
                panic!(
                    "Invalid enforce-parser '{}'. Valid parsers: {}",
                    name, valid
                );
            }
            name
        } else {
            Self::parser_name_for_model(model_type, &model_id)
        };
        if !tools.is_empty() {
            tracing::info!(
                "Tool parser selected: {} (model_id={}, enforce_parser={})",
                parser_name,
                model_id,
                enforce_parser.as_deref().unwrap_or("none")
            );
        }
        let parser = factory
            .registry()
            .create_parser(parser_name)
            .or_else(|| factory.registry().create_for_model(&model_id))
            .or_else(|| factory.registry().create_parser("passthrough"))
            .expect("tool parser available");

        Self {
            config,
            state: ParserState::Normal,
            buffer: String::new(),
            model_id,
            parse_strategy,
            parser,
            tools,
            streaming_calls: Vec::new(),
            accumulated_output: String::new(),
            active_reasoning_end: None,
            in_code_block: false,
            saw_buffer_parse_activity: false,
            buffer_had_parse_activity: false,
            pending_end_marker_candidate: false,
            buffer_started_from_special_token: false,
            buffer_saw_non_marker_content: false,
            detect_tools_in_reasoning: false,
        }
    }

    /// Enable tool call detection inside reasoning blocks.
    /// Should be set when reasoning content is streamed separately
    /// (STREAM_AS_REASONING_CONTENT), so that tool calls inside
    /// `<think>...</think>` are still detected and buffered.
    pub fn set_detect_tools_in_reasoning(&mut self, enabled: bool) {
        self.detect_tools_in_reasoning = enabled;
    }

    /// Check if currently inside a reasoning block
    pub fn in_reasoning(&self) -> bool {
        self.active_reasoning_end.is_some()
    }

    /// Set initial reasoning state when prompt already includes an opening think marker.
    pub fn set_initial_reasoning_end_marker(&mut self, end_marker: Option<String>) {
        self.active_reasoning_end = end_marker;
    }

    /// Advance reasoning-block tracking for one token without running tool
    /// detection. Used by streaming callers that don't have tools attached
    /// but still need `in_reasoning()` to reflect `<think>…</think>` state
    /// so SSE chunks can be routed to `delta.reasoning_content`.
    pub fn advance_reasoning_state(&mut self, token_text: &str) {
        self.accumulated_output.push_str(token_text);
        self.update_reasoning_state(token_text);
    }

    /// Check if currently inside a code block
    pub fn in_code_block(&self) -> bool {
        self.in_code_block
    }

    /// Get the current parser state
    pub fn state(&self) -> &ParserState {
        &self.state
    }

    /// Get accumulated output for debugging/logging
    pub fn accumulated_output(&self) -> &str {
        &self.accumulated_output
    }

    /// Return accumulated output with reasoning blocks stripped.
    /// Useful for fallback tool-call parsing when reasoning markers may have
    /// prevented the streaming parser from detecting tool calls.
    pub fn accumulated_output_without_reasoning(&self) -> String {
        strip_reasoning_blocks(&self.accumulated_output)
    }

    /// Get the buffered content
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Returns true if text contains tool-structure markup that should not be
    /// emitted verbatim as normal assistant text.
    pub fn contains_tool_markup(&self, text: &str) -> bool {
        if text.is_empty() {
            return false;
        }
        for marker in self.display_escape_markers() {
            if text.contains(&marker) || Self::contains_partial_marker_fragment(text, &marker) {
                return true;
            }
        }
        false
    }

    /// Escapes tool-structure markers in plain text so leaked tool payloads do
    /// not become executable-looking tags in later model turns.
    pub fn sanitize_tool_markup_for_display(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        let mut out = text.to_string();
        let mut markers = self.display_escape_markers();
        markers.sort_by_key(|m| std::cmp::Reverse(m.len()));
        markers.dedup();

        for marker in markers {
            if marker.is_empty() {
                continue;
            }
            out = out.replace(&marker, &Self::escape_marker_for_display(&marker));
            out = Self::escape_partial_marker_fragments(&out, &marker);
        }
        out
    }

    /// Returns whether the latest processed token produced incremental tool-parse activity.
    /// The flag is reset after being read.
    pub fn take_buffer_parse_activity(&mut self) -> bool {
        std::mem::take(&mut self.saw_buffer_parse_activity)
    }

    /// Process a single incoming token.
    /// Returns StreamResult indicating what action to take.
    pub async fn process_token(&mut self, token_id: u32, token_text: &str) -> StreamResult {
        self.saw_buffer_parse_activity = false;
        // Always accumulate
        self.accumulated_output.push_str(token_text);

        // Only track reasoning and code-block state while in Normal mode.
        // During Buffering the token content is tool-call payload (JSON, XML)
        // which may contain strings like "</think>" or "```" that must not
        // corrupt the reasoning/code-block tracking used for Normal-mode gating.
        if !matches!(self.state, ParserState::Buffering) {
            self.update_code_block_state(token_text);
            self.update_reasoning_state(token_text);
        }

        match self.state.clone() {
            ParserState::Normal => {
                // Don't detect tool-call starts inside code blocks.
                // Skip reasoning blocks unless detect_tools_in_reasoning is set
                // (which is the case when reasoning is streamed separately).
                if self.in_code_block {
                    return StreamResult::Content(token_text.to_string());
                }
                if self.in_reasoning() && !self.detect_tools_in_reasoning {
                    return StreamResult::Content(token_text.to_string());
                }
                // Check for start trigger
                if self.is_start_token(token_id, token_text) {
                    let started_from_special_token = self.config.has_start_tokens()
                        && self.config.start_token_ids.contains(&token_id);
                    self.state = ParserState::Buffering;
                    self.buffer.clear();
                    self.buffer.push_str(token_text);
                    self.streaming_calls.clear();
                    self.buffer_had_parse_activity = false;
                    self.pending_end_marker_candidate = false;
                    self.buffer_started_from_special_token = started_from_special_token;
                    self.buffer_saw_non_marker_content = false;
                    match self.parser.parse_incremental(token_text, &self.tools).await {
                        Ok(result) => {
                            if !result.calls.is_empty() {
                                self.saw_buffer_parse_activity = true;
                                self.buffer_had_parse_activity = true;
                            }
                            self.apply_streaming_result(&result);
                        }
                        Err(err) => {
                            tracing::warn!("Incremental tool parse failed at start tag: {:?}", err);
                        }
                    }

                    tracing::info!(
                        "Tool call {} ({}) found, start buffering!",
                        token_text,
                        token_id
                    );
                    return StreamResult::Buffering;
                }
                // Normal content
                StreamResult::Content(token_text.to_string())
            }
            ParserState::Buffering => {
                self.buffer.push_str(token_text);
                if self.token_contains_non_marker_content(token_text) {
                    self.buffer_saw_non_marker_content = true;
                }
                let nested_start_marker = !self.config.start_token_str.is_empty()
                    && token_text.contains(&self.config.start_token_str);
                if nested_start_marker {
                    tracing::warn!(
                        "Ignoring nested tool-call start marker while buffering: {:?}",
                        token_text
                    );
                } else {
                    match self.parser.parse_incremental(token_text, &self.tools).await {
                        Ok(result) => {
                            if !result.calls.is_empty() {
                                self.saw_buffer_parse_activity = true;
                                self.buffer_had_parse_activity = true;
                                // tracing::info!("Stream parsing: {:?}", result.calls);
                            }
                            self.apply_streaming_result(&result);
                        }
                        Err(err) => {
                            tracing::warn!(
                                "Incremental tool parse failed while buffering: {:?}",
                                err
                            );
                        }
                    }
                }
                let end_reached = self.is_end_token(token_id, token_text)
                    || self.buffer_has_end_tag()
                    || self.maybe_complete_mistral_list();
                if !end_reached && self.pending_end_marker_candidate {
                    self.pending_end_marker_candidate = false;
                }
                if end_reached {
                    let strict_complete = self.has_strict_complete_tool_call().await;
                    if !strict_complete && !self.pending_end_marker_candidate {
                        self.pending_end_marker_candidate = true;
                        tracing::warn!(
                            "Tool-call end marker seen before payload completion; waiting for confirmation"
                        );
                        return StreamResult::Buffering;
                    }
                    self.pending_end_marker_candidate = false;
                    tracing::info!(
                        "Tool call buffering end, reached {} ({})",
                        token_text,
                        token_id
                    );

                    let had_partial_calls = !self.streaming_calls.is_empty();
                    let tool_calls = self.build_tool_calls_with_fallback().await;
                    let result = if tool_calls.is_empty() {
                        if had_partial_calls {
                            tracing::warn!(
                                "End marker seen but tool call is still incomplete; continuing buffering"
                            );
                            StreamResult::Buffering
                        } else {
                            // False positive - flush buffered content as normal text.
                            tracing::error!("Unable to parse tool call buffer: {}", self.buffer,);
                            StreamResult::FlushBuffer(self.buffer.clone())
                        }
                    } else {
                        StreamResult::ToolCalls(tool_calls)
                    };
                    if matches!(result, StreamResult::Buffering) {
                        return result;
                    }
                    self.parser.reset();
                    self.buffer.clear();
                    self.state = ParserState::Normal;
                    self.streaming_calls.clear();
                    self.buffer_had_parse_activity = false;
                    self.pending_end_marker_candidate = false;
                    self.buffer_started_from_special_token = false;
                    self.buffer_saw_non_marker_content = false;
                    self.resync_reasoning_and_code_block_state();
                    return result;
                }

                StreamResult::Buffering
            }
        }
    }

    /// Finalize buffered tool-call state at EOS.
    /// Tries to build tool calls first; if unsuccessful, returns buffered text for flushing.
    pub async fn finalize_buffered_tool_calls(&mut self) -> Option<BufferedFinalizeResult> {
        if !matches!(self.state, ParserState::Buffering) {
            return None;
        }

        tracing::warn!("Stream ended while buffering a tool call; attempting final parse");

        let buffered_text = self.buffer.clone();
        let strict_complete = self.has_strict_complete_tool_call().await;
        let tool_calls = self.build_tool_calls_with_fallback().await;
        let recoverable_incomplete = !strict_complete
            && !tool_calls.is_empty()
            && self.can_recover_incomplete_buffered_tool_calls()
            && !self.has_ambiguous_incomplete_end_marker();

        self.parser.reset();
        self.buffer.clear();
        self.state = ParserState::Normal;
        self.streaming_calls.clear();
        self.buffer_had_parse_activity = false;
        self.pending_end_marker_candidate = false;
        let drop_bare_start_marker = self.should_drop_bare_start_marker();
        self.buffer_started_from_special_token = false;
        self.buffer_saw_non_marker_content = false;
        self.resync_reasoning_and_code_block_state();

        if tool_calls.is_empty() || (!strict_complete && !recoverable_incomplete) {
            if drop_bare_start_marker {
                tracing::warn!(
                    "Dropping buffered bare tool-call marker at stream end without flushing text"
                );
                return Some(BufferedFinalizeResult::FlushBuffer(String::new()));
            }
            tracing::warn!("Buffered tool call could not be finalized; flushing buffered text");
            Some(BufferedFinalizeResult::FlushBuffer(buffered_text))
        } else {
            if recoverable_incomplete {
                tracing::warn!(
                    "Recovered buffered tool call(s) from partial envelope using incremental parse state"
                );
            }
            tracing::warn!(
                "Recovered {} tool call(s) from buffered state at stream end",
                tool_calls.len()
            );
            Some(BufferedFinalizeResult::ToolCalls(tool_calls))
        }
    }

    /// Drain the buffer and reset parser state.
    pub fn take_buffer(&mut self) -> String {
        self.state = ParserState::Normal;
        self.buffer_had_parse_activity = false;
        self.pending_end_marker_candidate = false;
        self.buffer_started_from_special_token = false;
        self.buffer_saw_non_marker_content = false;
        let buf = std::mem::take(&mut self.buffer);
        self.resync_reasoning_and_code_block_state();
        buf
    }

    /// Re-derive reasoning and code-block state from the full accumulated
    /// output after exiting Buffering mode.  While buffering, these trackers
    /// are frozen so that tool-call payload (which may contain `</think>` or
    /// triple-backtick strings) does not corrupt them.  On transition back to
    /// Normal we must reconcile the true state.
    ///
    /// Tool-call envelopes (e.g. `<tool_call>...</tool_call>`) are masked out
    /// before scanning so that markers embedded in JSON arguments are ignored.
    fn resync_reasoning_and_code_block_state(&mut self) {
        let text = Self::mask_tool_envelopes(
            &self.accumulated_output,
            &self.config.start_token_str,
            &self.config.end_token_str,
        );

        // Re-derive code-block state: count fences in full output.
        let mut code_block_count = 0usize;
        for line in text.lines() {
            if line.trim().starts_with("```") {
                code_block_count += 1;
            }
        }
        self.in_code_block = code_block_count % 2 == 1;

        // Re-derive reasoning state: find the last unmatched reasoning start.
        self.active_reasoning_end = None;
        for &(start, end) in REASONING_MARKERS {
            let mut search_from = 0usize;
            let mut open = false;
            loop {
                if open {
                    match text[search_from..].find(end) {
                        Some(pos) => {
                            open = false;
                            search_from += pos + end.len();
                        }
                        None => break,
                    }
                } else {
                    match text[search_from..].find(start) {
                        Some(pos) => {
                            open = true;
                            search_from += pos + start.len();
                        }
                        None => break,
                    }
                }
            }
            if open {
                self.active_reasoning_end = Some(end.to_string());
                break;
            }
        }
    }

    /// Replace the content between each tool-call start/end envelope with
    /// spaces so that markers inside tool arguments are invisible to the
    /// reasoning/code-block resync scan.
    fn mask_tool_envelopes(text: &str, start_tag: &str, end_tag: &str) -> String {
        if start_tag.is_empty() || end_tag.is_empty() {
            return text.to_string();
        }
        let mut masked = text.to_string();
        let mut search_from = 0usize;
        loop {
            let Some(start_pos) = masked[search_from..]
                .find(start_tag)
                .map(|p| search_from + p)
            else {
                break;
            };
            let inner_start = start_pos + start_tag.len();
            let Some(end_pos) = masked[inner_start..].find(end_tag).map(|p| inner_start + p) else {
                break;
            };
            // Replace the inner content (between tags) with spaces.
            let replacement = " ".repeat(end_pos - inner_start);
            masked.replace_range(inner_start..end_pos, &replacement);
            search_from = end_pos + end_tag.len();
        }
        masked
    }

    /// Update code-block fence tracking from the current token.
    /// Only counts fences in the token text itself (incremental) to avoid
    /// re-scanning the entire accumulated output on every token.
    fn update_code_block_state(&mut self, token_text: &str) {
        for line in token_text.lines() {
            if line.trim().starts_with("```") {
                self.in_code_block = !self.in_code_block;
            }
        }
    }

    /// Update reasoning block tracking from the current token.
    fn update_reasoning_state(&mut self, token_text: &str) {
        if self.active_reasoning_end.is_none() {
            for &(start, end) in REASONING_MARKERS {
                if token_text.contains(start) || self.accumulated_output.ends_with(start) {
                    self.active_reasoning_end = Some(end.to_string());
                    break;
                }
            }
        } else if let Some(end_marker) = self.active_reasoning_end.as_deref() {
            if token_text.contains(end_marker) || self.accumulated_output.ends_with(end_marker) {
                self.active_reasoning_end = None;
            }
        }
    }

    /// Check if token/text matches start trigger
    fn is_start_token(&self, id: u32, _text: &str) -> bool {
        // Token ID match (if available)
        if self.config.has_start_tokens() {
            return self.config.start_token_ids.contains(&id);
        }

        // Text-only mode: detect on the current line, allowing split tags while
        // avoiding overly eager triggers like a lone "<".
        let current_line = self.accumulated_output.rsplit('\n').next().unwrap_or("");
        let candidate = current_line.trim_start_matches(|c| c == ' ' || c == '\t' || c == '\r');

        if candidate.starts_with(&self.config.start_token_str) {
            return true;
        }

        let min_prefix_len = Self::safe_partial_prefix_len(&self.config.start_token_str);
        !candidate.is_empty()
            && candidate.len() >= min_prefix_len
            && self.config.start_token_str.starts_with(candidate)
    }

    /// Check if token/text matches end trigger
    fn is_end_token(&self, id: u32, text: &str) -> bool {
        // Token ID match (if available)
        if self.config.has_end_tokens() {
            return self.config.end_token_ids.contains(&id);
        }
        if self.parse_strategy == "mistral_list" && self.config.end_token_str == "]" {
            return false;
        }
        // Text match
        text.contains(&self.config.end_token_str)
    }

    fn apply_streaming_result(&mut self, result: &StreamingParseResult) {
        if !result.calls.is_empty() {
            self.apply_stream_items(&result.calls);
        }
    }

    fn apply_stream_items(&mut self, items: &[ToolCallItem]) {
        if !items.is_empty() {
            self.buffer_had_parse_activity = true;
        }
        for item in items {
            if self.streaming_calls.len() <= item.tool_index {
                self.streaming_calls
                    .resize_with(item.tool_index + 1, StreamingToolCallState::default);
            }
            let state = &mut self.streaming_calls[item.tool_index];
            if let Some(name) = &item.name {
                state.name = Some(name.clone());
            }
            if !item.parameters.is_empty() {
                state.arguments.push_str(&item.parameters);
            }
        }
    }

    fn build_tool_calls_from_streaming(&mut self) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        tracing::info!("Building tool call: {:?}", self.streaming_calls);
        for state in &self.streaming_calls {
            let Some(name) = &state.name else { continue };
            let args = self.finalize_streamed_arguments(&state.arguments);
            calls.push(crate::tools::new_tool_call(
                crate::tools::generate_tool_call_id(),
                name.clone(),
                args,
            ));
        }
        calls
    }

    async fn build_tool_calls_with_fallback(&mut self) -> Vec<ToolCall> {
        if let Some(unstreamed) = self.parser.get_unstreamed_tool_args() {
            self.apply_stream_items(&unstreamed);
        }
        self.recover_streaming_arguments_from_buffer();
        let streaming_calls = self.build_tool_calls_from_streaming();
        let fallback_calls = self.parse_complete_with_fallback(&self.buffer).await;
        if self.should_prefer_fallback_tool_calls(&streaming_calls, &fallback_calls) {
            tracing::info!("Fallback to non-stream parsing for buffer: {}", self.buffer);
            return fallback_calls;
        }

        streaming_calls
    }

    fn should_prefer_fallback_tool_calls(
        &self,
        streaming_calls: &[ToolCall],
        fallback_calls: &[ToolCall],
    ) -> bool {
        if fallback_calls.is_empty() {
            return false;
        }
        if streaming_calls.is_empty() || fallback_calls.len() > streaming_calls.len() {
            return true;
        }

        let streaming_missing_args = self
            .streaming_calls
            .iter()
            .any(|call| call.name.is_none() || call.arguments.trim().is_empty());
        if streaming_missing_args {
            return true;
        }

        streaming_calls
            .iter()
            .zip(fallback_calls.iter())
            .any(|(streaming, fallback)| {
                streaming.function.arguments.as_deref() == Some("{}")
                    && fallback.function.arguments.as_deref() != Some("{}")
            })
    }

    async fn has_strict_complete_tool_call(&self) -> bool {
        if !self.has_complete_tool_envelope() {
            return false;
        }
        if self.streaming_calls.iter().any(|call| call.name.is_none()) {
            return false;
        }
        if !self.streaming_calls.is_empty()
            && self
                .streaming_calls
                .iter()
                .all(|call| call.arguments.trim().is_empty())
        {
            return true;
        }
        if !self.streaming_calls.is_empty()
            && self
                .streaming_calls
                .iter()
                .all(|call| serde_json::from_str::<Value>(call.arguments.trim()).is_ok())
        {
            return true;
        }

        !self
            .parse_complete_with_fallback(&self.buffer)
            .await
            .is_empty()
    }

    fn can_recover_incomplete_buffered_tool_calls(&self) -> bool {
        if self.streaming_calls.is_empty() {
            return false;
        }

        if !self.buffer_had_parse_activity
            && self
                .streaming_calls
                .iter()
                .all(|call| call.arguments.trim().is_empty())
        {
            return false;
        }

        if self.streaming_calls.iter().any(|call| call.name.is_none()) {
            return false;
        }

        self.streaming_calls.iter().all(|call| {
            let args = call.arguments.trim();
            args.is_empty()
                || serde_json::from_str::<Value>(&self.finalize_streamed_arguments(args)).is_ok()
        })
    }

    fn has_ambiguous_incomplete_end_marker(&self) -> bool {
        if self.config.end_token_str.is_empty() || !self.config.end_token_str.starts_with('<') {
            return false;
        }

        self.buffer.contains(&self.config.end_token_str) && !self.has_complete_tool_envelope()
    }

    fn has_complete_tool_envelope(&self) -> bool {
        // Non-XML formats should not be gated by XML envelope checks.
        if !self.config.start_token_str.starts_with('<')
            || !self.config.end_token_str.starts_with('<')
        {
            return true;
        }

        // `<|...|>` style special tokens (e.g. LLaMA 4's <|python_start|>/<|eom|>)
        // are not XML envelopes; skip structural validation for them.
        if self.config.start_token_str.starts_with("<|")
            || self.config.end_token_str.starts_with("<|")
        {
            return true;
        }

        let Some(start_idx) = self.buffer.find(&self.config.start_token_str) else {
            // If no explicit start marker is present, keep existing behavior.
            return true;
        };

        let section = &self.buffer[start_idx..];
        let Some(end_rel) = section.rfind(&self.config.end_token_str) else {
            return false;
        };
        let end_idx = start_idx + end_rel + self.config.end_token_str.len();
        if end_idx <= start_idx {
            return false;
        }

        let block = &self.buffer[start_idx..end_idx];
        let inner_start = start_idx + self.config.start_token_str.len();
        let inner_end = end_idx - self.config.end_token_str.len();
        if inner_end < inner_start {
            return false;
        }
        let inner = self.buffer[inner_start..inner_end].trim();

        // MiniMax XML style: <minimax:tool_call><invoke name="..."><parameter name="...">...</parameter></invoke></minimax:tool_call>
        if block.contains("<invoke ") || block.contains("<parameter name=") {
            let Some(invoke_start) = block.find("<invoke ") else {
                return false;
            };
            let invoke_section = &block[invoke_start..];
            let Some(invoke_end_rel) = invoke_section.rfind("</invoke>") else {
                return false;
            };
            let invoke_end = invoke_start + invoke_end_rel + "</invoke>".len();
            let invoke_block = &block[invoke_start..invoke_end];

            if !Self::has_balanced_parameter_tags(invoke_block, "<parameter name=") {
                return false;
            }
            return true;
        }

        // Qwen-coder XML style: <tool_call><function=...><parameter=...>...</parameter></function></tool_call>
        if block.contains("<function=") || block.contains("<parameter=") {
            let Some(function_start) = block.find("<function=") else {
                return false;
            };
            let function_section = &block[function_start..];
            // Use the last function closer inside the current tool-call block so
            // literal `</function>` text inside parameter content does not
            // truncate the structural envelope check.
            let Some(function_end_rel) = function_section.rfind("</function>") else {
                return false;
            };
            let function_end = function_start + function_end_rel + "</function>".len();
            let function_block = &block[function_start..function_end];

            // Validate parameter pairing in order and ignore unmatched closing tags.
            // This tolerates malformed tails like:
            //   </function>\n</parameter>\n</function>
            // which should not invalidate an otherwise complete function payload.
            if !Self::has_balanced_parameter_tags(function_block, "<parameter=") {
                return false;
            }
            return true;
        }

        // GLM4.7 XML style: <tool_call>func_name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
        if block.contains("<arg_key>") || block.contains("<arg_value>") {
            return block.contains("</arg_value>")
                && Self::has_balanced_xml_tags(block, "<arg_key>", "</arg_key>")
                && Self::has_balanced_xml_tags(block, "<arg_value>", "</arg_value>");
        }

        // Qwen JSON style: <tool_call>{"name":"...","arguments":{...}}</tool_call>
        // Accept only if the inner payload is complete JSON at this point.
        if inner.is_empty() {
            return false;
        }
        serde_json::from_str::<Value>(inner).is_ok()
    }

    fn has_balanced_parameter_tags(function_block: &str, open_tag: &str) -> bool {
        let mut idx = 0usize;
        let mut open_count = 0usize;
        const CLOSE: &str = "</parameter>";

        while idx < function_block.len() {
            let open_pos = function_block[idx..].find(open_tag).map(|p| idx + p);
            let close_pos = function_block[idx..].find(CLOSE).map(|p| idx + p);

            match (open_pos, close_pos) {
                (None, None) => break,
                (Some(op), None) => {
                    open_count += 1;
                    idx = op + open_tag.len();
                }
                (None, Some(cp)) => {
                    if open_count > 0 {
                        open_count -= 1;
                    }
                    idx = cp + CLOSE.len();
                }
                (Some(op), Some(cp)) => {
                    if op < cp {
                        open_count += 1;
                        idx = op + open_tag.len();
                    } else {
                        if open_count > 0 {
                            open_count -= 1;
                        }
                        idx = cp + CLOSE.len();
                    }
                }
            }
        }

        open_count == 0
    }

    fn has_balanced_xml_tags(block: &str, open: &str, close: &str) -> bool {
        let open_count = block.matches(open).count();
        let close_count = block.matches(close).count();
        open_count > 0 && open_count == close_count
    }

    pub async fn parse_complete_with_fallback(&self, text: &str) -> Vec<ToolCall> {
        if self.parse_strategy == "gemma4" {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Self::parse_gemma4_tool_calls(text)
            })) {
                Ok(Some(calls)) => return calls,
                Ok(None) => {}
                Err(e) => {
                    let msg = e
                        .downcast_ref::<String>()
                        .map(|s| s.as_str())
                        .or_else(|| e.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown");
                    tracing::warn!("Gemma4 tool call parse panicked: {}", msg);
                }
            }
        }

        let mut parsed_calls = match self.parser.parse_complete(text).await {
            Ok((_normal_text, calls)) => calls,
            Err(err) => {
                tracing::warn!("Tool parse failed: {:?}", err);
                Vec::new()
            }
        };

        // Pythonic fallback for LLaMA 4: the model may output tool calls
        // without the <|python_start|> token, so re-try with a fresh pythonic
        // parser after stripping any remaining special tokens.
        if parsed_calls.is_empty() && self.parse_strategy == "pythonic" {
            let factory = ParserFactory::new();
            if let Some(pythonic_parser) = factory.registry().create_parser("pythonic") {
                if let Ok((_normal_text, calls)) = pythonic_parser.parse_complete(text).await {
                    parsed_calls = calls;
                }
            }
        }

        if parsed_calls.is_empty() && text.contains("<invoke name=") {
            let factory = ParserFactory::new();
            if let Some(xml_parser) = factory.registry().create_parser("minimax_m2") {
                if let Ok((_normal_text, calls)) = xml_parser.parse_complete(text).await {
                    parsed_calls = calls;
                }
            }
            // Manual fallback if tool-parser crate fails
            if parsed_calls.is_empty() {
                tracing::info!("Falling back to manual MiniMax XML parser for buffer");
                return parse_minimax_xml_tool_calls(text, &self.tools);
            }
        }

        if parsed_calls.is_empty() && text.contains("<function=") {
            let factory = ParserFactory::new();
            if let Some(xml_parser) = factory.registry().create_parser("qwen_coder") {
                if let Ok((_normal_text, calls)) = xml_parser.parse_complete(text).await {
                    parsed_calls = calls;
                }
            }
        }

        if parsed_calls.is_empty()
            && self.config.start_token_str.starts_with('<')
            && self.config.end_token_str.starts_with('<')
            && (text.contains(&self.config.start_token_str)
                || text.contains(&self.config.end_token_str))
        {
            let stripped = self.strip_tool_tags(text);
            let factory = ParserFactory::new();
            if let Some(json_parser) = factory.registry().create_parser("json") {
                if let Ok((_normal_text, calls)) = json_parser.parse_complete(&stripped).await {
                    parsed_calls = calls;
                }
            }
        }

        // Final fallback: only for JSON-native parsing strategy and only when the
        // completion itself looks like JSON. Avoid scanning arbitrary mixed text
        // to prevent false positives from example snippets.
        if parsed_calls.is_empty()
            && self.parse_strategy == "json"
            && self.parser.has_tool_markers(text)
        {
            let factory = ParserFactory::new();
            if let Some(json_parser) = factory.registry().create_parser("json") {
                if let Ok((normal_text, calls)) = json_parser.parse_complete(text).await {
                    if normal_text.trim().is_empty() {
                        parsed_calls = calls;
                    }
                }
            }
        }

        parsed_calls
            .into_iter()
            .map(crate::tools::tool_call_from_parser)
            .collect()
    }

    fn finalize_streamed_arguments(&self, raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return "{}".to_string();
        }
        if serde_json::from_str::<Value>(trimmed).is_ok() {
            return trimmed.to_string();
        }

        let repaired = repair_streamed_json_arguments(trimmed);
        if repaired != trimmed {
            tracing::warn!("Applied structural JSON repair to streamed tool arguments");
        }
        repaired
    }

    fn buffer_has_end_tag(&self) -> bool {
        if self.config.end_token_str.is_empty() {
            return false;
        }
        if self.config.has_end_tokens() {
            return false;
        }
        if self.parse_strategy == "mistral_list" && self.config.end_token_str == "]" {
            return false;
        }
        self.buffer.contains(&self.config.end_token_str)
    }

    fn maybe_complete_mistral_list(&self) -> bool {
        if self.parse_strategy != "mistral_list" {
            return false;
        }
        let trimmed = self.buffer.trim();
        if !trimmed.ends_with(']') {
            return false;
        }
        serde_json::from_str::<Vec<Value>>(trimmed).is_ok()
    }

    pub fn parser_name_for_model(model_type: &ToolModelType, model_id: &str) -> &'static str {
        let model_lower = model_id.to_ascii_lowercase();
        match model_type {
            ToolModelType::LLaMa => "llama",
            ToolModelType::Mistral | ToolModelType::Mistral3VL => "mistral",
            ToolModelType::Qwen3_5 | ToolModelType::Qwen3_5MoE => "qwen_coder",
            ToolModelType::Qwen3 | ToolModelType::Qwen3MoE | ToolModelType::Qwen3VL => {
                if model_lower.contains("coder")
                    || model_lower.contains("qwen3.5")
                    || model_lower.contains("qwen3.6")
                {
                    "qwen_coder"
                } else {
                    "qwen"
                }
            }
            ToolModelType::Gemma | ToolModelType::Gemma3 | ToolModelType::Gemma4 => "json",
            ToolModelType::LLaMa4 => "pythonic",
            ToolModelType::Phi | ToolModelType::Phi4 => "qwen",
            ToolModelType::GLM4
            | ToolModelType::GLM4MoE
            | ToolModelType::GLM4MoeLite
            | ToolModelType::GLM5 => "glm47_moe",
            ToolModelType::Yi | ToolModelType::StableLM => "qwen",
            ToolModelType::DeepSeek => "deepseek",
            ToolModelType::MiniMax => "minimax_m2",
        }
    }

    /// Parse Gemma4 tool calls: `<|tool_call>call:NAME{key:<|"|>value<|"|>,...}<tool_call|>`
    /// Also handles stripped markers: `call:NAME{key:"value",...}`
    ///
    /// Follows vLLM's tiered approach: regex extraction first, then
    /// custom brace-matching as fallback. All string operations are
    /// UTF-8 safe.
    fn parse_gemma4_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
        const PREFIX: &str = "<|tool_call>call:";
        const PREFIX_STRIPPED: &str = "call:";
        const SUFFIX: &str = "<tool_call|>";

        let text = text.trim_end();
        let text = text
            .strip_suffix("<|tool_response>")
            .or_else(|| text.strip_suffix("<tool_response|>"))
            .unwrap_or(text);

        let has_full_prefix = text.contains(PREFIX);
        let has_stripped_prefix = !has_full_prefix && text.contains(PREFIX_STRIPPED);
        if !has_full_prefix && !has_stripped_prefix {
            return None;
        }
        let active_prefix = if has_full_prefix {
            PREFIX
        } else {
            PREFIX_STRIPPED
        };

        let mut calls = Vec::new();
        let mut search_start = 0;

        while let Some(rel_pos) = text[search_start..].find(active_prefix) {
            let abs_start = search_start + rel_pos + active_prefix.len();
            let Some(brace_rel) = text[abs_start..].find('{') else {
                break;
            };
            let name = text[abs_start..abs_start + brace_rel].trim().to_string();
            let brace_abs = abs_start + brace_rel;

            let matched = Self::gemma4_extract_braces(text, brace_abs);
            let Some((inner, after_brace)) = matched else {
                break;
            };

            let arguments = Self::gemma4_parse_args(inner);

            calls.push(ToolCall {
                index: None,
                id: crate::tools::generate_tool_call_id(),
                tool_type: "function".to_string(),
                function: crate::tools::FunctionCall {
                    name,
                    arguments: Some(serde_json::to_string(&arguments).unwrap_or_default()),
                },
            });

            let remaining = &text[after_brace..];
            if let Some(suf_pos) = remaining.find(SUFFIX) {
                search_start = after_brace + suf_pos + SUFFIX.len();
            } else {
                search_start = after_brace;
            }
        }

        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn gemma4_extract_braces(s: &str, start: usize) -> Option<(&str, usize)> {
        const DELIM: &str = "<|\"|>";

        if !s.is_char_boundary(start) || s.as_bytes().get(start) != Some(&b'{') {
            return None;
        }

        let mut depth: usize = 0;
        let mut in_delim_string = false;
        let mut in_regular_string = false;
        let tail = &s[start..];
        let mut iter = tail.char_indices();

        while let Some((offset, ch)) = iter.next() {
            let abs = start + offset;

            if in_delim_string {
                if tail[offset..].starts_with(DELIM) {
                    in_delim_string = false;
                    for _ in 0..DELIM.len().saturating_sub(ch.len_utf8()) {
                        iter.next();
                    }
                }
                continue;
            }

            if in_regular_string {
                if ch == '"' && (offset == 0 || tail.as_bytes()[offset - 1] != b'\\') {
                    in_regular_string = false;
                }
                continue;
            }

            if tail[offset..].starts_with(DELIM) {
                in_delim_string = true;
                for _ in 0..DELIM.len().saturating_sub(ch.len_utf8()) {
                    iter.next();
                }
                continue;
            }

            match ch {
                '"' => {
                    in_regular_string = true;
                }
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        let inner_start = start + '{'.len_utf8();
                        return Some((&s[inner_start..abs], abs + '}'.len_utf8()));
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Parse Gemma4 key-value format into a JSON Value.
    ///
    /// Follows vLLM's `_parse_gemma4_args` approach: iterates through the
    /// string character-by-character using `char_indices` (UTF-8 safe) and
    /// handles `<|"|>` delimited strings, nested objects, arrays, and bare
    /// values (numbers, booleans, null).
    fn gemma4_parse_args(args_str: &str) -> Value {
        if args_str.trim().is_empty() {
            return Value::Object(Map::new());
        }

        let cleaned = args_str.replace("<|\"|>", "\"");
        if let Ok(v) = serde_json::from_str::<Value>(&format!("{{{cleaned}}}")) {
            return v;
        }

        let mut map = Map::new();
        let chars: Vec<(usize, char)> = args_str.char_indices().collect();
        let n = chars.len();
        let mut ci = 0;

        while ci < n {
            while ci < n && matches!(chars[ci].1, ' ' | ',' | '\n' | '\t') {
                ci += 1;
            }
            if ci >= n {
                break;
            }

            let key_start_byte = chars[ci].0;
            while ci < n && chars[ci].1 != ':' {
                ci += 1;
            }
            if ci >= n {
                break;
            }
            let key = args_str[key_start_byte..chars[ci].0]
                .trim()
                .trim_matches('"');
            ci += 1; // skip ':'

            while ci < n && matches!(chars[ci].1, ' ' | '\n' | '\t') {
                ci += 1;
            }
            if ci >= n {
                map.insert(key.to_string(), Value::String(String::new()));
                break;
            }

            const DELIM: &str = "<|\"|>";
            let byte_pos = chars[ci].0;

            if args_str[byte_pos..].starts_with(DELIM) {
                let delim_char_len = DELIM.chars().count();
                ci += delim_char_len;
                let val_start = if ci < n { chars[ci].0 } else { args_str.len() };
                let end_byte = args_str[val_start..].find(DELIM);
                match end_byte {
                    Some(rel) => {
                        let val = &args_str[val_start..val_start + rel];
                        map.insert(key.to_string(), Value::String(val.to_string()));
                        let after = val_start + rel + DELIM.len();
                        ci = chars.iter().position(|&(b, _)| b >= after).unwrap_or(n);
                    }
                    None => {
                        let val = &args_str[val_start..];
                        map.insert(key.to_string(), Value::String(val.to_string()));
                        break;
                    }
                }
            } else if chars[ci].1 == '"' {
                ci += 1;
                let val_start = if ci < n { chars[ci].0 } else { args_str.len() };
                let mut end_ci = ci;
                while end_ci < n {
                    if chars[end_ci].1 == '"' && (end_ci == 0 || chars[end_ci - 1].1 != '\\') {
                        break;
                    }
                    end_ci += 1;
                }
                let val_end = if end_ci < n {
                    chars[end_ci].0
                } else {
                    args_str.len()
                };
                let val = &args_str[val_start..val_end];
                map.insert(key.to_string(), Value::String(val.to_string()));
                ci = if end_ci < n { end_ci + 1 } else { n };
            } else if chars[ci].1 == '{' {
                let (inner, after_ci) = Self::gemma4_scan_nested(&chars, ci, '{', '}', n, args_str);
                let nested = Self::gemma4_parse_args(inner);
                map.insert(key.to_string(), nested);
                ci = after_ci;
            } else if chars[ci].1 == '[' {
                let (inner, after_ci) = Self::gemma4_scan_nested(&chars, ci, '[', ']', n, args_str);
                let arr = Self::gemma4_parse_array(inner);
                map.insert(key.to_string(), arr);
                ci = after_ci;
            } else {
                let val_start = chars[ci].0;
                while ci < n && !matches!(chars[ci].1, ',' | '}' | ']') {
                    ci += 1;
                }
                let val_end = if ci < n { chars[ci].0 } else { args_str.len() };
                let val = args_str[val_start..val_end].trim();
                map.insert(key.to_string(), Self::gemma4_parse_bare_value(val));
            }
        }

        Value::Object(map)
    }

    fn gemma4_scan_nested<'a>(
        chars: &[(usize, char)],
        start_ci: usize,
        open: char,
        close: char,
        n: usize,
        source: &'a str,
    ) -> (&'a str, usize) {
        const DELIM: &str = "<|\"|>";
        let delim_char_len = DELIM.chars().count();
        let mut depth = 1usize;
        let mut ci = start_ci + 1;
        let inner_start = if ci < n { chars[ci].0 } else { source.len() };

        while ci < n && depth > 0 {
            let byte_pos = chars[ci].0;
            if source[byte_pos..].starts_with(DELIM) {
                ci += delim_char_len;
                while ci < n {
                    let bp = chars[ci].0;
                    if source[bp..].starts_with(DELIM) {
                        ci += delim_char_len;
                        break;
                    }
                    ci += 1;
                }
                continue;
            }
            if chars[ci].1 == open {
                depth += 1;
            } else if chars[ci].1 == close {
                depth -= 1;
            }
            ci += 1;
        }

        let inner_end = if depth == 0 && ci > 0 {
            chars[ci - 1].0
        } else if ci <= n && ci > 0 {
            if ci < n {
                chars[ci].0
            } else {
                source.len()
            }
        } else {
            source.len()
        };

        (&source[inner_start..inner_end], ci)
    }

    fn gemma4_parse_array(arr_str: &str) -> Value {
        const DELIM: &str = "<|\"|>";
        let mut items = Vec::new();
        let chars: Vec<(usize, char)> = arr_str.char_indices().collect();
        let n = chars.len();
        let mut ci = 0;

        while ci < n {
            while ci < n && matches!(chars[ci].1, ' ' | ',' | '\n' | '\t') {
                ci += 1;
            }
            if ci >= n {
                break;
            }

            let byte_pos = chars[ci].0;

            if arr_str[byte_pos..].starts_with(DELIM) {
                let delim_char_len = DELIM.chars().count();
                ci += delim_char_len;
                let val_start = if ci < n { chars[ci].0 } else { arr_str.len() };
                let end_byte = arr_str[val_start..].find(DELIM);
                match end_byte {
                    Some(rel) => {
                        items.push(Value::String(
                            arr_str[val_start..val_start + rel].to_string(),
                        ));
                        let after = val_start + rel + DELIM.len();
                        ci = chars.iter().position(|&(b, _)| b >= after).unwrap_or(n);
                    }
                    None => {
                        items.push(Value::String(arr_str[val_start..].to_string()));
                        break;
                    }
                }
            } else if chars[ci].1 == '"' {
                ci += 1;
                let val_start = if ci < n { chars[ci].0 } else { arr_str.len() };
                let mut end_ci = ci;
                while end_ci < n
                    && !(chars[end_ci].1 == '"' && (end_ci == 0 || chars[end_ci - 1].1 != '\\'))
                {
                    end_ci += 1;
                }
                let val_end = if end_ci < n {
                    chars[end_ci].0
                } else {
                    arr_str.len()
                };
                items.push(Value::String(arr_str[val_start..val_end].to_string()));
                ci = if end_ci < n { end_ci + 1 } else { n };
            } else if chars[ci].1 == '{' {
                let (inner, after_ci) = Self::gemma4_scan_nested(&chars, ci, '{', '}', n, arr_str);
                items.push(Self::gemma4_parse_args(inner));
                ci = after_ci;
            } else if chars[ci].1 == '[' {
                let (inner, after_ci) = Self::gemma4_scan_nested(&chars, ci, '[', ']', n, arr_str);
                items.push(Self::gemma4_parse_array(inner));
                ci = after_ci;
            } else {
                let val_start = chars[ci].0;
                while ci < n && !matches!(chars[ci].1, ',' | ']') {
                    ci += 1;
                }
                let val_end = if ci < n { chars[ci].0 } else { arr_str.len() };
                let val = arr_str[val_start..val_end].trim();
                if !val.is_empty() {
                    items.push(Self::gemma4_parse_bare_value(val));
                }
            }
        }

        Value::Array(items)
    }

    fn gemma4_parse_bare_value(val: &str) -> Value {
        let lower = val.to_ascii_lowercase();
        match lower.as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" | "none" | "nil" => Value::Null,
            _ => {
                if let Ok(n) = val.parse::<i64>() {
                    Value::Number(n.into())
                } else if let Ok(f) = val.parse::<f64>() {
                    serde_json::Number::from_f64(f)
                        .map(Value::Number)
                        .unwrap_or_else(|| Value::String(val.to_string()))
                } else {
                    Value::String(val.to_string())
                }
            }
        }
    }

    fn strip_tool_tags(&self, text: &str) -> String {
        let mut output = text.to_string();
        if !self.config.start_token_str.is_empty() {
            output = output.replace(&self.config.start_token_str, "");
        }
        if !self.config.end_token_str.is_empty() {
            output = output.replace(&self.config.end_token_str, "");
        }
        output
    }

    fn should_drop_bare_start_marker(&self) -> bool {
        self.buffer_started_from_special_token && !self.buffer_saw_non_marker_content
    }

    fn token_contains_non_marker_content(&self, token_text: &str) -> bool {
        let trimmed = token_text.trim();
        if trimmed.is_empty() {
            return false;
        }
        if !self.config.start_token_str.is_empty() && trimmed == self.config.start_token_str {
            return false;
        }
        if !self.config.end_token_str.is_empty() && trimmed == self.config.end_token_str {
            return false;
        }
        true
    }

    fn safe_partial_prefix_len(start_tag: &str) -> usize {
        if let Some(idx) = start_tag.find('_') {
            // E.g. "<tool_call>" => require at least "<tool"
            return idx.max(2);
        }
        // Default minimum for tags without underscore.
        start_tag.find('>').map_or(6, |idx| idx).clamp(2, 6)
    }

    fn escape_marker_for_display(marker: &str) -> String {
        if let Some(rest) = marker.strip_prefix('<') {
            format!("<\u{200C}{}", rest)
        } else {
            format!("{}\u{200C}", marker)
        }
    }

    fn contains_partial_marker_fragment(text: &str, marker: &str) -> bool {
        if marker.is_empty()
            || !Self::should_escape_marker_for_display(marker)
            || !marker.starts_with('<')
        {
            return false;
        }

        let marker_len = marker.len();
        if marker_len < 4 {
            return false;
        }

        let min_prefix_len =
            Self::safe_partial_prefix_len(marker).min(marker_len.saturating_sub(1));
        if min_prefix_len >= marker_len {
            return false;
        }

        (min_prefix_len..marker_len).rev().any(|len| {
            let prefix = &marker[..len];
            text.contains(prefix)
        })
    }

    fn escape_partial_marker_fragments(text: &str, marker: &str) -> String {
        if marker.is_empty()
            || !Self::should_escape_marker_for_display(marker)
            || !marker.starts_with('<')
        {
            return text.to_string();
        }

        let marker_len = marker.len();
        if marker_len < 4 {
            return text.to_string();
        }

        let min_prefix_len =
            Self::safe_partial_prefix_len(marker).min(marker_len.saturating_sub(1));
        if min_prefix_len >= marker_len {
            return text.to_string();
        }

        let mut out = text.to_string();
        for len in (min_prefix_len..marker_len).rev() {
            let prefix = &marker[..len];
            out = out.replace(prefix, &Self::escape_marker_for_display(prefix));
        }
        out
    }

    fn should_escape_marker_for_display(marker: &str) -> bool {
        if marker.is_empty() || marker.len() < 3 {
            return false;
        }
        let Some(first) = marker.chars().next() else {
            return false;
        };
        matches!(first, '<' | '[' | '{' | '(') || marker.contains('|')
    }

    fn display_escape_markers(&self) -> Vec<String> {
        let mut markers = Vec::new();
        for marker in [&self.config.start_token_str, &self.config.end_token_str] {
            if Self::should_escape_marker_for_display(marker) {
                markers.push(marker.to_string());
            }
        }
        // XML-style nested tool markers commonly appear in model-specific tool payloads.
        if self.uses_minimax_xml() {
            markers.extend(
                [
                    "<invoke name=",
                    "</invoke>",
                    "<parameter name=",
                    "</parameter>",
                ]
                .into_iter()
                .map(|s| s.to_string()),
            );
        } else if self.config.start_token_str.contains("tool_call")
            && self.config.end_token_str.contains("tool_call")
        {
            let is_glm = self.uses_glm_xml();
            if is_glm {
                markers.extend(
                    ["<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"]
                        .into_iter()
                        .map(|s| s.to_string()),
                );
            } else {
                markers.extend(
                    ["<function=", "</function>", "<parameter=", "</parameter>"]
                        .into_iter()
                        .map(|s| s.to_string()),
                );
            }
        }
        markers
    }

    fn recover_streaming_arguments_from_buffer(&mut self) {
        let uses_minimax_xml = self.uses_minimax_xml();
        if self.streaming_calls.is_empty()
            || !self
                .buffer
                .contains(Self::xml_parameter_open_prefix(uses_minimax_xml))
        {
            return;
        }
        let buffer = self.buffer.clone();

        for state in &mut self.streaming_calls {
            let Some(name) = state.name.as_deref() else {
                continue;
            };

            let recovered =
                Self::extract_xml_parameters_for_function(&buffer, name, uses_minimax_xml);
            if recovered.is_empty() {
                continue;
            }

            let mut args_obj = match serde_json::from_str::<Value>(state.arguments.trim()) {
                Ok(Value::Object(map)) => map,
                _ => Map::new(),
            };

            let mut merged_any = false;
            for (key, value) in recovered {
                if !args_obj.contains_key(&key) {
                    args_obj.insert(key, value);
                    merged_any = true;
                }
            }

            if merged_any {
                state.arguments = Value::Object(args_obj).to_string();
                tracing::warn!("Recovered missing parameter(s) from buffered tool-call content");
            }
        }
    }

    fn extract_xml_parameters_for_function(
        buffer: &str,
        function_name: &str,
        uses_minimax_xml: bool,
    ) -> std::collections::HashMap<String, Value> {
        let mut recovered = std::collections::HashMap::new();
        let func_start = if uses_minimax_xml {
            let function_tag = format!(r#"<invoke name="{function_name}">"#);
            let alt_function_tag = format!(r#"<invoke name='{function_name}'>"#);
            buffer
                .rfind(&function_tag)
                .or_else(|| buffer.rfind(&alt_function_tag))
        } else {
            let function_tag = format!("<function={}>", function_name);
            let alt_function_tag = format!("<function=\"{}\">", function_name);
            buffer
                .rfind(&function_tag)
                .or_else(|| buffer.rfind(&alt_function_tag))
        };

        let Some(func_start) = func_start else {
            return recovered;
        };

        let section = &buffer[func_start..];
        let mut cursor = 0usize;
        const PARAM_END: &str = "</parameter>";
        let param_prefix = Self::xml_parameter_open_prefix(uses_minimax_xml);

        while let Some(rel) = section[cursor..].find(param_prefix) {
            let tag_start = cursor + rel;
            let name_start = tag_start + param_prefix.len();
            let Some(name_end_rel) = section[name_start..].find('>') else {
                break;
            };
            let name_end = name_start + name_end_rel;
            let parameter_name = section[name_start..name_end]
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
            if parameter_name.is_empty() {
                break;
            }

            let value_start = name_end + 1;
            if value_start > section.len() {
                break;
            }

            if let Some(value_end_rel) = section[value_start..].find(PARAM_END) {
                let value_end = value_start + value_end_rel;
                let value = section[value_start..value_end]
                    .trim_matches(|c| c == '\n' || c == '\r')
                    .trim()
                    .to_string();
                recovered.insert(parameter_name, Self::parse_recovered_xml_value(&value));
                cursor = value_end + PARAM_END.len();
            } else {
                let value = section[value_start..]
                    .trim_matches(|c| c == '\n' || c == '\r')
                    .trim()
                    .to_string();
                recovered.insert(parameter_name, Self::parse_recovered_xml_value(&value));
                break;
            }
        }

        recovered
    }

    fn uses_minimax_xml(&self) -> bool {
        self.config.start_token_str == "<minimax:tool_call>"
            && self.config.end_token_str == "</minimax:tool_call>"
    }

    fn uses_glm_xml(&self) -> bool {
        let id = self.model_id.to_ascii_lowercase();
        id.contains("glm")
            && !self.uses_minimax_xml()
            && self.config.start_token_str == "<tool_call>"
            && self.config.end_token_str == "</tool_call>"
    }

    fn xml_parameter_open_prefix(uses_minimax_xml: bool) -> &'static str {
        if uses_minimax_xml {
            "<parameter name="
        } else {
            "<parameter="
        }
    }

    fn parse_recovered_xml_value(raw: &str) -> Value {
        let decoded = raw
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&quot;", "\"")
            .replace("&apos;", "'");

        match decoded.as_str() {
            "true" | "True" => return Value::Bool(true),
            "false" | "False" => return Value::Bool(false),
            "null" | "None" => return Value::Null,
            _ => {}
        }

        if decoded.starts_with('{') || decoded.starts_with('[') {
            if let Ok(value) = serde_json::from_str::<Value>(&decoded) {
                return value;
            }
        }

        if let Ok(num) = decoded.parse::<i64>() {
            return Value::Number(num.into());
        }

        if let Ok(num) = decoded.parse::<f64>() {
            if let Some(num) = serde_json::Number::from_f64(num) {
                return Value::Number(num);
            }
        }

        Value::String(decoded)
    }
}

fn repair_streamed_json_arguments(raw: &str) -> String {
    let mut repaired = raw.trim().to_string();
    if repaired.is_empty() {
        return "{}".to_string();
    }

    let mut in_string = false;
    let mut escaped = false;
    let mut stack: Vec<char> = Vec::new();

    for ch in repaired.chars() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' | '[' => stack.push(ch),
            '}' => {
                if stack.last() == Some(&'{') {
                    stack.pop();
                }
            }
            ']' => {
                if stack.last() == Some(&'[') {
                    stack.pop();
                }
            }
            _ => {}
        }
    }

    if in_string {
        repaired.push('"');
    }

    while repaired
        .chars()
        .last()
        .is_some_and(|c| c.is_whitespace() || c == ',')
    {
        repaired.pop();
    }

    while let Some(open) = stack.pop() {
        repaired.push(match open {
            '{' => '}',
            '[' => ']',
            _ => continue,
        });
    }

    repaired
}

#[derive(Debug, Clone, Default)]
struct StreamingToolCallState {
    name: Option<String>,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_config_qwen() {
        let config = ToolConfig::for_model_type(&ToolModelType::Qwen3);
        assert!(config.has_special_tokens());
        assert!(config.start_token_ids.contains(&151657));
        assert_eq!(config.start_token_str, "<tool_call>");
    }

    #[test]
    fn test_tool_config_default() {
        let config = ToolConfig::for_model_type(&ToolModelType::Phi);
        assert!(!config.has_special_tokens());
        assert_eq!(config.start_token_str, "<tool_call>");
    }

    #[test]
    fn test_tool_config_minimax() {
        let config = ToolConfig::for_model_type(&ToolModelType::MiniMax);
        assert!(config.has_special_tokens());
        assert!(config.start_token_ids.contains(&200052));
        assert!(config.end_token_ids.contains(&200053));
        assert_eq!(config.start_token_str, "<minimax:tool_call>");
        assert_eq!(config.end_token_str, "</minimax:tool_call>");
    }

    #[tokio::test]
    async fn test_minimax_parser_detects_start_and_end_by_token_id() {
        let tools = vec![crate::tools::function_tool("search_web", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::MiniMax,
            "MiniMax-M2.5".to_string(),
            ToolConfig::for_model_type(&ToolModelType::MiniMax),
            tools,
            None,
        );

        match parser.process_token(200052, "<minimax:tool_call>").await {
            StreamResult::Buffering => {}
            other => panic!("expected buffering on MiniMax start token, got {:?}", other),
        }

        parser.buffer = r#"<minimax:tool_call>
<invoke name="search_web">
<parameter name="query_tag">["technology","events"]</parameter>
<parameter name="query_list">["\"OpenAI\" \"latest\" \"release\""]</parameter>
</invoke>"#
            .to_string();
        parser.streaming_calls.push(StreamingToolCallState {
            name: Some("search_web".to_string()),
            arguments: r#"{"query_tag":["technology","events"],"query_list":["\"OpenAI\" \"latest\" \"release\""]}"#.to_string(),
        });

        match parser.process_token(200053, "</minimax:tool_call>").await {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "search_web");
            }
            other => panic!(
                "expected parsed tool calls on MiniMax end token, got {:?}",
                other
            ),
        }
    }

    #[tokio::test]
    async fn test_parser_normal_content() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );
        match parser.process_token(0, "Hello world").await {
            StreamResult::Content(s) => assert_eq!(s, "Hello world"),
            _ => panic!("Expected Content"),
        }
    }

    #[tokio::test]
    async fn test_parser_tool_call_detection() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        // Start tag triggers buffering
        match parser.process_token(151657, "<tool_call>").await {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on start tag"),
        }

        // Content is buffered
        match parser
            .process_token(0, r#"{"name": "test", "arguments": {}}"#)
            .await
        {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering"),
        }

        // End tag triggers parsing
        match parser.process_token(151658, "</tool_call>").await {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "test");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[tokio::test]
    async fn test_parser_partial_start_text_mode() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Phi,
            "phi".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Phi),
            tools,
            None,
        );

        // Partial start tag splits across tokens
        match parser.process_token(0, "<tool_").await {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on partial start"),
        }
        match parser.process_token(0, "call>").await {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on completed start"),
        }
        match parser
            .process_token(0, r#"{"name": "test", "arguments": {}}"#)
            .await
        {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering"),
        }
        match parser.process_token(0, "</tool_call>").await {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "test");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[tokio::test]
    async fn test_parser_token_id_strict_match() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        // Text match should not trigger when token IDs are available
        match parser.process_token(0, "<tool_call>").await {
            StreamResult::Content(text) => assert_eq!(text, "<tool_call>"),
            _ => panic!("Expected Content without token ID match"),
        }
    }

    #[tokio::test]
    async fn test_parser_keeps_buffering_when_args_include_code_fence() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        match parser.process_token(151657, "<tool_call>").await {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on start tag"),
        }

        // Code-fence-like content inside buffered arguments should not switch the
        // parser back to normal content mode.
        match parser.process_token(0, "\n```markdown\n").await {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering while inside tool call arguments"),
        }
    }

    #[test]
    fn test_repair_streamed_json_arguments_balances_only_structural_tokens() {
        let raw = r#"{"file_path":"/tmp/a.rs","new_string":"fn a() { let x = vec![1,2,3]; }","replace_all":false"#;
        let repaired = repair_streamed_json_arguments(raw);
        assert_ne!(repaired, raw);
        let parsed: Value = serde_json::from_str(&repaired).expect("repaired JSON should parse");
        assert_eq!(parsed["file_path"], "/tmp/a.rs");
        assert_eq!(parsed["new_string"], "fn a() { let x = vec![1,2,3]; }");
        assert_eq!(parsed["replace_all"], false);
    }

    #[tokio::test]
    async fn test_finalize_buffered_tool_calls_recovers_calls_on_eos() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call><function=Write>".to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: r#"{"file_path":"/tmp/a.rs","content":"abc""#.to_string(),
        }];

        let finalized = parser.finalize_buffered_tool_calls().await;
        match finalized {
            Some(BufferedFinalizeResult::ToolCalls(calls)) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "Write");
                let args = calls[0].function.arguments.as_ref().unwrap();
                let parsed: Value = serde_json::from_str(args).unwrap();
                assert_eq!(parsed["file_path"], "/tmp/a.rs");
                assert_eq!(parsed["content"], "abc");
            }
            other => panic!("Expected recovered tool calls, got {:?}", other),
        }

        assert!(matches!(parser.state, ParserState::Normal));
        assert!(parser.buffer.is_empty());
        assert!(parser.streaming_calls.is_empty());
    }

    #[tokio::test]
    async fn test_finalize_buffered_tool_calls_flushes_when_unrecoverable() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call><function=Write><parameter=content>".to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: None,
            arguments: String::new(),
        }];

        let finalized = parser.finalize_buffered_tool_calls().await;
        match finalized {
            Some(BufferedFinalizeResult::FlushBuffer(text)) => {
                assert_eq!(text, "<tool_call><function=Write><parameter=content>");
            }
            other => panic!("Expected FlushBuffer, got {:?}", other),
        }

        assert!(matches!(parser.state, ParserState::Normal));
        assert!(parser.buffer.is_empty());
        assert!(parser.streaming_calls.is_empty());
    }

    #[tokio::test]
    async fn test_fake_end_marker_inside_parameter_keeps_buffering() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call>\n<function=Write>\n<parameter=file_path>\n/tmp/a.md\n</parameter>\n<parameter=content>\n- Qwen format (`<tool_call>..."
            .to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: r#"{"file_path":"/tmp/a.md","content":"- Qwen format (`<tool_call>..."}"#
                .to_string(),
        }];

        let result = parser.process_token(151658, "</tool_call>").await;
        assert!(matches!(result, StreamResult::Buffering));
        assert!(matches!(parser.state, ParserState::Buffering));
        assert!(parser.pending_end_marker_candidate);
    }

    #[tokio::test]
    async fn test_finalize_rejects_incomplete_xml_envelope_even_if_args_parse() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call>\n<function=Write>\n<parameter=file_path>\n/tmp/a.md\n</parameter>\n<parameter=content>\n- Qwen format (`<tool_call>...</tool_call>"
            .to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: r#"{"file_path":"/tmp/a.md","content":"- Qwen format (`<tool_call>...</tool_call>"}"#.to_string(),
        }];

        let finalized = parser.finalize_buffered_tool_calls().await;
        match finalized {
            Some(BufferedFinalizeResult::FlushBuffer(text)) => {
                assert!(text.contains("<parameter=content>"));
                assert!(text.contains("</tool_call>"));
            }
            other => panic!("Expected FlushBuffer, got {:?}", other),
        }
    }

    #[test]
    fn test_envelope_accepts_stray_parameter_closer_after_function() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.buffer = r#"<tool_call>
<function=edit>
<parameter=filePath>
/root/candle-vllm/src/models/qwen3_5_moe.rs
</parameter>
<parameter=newString>
abc
</parameter>
<parameter=oldString>
def
</parameter>
</function>

</parameter>
</function>
</tool_call>"#
            .to_string();

        assert!(parser.has_complete_tool_envelope());
    }

    #[test]
    fn test_envelope_rejects_unclosed_parameter() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.buffer = r#"<tool_call>
<function=edit>
<parameter=filePath>
/root/candle-vllm/src/models/qwen3_5_moe.rs
</parameter>
<parameter=newString>
abc
</function>
</tool_call>"#
            .to_string();

        assert!(!parser.has_complete_tool_envelope());
    }

    #[test]
    fn test_envelope_glm47_xml_format() {
        let tools = vec![crate::tools::function_tool("read", "Read a file").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::GLM4MoeLite,
            "glm-4.7-flash".to_string(),
            ToolConfig::for_model_type(&ToolModelType::GLM4MoeLite),
            tools,
            None,
        );

        parser.buffer =
            "<tool_call>read<arg_key>filePath</arg_key><arg_value>/tmp/test.rs</arg_value></tool_call>"
                .to_string();
        assert!(parser.has_complete_tool_envelope());

        parser.buffer =
            "<tool_call>read<arg_key>filePath</arg_key><arg_value>/tmp/test.rs</arg_value>"
                .to_string();
        assert!(!parser.has_complete_tool_envelope());

        parser.buffer =
            "<tool_call>read<arg_key>filePath</arg_key><arg_value>/tmp/test.rs</arg_value></tool_call>"
                .to_string();
        assert!(parser.has_complete_tool_envelope());
    }

    #[test]
    fn test_glm47_display_escape_markers() {
        let tools = vec![crate::tools::function_tool("read", "Read a file").build()];
        let parser = StreamToolParser::new_with_config(
            &ToolModelType::GLM4MoeLite,
            "glm-4.7-flash".to_string(),
            ToolConfig::for_model_type(&ToolModelType::GLM4MoeLite),
            tools,
            None,
        );

        let markers = parser.display_escape_markers();
        assert!(markers.iter().any(|m| m == "<arg_key>"));
        assert!(markers.iter().any(|m| m == "</arg_key>"));
        assert!(markers.iter().any(|m| m == "<arg_value>"));
        assert!(markers.iter().any(|m| m == "</arg_value>"));
        assert!(markers.iter().any(|m| m == "<tool_call>"));
        assert!(markers.iter().any(|m| m == "</tool_call>"));
    }

    #[tokio::test]
    async fn test_nested_start_marker_is_ignored_while_buffering() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Phi,
            "phi".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Phi),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call><function=Write>".to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: String::new(),
        }];

        let result = parser.process_token(0, "<tool_call>").await;
        assert!(matches!(result, StreamResult::Buffering));
        assert!(matches!(parser.state, ParserState::Buffering));
    }

    #[tokio::test]
    async fn test_false_end_marker_inside_arguments_requires_confirmation() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Phi,
            "phi".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Phi),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call>".to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: r#"{"file_path":"/tmp/a.rs","content":"text with "#.to_string(),
        }];

        let first = parser.process_token(0, "</tool_call>").await;
        assert!(matches!(first, StreamResult::Buffering));
        assert!(matches!(parser.state, ParserState::Buffering));
        assert!(parser.pending_end_marker_candidate);
    }

    #[tokio::test]
    async fn test_finalize_recovers_unclosed_xml_parameter_content() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call>\n<function=Write>\n<parameter=file_path>\n/tmp/a.md\n</parameter>\n<parameter=content>\n# Title\n".to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: r#"{"file_path":"/tmp/a.md"}"#.to_string(),
        }];

        let finalized = parser.finalize_buffered_tool_calls().await;
        match finalized {
            Some(BufferedFinalizeResult::ToolCalls(calls)) => {
                assert_eq!(calls.len(), 1);
                let args = calls[0].function.arguments.as_ref().unwrap();
                let parsed: Value = serde_json::from_str(args).unwrap();
                assert_eq!(parsed["file_path"], "/tmp/a.md");
                assert_eq!(parsed["content"], "# Title");
            }
            other => panic!("Expected recovered tool calls, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_finalize_recovers_qwen3_json_missing_end_tag() {
        let tools = vec![crate::tools::function_tool("get_weather", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}"#
            .to_string();
        parser.buffer_had_parse_activity = true;
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("get_weather".to_string()),
            arguments: r#"{"location": "NYC"}"#.to_string(),
        }];

        let finalized = parser.finalize_buffered_tool_calls().await;
        match finalized {
            Some(BufferedFinalizeResult::ToolCalls(calls)) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "get_weather");
                let args = calls[0].function.arguments.as_ref().unwrap();
                let parsed: Value = serde_json::from_str(args).unwrap();
                assert_eq!(parsed["location"], "NYC");
            }
            other => panic!(
                "Expected recovered tool calls when </tool_call> is missing, got {:?}",
                other
            ),
        }
    }

    #[tokio::test]
    async fn test_finalize_recovers_qwen3_json_missing_outer_brace_and_end_tag() {
        let tools = vec![crate::tools::function_tool("get_weather", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}"#
            .to_string();
        parser.buffer_had_parse_activity = true;
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("get_weather".to_string()),
            arguments: r#"{"location": "NYC"}"#.to_string(),
        }];

        let finalized = parser.finalize_buffered_tool_calls().await;
        match finalized {
            Some(BufferedFinalizeResult::ToolCalls(calls)) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "get_weather");
                let args = calls[0].function.arguments.as_ref().unwrap();
                let parsed: Value = serde_json::from_str(args).unwrap();
                assert_eq!(parsed["location"], "NYC");
            }
            other => panic!(
                "Expected recovered tool calls when outer }} and </tool_call> are missing, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_sanitize_tool_markup_for_display_escapes_xml_tool_payload() {
        let tools = vec![crate::tools::function_tool("write", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        let raw = "<tool_call><function=write><parameter=filePath>/tmp/a.md</parameter></function></tool_call>";
        assert!(parser.contains_tool_markup(raw));

        let safe = parser.sanitize_tool_markup_for_display(raw);
        assert!(safe.contains("<\u{200C}tool_call>"));
        assert!(safe.contains("<\u{200C}function=write>"));
        assert!(safe.contains("<\u{200C}parameter=filePath>"));
        assert!(!parser.contains_tool_markup(&safe));
    }

    #[test]
    fn test_sanitize_tool_markup_for_display_keeps_non_xml_models_simple() {
        let tools = vec![crate::tools::function_tool("write", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ToolModelType::Mistral,
            "mistral".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Mistral),
            tools,
            None,
        );

        let raw = "[TOOL_CALLS]";
        assert!(parser.contains_tool_markup(raw));
        let safe = parser.sanitize_tool_markup_for_display(raw);
        assert_eq!(safe, "[TOOL_CALLS]\u{200C}");
    }

    #[test]
    fn test_sanitize_tool_markup_for_display_escapes_minimax_xml_payload() {
        let tools = vec![crate::tools::function_tool("search_web", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ToolModelType::MiniMax,
            "MiniMax-M2.5".to_string(),
            ToolConfig::for_model_type(&ToolModelType::MiniMax),
            tools,
            None,
        );

        let raw = r#"<minimax:tool_call><invoke name="search_web"><parameter name="query_list">["rust"]</parameter></invoke></minimax:tool_call>"#;
        assert!(parser.contains_tool_markup(raw));

        let safe = parser.sanitize_tool_markup_for_display(raw);
        assert!(safe.contains("<\u{200C}minimax:tool_call>"));
        assert!(safe.contains("<\u{200C}invoke name=\"search_web\">"));
        assert!(safe.contains("<\u{200C}parameter name=\"query_list\">"));
        assert!(!parser.contains_tool_markup(&safe));
    }

    #[test]
    fn test_contains_tool_markup_detects_partial_xml_marker() {
        let tools = vec![crate::tools::function_tool("write", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        assert!(parser.contains_tool_markup("example <tool_ca"));
    }

    #[test]
    fn test_sanitize_tool_markup_for_display_escapes_partial_xml_marker() {
        let tools = vec![crate::tools::function_tool("write", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        let raw = "example <tool_ca";
        let safe = parser.sanitize_tool_markup_for_display(raw);
        assert!(safe.contains("<\u{200C}tool_ca"));
        assert!(!parser.contains_tool_markup(&safe));
    }

    #[test]
    fn test_parser_defaults_to_qwen_coder_for_qwen35() {
        assert_eq!(
            StreamToolParser::parser_name_for_model(&ToolModelType::Qwen3_5, "qwen3.5-instruct"),
            "qwen_coder"
        );
    }

    #[test]
    fn test_parser_defaults_to_qwen_coder_for_qwen35_moe() {
        assert_eq!(
            StreamToolParser::parser_name_for_model(&ToolModelType::Qwen3_5MoE, "qwen3.5-moe"),
            "qwen_coder"
        );
    }

    #[test]
    fn test_parser_defaults_to_minimax_parser() {
        assert_eq!(
            StreamToolParser::parser_name_for_model(&ToolModelType::MiniMax, "MiniMax-M2.5-NVFP4"),
            "minimax_m2"
        );
    }

    #[test]
    fn test_minimax_envelope_accepts_complete_invoke_block() {
        let tools = vec![crate::tools::function_tool("search_web", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::MiniMax,
            "MiniMax-M2.5".to_string(),
            ToolConfig::for_model_type(&ToolModelType::MiniMax),
            tools,
            None,
        );

        parser.buffer = r#"<minimax:tool_call>
<invoke name="search_web">
<parameter name="query_tag">["technology", "events"]</parameter>
<parameter name="query_list">["\"OpenAI\" \"latest\" \"release\""]</parameter>
</invoke>
</minimax:tool_call>"#
            .to_string();

        assert!(parser.has_complete_tool_envelope());
    }

    // ---------------------------------------------------------------
    // Reasoning marker isolation during tool-call buffering
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_reasoning_markers_inside_tool_args_do_not_corrupt_state() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        // Model outputs <think> before tool call
        assert!(matches!(
            parser.process_token(0, "<think>").await,
            StreamResult::Content(_)
        ));
        assert!(parser.in_reasoning());

        // Reasoning ends
        assert!(matches!(
            parser.process_token(0, "some thought</think>").await,
            StreamResult::Content(_)
        ));
        assert!(!parser.in_reasoning());

        // Now tool call starts
        assert!(matches!(
            parser.process_token(151657, "<tool_call>").await,
            StreamResult::Buffering
        ));

        // Tool arguments contain "</think>" as a string value - must NOT flip reasoning state
        assert!(matches!(
            parser
                .process_token(
                    0,
                    r#"{"name": "test", "arguments": {"text": "<think>inside</think>"}}"#
                )
                .await,
            StreamResult::Buffering
        ));
        assert!(
            !parser.in_reasoning(),
            "Reasoning state must not be corrupted by markers inside tool-call buffer"
        );

        // End tool call
        match parser.process_token(151658, "</tool_call>").await {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "test");
            }
            other => panic!("Expected ToolCalls, got {:?}", other),
        }

        // After exiting buffering, reasoning state must be correct
        assert!(
            !parser.in_reasoning(),
            "Reasoning state must be clean after tool call completes"
        );
    }

    #[tokio::test]
    async fn test_reasoning_state_resyncs_after_buffering_exit() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        // Open reasoning block
        assert!(matches!(
            parser.process_token(0, "<think>").await,
            StreamResult::Content(_)
        ));
        assert!(parser.in_reasoning());

        // Close reasoning
        assert!(matches!(
            parser.process_token(0, "thought</think>\n").await,
            StreamResult::Content(_)
        ));
        assert!(!parser.in_reasoning());

        // Tool call with <think> in arguments
        assert!(matches!(
            parser.process_token(151657, "<tool_call>").await,
            StreamResult::Buffering
        ));
        assert!(matches!(
            parser
                .process_token(0, r#"{"name": "test", "arguments": {"q": "<think>"}}"#)
                .await,
            StreamResult::Buffering
        ));
        match parser.process_token(151658, "</tool_call>").await {
            StreamResult::ToolCalls(_) => {}
            other => panic!("Expected ToolCalls, got {:?}", other),
        }

        // After tool call, reasoning state must reflect the full accumulated
        // output: <think>thought</think>\n<tool_call>...<think>...</tool_call>
        // The <think> inside the tool call is part of JSON, not a real marker.
        // The resync should see the original <think>...</think> pair as balanced.
        assert!(
            !parser.in_reasoning(),
            "Resync must recognize balanced reasoning markers in accumulated output"
        );
    }

    #[tokio::test]
    async fn test_tool_call_suppressed_during_active_reasoning() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        // Open reasoning
        assert!(matches!(
            parser.process_token(0, "<think>").await,
            StreamResult::Content(_)
        ));
        assert!(parser.in_reasoning());

        // Tool call start token during reasoning should be treated as content
        match parser.process_token(151657, "<tool_call>").await {
            StreamResult::Content(text) => {
                assert_eq!(text, "<tool_call>");
            }
            other => panic!(
                "Expected Content (tool suppressed during reasoning), got {:?}",
                other
            ),
        }
        assert_eq!(
            parser.state(),
            &ParserState::Normal,
            "Parser must stay in Normal when tool start is suppressed during reasoning"
        );
    }

    #[tokio::test]
    async fn test_code_block_state_not_corrupted_by_tool_buffer() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        // Not in a code block initially
        assert!(!parser.in_code_block());

        // Start tool call
        assert!(matches!(
            parser.process_token(151657, "<tool_call>").await,
            StreamResult::Buffering
        ));

        // Tool arguments contain triple backticks
        assert!(matches!(
            parser
                .process_token(
                    0,
                    r#"{"name": "test", "arguments": {"code": "```rust\nfn main() {}\n```"}}"#
                )
                .await,
            StreamResult::Buffering
        ));

        // Code block state must not be affected by content inside the tool buffer
        assert!(
            !parser.in_code_block(),
            "Code block state must not be corrupted by backticks inside tool-call buffer"
        );

        // End tool call
        match parser.process_token(151658, "</tool_call>").await {
            StreamResult::ToolCalls(_) => {}
            other => panic!("Expected ToolCalls, got {:?}", other),
        }

        // After resync, code block state should still be false
        assert!(
            !parser.in_code_block(),
            "Code block state must be clean after tool call with backticks in args"
        );
    }

    #[tokio::test]
    async fn test_prefilled_reasoning_end_marker_suppresses_tool_detection() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        // Simulate prompt that ends with <think> (prefilled reasoning)
        parser.set_initial_reasoning_end_marker(Some("</think>".to_string()));
        assert!(parser.in_reasoning());

        // Tool call start during active reasoning should be suppressed
        match parser.process_token(151657, "<tool_call>").await {
            StreamResult::Content(text) => {
                assert_eq!(text, "<tool_call>");
            }
            other => panic!(
                "Expected Content (tool suppressed during prefilled reasoning), got {:?}",
                other
            ),
        }

        // Close reasoning
        assert!(matches!(
            parser.process_token(0, "</think>\n").await,
            StreamResult::Content(_)
        ));
        assert!(!parser.in_reasoning());

        // Now tool call should work
        assert!(matches!(
            parser.process_token(151657, "<tool_call>").await,
            StreamResult::Buffering
        ));
    }

    #[tokio::test]
    async fn test_multiple_tool_calls_with_reasoning_between() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        // First tool call
        assert!(matches!(
            parser.process_token(151657, "<tool_call>").await,
            StreamResult::Buffering
        ));
        assert!(matches!(
            parser
                .process_token(0, r#"{"name": "test", "arguments": {}}"#)
                .await,
            StreamResult::Buffering
        ));
        match parser.process_token(151658, "</tool_call>").await {
            StreamResult::ToolCalls(calls) => assert_eq!(calls.len(), 1),
            other => panic!("Expected ToolCalls, got {:?}", other),
        }

        // Reasoning block between tool calls
        assert!(matches!(
            parser.process_token(0, "\n<think>").await,
            StreamResult::Content(_)
        ));
        assert!(parser.in_reasoning());
        assert!(matches!(
            parser
                .process_token(0, "planning next step</think>\n")
                .await,
            StreamResult::Content(_)
        ));
        assert!(!parser.in_reasoning());

        // Second tool call should work
        assert!(matches!(
            parser.process_token(151657, "<tool_call>").await,
            StreamResult::Buffering
        ));
        assert!(matches!(
            parser
                .process_token(0, r#"{"name": "test", "arguments": {}}"#)
                .await,
            StreamResult::Buffering
        ));
        match parser.process_token(151658, "</tool_call>").await {
            StreamResult::ToolCalls(calls) => assert_eq!(calls.len(), 1),
            other => panic!("Expected ToolCalls, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_finalize_with_reasoning_markers_in_buffer() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer =
            r#"<tool_call>{"name": "Write", "arguments": {"text": "<think>test</think>"}}"#
                .to_string();
        parser.accumulated_output = parser.buffer.clone();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: r#"{"text": "<think>test</think>"}"#.to_string(),
        }];

        let finalized = parser.finalize_buffered_tool_calls().await;
        match finalized {
            Some(BufferedFinalizeResult::ToolCalls(calls)) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "Write");
            }
            other => panic!("Expected ToolCalls, got {:?}", other),
        }

        assert!(
            !parser.in_reasoning(),
            "After finalize, reasoning state must be resynced and clean"
        );
    }

    // ---------------------------------------------------------------
    // Resync correctness tests
    // ---------------------------------------------------------------

    #[test]
    fn test_resync_reasoning_balanced_markers() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.accumulated_output = "<think>some thought</think>\nHello".to_string();
        parser.resync_reasoning_and_code_block_state();
        assert!(
            !parser.in_reasoning(),
            "Balanced <think>...</think> should not leave reasoning open"
        );
    }

    #[test]
    fn test_resync_reasoning_unbalanced_open() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.accumulated_output = "<think>still thinking...".to_string();
        parser.resync_reasoning_and_code_block_state();
        assert!(
            parser.in_reasoning(),
            "Unbalanced <think> without </think> should leave reasoning open"
        );
    }

    #[test]
    fn test_resync_code_block_state() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.accumulated_output = "text\n```rust\nfn main() {}\n```\nmore text".to_string();
        parser.resync_reasoning_and_code_block_state();
        assert!(
            !parser.in_code_block(),
            "Balanced code fences should not leave code block open"
        );

        parser.accumulated_output = "text\n```rust\nfn main() {}".to_string();
        parser.resync_reasoning_and_code_block_state();
        assert!(
            parser.in_code_block(),
            "Unbalanced code fence should leave code block open"
        );
    }

    #[test]
    fn test_resync_multiple_reasoning_blocks() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        parser.accumulated_output =
            "<think>first</think>\ntext\n<think>second</think>\nmore".to_string();
        parser.resync_reasoning_and_code_block_state();
        assert!(
            !parser.in_reasoning(),
            "Two balanced reasoning blocks should not leave reasoning open"
        );

        parser.accumulated_output = "<think>first</think>\ntext\n<think>still open".to_string();
        parser.resync_reasoning_and_code_block_state();
        assert!(
            parser.in_reasoning(),
            "Second unbalanced reasoning block should leave reasoning open"
        );
    }

    // ---------------------------------------------------------------
    // Detect prefilled reasoning end marker
    // ---------------------------------------------------------------

    #[test]
    fn test_detect_prefilled_reasoning_end_marker_think() {
        let prompt = "...<|im_start|>assistant\n<think>";
        assert_eq!(
            detect_prefilled_reasoning_end_marker(prompt),
            Some("</think>".to_string())
        );
    }

    #[test]
    fn test_detect_prefilled_reasoning_end_marker_qwen() {
        let prompt = "...<|im_start|>assistant\n<|think|>";
        assert_eq!(
            detect_prefilled_reasoning_end_marker(prompt),
            Some("<|/think|>".to_string())
        );
    }

    #[test]
    fn test_detect_prefilled_reasoning_end_marker_none() {
        let prompt = "...<|im_start|>assistant\n";
        assert_eq!(detect_prefilled_reasoning_end_marker(prompt), None);
    }

    #[test]
    fn test_detect_prefilled_reasoning_end_marker_trailing_whitespace() {
        let prompt = "...<|im_start|>assistant\n<think>  \n";
        assert_eq!(
            detect_prefilled_reasoning_end_marker(prompt),
            Some("</think>".to_string())
        );
    }

    // ---------------------------------------------------------------
    // mask_tool_envelopes
    // ---------------------------------------------------------------

    #[test]
    fn test_mask_tool_envelopes_basic() {
        let text = "before<tool_call>{\"think\": \"</think>\"}</tool_call>after";
        let masked = StreamToolParser::mask_tool_envelopes(text, "<tool_call>", "</tool_call>");
        assert!(!masked.contains("</think>"));
        assert!(masked.contains("before"));
        assert!(masked.contains("after"));
        assert!(masked.contains("<tool_call>"));
        assert!(masked.contains("</tool_call>"));
    }

    #[test]
    fn test_mask_tool_envelopes_multiple() {
        let text = "<tool_call>first</tool_call>middle<tool_call>second</tool_call>";
        let masked = StreamToolParser::mask_tool_envelopes(text, "<tool_call>", "</tool_call>");
        assert!(!masked.contains("first"));
        assert!(!masked.contains("second"));
        assert!(masked.contains("middle"));
    }

    #[test]
    fn test_mask_tool_envelopes_no_tags() {
        let text = "no tool calls here <think>reasoning</think>";
        let masked = StreamToolParser::mask_tool_envelopes(text, "<tool_call>", "</tool_call>");
        assert_eq!(masked, text);
    }

    #[test]
    fn test_mask_tool_envelopes_unclosed() {
        let text = "<tool_call>unclosed content with </think>";
        let masked = StreamToolParser::mask_tool_envelopes(text, "<tool_call>", "</tool_call>");
        assert_eq!(masked, text, "Unclosed envelope should not be masked");
    }

    #[test]
    fn test_mask_tool_envelopes_empty_tags() {
        let text = "content with </think> markers";
        let masked = StreamToolParser::mask_tool_envelopes(text, "", "</tool_call>");
        assert_eq!(masked, text, "Empty start tag should return text unchanged");
    }

    // ---------------------------------------------------------------
    // End-to-end: reasoning → tool → reasoning → tool
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_full_agentic_loop_reasoning_tool_interleave() {
        let tools = vec![crate::tools::function_tool("search", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        // Prefilled reasoning from prompt
        parser.set_initial_reasoning_end_marker(Some("</think>".to_string()));
        assert!(parser.in_reasoning());

        // Model generates reasoning content
        assert!(matches!(
            parser.process_token(0, "Let me think about this...").await,
            StreamResult::Content(_)
        ));
        assert!(parser.in_reasoning());

        // Model closes reasoning
        assert!(matches!(
            parser.process_token(0, "</think>\n").await,
            StreamResult::Content(_)
        ));
        assert!(!parser.in_reasoning());

        // Model generates first tool call
        assert!(matches!(
            parser.process_token(151657, "<tool_call>").await,
            StreamResult::Buffering
        ));
        assert!(matches!(
            parser
                .process_token(0, r#"{"name": "search", "arguments": {"q": "test"}}"#)
                .await,
            StreamResult::Buffering
        ));
        match parser.process_token(151658, "</tool_call>").await {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "search");
            }
            other => panic!("Expected ToolCalls, got {:?}", other),
        }
        assert!(!parser.in_reasoning());

        // Model generates text after tool call
        assert!(matches!(
            parser
                .process_token(0, "\nBased on the search results")
                .await,
            StreamResult::Content(_)
        ));
        assert!(!parser.in_reasoning());
    }

    #[tokio::test]
    async fn test_tool_call_with_think_marker_in_json_string_value() {
        let tools = vec![crate::tools::function_tool("write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        // Tool call whose argument value contains a think marker
        assert!(matches!(
            parser.process_token(151657, "<tool_call>").await,
            StreamResult::Buffering
        ));

        // JSON with </think> embedded in a string value
        let json_with_think = r#"{"name": "write", "arguments": {"content": "The model uses <think> tags for reasoning and </think> to close them."}}"#;
        assert!(matches!(
            parser.process_token(0, json_with_think).await,
            StreamResult::Buffering
        ));

        match parser.process_token(151658, "</tool_call>").await {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "write");
            }
            other => panic!("Expected ToolCalls, got {:?}", other),
        }

        // After the tool call, reasoning state must be clean
        assert!(
            !parser.in_reasoning(),
            "Think markers in tool call JSON must not corrupt reasoning state"
        );

        // Subsequent content should flow normally
        match parser.process_token(0, "\nDone!").await {
            StreamResult::Content(text) => assert_eq!(text, "\nDone!"),
            other => panic!("Expected Content, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_incremental_code_block_tracking_in_normal_mode() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Qwen3),
            tools,
            None,
        );

        // Open code block
        assert!(matches!(
            parser.process_token(0, "```python\n").await,
            StreamResult::Content(_)
        ));
        assert!(parser.in_code_block());

        // Tool start inside code block should be suppressed
        match parser.process_token(151657, "<tool_call>").await {
            StreamResult::Content(text) => {
                assert_eq!(text, "<tool_call>");
            }
            other => panic!(
                "Expected Content (tool suppressed in code block), got {:?}",
                other
            ),
        }

        // Close code block
        assert!(matches!(
            parser.process_token(0, "\n```\n").await,
            StreamResult::Content(_)
        ));
        assert!(!parser.in_code_block());

        // Now tool call should work
        assert!(matches!(
            parser.process_token(151657, "<tool_call>").await,
            StreamResult::Buffering
        ));
    }

    #[tokio::test]
    async fn test_text_mode_tool_call_with_reasoning_markers_in_args() {
        let tools = vec![crate::tools::function_tool("edit", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Phi,
            "phi".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Phi),
            tools,
            None,
        );

        // Text-mode tool call (no special token IDs)
        assert!(matches!(
            parser.process_token(0, "<tool_call>").await,
            StreamResult::Buffering
        ));

        // Arguments contain multiple reasoning marker types
        let args = r#"{"name": "edit", "arguments": {"old": "<think>old</think>", "new": "<|think|>new<|/think|>"}}"#;
        assert!(matches!(
            parser.process_token(0, args).await,
            StreamResult::Buffering
        ));

        match parser.process_token(0, "</tool_call>").await {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "edit");
            }
            other => panic!("Expected ToolCalls, got {:?}", other),
        }

        assert!(
            !parser.in_reasoning(),
            "Multiple reasoning marker types in tool args must not corrupt state"
        );
    }

    // ---------------------------------------------------------------
    // strip_reasoning_blocks
    // ---------------------------------------------------------------

    #[test]
    fn test_strip_reasoning_blocks_basic() {
        let text = "<think>some reasoning</think>\nHello world";
        assert_eq!(strip_reasoning_blocks(text), "\nHello world");
    }

    #[test]
    fn test_strip_reasoning_blocks_multiple() {
        let text = "<think>first</think>\nmiddle\n<think>second</think>\nend";
        assert_eq!(strip_reasoning_blocks(text), "\nmiddle\n\nend");
    }

    #[test]
    fn test_strip_reasoning_blocks_unmatched_open() {
        let text = "before\n<think>unclosed reasoning and tool call";
        assert_eq!(
            strip_reasoning_blocks(text),
            "before\n",
            "Unmatched opening marker should truncate from that point"
        );
    }

    #[test]
    fn test_strip_reasoning_blocks_with_tool_call() {
        let text = "<think>reasoning</think>\n<tool_call>{\"name\": \"test\"}</tool_call>";
        let stripped = strip_reasoning_blocks(text);
        assert!(stripped.contains("<tool_call>"));
        assert!(!stripped.contains("<think>"));
    }

    #[test]
    fn test_strip_reasoning_blocks_empty_think() {
        let text = "<think>\n</think>\n<tool_call>{\"name\": \"test\"}</tool_call>";
        let stripped = strip_reasoning_blocks(text);
        assert!(stripped.contains("<tool_call>"));
        assert!(!stripped.contains("<think>"));
    }

    #[test]
    fn test_strip_reasoning_blocks_no_markers() {
        let text = "Hello world, no reasoning here";
        assert_eq!(strip_reasoning_blocks(text), text);
    }

    #[test]
    fn test_strip_reasoning_blocks_qwen_markers() {
        let text = "<|think|>reasoning<|/think|>\nHello";
        assert_eq!(strip_reasoning_blocks(text), "\nHello");
    }

    #[test]
    fn test_strip_reasoning_blocks_double_think_with_tool() {
        let text = "<think>\n\n</think>\n\n<think>\n</think>\n\n<tool_call>{\"name\": \"test\"}</tool_call>";
        let stripped = strip_reasoning_blocks(text);
        assert!(
            stripped.contains("<tool_call>"),
            "Tool call should survive after stripping double-think pattern. Got: {}",
            stripped
        );
    }

    // ---------------------------------------------------------------
    // Manual MiniMax XML parser tests
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_minimax_xml_tool_calls_manual() {
        let text = r#"<minimax:tool_call>
<invoke name="write">
<parameter name="content"># Test Content
Some markdown text here.</parameter>
<parameter name="filePath">/root/test.md</parameter>
</invoke>
</minimax:tool_call>"#;

        let calls = parse_minimax_xml_tool_calls(text, &[]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write");
        assert!(calls[0]
            .function
            .arguments
            .as_deref()
            .unwrap_or("")
            .contains("content"));
        assert!(calls[0]
            .function
            .arguments
            .as_deref()
            .unwrap_or("")
            .contains("filePath"));
    }

    #[test]
    fn test_parse_minimax_xml_without_closing_tag() {
        let text = r#"<minimax:tool_call>
<invoke name="read">
<parameter name="filePath">/root/AGENTS.md</parameter>
</invoke>"#;

        let calls = parse_minimax_xml_tool_calls(text, &[]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert!(calls[0]
            .function
            .arguments
            .as_deref()
            .unwrap_or("")
            .contains("filePath"));
    }

    #[test]
    fn test_parse_minimax_xml_with_array_value() {
        let text = r#"<invoke name="search">
<parameter name="tags">["rust", "programming"]</parameter>
</invoke>"#;

        let calls = parse_minimax_xml_tool_calls(text, &[]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
        let args: Value =
            serde_json::from_str(calls[0].function.arguments.as_deref().unwrap_or("{}")).unwrap();
        assert!(args["tags"].is_array());
    }

    #[test]
    fn test_parse_minimax_xml_multiple_invokes() {
        let text = r#"<minimax:tool_call>
<invoke name="read">
<parameter name="filePath">/root/file1.md</parameter>
</invoke>
<invoke name="write">
<parameter name="filePath">/root/file2.md</parameter>
<parameter name="content">Hello</parameter>
</invoke>
</minimax:tool_call>"#;

        let calls = parse_minimax_xml_tool_calls(text, &[]);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[1].function.name, "write");
    }

    #[test]
    fn test_parse_minimax_xml_type_coercion_with_schema() {
        let tools = vec![crate::tools::function_tool("get_weather", "Get weather")
            .parameters_schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "verbose": {"type": "boolean"},
                    "temp_unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }))
            .build()];

        let text = r#"<invoke name="get_weather">
<parameter name="city">London</parameter>
<parameter name="days">5</parameter>
<parameter name="verbose">true</parameter>
<parameter name="temp_unit">celsius</parameter>
</invoke>"#;

        let calls = parse_minimax_xml_tool_calls(text, &tools);
        assert_eq!(calls.len(), 1);
        let args: Value =
            serde_json::from_str(calls[0].function.arguments.as_deref().unwrap()).unwrap();
        assert_eq!(args["city"], Value::String("London".to_string()));
        assert_eq!(args["days"], serde_json::json!(5));
        assert_eq!(args["verbose"], Value::Bool(true));
        assert_eq!(args["temp_unit"], Value::String("celsius".to_string()));
    }

    #[test]
    fn test_parse_minimax_xml_anyof_schema() {
        let tools = vec![crate::tools::function_tool("test_fn", "Test")
            .parameters_schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "value": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "null"}
                        ]
                    }
                }
            }))
            .build()];

        let text = r#"<invoke name="test_fn">
<parameter name="value">42</parameter>
</invoke>"#;

        let calls = parse_minimax_xml_tool_calls(text, &tools);
        let args: Value =
            serde_json::from_str(calls[0].function.arguments.as_deref().unwrap()).unwrap();
        assert_eq!(args["value"], serde_json::json!(42));

        let text_null = r#"<invoke name="test_fn">
<parameter name="value">null</parameter>
</invoke>"#;
        let calls_null = parse_minimax_xml_tool_calls(text_null, &tools);
        let args_null: Value =
            serde_json::from_str(calls_null[0].function.arguments.as_deref().unwrap()).unwrap();
        assert!(args_null["value"].is_null());
    }

    // ---------------------------------------------------------------
    // LLaMA 4 pythonic tool config
    // ---------------------------------------------------------------

    #[test]
    fn test_tool_config_llama4() {
        let config = ToolConfig::for_model_type(&ToolModelType::LLaMa4);
        assert!(config.start_token_ids.contains(&200016)); // <|python_start|>
        assert!(config.end_token_ids.contains(&200007)); // <|eom|>
        assert!(config.end_token_ids.contains(&200008)); // <|eot|>
        assert_eq!(config.start_token_str, "<|python_start|>");
        assert!(config.start_is_special);
        assert!(config.end_is_special);
    }

    #[test]
    fn test_llama4_uses_pythonic_parser() {
        let name = StreamToolParser::parser_name_for_model(
            &ToolModelType::LLaMa4,
            "meta-llama/Llama-4-Scout",
        );
        assert_eq!(name, "pythonic");
    }

    #[test]
    fn test_llama3_still_uses_llama_parser() {
        let name = StreamToolParser::parser_name_for_model(
            &ToolModelType::LLaMa,
            "meta-llama/Llama-3.2-3B-Instruct",
        );
        assert_eq!(name, "llama");
    }

    #[tokio::test]
    async fn test_llama4_parse_pythonic_tool_call() {
        let tools = vec![crate::tools::function_tool("get_weather", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ToolModelType::LLaMa4,
            "llama4".to_string(),
            ToolConfig::for_model_type(&ToolModelType::LLaMa4),
            tools,
            None,
        );

        let text = r#"[get_weather(location="Vancouver", units="celsius")]"#;
        let calls = parser.parse_complete_with_fallback(text).await;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: Value =
            serde_json::from_str(calls[0].function.arguments.as_deref().unwrap()).unwrap();
        assert_eq!(args["location"], "Vancouver");
        assert_eq!(args["units"], "celsius");
    }

    #[tokio::test]
    async fn test_llama4_parse_multiple_pythonic_tool_calls() {
        let tools = vec![
            crate::tools::function_tool("get_weather", "desc").build(),
            crate::tools::function_tool("calculate_route", "desc").build(),
        ];
        let parser = StreamToolParser::new_with_config(
            &ToolModelType::LLaMa4,
            "llama4".to_string(),
            ToolConfig::for_model_type(&ToolModelType::LLaMa4),
            tools,
            None,
        );

        let text = r#"[get_weather(location="Vancouver"), calculate_route(start="Boston", end="New York")]"#;
        let calls = parser.parse_complete_with_fallback(text).await;
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "calculate_route");
    }

    #[tokio::test]
    async fn test_llama4_buffering_with_python_start_token() {
        let tools = vec![crate::tools::function_tool("get_weather", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::LLaMa4,
            "llama4".to_string(),
            ToolConfig::for_model_type(&ToolModelType::LLaMa4),
            tools,
            None,
        );

        // <|python_start|> token triggers buffering
        assert!(matches!(
            parser.process_token(200016, "<|python_start|>").await,
            StreamResult::Buffering
        ));

        // Tool call content gets buffered
        assert!(matches!(
            parser
                .process_token(0, r#"[get_weather(location="Vancouver")]"#)
                .await,
            StreamResult::Buffering
        ));

        // <|eom|> token ends buffering and parses
        match parser.process_token(200007, "<|eom|>").await {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "get_weather");
            }
            other => panic!("Expected ToolCalls, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Gemma 4 case-insensitive bare values
    // ---------------------------------------------------------------

    #[test]
    fn test_gemma4_parse_bare_value_case_insensitive() {
        assert_eq!(
            StreamToolParser::gemma4_parse_bare_value("true"),
            Value::Bool(true)
        );
        assert_eq!(
            StreamToolParser::gemma4_parse_bare_value("True"),
            Value::Bool(true)
        );
        assert_eq!(
            StreamToolParser::gemma4_parse_bare_value("TRUE"),
            Value::Bool(true)
        );
        assert_eq!(
            StreamToolParser::gemma4_parse_bare_value("false"),
            Value::Bool(false)
        );
        assert_eq!(
            StreamToolParser::gemma4_parse_bare_value("False"),
            Value::Bool(false)
        );
        assert_eq!(
            StreamToolParser::gemma4_parse_bare_value("null"),
            Value::Null
        );
        assert_eq!(
            StreamToolParser::gemma4_parse_bare_value("None"),
            Value::Null
        );
        assert_eq!(
            StreamToolParser::gemma4_parse_bare_value("42"),
            Value::Number(42.into())
        );
    }

    #[tokio::test]
    async fn test_gemma4_tool_call_parse() {
        let tools = vec![crate::tools::function_tool("search", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ToolModelType::Gemma4,
            "gemma4".to_string(),
            ToolConfig::for_model_type(&ToolModelType::Gemma4),
            tools,
            None,
        );

        let text =
            r#"<|tool_call>call:search{query:<|"|>rust programming<|"|>,count:5}<tool_call|>"#;
        let calls = parser.parse_complete_with_fallback(text).await;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
        let args: Value =
            serde_json::from_str(calls[0].function.arguments.as_deref().unwrap()).unwrap();
        assert_eq!(args["query"], "rust programming");
        assert_eq!(args["count"], 5);
    }
}
