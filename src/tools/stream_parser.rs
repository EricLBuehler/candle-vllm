// src/tools/stream_parser.rs
//! Streaming tool call parser — detects and buffers tool calls during streaming.
//! Ported from vllm.rs server/parser.rs with tool-parser crate integration.

use super::{Tool, ToolCall};
use serde_json::{Map, Value};
use std::collections::HashSet;
use tokenizers::Tokenizer;
use tool_parser::{
    types::{StreamingParseResult, ToolCallItem},
    ParserFactory, ToolParser as ExternalToolParser,
};

/// Look up the JSON schema types for a parameter.
/// Supports direct `type`, compound schemas, and enum values.
fn extract_schema_types(schema: &Value) -> Vec<String> {
    let Some(obj) = schema.as_object() else {
        return vec!["string".to_string()];
    };

    let mut types = Vec::new();
    if let Some(t) = obj.get("type") {
        match t {
            Value::String(s) => types.push(s.clone()),
            Value::Array(arr) => {
                types.extend(
                    arr.iter()
                        .filter_map(|item| item.as_str().map(str::to_string)),
                );
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
            let ty = match val {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            };
            types.push(ty.to_string());
        }
    }

    if types.is_empty() {
        types.push("string".to_string());
    }
    types.sort();
    types.dedup();
    types
}

fn coerce_param_value(raw: &str, schema_types: &[String]) -> Value {
    let raw = raw.trim();
    let lower = raw.to_ascii_lowercase();
    if matches!(lower.as_str(), "null" | "none" | "nil") {
        return Value::Null;
    }

    let has_explicit_non_string = schema_types
        .iter()
        .any(|t| !matches!(t.as_str(), "string" | "str" | "text"));
    if has_explicit_non_string {
        for ptype in ["integer", "number", "boolean", "object", "array", "string"] {
            if !schema_types.iter().any(|t| t == ptype) {
                continue;
            }
            match ptype {
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
                "boolean" => match lower.as_str() {
                    "true" | "1" | "yes" | "on" => return Value::Bool(true),
                    "false" | "0" | "no" | "off" => return Value::Bool(false),
                    _ => {}
                },
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

    serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

fn resolve_param_properties<'a>(
    function_name: &str,
    tools: &'a [openai_protocol::common::Tool],
) -> Option<&'a serde_json::Map<String, Value>> {
    tools
        .iter()
        .find(|tool| tool.function.name == function_name)
        .and_then(|tool| tool.function.parameters.get("properties"))
        .and_then(Value::as_object)
}

/// Convert our local Tool to openai_protocol::Tool for the tool-parser crate.
fn to_openai_tools(tools: &[crate::tools::Tool]) -> Vec<openai_protocol::common::Tool> {
    tools
        .iter()
        .map(|t| openai_protocol::common::Tool {
            tool_type: "function".to_string(),
            function: openai_protocol::common::Function {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                parameters: t.function.parameters.clone(),
                strict: t.function.strict,
            },
        })
        .collect()
}

/// Manually parse MiniMax XML tool call format.
/// Format: `<minimax:tool_call><invoke name="..."><parameter name="...">...</parameter></invoke></minimax:tool_call>`
fn parse_minimax_xml_tool_calls(
    text: &str,
    tools: &[openai_protocol::common::Tool],
) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut search_from = 0;

    while let Some(invoke_start) = text[search_from..].find("<invoke name=") {
        let abs_invoke_start = search_from + invoke_start;
        let invoke_section = &text[abs_invoke_start..];

        // Extract function name from <invoke name="..."> or <invoke name='...'>
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

        // Find the end of this invoke block
        let invoke_end = if let Some(end_rel) = invoke_section.find("</invoke>") {
            abs_invoke_start + end_rel + "</invoke>".len()
        } else {
            text.len()
        };

        let invoke_block = &text[abs_invoke_start..invoke_end];
        let param_props = resolve_param_properties(function_name, tools);

        // Extract parameters from <parameter name="...">...</parameter>
        let mut args = Map::new();
        let mut param_search = 0;
        while let Some(param_start) = invoke_block[param_search..].find("<parameter name=") {
            let abs_param_start = param_search + param_start;
            let param_section = &invoke_block[abs_param_start..];

            // Extract parameter name
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

            // Find value between > and </parameter>
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

/// Model type classification for tool call handling.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolModelType {
    LLaMa,
    LLaMa4,
    Qwen,
    Qwen3MoE,
    Mistral,
    Gemma,
    Gemma3,
    Gemma4,
    Phi,
    Phi4,
    GLM4,
    Yi,
    StableLM,
    DeepSeek,
    MiniMax,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParserState {
    Normal,
    Buffering,
}

#[derive(Debug, Clone)]
pub enum StreamResult {
    Content(String),
    Buffering,
    ToolCalls(Vec<ToolCall>),
    FlushBuffer(String),
}

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
}

impl ToolConfig {
    pub fn for_model_type(model_type: &ToolModelType) -> Self {
        let mut start_ids = HashSet::new();
        let mut end_ids = HashSet::new();
        match model_type {
            ToolModelType::LLaMa => {
                start_ids.insert(128010);
                end_ids.insert(128008);
                Self {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<|python_tag|>".into(),
                    end_token_str: "<|eom_id|>".into(),
                }
            }
            ToolModelType::LLaMa4 => {
                start_ids.insert(200016);
                end_ids.insert(200007);
                end_ids.insert(200008);
                Self {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<|python_start|>".into(),
                    end_token_str: "<|eom|>".into(),
                }
            }
            ToolModelType::Qwen | ToolModelType::Qwen3MoE => {
                start_ids.insert(151657);
                end_ids.insert(151658);
                Self {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<tool_call>".into(),
                    end_token_str: "</tool_call>".into(),
                }
            }
            ToolModelType::Mistral => {
                start_ids.insert(9);
                Self {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "[TOOL_CALLS]".into(),
                    end_token_str: "]".into(),
                }
            }
            ToolModelType::Gemma | ToolModelType::Gemma3 => Self {
                start_token_ids: start_ids,
                end_token_ids: end_ids,
                start_token_str: "<start_function_call>".into(),
                end_token_str: "<end_function_call>".into(),
            },
            ToolModelType::Gemma4 => {
                start_ids.insert(48);
                end_ids.insert(49);
                Self {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<|tool_call>".into(),
                    end_token_str: "<tool_call|>".into(),
                }
            }
            ToolModelType::MiniMax => {
                // MiniMax tokenizer ships dedicated tool envelope tokens:
                //   200052 => <minimax:tool_call>
                //   200053 => </minimax:tool_call>
                start_ids.insert(200052);
                end_ids.insert(200053);
                Self {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<minimax:tool_call>".into(),
                    end_token_str: "</minimax:tool_call>".into(),
                }
            }
            _ => Self {
                start_token_ids: HashSet::new(),
                end_token_ids: HashSet::new(),
                start_token_str: "<tool_call>".into(),
                end_token_str: "</tool_call>".into(),
            },
        }
    }

    pub fn has_special_tokens(&self) -> bool {
        self.has_start_tokens()
    }
    pub fn has_start_tokens(&self) -> bool {
        !self.start_token_ids.is_empty()
    }
    pub fn has_end_tokens(&self) -> bool {
        !self.end_token_ids.is_empty()
    }

    pub fn validate_with_tokenizer(&mut self, tokenizer: &Tokenizer, model_type: &ToolModelType) {
        if self.has_start_tokens()
            && !Self::matches_single_token(tokenizer, &self.start_token_str, &self.start_token_ids)
        {
            if Self::try_rebind_single_token_id(
                tokenizer,
                &self.start_token_str,
                &mut self.start_token_ids,
            ) {
                tracing::warn!(
                    "Tool start token IDs corrected for model {:?}: {:?}",
                    model_type,
                    self.start_token_ids
                );
            } else {
                tracing::warn!("Tool start token IDs not supported for model {:?}, falling back to text matching", model_type);
                self.start_token_ids.clear();
            }
        }
        if self.has_end_tokens()
            && !Self::matches_single_token(tokenizer, &self.end_token_str, &self.end_token_ids)
        {
            if Self::try_rebind_single_token_id(
                tokenizer,
                &self.end_token_str,
                &mut self.end_token_ids,
            ) {
                tracing::warn!(
                    "Tool end token IDs corrected for model {:?}: {:?}",
                    model_type,
                    self.end_token_ids
                );
            } else {
                tracing::warn!("Tool end token IDs not supported for model {:?}, falling back to text matching", model_type);
                self.end_token_ids.clear();
            }
        }
    }

    pub fn tool_call_end_ids(&self, tokenizer: &Tokenizer) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::new();
        let mut used_special = false;
        if self.has_end_tokens() {
            let mut use_special = true;
            if !self.end_token_str.is_empty() {
                if let Ok(encoded) = tokenizer.encode(self.end_token_str.as_str(), false) {
                    let tok_ids = encoded.get_ids();
                    if tok_ids.len() != 1 || !self.end_token_ids.contains(&tok_ids[0]) {
                        use_special = false;
                    }
                } else {
                    use_special = false;
                }
            }
            if use_special {
                ids.extend(self.end_token_ids.iter().copied());
                used_special = true;
            }
        }
        if !used_special && !self.end_token_str.is_empty() && self.end_token_str.starts_with('<') {
            if let Ok(encoded) = tokenizer.encode(self.end_token_str.as_str(), false) {
                let tok_ids = encoded.get_ids();
                if tok_ids.len() == 1 {
                    ids.push(tok_ids[0]);
                }
            }
        }
        ids
    }

    pub fn tool_call_start_ids(&self, tokenizer: &Tokenizer) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::new();
        let mut used_special = false;
        if self.has_start_tokens() {
            let mut use_special = true;
            if !self.start_token_str.is_empty() {
                if let Ok(encoded) = tokenizer.encode(self.start_token_str.as_str(), false) {
                    let tok_ids = encoded.get_ids();
                    if tok_ids.len() != 1 || !self.start_token_ids.contains(&tok_ids[0]) {
                        use_special = false;
                    }
                } else {
                    use_special = false;
                }
            }
            if use_special {
                ids.extend(self.start_token_ids.iter().copied());
                used_special = true;
            }
        }
        if !used_special
            && !self.start_token_str.is_empty()
            && self.start_token_str.starts_with('<')
        {
            if let Ok(encoded) = tokenizer.encode(self.start_token_str.as_str(), false) {
                let tok_ids = encoded.get_ids();
                if tok_ids.len() == 1 {
                    ids.push(tok_ids[0]);
                }
            }
        }
        ids
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

const REASONING_MARKERS: &[(&str, &str)] = &[
    ("<think>", "</think>"),
    ("<|think|>", "<|/think|>"),
    ("[THINK]", "[/THINK]"),
    ("<thought>", "</thought>"),
];

pub fn reasoning_markers() -> &'static [(&'static str, &'static str)] {
    REASONING_MARKERS
}

pub fn detect_prefilled_reasoning_end_marker(prompt: &str) -> Option<String> {
    let trimmed = prompt.trim_end();
    for &(start, end) in REASONING_MARKERS {
        if trimmed.ends_with(start) {
            return Some(end.to_string());
        }
    }
    None
}

/// Strip all reasoning start/end markers from a text fragment.
/// Used by the reasoning content router to remove markers before sending to client.
pub fn strip_reasoning_markers(text: &str) -> String {
    let mut result = text.to_string();
    for &(open, close) in REASONING_MARKERS {
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
                result.truncate(start_idx);
                break;
            }
        }
    }
    result
}

/// Extract reasoning content from text containing reasoning blocks.
/// Returns `(reasoning_content, remaining_content)` if any matched pair is found.
pub fn extract_reasoning_content(content: &str) -> Option<(String, String)> {
    for &(open, close) in REASONING_MARKERS {
        if !content.contains(open) || !content.contains(close) {
            continue;
        }
        let mut reasoning_parts: Vec<&str> = Vec::new();
        let mut search_from = 0;
        let mut last_close_end = 0;

        while let Some(open_idx) = content[search_from..].find(open) {
            let abs_open = search_from + open_idx;
            let inner_start = abs_open + open.len();
            let Some(close_rel) = content[inner_start..].find(close) else {
                break;
            };
            let abs_close = inner_start + close_rel;
            let block = content[inner_start..abs_close].trim_matches('\n');
            if !block.is_empty() {
                reasoning_parts.push(block);
            }
            last_close_end = abs_close + close.len();
            search_from = last_close_end;
        }

        if last_close_end == 0 {
            continue;
        }

        let reasoning = reasoning_parts.join("\n");
        let remaining = content[last_close_end..]
            .trim_start_matches('\n')
            .to_string();
        return Some((reasoning, remaining));
    }
    None
}

#[derive(Debug, Clone, Default)]
struct StreamingToolCallState {
    name: Option<String>,
    arguments: String,
}

/// Streaming tool parser with tool-parser crate integration
pub struct StreamToolParser {
    config: ToolConfig,
    state: ParserState,
    buffer: String,
    model_id: String,
    parse_strategy: String,
    parser: Box<dyn ExternalToolParser>,
    tools: Vec<openai_protocol::common::Tool>,
    streaming_calls: Vec<StreamingToolCallState>,
    accumulated_output: String,
    active_reasoning_end: Option<String>,
    detect_tools_in_reasoning: bool,
    in_code_block: bool,
    saw_buffer_parse_activity: bool,
    buffer_had_parse_activity: bool,
    pending_end_marker_candidate: bool,
    buffer_started_from_special_token: bool,
    buffer_saw_non_marker_content: bool,
}

impl StreamToolParser {
    pub fn new(model_type: ToolModelType) -> Self {
        let config = ToolConfig::for_model_type(&model_type);
        Self::new_with_config(&model_type, "unknown".into(), config, vec![], None)
    }

    pub fn new_with_config(
        model_type: &ToolModelType,
        model_id: String,
        config: ToolConfig,
        tools: Vec<Tool>,
        enforce_parser: Option<String>,
    ) -> Self {
        let parse_strategy = match model_type {
            ToolModelType::Mistral => "mistral_list",
            ToolModelType::Gemma4 => "gemma4",
            ToolModelType::LLaMa4 => "pythonic",
            _ => "json",
        }
        .to_string();
        let openai_tools = to_openai_tools(&tools);

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
            tools: openai_tools,
            streaming_calls: Vec::new(),
            accumulated_output: String::new(),
            active_reasoning_end: None,
            detect_tools_in_reasoning: false,
            in_code_block: false,
            saw_buffer_parse_activity: false,
            buffer_had_parse_activity: false,
            pending_end_marker_candidate: false,
            buffer_started_from_special_token: false,
            buffer_saw_non_marker_content: false,
        }
    }

    // Backward-compatible 2-arg constructor used by existing llm_engine.rs code
    pub fn new_with_config_compat(model_type: &ToolModelType, config: ToolConfig) -> Self {
        Self::new_with_config(model_type, "unknown".into(), config, Vec::new(), None)
    }

    pub fn in_reasoning(&self) -> bool {
        self.active_reasoning_end.is_some()
    }
    pub fn set_initial_reasoning_end_marker(&mut self, end_marker: Option<String>) {
        self.active_reasoning_end = end_marker;
    }
    pub fn set_detect_tools_in_reasoning(&mut self, enabled: bool) {
        self.detect_tools_in_reasoning = enabled;
    }
    pub fn in_code_block(&self) -> bool {
        self.in_code_block
    }
    pub fn accumulated_output_without_reasoning(&self) -> String {
        strip_reasoning_blocks(&self.accumulated_output)
    }
    pub fn state(&self) -> &ParserState {
        &self.state
    }
    pub fn accumulated_output(&self) -> &str {
        &self.accumulated_output
    }
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

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

    pub fn take_buffer_parse_activity(&mut self) -> bool {
        std::mem::take(&mut self.saw_buffer_parse_activity)
    }

    fn update_code_block_state(&mut self, _token_text: &str) {
        let mut code_block_count = 0;
        for line in self.accumulated_output.lines() {
            if line.trim().starts_with("```") {
                code_block_count += 1;
            }
        }
        self.in_code_block = code_block_count % 2 == 1;
    }

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

    fn mask_tool_envelopes(text: &str, start_tag: &str, end_tag: &str) -> String {
        if start_tag.is_empty() || end_tag.is_empty() {
            return text.to_string();
        }
        let mut result = text.to_string();
        loop {
            let Some(s) = result.find(start_tag) else {
                break;
            };
            let inner = s + start_tag.len();
            if let Some(e_rel) = result[inner..].find(end_tag) {
                let e = inner + e_rel + end_tag.len();
                let mask = " ".repeat(e - s);
                result.replace_range(s..e, &mask);
            } else {
                break;
            }
        }
        result
    }

    fn resync_reasoning_and_code_block_state(&mut self) {
        let text = Self::mask_tool_envelopes(
            &self.accumulated_output,
            &self.config.start_token_str,
            &self.config.end_token_str,
        );

        let mut code_block_count = 0;
        for line in text.lines() {
            if line.trim().starts_with("```") {
                code_block_count += 1;
            }
        }
        self.in_code_block = code_block_count % 2 == 1;

        self.active_reasoning_end = None;
        for &(start, end) in REASONING_MARKERS {
            let mut open = false;
            let mut search_from = 0;
            while let Some(idx) = text[search_from..].find(start) {
                let abs = search_from + idx;
                let after = abs + start.len();
                if let Some(close_rel) = text[after..].find(end) {
                    search_from = after + close_rel + end.len();
                    open = false;
                } else {
                    open = true;
                    break;
                }
            }
            if open {
                self.active_reasoning_end = Some(end.to_string());
                break;
            }
        }
    }

    pub fn process_token(&mut self, token_id: u32, token_text: &str) -> StreamResult {
        self.saw_buffer_parse_activity = false;
        self.accumulated_output.push_str(token_text);

        // Only track reasoning and code-block state while in Normal mode.
        // During Buffering the token content is tool-call payload (JSON, XML)
        // which may contain strings like `</think>` or "```" that must not
        // corrupt the reasoning/code-block tracking used for Normal-mode gating.
        if !matches!(self.state, ParserState::Buffering) {
            self.update_code_block_state(token_text);
            self.update_reasoning_state(token_text);
        }

        match self.state.clone() {
            ParserState::Normal => {
                if self.in_code_block {
                    return StreamResult::Content(token_text.to_string());
                }
                if self.in_reasoning() && !self.detect_tools_in_reasoning {
                    return StreamResult::Content(token_text.to_string());
                }
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
                    if let Ok(result) = futures::executor::block_on(
                        self.parser.parse_incremental(token_text, &self.tools),
                    ) {
                        if !result.calls.is_empty() {
                            self.saw_buffer_parse_activity = true;
                            self.buffer_had_parse_activity = true;
                        }
                        self.apply_streaming_result(&result);
                    }
                    tracing::info!(
                        "Tool call {} ({}) found, start buffering!",
                        token_text,
                        token_id
                    );
                    return StreamResult::Buffering;
                }
                StreamResult::Content(token_text.to_string())
            }
            ParserState::Buffering => {
                self.buffer.push_str(token_text);
                if self.token_contains_non_marker_content(token_text) {
                    self.buffer_saw_non_marker_content = true;
                }
                let nested_start = !self.config.start_token_str.is_empty()
                    && token_text.contains(&self.config.start_token_str);
                if nested_start {
                    tracing::warn!(
                        "Ignoring nested tool-call start marker while buffering: {:?}",
                        token_text
                    );
                } else {
                    if let Ok(result) = futures::executor::block_on(
                        self.parser.parse_incremental(token_text, &self.tools),
                    ) {
                        if !result.calls.is_empty() {
                            self.saw_buffer_parse_activity = true;
                            self.buffer_had_parse_activity = true;
                        }
                        self.apply_streaming_result(&result);
                    }
                }

                let end_reached = self.is_end_token(token_id, token_text)
                    || self.buffer_has_end_tag()
                    || self.maybe_complete_mistral_list();

                if !end_reached && self.pending_end_marker_candidate {
                    self.pending_end_marker_candidate = false;
                }

                if end_reached {
                    let strict_complete =
                        futures::executor::block_on(self.has_strict_complete_tool_call());
                    if !strict_complete && !self.pending_end_marker_candidate {
                        self.pending_end_marker_candidate = true;
                        tracing::warn!("Tool-call end marker seen before payload completion; waiting for confirmation");
                        return StreamResult::Buffering;
                    }
                    self.pending_end_marker_candidate = false;
                    tracing::info!(
                        "Tool call buffering end, reached {} ({})",
                        token_text,
                        token_id
                    );

                    let had_partial = !self.streaming_calls.is_empty();
                    let tool_calls =
                        futures::executor::block_on(self.build_tool_calls_with_fallback());
                    let result = if tool_calls.is_empty() {
                        if had_partial {
                            tracing::warn!(
                                "End marker seen but tool call still incomplete; continuing"
                            );
                            StreamResult::Buffering
                        } else {
                            tracing::error!("Unable to parse tool call buffer: {}", self.buffer);
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

    pub fn finalize_buffered_tool_calls(&mut self) -> Option<BufferedFinalizeResult> {
        if !matches!(self.state, ParserState::Buffering) {
            return None;
        }
        tracing::warn!("Stream ended while buffering a tool call; attempting final parse");
        let buffered_text = self.buffer.clone();
        let strict_complete = futures::executor::block_on(self.has_strict_complete_tool_call());
        let tool_calls = futures::executor::block_on(self.build_tool_calls_with_fallback());
        let recoverable = !strict_complete
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

        if tool_calls.is_empty() || (!strict_complete && !recoverable) {
            if drop_bare_start_marker {
                tracing::warn!(
                    "Dropping buffered bare tool-call marker at stream end without flushing text"
                );
                return Some(BufferedFinalizeResult::FlushBuffer(String::new()));
            }
            tracing::warn!("Buffered tool call could not be finalized; flushing buffered text");
            Some(BufferedFinalizeResult::FlushBuffer(buffered_text))
        } else {
            if recoverable {
                tracing::warn!("Recovered buffered tool call(s) from partial envelope");
            }
            tracing::warn!(
                "Recovered {} tool call(s) from buffered state at stream end",
                tool_calls.len()
            );
            Some(BufferedFinalizeResult::ToolCalls(tool_calls))
        }
    }

    /// Legacy finalize returning Option<Vec<ToolCall>> for backward compat with llm_engine.
    pub fn finalize(&mut self) -> Option<Vec<ToolCall>> {
        match self.finalize_buffered_tool_calls() {
            Some(BufferedFinalizeResult::ToolCalls(calls)) => Some(calls),
            _ => None,
        }
    }

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

    pub fn reparse_accumulated_output(&self) -> Vec<ToolCall> {
        futures::executor::block_on(self.parse_complete_with_fallback(self.accumulated_output()))
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

        if parsed_calls.is_empty() && self.parse_strategy == "pythonic" {
            let factory = ParserFactory::new();
            if let Some(pythonic_parser) = factory.registry().create_parser("pythonic") {
                if let Ok((_normal_text, calls)) = pythonic_parser.parse_complete(text).await {
                    parsed_calls = calls;
                }
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

        // MiniMax XML style: <minimax:tool_call><invoke name="..."><parameter name="...">...</parameter></invoke></minimax:tool_call>
        if parsed_calls.is_empty() && text.contains("<invoke name=") {
            let factory = ParserFactory::new();
            if let Some(xml_parser) = factory.registry().create_parser("minimax_m2") {
                if let Ok((_normal_text, calls)) = xml_parser.parse_complete(text).await {
                    parsed_calls = calls;
                }
            }
            // Manual fallback if tool-parser crate fails
            if parsed_calls.is_empty() {
                tracing::info!("Falling back to manual MiniMax XML parser for: {}", text);
                return parse_minimax_xml_tool_calls(text, &self.tools);
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

    // ── Private helpers ──────────────────────────────────────────────────

    fn is_start_token(&self, id: u32, _text: &str) -> bool {
        if self.config.has_start_tokens() {
            return self.config.start_token_ids.contains(&id);
        }
        if self.config.start_token_str.is_empty() {
            return false;
        }
        let current_line = self.accumulated_output.rsplit('\n').next().unwrap_or("");
        let candidate = current_line.trim_start_matches(|c| c == ' ' || c == '\t' || c == '\r');
        if candidate.starts_with(&self.config.start_token_str) {
            return true;
        }
        let min_prefix = Self::safe_partial_prefix_len(&self.config.start_token_str);
        !candidate.is_empty()
            && candidate.len() >= min_prefix
            && self.config.start_token_str.starts_with(candidate)
    }

    fn is_end_token(&self, id: u32, text: &str) -> bool {
        if self.config.has_end_tokens() {
            return self.config.end_token_ids.contains(&id);
        }
        if self.config.end_token_str.is_empty() {
            return false;
        }
        if self.parse_strategy == "mistral_list" && self.config.end_token_str == "]" {
            return false;
        }
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

    async fn has_strict_complete_tool_call(&self) -> bool {
        if !self.has_complete_tool_envelope() {
            return false;
        }
        if self.streaming_calls.iter().any(|c| c.name.is_none()) {
            return false;
        }
        if !self.streaming_calls.is_empty()
            && self
                .streaming_calls
                .iter()
                .all(|c| c.arguments.trim().is_empty())
        {
            return true;
        }
        if !self.streaming_calls.is_empty()
            && self
                .streaming_calls
                .iter()
                .all(|c| serde_json::from_str::<Value>(c.arguments.trim()).is_ok())
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
                .all(|c| c.arguments.trim().is_empty())
        {
            return false;
        }
        if self.streaming_calls.iter().any(|c| c.name.is_none()) {
            return false;
        }
        self.streaming_calls.iter().all(|c| {
            let args = c.arguments.trim();
            args.is_empty()
                || serde_json::from_str::<Value>(&self.finalize_streamed_arguments(args)).is_ok()
        })
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
                streaming.function.arguments == "{}" && fallback.function.arguments != "{}"
            })
    }

    fn has_ambiguous_incomplete_end_marker(&self) -> bool {
        if self.config.end_token_str.is_empty() || !self.config.end_token_str.starts_with('<') {
            return false;
        }
        self.buffer.contains(&self.config.end_token_str) && !self.has_complete_tool_envelope()
    }

    fn has_complete_tool_envelope(&self) -> bool {
        if !self.config.start_token_str.starts_with('<')
            || !self.config.end_token_str.starts_with('<')
        {
            return true;
        }
        let Some(start_idx) = self.buffer.find(&self.config.start_token_str) else {
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
            return Self::has_balanced_parameter_tags(invoke_block, "<parameter name=");
        }

        if block.contains("<function=") || block.contains("<parameter=") {
            let Some(fs) = block.find("<function=") else {
                return false;
            };
            let func_section = &block[fs..];
            let Some(fe_rel) = func_section.rfind("</function>") else {
                return false;
            };
            let func_end = fs + fe_rel + "</function>".len();
            let func_block = &block[fs..func_end];
            return Self::has_balanced_parameter_tags(func_block, "<parameter=");
        }
        if block.contains("<arg_key>") || block.contains("<arg_value>") {
            return block.contains("</arg_value>")
                && Self::has_balanced_xml_tags(block, "<arg_key>", "</arg_key>")
                && Self::has_balanced_xml_tags(block, "<arg_value>", "</arg_value>");
        }
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

    fn parser_name_for_model(model_type: &ToolModelType, model_id: &str) -> &'static str {
        let model_lower = model_id.to_ascii_lowercase();
        match model_type {
            ToolModelType::LLaMa => "llama",
            ToolModelType::LLaMa4 => "pythonic",
            ToolModelType::Mistral => "mistral",
            ToolModelType::Qwen | ToolModelType::Qwen3MoE => {
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
            ToolModelType::Phi | ToolModelType::Phi4 => "qwen",
            ToolModelType::GLM4 => "glm47_moe",
            ToolModelType::Yi | ToolModelType::StableLM => "qwen",
            ToolModelType::DeepSeek => "deepseek",
            ToolModelType::MiniMax => "minimax_m2",
        }
    }

    /// Parse Gemma4 tool calls: `<|tool_call>call:NAME{key:<|"|>value<|"|>,...}<tool_call|>`.
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
        let mut search_start = 0usize;
        while let Some(rel_pos) = text[search_start..].find(active_prefix) {
            let abs_start = search_start + rel_pos + active_prefix.len();
            let Some(brace_rel) = text[abs_start..].find('{') else {
                break;
            };
            let name = text[abs_start..abs_start + brace_rel].trim();
            let brace_abs = abs_start + brace_rel;
            let Some((inner, after_brace)) = Self::gemma4_extract_braces(text, brace_abs) else {
                break;
            };
            let arguments = Self::gemma4_parse_args(inner);
            calls.push(crate::tools::new_tool_call(
                crate::tools::generate_tool_call_id(),
                name.to_string(),
                serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string()),
            ));

            let remaining = &text[after_brace..];
            search_start = if let Some(suf_pos) = remaining.find(SUFFIX) {
                after_brace + suf_pos + SUFFIX.len()
            } else {
                after_brace
            };
        }

        (!calls.is_empty()).then_some(calls)
    }

    fn gemma4_extract_braces(s: &str, start: usize) -> Option<(&str, usize)> {
        const DELIM: &str = "<|\"|>";
        if !s.is_char_boundary(start) || s.as_bytes().get(start) != Some(&b'{') {
            return None;
        }

        let mut depth = 0usize;
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
                '"' => in_regular_string = true,
                '{' => depth += 1,
                '}' => {
                    depth = depth.saturating_sub(1);
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
        let mut ci = 0usize;
        while ci < n {
            while ci < n && matches!(chars[ci].1, ' ' | ',' | '\n' | '\t') {
                ci += 1;
            }
            if ci >= n {
                break;
            }

            let key_start = chars[ci].0;
            while ci < n && chars[ci].1 != ':' {
                ci += 1;
            }
            if ci >= n {
                break;
            }
            let key = args_str[key_start..chars[ci].0].trim().trim_matches('"');
            ci += 1;
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
                match args_str[val_start..].find(DELIM) {
                    Some(rel) => {
                        let val = &args_str[val_start..val_start + rel];
                        map.insert(key.to_string(), Value::String(val.to_string()));
                        let after = val_start + rel + DELIM.len();
                        ci = chars.iter().position(|&(b, _)| b >= after).unwrap_or(n);
                    }
                    None => {
                        map.insert(
                            key.to_string(),
                            Value::String(args_str[val_start..].to_string()),
                        );
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
                map.insert(
                    key.to_string(),
                    Value::String(args_str[val_start..val_end].to_string()),
                );
                ci = if end_ci < n { end_ci + 1 } else { n };
            } else if chars[ci].1 == '{' {
                let (inner, after_ci) = Self::gemma4_scan_nested(&chars, ci, '{', '}', n, args_str);
                map.insert(key.to_string(), Self::gemma4_parse_args(inner));
                ci = after_ci;
            } else if chars[ci].1 == '[' {
                let (inner, after_ci) = Self::gemma4_scan_nested(&chars, ci, '[', ']', n, args_str);
                map.insert(key.to_string(), Self::gemma4_parse_array(inner));
                ci = after_ci;
            } else {
                let val_start = chars[ci].0;
                while ci < n && !matches!(chars[ci].1, ',' | '}' | ']') {
                    ci += 1;
                }
                let val_end = if ci < n { chars[ci].0 } else { args_str.len() };
                map.insert(
                    key.to_string(),
                    Self::gemma4_parse_bare_value(args_str[val_start..val_end].trim()),
                );
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
                depth = depth.saturating_sub(1);
            }
            ci += 1;
        }
        let inner_end = if depth == 0 && ci > 0 {
            chars[ci - 1].0
        } else if ci < n {
            chars[ci].0
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
        let mut ci = 0usize;
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
                match arr_str[val_start..].find(DELIM) {
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
        match val.to_ascii_lowercase().as_str() {
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
            return idx.max(2);
        }
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
        let min_prefix = Self::safe_partial_prefix_len(marker).min(marker_len.saturating_sub(1));
        if min_prefix >= marker_len {
            return false;
        }
        (min_prefix..marker_len)
            .rev()
            .any(|len| text.contains(&marker[..len]))
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
        let min_prefix = Self::safe_partial_prefix_len(marker).min(marker_len.saturating_sub(1));
        if min_prefix >= marker_len {
            return text.to_string();
        }
        let mut out = text.to_string();
        for len in (min_prefix..marker_len).rev() {
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
            if self.uses_glm_xml() {
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
        if (decoded.starts_with('{') || decoded.starts_with('['))
            && serde_json::from_str::<Value>(&decoded).is_ok()
        {
            return serde_json::from_str::<Value>(&decoded).unwrap();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_config_qwen() {
        let config = ToolConfig::for_model_type(&ToolModelType::Qwen);
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
    fn test_parser_normal_content() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );
        match parser.process_token(0, "Hello world") {
            StreamResult::Content(s) => assert_eq!(s, "Hello world"),
            _ => panic!("Expected Content"),
        }
    }

    #[test]
    fn test_empty_tool_markers_do_not_trigger_buffering() {
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3".into(),
            ToolConfig {
                start_token_ids: HashSet::new(),
                end_token_ids: HashSet::new(),
                start_token_str: String::new(),
                end_token_str: String::new(),
            },
            Vec::new(),
            None,
        );

        match parser.process_token(248069, "</think>") {
            StreamResult::Content(text) => assert_eq!(text, "</think>"),
            other => panic!("Expected Content for empty-marker parser, got {:?}", other),
        }
        match parser.process_token(1825, "AG") {
            StreamResult::Content(text) => assert_eq!(text, "AG"),
            other => panic!("Expected Content for empty-marker parser, got {:?}", other),
        }
        assert!(!matches!(parser.state(), ParserState::Buffering));
    }

    #[test]
    fn test_parser_tool_call_detection() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );
        match parser.process_token(151657, "<tool_call>") {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on start tag"),
        }
        match parser.process_token(0, r#"{"name": "test", "arguments": {}}"#) {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering"),
        }
        match parser.process_token(151658, "</tool_call>") {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "test");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_parser_token_id_strict_match() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );
        // Text match should not trigger when token IDs are available
        match parser.process_token(0, "<tool_call>") {
            StreamResult::Content(text) => assert_eq!(text, "<tool_call>"),
            _ => panic!("Expected Content without token ID match"),
        }
    }

    #[test]
    fn test_repair_streamed_json_balances_brackets() {
        let raw = r#"{"file_path":"/tmp/a.rs","new_string":"fn a() { let x = vec![1,2,3]; }","replace_all":false"#;
        let repaired = repair_streamed_json_arguments(raw);
        let parsed: Value = serde_json::from_str(&repaired).expect("repaired JSON should parse");
        assert_eq!(parsed["file_path"], "/tmp/a.rs");
        assert_eq!(parsed["replace_all"], false);
    }

    #[test]
    fn test_sanitize_tool_markup_escapes_xml() {
        let tools = vec![crate::tools::function_tool("write", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3-coder".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );
        let raw = "<tool_call><function=write></function></tool_call>";
        assert!(parser.contains_tool_markup(raw));
        let safe = parser.sanitize_tool_markup_for_display(raw);
        assert!(!parser.contains_tool_markup(&safe));
    }

    #[test]
    fn test_finalize_recovers_qwen_tool_call_without_closing_tag() {
        let tools = vec![crate::tools::function_tool("glob", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );

        match parser.process_token(151657, "<tool_call>") {
            StreamResult::Buffering => {}
            other => panic!("Expected Buffering on start tag, got {:?}", other),
        }
        match parser.process_token(0, "\n") {
            StreamResult::Buffering => {}
            other => panic!("Expected Buffering on newline, got {:?}", other),
        }
        match parser.process_token(
            0,
            r#"{"name": "glob", "arguments": {"pattern": "AGENTS.md"}}"#,
        ) {
            StreamResult::Buffering => {}
            other => panic!("Expected Buffering on JSON payload, got {:?}", other),
        }

        match parser.finalize_buffered_tool_calls() {
            Some(BufferedFinalizeResult::ToolCalls(calls)) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "glob");
                assert!(calls[0].function.arguments.contains("AGENTS.md"));
            }
            other => panic!("Expected tool-call recovery, got {:?}", other),
        }
    }

    #[test]
    fn test_finalize_flushes_unrecoverable_tool_call_payload() {
        let tools = vec![crate::tools::function_tool("glob", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3.5-coder".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );

        match parser.process_token(151657, "<tool_call>") {
            StreamResult::Buffering => {}
            other => panic!("Expected Buffering on start tag, got {:?}", other),
        }
        match parser.process_token(0, r#"{"arguments": {"pattern": "AGENTS.md"}}"#) {
            StreamResult::Buffering => {}
            other => panic!(
                "Expected Buffering on malformed JSON payload, got {:?}",
                other
            ),
        }

        match parser.finalize_buffered_tool_calls() {
            Some(BufferedFinalizeResult::FlushBuffer(buffer)) => {
                assert!(buffer.contains("AGENTS.md"));
            }
            other => panic!("Expected buffered text flush, got {:?}", other),
        }
    }

    #[test]
    fn test_reasoning_state_tracking() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );
        assert!(!parser.in_reasoning());

        parser.process_token(0, "<think>");
        assert!(parser.in_reasoning());

        parser.process_token(0, "Let me think about this");
        assert!(parser.in_reasoning());

        parser.process_token(0, "</think>");
        assert!(!parser.in_reasoning());
    }

    #[test]
    fn test_reasoning_then_tool_call() {
        let tools = vec![crate::tools::function_tool("search", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );
        parser.set_detect_tools_in_reasoning(true);

        match parser.process_token(0, "<think>") {
            StreamResult::Content(_) => {}
            other => panic!("Expected Content for think open, got {:?}", other),
        }
        assert!(parser.in_reasoning());

        match parser.process_token(0, "I should search for this") {
            StreamResult::Content(text) => assert_eq!(text, "I should search for this"),
            other => panic!("Expected Content in reasoning, got {:?}", other),
        }

        match parser.process_token(0, "</think>") {
            StreamResult::Content(_) => {}
            other => panic!("Expected Content for think close, got {:?}", other),
        }
        assert!(!parser.in_reasoning());

        match parser.process_token(151657, "<tool_call>") {
            StreamResult::Buffering => {}
            other => panic!("Expected Buffering on tool start, got {:?}", other),
        }

        match parser.process_token(0, r#"{"name": "search", "arguments": {"q": "test"}}"#) {
            StreamResult::Buffering => {}
            other => panic!("Expected Buffering on JSON, got {:?}", other),
        }

        match parser.process_token(151658, "</tool_call>") {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "search");
            }
            other => panic!("Expected ToolCalls, got {:?}", other),
        }

        assert!(!parser.in_reasoning());
    }

    #[test]
    fn test_reasoning_state_not_corrupted_during_buffering() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );
        parser.set_detect_tools_in_reasoning(true);

        parser.process_token(0, "<think>");
        assert!(parser.in_reasoning());

        parser.process_token(0, "thinking...");
        parser.process_token(0, "</think>");
        assert!(!parser.in_reasoning());

        match parser.process_token(151657, "<tool_call>") {
            StreamResult::Buffering => {}
            other => panic!("Expected Buffering, got {:?}", other),
        }

        // </think> inside tool JSON payload must not flip reasoning state
        parser.process_token(0, r#"{"name": "test", "arguments": {"text": "</think>"}}"#);
        assert!(!parser.in_reasoning());

        match parser.process_token(151658, "</tool_call>") {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "test");
            }
            other => panic!("Expected ToolCalls, got {:?}", other),
        }

        // After tool call completes, reasoning state should be resynced correctly
        assert!(!parser.in_reasoning());
    }

    #[test]
    fn test_tool_call_suppressed_during_active_reasoning() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );
        // detect_tools_in_reasoning defaults to false
        assert!(!parser.detect_tools_in_reasoning);

        parser.process_token(0, "<think>");
        assert!(parser.in_reasoning());

        // Tool-call-like text inside reasoning should NOT trigger buffering
        match parser.process_token(0, "I could use <tool_call> here") {
            StreamResult::Content(text) => {
                assert!(text.contains("<tool_call>"));
            }
            other => panic!(
                "Expected Content (tool suppressed in reasoning), got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_prefilled_reasoning_end_marker() {
        assert_eq!(
            detect_prefilled_reasoning_end_marker("Hello <think>"),
            Some("</think>".to_string())
        );
        assert_eq!(
            detect_prefilled_reasoning_end_marker("Hello <|think|>"),
            Some("<|/think|>".to_string())
        );
        assert_eq!(detect_prefilled_reasoning_end_marker("Hello world"), None);
    }

    #[test]
    fn test_strip_reasoning_markers() {
        assert_eq!(strip_reasoning_markers("<think>hello</think>"), "hello");
        assert_eq!(strip_reasoning_markers("<|think|>hello<|/think|>"), "hello");
        assert_eq!(
            strip_reasoning_markers("no markers here"),
            "no markers here"
        );
        assert_eq!(strip_reasoning_markers("<think>"), "");
        assert_eq!(strip_reasoning_markers("</think>"), "");
    }

    #[test]
    fn test_extract_reasoning_content() {
        let (reasoning, remaining) =
            extract_reasoning_content("<think>I should search</think>\nHere is the answer")
                .unwrap();
        assert_eq!(reasoning, "I should search");
        assert_eq!(remaining, "Here is the answer");

        let (reasoning, remaining) =
            extract_reasoning_content("<think>Only reasoning</think>").unwrap();
        assert_eq!(reasoning, "Only reasoning");
        assert_eq!(remaining, "");

        assert!(extract_reasoning_content("No reasoning markers here").is_none());
    }

    #[test]
    fn test_strip_reasoning_blocks() {
        assert_eq!(
            strip_reasoning_blocks("<think>reasoning</think>content"),
            "content"
        );
        assert_eq!(
            strip_reasoning_blocks("prefix<think>reasoning</think>suffix"),
            "prefixsuffix"
        );
        // Unmatched opening marker strips to end
        assert_eq!(strip_reasoning_blocks("before<think>unmatched"), "before");
    }

    #[test]
    fn test_accumulated_output_without_reasoning() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );

        parser.process_token(0, "<think>reasoning</think>");
        parser.process_token(151657, "<tool_call>");
        parser.process_token(0, r#"{"name": "test", "arguments": {}}"#);
        parser.process_token(151658, "</tool_call>");

        let without = parser.accumulated_output_without_reasoning();
        assert!(!without.contains("<think>"));
        assert!(!without.contains("reasoning"));
        assert!(without.contains("<tool_call>"));
    }

    #[test]
    fn test_resync_reasoning_after_tool_call() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );
        parser.set_detect_tools_in_reasoning(true);

        // Start reasoning
        parser.process_token(0, "<think>");
        assert!(parser.in_reasoning());

        // Close reasoning
        parser.process_token(0, "</think>");
        assert!(!parser.in_reasoning());

        // Enter tool call buffering
        match parser.process_token(151657, "<tool_call>") {
            StreamResult::Buffering => {}
            other => panic!("Expected Buffering, got {:?}", other),
        }

        // Complete tool call
        parser.process_token(0, r#"{"name": "test", "arguments": {}}"#);
        match parser.process_token(151658, "</tool_call>") {
            StreamResult::ToolCalls(_) => {}
            other => panic!("Expected ToolCalls, got {:?}", other),
        }

        // After tool call, reasoning state should be correctly resynced
        assert!(!parser.in_reasoning());

        // New content after tool call should be normal content
        match parser.process_token(0, "After tool call") {
            StreamResult::Content(text) => assert_eq!(text, "After tool call"),
            other => panic!("Expected Content after tool call, got {:?}", other),
        }
    }

    #[test]
    fn test_end_marker_then_finalize_flushes_unrecoverable_tool_call_payload() {
        let tools = vec![crate::tools::function_tool("glob", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::Qwen,
            "qwen3.5-coder".into(),
            ToolConfig::for_model_type(&ToolModelType::Qwen),
            tools,
            None,
        );

        match parser.process_token(151657, "<tool_call>") {
            StreamResult::Buffering => {}
            other => panic!("Expected Buffering on start tag, got {:?}", other),
        }
        match parser.process_token(0, r#"{"arguments": {"pattern": "AGENTS.md"}}"#) {
            StreamResult::Buffering => {}
            other => panic!(
                "Expected Buffering on malformed JSON payload, got {:?}",
                other
            ),
        }

        match parser.process_token(151658, "</tool_call>") {
            StreamResult::Buffering => {}
            other => panic!(
                "Expected confirmation buffering on first end marker, got {:?}",
                other
            ),
        }

        match parser.finalize_buffered_tool_calls() {
            Some(BufferedFinalizeResult::FlushBuffer(buffer)) => {
                assert!(buffer.contains("AGENTS.md"));
            }
            other => panic!(
                "Expected buffered text flush after finalize, got {:?}",
                other
            ),
        }
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

    #[test]
    fn test_minimax_parser_detects_start_and_end_by_token_id() {
        let tools = vec![crate::tools::function_tool("search_web", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::MiniMax,
            "MiniMax-M2.5".to_string(),
            ToolConfig::for_model_type(&ToolModelType::MiniMax),
            tools,
            None,
        );

        match parser.process_token(200052, "<minimax:tool_call>") {
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

        match parser.process_token(200053, "</minimax:tool_call>") {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "search_web");
            }
            other => panic!("expected ToolCalls on MiniMax end token, got {:?}", other),
        }
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
        assert!(calls[0].function.arguments.contains("content"));
        assert!(calls[0].function.arguments.contains("filePath"));
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
        assert!(calls[0].function.arguments.contains("filePath"));
    }

    #[test]
    fn test_parse_minimax_xml_with_array_value() {
        let text = r#"<invoke name="search">
<parameter name="tags">["rust", "programming"]</parameter>
</invoke>"#;

        let calls = parse_minimax_xml_tool_calls(text, &[]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args["tags"].is_array());
    }

    #[test]
    fn test_parse_minimax_xml_type_coercion_with_schema() {
        let tools = to_openai_tools(&[crate::tools::function_tool("get_weather", "desc")
            .parameters_schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "days": {"type": "integer"},
                    "include_hourly": {"type": "boolean"},
                    "units": {"type": "string"}
                }
            }))
            .build()]);
        let text = r#"<invoke name="get_weather">
<parameter name="days">3</parameter>
<parameter name="include_hourly">true</parameter>
<parameter name="units">metric</parameter>
</invoke>"#;

        let calls = parse_minimax_xml_tool_calls(text, &tools);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["days"], 3);
        assert_eq!(args["include_hourly"], true);
        assert_eq!(args["units"], "metric");
    }

    #[test]
    fn test_tool_config_llama4() {
        let config = ToolConfig::for_model_type(&ToolModelType::LLaMa4);
        assert!(config.start_token_ids.contains(&200016));
        assert!(config.end_token_ids.contains(&200007));
        assert!(config.end_token_ids.contains(&200008));
        assert_eq!(config.start_token_str, "<|python_start|>");
    }

    #[test]
    fn test_llama4_uses_pythonic_parser() {
        assert_eq!(
            StreamToolParser::parser_name_for_model(
                &ToolModelType::LLaMa4,
                "meta-llama/Llama-4-Scout"
            ),
            "pythonic"
        );
    }

    #[test]
    fn test_llama4_parse_pythonic_tool_call() {
        let tools = vec![crate::tools::function_tool("get_weather", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ToolModelType::LLaMa4,
            "llama4".to_string(),
            ToolConfig::for_model_type(&ToolModelType::LLaMa4),
            tools,
            None,
        );

        let calls = futures::executor::block_on(parser.parse_complete_with_fallback(
            r#"[get_weather(location="Vancouver", units="celsius")]"#,
        ));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "Vancouver");
        assert_eq!(args["units"], "celsius");
    }

    #[test]
    fn test_envelope_glm47_xml_format() {
        let tools = vec![crate::tools::function_tool("read", "Read a file").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ToolModelType::GLM4,
            "glm-4.7-flash".to_string(),
            ToolConfig::for_model_type(&ToolModelType::GLM4),
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
    }

    #[test]
    fn test_glm47_display_escape_markers() {
        let tools = vec![crate::tools::function_tool("read", "Read a file").build()];
        let parser = StreamToolParser::new_with_config(
            &ToolModelType::GLM4,
            "glm-4.7-flash".to_string(),
            ToolConfig::for_model_type(&ToolModelType::GLM4),
            tools,
            None,
        );

        let markers = parser.display_escape_markers();
        assert!(markers.iter().any(|m| m == "<arg_key>"));
        assert!(markers.iter().any(|m| m == "</arg_key>"));
        assert!(markers.iter().any(|m| m == "<arg_value>"));
        assert!(markers.iter().any(|m| m == "</arg_value>"));
    }

    #[test]
    fn test_gemma4_parse_bare_value_case_insensitive() {
        assert_eq!(
            StreamToolParser::gemma4_parse_bare_value("TRUE"),
            Value::Bool(true)
        );
        assert_eq!(
            StreamToolParser::gemma4_parse_bare_value("False"),
            Value::Bool(false)
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

    #[test]
    fn test_gemma4_tool_call_parse() {
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
        let calls = futures::executor::block_on(parser.parse_complete_with_fallback(text));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust programming");
        assert_eq!(args["count"], 5);
    }
}
