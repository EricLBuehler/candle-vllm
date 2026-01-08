// src/tools/stream_parser.rs
//! Streaming tool call parser for detecting and buffering tool calls during streaming.
//! Handles model-specific tool call tokens and formats.

use super::{FunctionCall, ToolCall};
use serde_json::Value;
use std::collections::HashSet;
use tokenizers::Tokenizer;
use tracing::{error, info, warn};

/// Model type classification for tool call handling.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolModelType {
    LLaMa,
    Qwen,
    Qwen3MoE,
    Mistral,
    Gemma,
    Gemma3,
    Phi,
    Phi4,
    GLM4,
    Yi,
    StableLM,
    DeepSeek,
}

/// Parser state for streaming tool call detection
#[derive(Debug, Clone, PartialEq)]
pub enum ParserState {
    /// Normal streaming mode - tokens pass through
    Normal,
    /// Potential start tag detected (partial match)
    MaybeStart,
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

/// Configuration for model-specific tool call detection
#[derive(Clone, Debug)]
pub struct ToolConfig {
    pub start_token_ids: HashSet<u32>,
    pub end_token_ids: HashSet<u32>,
    pub start_token_str: String,
    pub end_token_str: String,
}

impl ToolConfig {
    /// Create tool config for a specific model type
    pub fn for_model_type(model_type: &ToolModelType) -> Self {
        let mut start_ids = HashSet::new();
        let mut end_ids = HashSet::new();

        match model_type {
            ToolModelType::LLaMa => {
                start_ids.insert(128010); // <|python_tag|>
                end_ids.insert(128008); // <|eom_id|>
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<|python_tag|>".to_string(),
                    end_token_str: "<|eom_id|>".to_string(),
                }
            }
            ToolModelType::Qwen | ToolModelType::Qwen3MoE => {
                start_ids.insert(151657); // <tool_call>
                end_ids.insert(151658); // </tool_call>
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<tool_call>".to_string(),
                    end_token_str: "</tool_call>".to_string(),
                }
            }
            ToolModelType::Mistral => {
                start_ids.insert(9); // [TOOL_CALLS]
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "[TOOL_CALLS]".to_string(),
                    end_token_str: "]".to_string(),
                }
            }
            ToolModelType::Gemma | ToolModelType::Gemma3 => ToolConfig {
                start_token_ids: start_ids,
                end_token_ids: end_ids,
                start_token_str: "<start_function_call>".to_string(),
                end_token_str: "<end_function_call>".to_string(),
            },
            ToolModelType::Phi
            | ToolModelType::Phi4
            | ToolModelType::GLM4
            | ToolModelType::Yi
            | ToolModelType::StableLM
            | ToolModelType::DeepSeek => ToolConfig {
                start_token_ids: HashSet::new(),
                end_token_ids: HashSet::new(),
                start_token_str: "<tool_call>".to_string(),
                end_token_str: "</tool_call>".to_string(),
            },
        }
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
    pub fn validate_with_tokenizer(&mut self, tokenizer: &Tokenizer, model_type: &ToolModelType) {
        if self.has_start_tokens()
            && !Self::matches_single_token(tokenizer, &self.start_token_str, &self.start_token_ids)
        {
            warn!(
                "Tool start token IDs not supported by tokenizer for model {:?}, falling back to text matching",
                model_type
            );
            self.start_token_ids.clear();
        }

        if self.has_end_tokens()
            && !Self::matches_single_token(tokenizer, &self.end_token_str, &self.end_token_ids)
        {
            warn!(
                "Tool end token IDs not supported by tokenizer for model {:?}, falling back to text matching",
                model_type
            );
            self.end_token_ids.clear();
        }
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

    fn matches_single_token(
        tokenizer: &Tokenizer,
        text: &str,
        token_ids: &HashSet<u32>,
    ) -> bool {
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
}

/// Streaming tool parser that handles tool call detection and buffering
pub struct StreamToolParser {
    config: ToolConfig,
    state: ParserState,
    buffer: String,
    parse_strategy: String,
    accumulated_output: String,
    active_reasoning_end: Option<&'static str>,
    in_code_block: bool,
    tool_call_index: usize,
}

/// Reasoning marker pairs: (start, end)
const REASONING_MARKERS: &[(&str, &str)] = &[
    ("<think>", "</think>"),
    ("<|think|>", "<|/think|>"),
    ("[THINK]", "[/THINK]"),
    ("<thought>", "</thought>"),
];

impl StreamToolParser {
    /// Create a new parser for the given model type
    pub fn new(model_type: ToolModelType) -> Self {
        Self::new_with_config(&model_type, ToolConfig::for_model_type(&model_type))
    }

    /// Create a new parser with a pre-validated tool config
    pub fn new_with_config(
        model_type: &ToolModelType,
        config: ToolConfig,
    ) -> Self {
        let parse_strategy = match model_type {
            ToolModelType::Mistral => "mistral_list",
            _ => "json",
        }
        .to_string();

        Self {
            config,
            state: ParserState::Normal,
            buffer: String::new(),
            parse_strategy,
            accumulated_output: String::new(),
            active_reasoning_end: None,
            in_code_block: false,
            tool_call_index: 0,
        }
    }

    /// Check if currently inside a reasoning block
    pub fn in_reasoning(&self) -> bool {
        self.active_reasoning_end.is_some()
    }

    /// Check if currently inside a code block
    pub fn in_code_block(&self) -> bool {
        self.in_code_block
    }

    /// Get the current parser state
    pub fn state(&self) -> &ParserState {
        &self.state
    }

    /// Get the buffered content
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Process a single incoming token.
    pub fn process_token(&mut self, token_id: u32, token_text: &str) -> StreamResult {
        self.accumulated_output.push_str(token_text);

        if self.active_reasoning_end.is_none() {
            for &(start, end) in REASONING_MARKERS {
                if token_text.contains(start) || self.accumulated_output.ends_with(start) {
                    self.active_reasoning_end = Some(end);
                    break;
                }
            }
        } else if let Some(end_marker) = self.active_reasoning_end {
            if token_text.contains(end_marker) || self.accumulated_output.ends_with(end_marker) {
                self.active_reasoning_end = None;
            }
        }

        if token_text.contains("```") || self.accumulated_output.ends_with("```") {
            self.in_code_block = !self.in_code_block;
        }

        if self.in_reasoning() || self.in_code_block {
            return StreamResult::Content(token_text.to_string());
        }

        match self.state.clone() {
            ParserState::Normal => {
                if self.is_start_token(token_id, token_text) {
                    self.state = ParserState::Buffering;
                    self.buffer.clear();

                    if let Some(pos) = token_text.find(&self.config.start_token_str) {
                        let before = &token_text[..pos];
                        let after = &token_text[pos + self.config.start_token_str.len()..];
                        if !after.is_empty() {
                            self.buffer.push_str(after);
                        }
                        if !before.is_empty() {
                            return StreamResult::Content(before.to_string());
                        }
                    }

                    info!(
                        "Tool call {} ({}) found, start buffering!",
                        token_text, token_id
                    );
                    return StreamResult::Buffering;
                }

                if !self.config.has_start_tokens() {
                    if let Some((prefix, partial)) = self.split_partial_start(token_text) {
                        self.state = ParserState::MaybeStart;
                        self.buffer.clear();
                        self.buffer.push_str(&partial);
                        return if prefix.is_empty() {
                            StreamResult::Buffering
                        } else {
                            StreamResult::Content(prefix)
                        };
                    }
                }

                StreamResult::Content(token_text.to_string())
            }
            ParserState::MaybeStart => {
                self.buffer.push_str(token_text);

                if let Some(tag_pos) = self.buffer.find(&self.config.start_token_str) {
                    let before = self.buffer[..tag_pos].to_string();
                    let after = self.buffer[tag_pos + self.config.start_token_str.len()..].to_string();
                    self.buffer.clear();
                    if !after.is_empty() {
                        self.buffer.push_str(&after);
                    }
                    self.state = ParserState::Buffering;
                    return if before.is_empty() {
                        StreamResult::Buffering
                    } else {
                        StreamResult::Content(before)
                    };
                }

                if self.partial_suffix_len(&self.buffer) > 0 {
                    return StreamResult::Buffering;
                }

                self.state = ParserState::Normal;
                let flushed = self.buffer.clone();
                self.buffer.clear();
                StreamResult::FlushBuffer(flushed)
            }
            ParserState::Buffering => {
                self.buffer.push_str(token_text);

                let end_by_token = self.is_end_token(token_id, token_text);
                let end_reached = end_by_token
                    || self.buffer_has_end_tag()
                    || self.maybe_complete_mistral_list();
                if end_reached {
                    if end_by_token {
                        info!(
                            "Tool call end token detected: {} ({})",
                            token_text, token_id
                        );
                    }
                    info!(
                        "Tool call buffering end, reached {} ({})",
                        token_text, token_id
                    );

                    let tool_calls = self.parse_buffer();
                    let result = if tool_calls.is_empty() {
                        error!("Unable to parse tool call buffer: {}", self.buffer);
                        StreamResult::FlushBuffer(self.buffer.clone())
                    } else {
                        StreamResult::ToolCalls(tool_calls)
                    };
                    self.buffer.clear();
                    self.state = ParserState::Normal;
                    return result;
                }

                StreamResult::Buffering
            }
        }
    }

    /// Finalize parsing when stream ends
    pub fn finalize(&mut self) -> Option<Vec<ToolCall>> {
        match self.state {
            ParserState::Buffering => {
                if self.buffer.is_empty() {
                    self.state = ParserState::Normal;
                    return None;
                }
                let tool_calls = self.parse_buffer();
                if !tool_calls.is_empty() {
                    self.buffer.clear();
                    self.state = ParserState::Normal;
                    return Some(tool_calls);
                }
                self.state = ParserState::Normal;
            }
            ParserState::MaybeStart => {
                self.state = ParserState::Normal;
            }
            ParserState::Normal => {}
        }
        None
    }

    /// Drain the buffer and reset parser state.
    pub fn take_buffer(&mut self) -> String {
        self.state = ParserState::Normal;
        std::mem::take(&mut self.buffer)
    }

    fn is_start_token(&self, id: u32, text: &str) -> bool {
        if self.config.has_start_tokens() {
            return self.config.start_token_ids.contains(&id);
        }
        text.contains(&self.config.start_token_str)
    }

    fn is_end_token(&self, id: u32, text: &str) -> bool {
        if self.config.has_end_tokens() {
            return self.config.end_token_ids.contains(&id);
        }
        if self.parse_strategy == "mistral_list" && self.config.end_token_str == "]" {
            return false;
        }
        text.contains(&self.config.end_token_str)
    }

    fn parse_buffer(&mut self) -> Vec<ToolCall> {
        let mut clean_text = self.buffer.trim().to_string();
        if self.should_strip_end_tag() {
            if let Some(pos) = clean_text.rfind(&self.config.end_token_str) {
                clean_text.truncate(pos);
            }
        }
        let mut calls = Vec::new();

        if self.parse_strategy == "mistral_list" && clean_text.starts_with('[') {
            if let Ok(list) = serde_json::from_str::<Vec<Value>>(&clean_text) {
                for item in list.iter() {
                    if let Some(call) = self.json_to_tool_call(item) {
                        calls.push(call);
                    }
                }
            }
        } else if let Ok(item) = serde_json::from_str::<Value>(&clean_text) {
            if let Some(call) = self.json_to_tool_call(&item) {
                calls.push(call);
            }
        }  else if let Some(repaired) = self.repair_unbalanced_json(&clean_text) {
            if repaired != clean_text {
                tracing::warn!("Tool call JSON missing closing braces; attempting repair");
            }
            if let Ok(item) = serde_json::from_str::<Value>(&repaired) {
                if let Some(call) = self.json_to_tool_call(&item) {
                    calls.push(call);
                }
            }
        }

        calls
    }

    fn split_partial_start(&self, text: &str) -> Option<(String, String)> {
        let tag = &self.config.start_token_str;
        let suffix_len = self.partial_suffix_len(text);
        if suffix_len > 0 && suffix_len < tag.len() {
            let prefix = text[..text.len() - suffix_len].to_string();
            let partial = text[text.len() - suffix_len..].to_string();
            return Some((prefix, partial));
        }
        None
    }

    fn partial_suffix_len(&self, text: &str) -> usize {
        let tag = &self.config.start_token_str;
        let max = std::cmp::min(tag.len(), text.len());
        for i in (1..=max).rev() {
            if text.ends_with(&tag[..i]) {
                return i;
            }
        }
        0
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

    fn should_strip_end_tag(&self) -> bool {
        let end_tag = self.config.end_token_str.as_str();
        if end_tag.is_empty() {
            return false;
        }
        if self.parse_strategy == "mistral_list" && end_tag == "]" {
            return false;
        }
        end_tag.starts_with('<')
    }

    fn repair_unbalanced_json(&self, text: &str) -> Option<String> {
        let trimmed = text.trim();
        if !(trimmed.starts_with('{') || trimmed.starts_with('[')) {
            return None;
        }

        let mut in_string = false;
        let mut escape = false;
        let mut open_braces = 0usize;
        let mut close_braces = 0usize;
        let mut open_brackets = 0usize;
        let mut close_brackets = 0usize;

        for ch in trimmed.chars() {
            if escape {
                escape = false;
                continue;
            }
            match ch {
                '\\' if in_string => {
                    escape = true;
                }
                '"' => {
                    in_string = !in_string;
                }
                '{' if !in_string => open_braces += 1,
                '}' if !in_string => close_braces += 1,
                '[' if !in_string => open_brackets += 1,
                ']' if !in_string => close_brackets += 1,
                _ => {}
            }
        }

        if in_string {
            return None;
        }
        if close_braces > open_braces || close_brackets > open_brackets {
            return None;
        }

        if open_braces == close_braces && open_brackets == close_brackets {
            return None;
        }

        let mut fixed = trimmed.to_string();
        if open_brackets > close_brackets {
            fixed.push_str(&"]".repeat(open_brackets - close_brackets));
        }
        if open_braces > close_braces {
            fixed.push_str(&"}".repeat(open_braces - close_braces));
        }
        Some(fixed)
    }

    fn json_to_tool_call(&mut self, item: &Value) -> Option<ToolCall> {
        let name = item["name"].as_str()?.to_string();
        let arguments = if let Some(args) = item.get("arguments") {
            if args.is_string() {
                args.as_str().unwrap_or("{}").to_string()
            } else {
                args.to_string()
            }
        } else {
            "{}".to_string()
        };

        let call = ToolCall {
            index: Some(self.tool_call_index),
            id: format!("call_{}", uuid::Uuid::new_v4().simple()),
            call_type: "function".to_string(),
            function: FunctionCall { name, arguments },
        };
        self.tool_call_index += 1;
        Some(call)
    }
}
