// src/tools/stream_parser.rs
//! Streaming tool call parser — detects and buffers tool calls during streaming.
//! Ported from vllm.rs server/parser.rs with tool-parser crate integration.

use super::{Tool, ToolCall};
pub use openai_protocol::common::{Function, FunctionCallResponse as FunctionCall};
use serde_json::{Map, Value};
use std::collections::HashSet;
use tokenizers::Tokenizer;
use tool_parser::{
    types::{StreamingParseResult, ToolCallItem},
    ParserFactory, ToolParser as ExternalToolParser,
};

const INVALID_TOOL_CALL_NAME: &str = "__invalid_tool_call__";
const INVALID_TOOL_CALL_RAW_ARGUMENT_KEY: &str = "_raw_tool_call";

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

pub fn detect_prefilled_reasoning_end_marker(prompt: &str) -> Option<String> {
    let trimmed = prompt.trim_end();
    for &(start, end) in REASONING_MARKERS {
        if trimmed.ends_with(start) {
            return Some(end.to_string());
        }
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
    parse_strategy: String,
    parser: Box<dyn ExternalToolParser>,
    tools: Vec<openai_protocol::common::Tool>,
    streaming_calls: Vec<StreamingToolCallState>,
    accumulated_output: String,
    active_reasoning_end: Option<String>,
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
        let openai_tools = to_openai_tools(&tools);
        let parse_strategy = match model_type {
            ToolModelType::Mistral => "mistral_list",
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
            parse_strategy,
            parser,
            tools: openai_tools,
            streaming_calls: Vec::new(),
            accumulated_output: String::new(),
            active_reasoning_end: None,
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
    pub fn in_code_block(&self) -> bool {
        self.in_code_block
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

    pub fn process_token(&mut self, token_id: u32, token_text: &str) -> StreamResult {
        self.saw_buffer_parse_activity = false;
        self.accumulated_output.push_str(token_text);

        // Track code blocks
        let mut code_block_count = 0;
        for line in self.accumulated_output.lines() {
            if line.trim().starts_with("```") {
                code_block_count += 1;
            }
        }
        self.in_code_block = code_block_count % 2 == 1;

        // Track reasoning blocks
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

        match self.state.clone() {
            ParserState::Normal => {
                if self.in_reasoning() || self.in_code_block {
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
                        let invalid_tool_calls =
                            self.build_invalid_tool_calls_from_buffer(&self.buffer);
                        if !invalid_tool_calls.is_empty() {
                            tracing::warn!(
                                "Unable to fully parse tool call; forwarding raw tool-call payload to client"
                            );
                            StreamResult::ToolCalls(invalid_tool_calls)
                        } else if had_partial {
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
            && !self.has_ambiguous_incomplete_end_marker()
            && (self.can_recover_incomplete_buffered_tool_calls()
                || self.can_recover_fallback_buffered_tool_calls());

        self.parser.reset();
        self.buffer.clear();
        self.state = ParserState::Normal;
        self.streaming_calls.clear();
        self.buffer_had_parse_activity = false;
        self.pending_end_marker_candidate = false;
        let drop_bare_start_marker = self.should_drop_bare_start_marker();
        self.buffer_started_from_special_token = false;
        self.buffer_saw_non_marker_content = false;

        if tool_calls.is_empty() || (!strict_complete && !recoverable) {
            if drop_bare_start_marker {
                tracing::warn!(
                    "Dropping buffered bare tool-call marker at stream end without flushing text"
                );
                return Some(BufferedFinalizeResult::FlushBuffer(String::new()));
            }
            let invalid_tool_calls = self.build_invalid_tool_calls_from_buffer(&buffered_text);
            if !invalid_tool_calls.is_empty() {
                tracing::warn!(
                    "Buffered tool call could not be finalized; forwarding raw tool-call payload to client"
                );
                return Some(BufferedFinalizeResult::ToolCalls(invalid_tool_calls));
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
        std::mem::take(&mut self.buffer)
    }

    pub fn reparse_accumulated_output(&self) -> Vec<ToolCall> {
        futures::executor::block_on(self.parse_complete_with_fallback(self.accumulated_output()))
    }

    pub async fn parse_complete_with_fallback(&self, text: &str) -> Vec<ToolCall> {
        let mut parsed_calls = match self.parser.parse_complete(text).await {
            Ok((_normal_text, calls)) => calls,
            Err(err) => {
                tracing::warn!("Tool parse failed: {:?}", err);
                Vec::new()
            }
        };

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

    fn build_invalid_tool_calls_from_buffer(&self, buffer: &str) -> Vec<ToolCall> {
        let trimmed = buffer.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        let looks_like_tool_call = self.parser.has_tool_markers(trimmed)
            || self.contains_tool_markup(trimmed)
            || (!self.config.start_token_str.is_empty()
                && trimmed.contains(&self.config.start_token_str))
            || (!self.config.end_token_str.is_empty()
                && trimmed.contains(&self.config.end_token_str))
            || trimmed.contains("<function=")
            || trimmed.contains("<parameter=");
        if !looks_like_tool_call {
            return Vec::new();
        }

        if let Some(call) = self.invalid_tool_call_from_xml_buffer(trimmed) {
            return vec![call];
        }
        if let Some(call) = self.invalid_tool_call_from_json_buffer(trimmed) {
            return vec![call];
        }

        let payload = self.strip_tool_tags(trimmed);
        let raw_payload = payload.trim();
        if raw_payload.is_empty() {
            return Vec::new();
        }
        vec![crate::tools::new_tool_call(
            crate::tools::generate_tool_call_id(),
            INVALID_TOOL_CALL_NAME,
            serde_json::json!({
                INVALID_TOOL_CALL_RAW_ARGUMENT_KEY: raw_payload,
            })
            .to_string(),
        )]
    }

    fn invalid_tool_call_from_json_buffer(&self, buffer: &str) -> Option<ToolCall> {
        let payload = self.strip_tool_tags(buffer);
        let payload = payload.trim();
        if payload.is_empty() || payload.contains("<function=") {
            return None;
        }

        let value = serde_json::from_str::<Value>(payload).ok()?;
        let object = value.as_object()?;
        let name = object
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .unwrap_or(INVALID_TOOL_CALL_NAME);
        let arguments = object
            .get("arguments")
            .and_then(|arguments| match arguments {
                Value::String(raw) => Some(raw.clone()),
                other => serde_json::to_string(other).ok(),
            })
            .unwrap_or_else(|| {
                serde_json::json!({
                    INVALID_TOOL_CALL_RAW_ARGUMENT_KEY: payload,
                })
                .to_string()
            });

        Some(crate::tools::new_tool_call(
            crate::tools::generate_tool_call_id(),
            name,
            arguments,
        ))
    }

    fn invalid_tool_call_from_xml_buffer(&self, buffer: &str) -> Option<ToolCall> {
        let function_name = Self::extract_xml_function_name(buffer)
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| INVALID_TOOL_CALL_NAME.to_string());
        if function_name.is_empty() {
            return None;
        }

        let recovered = if function_name == INVALID_TOOL_CALL_NAME {
            Map::new()
        } else {
            Self::extract_xml_parameters_for_function(buffer, &function_name)
                .into_iter()
                .map(|(key, value)| (key, Value::String(value)))
                .collect::<Map<String, Value>>()
        };
        let arguments = if recovered.is_empty() {
            let payload = self.strip_tool_tags(buffer);
            let raw_payload = payload.trim();
            if raw_payload.is_empty() {
                return None;
            }
            serde_json::json!({
                INVALID_TOOL_CALL_RAW_ARGUMENT_KEY: raw_payload,
            })
            .to_string()
        } else {
            Value::Object(recovered).to_string()
        };

        Some(crate::tools::new_tool_call(
            crate::tools::generate_tool_call_id(),
            function_name,
            arguments,
        ))
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

    fn can_recover_fallback_buffered_tool_calls(&self) -> bool {
        let trimmed = self.buffer.trim();
        if trimmed.is_empty() {
            return false;
        }
        if !self.config.start_token_str.is_empty() && trimmed.contains(&self.config.start_token_str)
        {
            return true;
        }
        self.parser.has_tool_markers(trimmed)
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

        if block.contains("<function=") || block.contains("<parameter=") {
            let Some(fs) = block.find("<function=") else {
                return false;
            };
            let func_section = &block[fs..];
            let Some(fe_rel) = func_section.rfind("</function>") else {
                return false;
            };
            let func_block = &block[fs..fs + fe_rel + "</function>".len()];
            return Self::has_balanced_parameter_tags(func_block);
        }
        if inner.is_empty() {
            return false;
        }
        serde_json::from_str::<Value>(inner).is_ok()
    }

    fn has_balanced_parameter_tags(function_block: &str) -> bool {
        let mut idx = 0usize;
        let mut open_count = 0usize;
        const OPEN: &str = "<parameter=";
        const CLOSE: &str = "</parameter>";
        while idx < function_block.len() {
            let open_pos = function_block[idx..].find(OPEN).map(|p| idx + p);
            let close_pos = function_block[idx..].find(CLOSE).map(|p| idx + p);
            match (open_pos, close_pos) {
                (None, None) => break,
                (Some(op), None) => {
                    open_count += 1;
                    idx = op + OPEN.len();
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
                        idx = op + OPEN.len();
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
            ToolModelType::Mistral => "mistral",
            ToolModelType::Qwen | ToolModelType::Qwen3MoE => {
                if model_lower.contains("coder") || model_lower.contains("qwen3.5") {
                    "qwen_coder"
                } else {
                    "qwen"
                }
            }
            ToolModelType::Gemma | ToolModelType::Gemma3 => "json",
            ToolModelType::Phi | ToolModelType::Phi4 => "qwen",
            ToolModelType::GLM4 => "json",
            ToolModelType::Yi | ToolModelType::StableLM => "qwen",
            ToolModelType::DeepSeek => "deepseek",
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
        if self.config.start_token_str.contains("tool_call")
            && self.config.end_token_str.contains("tool_call")
        {
            markers.extend(
                ["<function=", "</function>", "<parameter=", "</parameter>"]
                    .into_iter()
                    .map(|s| s.to_string()),
            );
        }
        markers
    }

    fn recover_streaming_arguments_from_buffer(&mut self) {
        if self.streaming_calls.is_empty() || !self.buffer.contains("<parameter=") {
            return;
        }
        for state in &mut self.streaming_calls {
            let Some(name) = state.name.as_deref() else {
                continue;
            };
            let recovered = Self::extract_xml_parameters_for_function(&self.buffer, name);
            if recovered.is_empty() {
                continue;
            }
            let mut args_obj = match serde_json::from_str::<Value>(state.arguments.trim()) {
                Ok(Value::Object(map)) => map,
                _ => Map::new(),
            };
            let mut merged_any = false;
            for (key, value) in recovered {
                if !args_obj.contains_key(&key) && !value.is_empty() {
                    args_obj.insert(key, Value::String(value));
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
    ) -> std::collections::HashMap<String, String> {
        let mut recovered = std::collections::HashMap::new();
        let function_tag = format!("<function={}>", function_name);
        let alt_function_tag = format!("<function=\"{}\">", function_name);
        let Some(func_start) = buffer
            .rfind(&function_tag)
            .or_else(|| buffer.rfind(&alt_function_tag))
        else {
            return recovered;
        };
        let section = &buffer[func_start..];
        let mut cursor = 0usize;
        const PARAM_PREFIX: &str = "<parameter=";
        const PARAM_END: &str = "</parameter>";
        while let Some(rel) = section[cursor..].find(PARAM_PREFIX) {
            let tag_start = cursor + rel;
            let name_start = tag_start + PARAM_PREFIX.len();
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
                    .to_string();
                recovered.insert(parameter_name, value);
                cursor = value_end + PARAM_END.len();
            } else {
                let value = section[value_start..]
                    .trim_matches(|c| c == '\n' || c == '\r')
                    .to_string();
                recovered.insert(parameter_name, value);
                break;
            }
        }
        recovered
    }

    fn extract_xml_function_name(buffer: &str) -> Option<String> {
        let open_tag = "<function=";
        let start = buffer.rfind(open_tag)?;
        let section = &buffer[start + open_tag.len()..];
        let end = section.find('>')?;
        let name = section[..end]
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .trim();
        if name.is_empty() {
            None
        } else {
            Some(name.to_string())
        }
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
    fn test_finalize_preserves_invalid_tool_call_payload() {
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
            Some(BufferedFinalizeResult::ToolCalls(calls)) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, INVALID_TOOL_CALL_NAME);
                assert!(calls[0]
                    .function
                    .arguments
                    .contains(INVALID_TOOL_CALL_RAW_ARGUMENT_KEY));
                assert!(calls[0].function.arguments.contains("AGENTS.md"));
            }
            other => panic!("Expected invalid tool call passthrough, got {:?}", other),
        }
    }

    #[test]
    fn test_end_marker_then_finalize_preserves_invalid_tool_call_payload() {
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
            Some(BufferedFinalizeResult::ToolCalls(calls)) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, INVALID_TOOL_CALL_NAME);
                assert!(calls[0].function.arguments.contains("AGENTS.md"));
            }
            other => panic!(
                "Expected invalid tool call passthrough after finalize, got {:?}",
                other
            ),
        }
    }
}
