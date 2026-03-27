pub mod default_conversation;
use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use std::sync::OnceLock;

#[derive(Serialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub num_images: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: String, content: String, num_images: usize) -> Self {
        Self {
            role,
            content,
            num_images,
            tool_calls: None,
            tool_call_id: None,
        }
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

// ---------------------------------------------------------------------------
// Rendered-prompt repair for prefix-cache alignment
// ---------------------------------------------------------------------------

fn generation_prompt_block_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#"(?s)\{%-?\s*if\s+add_generation_prompt\s*-?%\}(?P<body>.*?)\{%-?\s*endif\s*-?%\}"#,
        )
        .expect("valid generation prompt block regex")
    })
}

fn template_string_literal_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"\{\{\-?\s*['"](?P<lit>.*?)['"]\s*-?\}\}"#)
            .expect("valid template string literal regex")
    })
}

fn enable_thinking_false_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(?s)enable_thinking\s+is\s+(defined\s+and\s+enable_thinking\s+is\s+)?false"#)
            .expect("valid enable_thinking false regex")
    })
}

fn decode_template_string_literal(literal: &str) -> String {
    let mut decoded = String::with_capacity(literal.len());
    let mut chars = literal.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('n') => decoded.push('\n'),
                Some('r') => decoded.push('\r'),
                Some('t') => decoded.push('\t'),
                Some('\\') => decoded.push('\\'),
                Some('\'') => decoded.push('\''),
                Some('"') => decoded.push('"'),
                Some(other) => {
                    decoded.push('\\');
                    decoded.push(other);
                }
                None => decoded.push('\\'),
            }
        } else {
            decoded.push(ch);
        }
    }
    decoded
}

fn escaped_special_token(token: &str) -> String {
    if let Some(rest) = token.strip_prefix('<') {
        format!("<\u{200C}{}", rest)
    } else {
        format!("{}\u{200C}", token)
    }
}

/// Extracts the full generation-prompt literal that the template would emit
/// when `add_generation_prompt` is true.
///
/// Handles three patterns found across Qwen model families:
///   1. Single combined literal  (e.g. Qwen3-4B-Thinking)
///   2. Header literal + `enable_thinking` branch  (e.g. Qwen3.5)
///   3. Header literal only, no thinking branch  (e.g. Qwen3-Coder-Next, Qwen3-VL)
fn extract_generation_prompt_literal(chat_template: &str, enable_thinking: bool) -> Option<String> {
    let block_caps = generation_prompt_block_re().captures(chat_template)?;
    let body = block_caps.name("body")?.as_str();

    let lit_re = template_string_literal_re();
    let literals: Vec<String> = lit_re
        .captures_iter(body)
        .filter_map(|c| c.name("lit").map(|m| m.as_str().to_string()))
        .collect();

    if literals.is_empty() {
        return None;
    }

    let has_thinking_branch = body.contains("enable_thinking");
    if !has_thinking_branch {
        return Some(
            literals
                .iter()
                .map(|l| decode_template_string_literal(l))
                .collect::<String>(),
        );
    }

    let thinking_block_start = body.find("enable_thinking")?;

    let header_body = &body[..thinking_block_start];
    let header_literals: Vec<String> = lit_re
        .captures_iter(header_body)
        .filter_map(|c| c.name("lit").map(|m| m.as_str().to_string()))
        .collect();

    let thinking_body = &body[thinking_block_start..];

    let is_false_first = enable_thinking_false_re()
        .is_match(&thinking_body[..thinking_body.find("else").unwrap_or(thinking_body.len())]);

    let branch_literals: Vec<Vec<String>> = thinking_body
        .split("{%- else")
        .chain(thinking_body.split("{% else"))
        .take(2)
        .map(|section| {
            lit_re
                .captures_iter(section)
                .filter_map(|c| c.name("lit").map(|m| m.as_str().to_string()))
                .collect()
        })
        .collect();

    let (disabled_lits, enabled_lits) = if branch_literals.len() >= 2 {
        if is_false_first {
            (&branch_literals[0], &branch_literals[1])
        } else {
            (&branch_literals[1], &branch_literals[0])
        }
    } else {
        return None;
    };

    let suffix_lits = if enable_thinking {
        enabled_lits
    } else {
        disabled_lits
    };

    let mut result = String::new();
    for lit in &header_literals {
        result.push_str(&decode_template_string_literal(lit));
    }
    for lit in suffix_lits {
        result.push_str(&decode_template_string_literal(lit));
    }
    Some(result)
}

fn extract_eot_delimiter(chat_template: &str, eos_token: Option<&str>) -> Option<String> {
    fn eot_re() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r#"(?s)message\.role\s*==\s*['"]assistant['"].*?\{\{\-?\s*['"](?P<eot><\|[^|]+\|>)['"]\s*-?\}\}"#)
                .expect("valid eot regex")
        })
    }

    if let Some(caps) = eot_re().captures(chat_template) {
        if let Some(eot) = caps.name("eot") {
            let decoded = decode_template_string_literal(eot.as_str());
            if decoded.contains("end") || decoded.contains("eot") {
                return Some(decoded);
            }
        }
    }

    eos_token.map(|s| s.to_string())
}

/// Holds extracted repair parameters for a specific template + thinking mode.
#[derive(Debug, Clone)]
pub struct RenderedPromptRepairer {
    assistant_header: String,
    eot_delimiter: String,
    start_marker: Option<String>,
    end_marker: Option<String>,
    scaffold: Option<String>,
}

impl RenderedPromptRepairer {
    /// Build a repairer from a chat template source and thinking mode.
    /// Returns `None` if no repair is possible (e.g. no assistant header found).
    pub fn from_template(
        chat_template: &str,
        eos_token: Option<&str>,
        enable_thinking: bool,
    ) -> Option<Self> {
        let generation_literal = extract_generation_prompt_literal(chat_template, enable_thinking)?;

        if generation_literal.is_empty() {
            return None;
        }

        let eot_delimiter = extract_eot_delimiter(chat_template, eos_token)
            .unwrap_or_else(|| "<|im_end|>".to_string());

        let known_markers = [
            ("<think>", "</think>"),
            ("<thinking>", "</thinking>"),
            ("<reasoning>", "</reasoning>"),
            ("<reflection>", "</reflection>"),
            ("<internal>", "</internal>"),
        ];

        let mut found_start: Option<(usize, &str, &str)> = None;
        for (start, end) in &known_markers {
            if let Some(idx) = generation_literal.find(start) {
                found_start = Some((idx, start, end));
                break;
            }
        }

        let (assistant_header, start_marker, end_marker, scaffold) =
            if let Some((idx, start, end)) = found_start {
                let header = generation_literal[..idx].to_string();
                let suffix = generation_literal[idx..].to_string();
                (
                    header,
                    Some(start.to_string()),
                    Some(end.to_string()),
                    Some(suffix),
                )
            } else if generation_literal.contains("assistant") {
                (generation_literal, None, None, None)
            } else {
                return None;
            };

        if assistant_header.is_empty() {
            return None;
        }

        Some(Self {
            assistant_header,
            eot_delimiter,
            start_marker,
            end_marker,
            scaffold,
        })
    }

    /// Build a repairer from a `DefaultConversation` instance.
    pub fn from_conversation(
        conversation: &default_conversation::DefaultConversation,
        enable_thinking: bool,
    ) -> Option<Self> {
        let source = conversation.template_source()?;
        Self::from_template(source, conversation.eos_token(), enable_thinking)
    }

    /// Returns true if this repairer has a reasoning scaffold to insert.
    pub fn has_reasoning_scaffold(&self) -> bool {
        self.scaffold.is_some()
    }

    /// Apply the repair to a rendered prompt. Returns `None` if no changes needed.
    ///
    /// Inserts the missing scaffold prefix (e.g. `<think>\n`) after every
    /// assistant header whose content does not already start with the
    /// reasoning start marker.  This is safe because the scaffold was part of
    /// the model's input context during generation and must be present when
    /// that context is replayed in a subsequent turn for prefix-cache
    /// alignment.
    ///
    /// No other content is modified — escaped markers, tool-call tags, and
    /// all other text in every assistant block are left byte-identical.
    pub fn repair(&self, base_prompt: &str) -> Option<String> {
        let (Some(start_marker), Some(end_marker), Some(scaffold)) =
            (&self.start_marker, &self.end_marker, &self.scaffold)
        else {
            return None;
        };

        let escaped_end = escaped_special_token(end_marker);

        let opening_scaffold = if let Some(idx) = scaffold.find(end_marker.as_str()) {
            &scaffold[..idx]
        } else {
            scaffold.as_str()
        };

        let mut cursor = 0usize;
        let mut repaired = String::with_capacity(base_prompt.len() + 128);
        let mut changed = false;

        while let Some(rel_idx) = base_prompt[cursor..].find(&self.assistant_header) {
            let header_idx = cursor + rel_idx;
            let after_header = header_idx + self.assistant_header.len();
            repaired.push_str(&base_prompt[cursor..after_header]);

            let rest = &base_prompt[after_header..];
            let block_end = rest.find(self.eot_delimiter.as_str()).unwrap_or(rest.len());
            let block = &rest[..block_end];
            let trimmed = block.trim_start();

            if !trimmed.starts_with(start_marker.as_str()) {
                let has_end = block.contains(end_marker.as_str()) || block.contains(&escaped_end);
                let prefix = if has_end {
                    opening_scaffold
                } else {
                    scaffold.as_str()
                };
                repaired.push_str(prefix);
                changed = true;
            }

            repaired.push_str(block);
            cursor = after_header + block_end;
        }

        if !changed {
            return None;
        }

        repaired.push_str(&base_prompt[cursor..]);
        Some(repaired)
    }

    /// Convenience: try to build a repairer and apply it in one step.
    pub fn try_repair(
        chat_template: &str,
        eos_token: Option<&str>,
        enable_thinking: bool,
        base_prompt: &str,
    ) -> Option<String> {
        let repairer = Self::from_template(chat_template, eos_token, enable_thinking)?;
        repairer.repair(base_prompt)
    }
}

#[cfg(test)]
mod repair_tests {
    use super::*;

    const QWEN35_TEMPLATE: &str = r#"
{%- for message in messages %}
    {%- if message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
"#;

    const QWEN3_4B_TEMPLATE: &str = r#"
{%- for message in messages %}
    {%- if message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}
"#;

    const QWEN3_CODER_TEMPLATE: &str = r#"
{%- for message in messages %}
    {%- if message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;

    #[test]
    fn qwen35_thinking_enabled_extracts_correct_scaffold() {
        let r = RenderedPromptRepairer::from_template(QWEN35_TEMPLATE, Some("<|im_end|>"), true)
            .unwrap();
        assert_eq!(r.assistant_header, "<|im_start|>assistant\n");
        assert_eq!(r.start_marker.as_deref(), Some("<think>"));
        assert_eq!(r.end_marker.as_deref(), Some("</think>"));
        assert_eq!(r.scaffold.as_deref(), Some("<think>\n"));
    }

    #[test]
    fn qwen35_thinking_disabled_extracts_paired_scaffold() {
        let r = RenderedPromptRepairer::from_template(QWEN35_TEMPLATE, Some("<|im_end|>"), false)
            .unwrap();
        assert_eq!(r.assistant_header, "<|im_start|>assistant\n");
        assert_eq!(r.scaffold.as_deref(), Some("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn qwen3_4b_combined_literal_extracts_scaffold() {
        let r = RenderedPromptRepairer::from_template(QWEN3_4B_TEMPLATE, Some("<|im_end|>"), true)
            .unwrap();
        assert_eq!(r.assistant_header, "<|im_start|>assistant\n");
        assert_eq!(r.scaffold.as_deref(), Some("<think>\n"));
    }

    #[test]
    fn qwen3_coder_no_thinking_returns_no_scaffold() {
        let r =
            RenderedPromptRepairer::from_template(QWEN3_CODER_TEMPLATE, Some("<|im_end|>"), true)
                .unwrap();
        assert_eq!(r.assistant_header, "<|im_start|>assistant\n");
        assert!(!r.has_reasoning_scaffold());
    }

    #[test]
    fn no_repair_needed_when_template_has_no_thinking() {
        let prompt = "<|im_start|>assistant\nHello world<|im_end|>\n";
        let result = RenderedPromptRepairer::try_repair(
            QWEN3_CODER_TEMPLATE,
            Some("<|im_end|>"),
            true,
            prompt,
        );
        assert!(result.is_none());
    }

    #[test]
    fn repair_inserts_missing_think_prefix_enabled() {
        let prompt = "<|im_start|>user\nhi<|im_end|>\n\
                       <|im_start|>assistant\nThinking...\n</think>\nhello<|im_end|>\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), true, prompt)
                .unwrap();
        assert!(repaired.contains("<|im_start|>assistant\n<think>\nThinking..."));
    }

    #[test]
    fn repair_inserts_full_paired_scaffold_when_disabled_and_no_end_marker() {
        let prompt = "<|im_start|>assistant\nVisible answer<|im_end|>\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), false, prompt)
                .unwrap();
        assert!(
            repaired.starts_with("<|im_start|>assistant\n<think>\n\n</think>\n\nVisible answer")
        );
    }

    #[test]
    fn repair_inserts_opening_scaffold_when_end_marker_present() {
        let prompt = "<|im_start|>assistant\nThinking...\n</think>\nhello<|im_end|>\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), false, prompt)
                .unwrap();
        assert!(repaired.starts_with("<|im_start|>assistant\n<think>\n\nThinking..."));
    }

    #[test]
    fn no_repair_when_escaped_end_marker_but_start_present() {
        let prompt =
            "<|im_start|>assistant\n<think>\nreasoning\n<\u{200C}/think>\nanswer<|im_end|>\n";
        let result =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), true, prompt);
        assert!(
            result.is_none(),
            "escaped markers in completed blocks should not be altered"
        );
    }

    #[test]
    fn repair_with_qwen3_4b_combined_template() {
        let prompt = "<|im_start|>assistant\nSome reasoning\n</think>\nhello<|im_end|>\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN3_4B_TEMPLATE, Some("<|im_end|>"), true, prompt)
                .unwrap();
        assert!(repaired.contains("<|im_start|>assistant\n<think>\nSome reasoning"));
    }

    #[test]
    fn repair_no_change_when_prefix_already_present() {
        let prompt = "<|im_start|>assistant\n<think>\nreasoning\n</think>\nhello<|im_end|>\n";
        let result =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), true, prompt);
        assert!(result.is_none());
    }

    #[test]
    fn extract_generation_literal_qwen35_enabled() {
        let literal = extract_generation_prompt_literal(QWEN35_TEMPLATE, true).unwrap();
        assert_eq!(literal, "<|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn extract_generation_literal_qwen35_disabled() {
        let literal = extract_generation_prompt_literal(QWEN35_TEMPLATE, false).unwrap();
        assert_eq!(literal, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }

    #[test]
    fn extract_generation_literal_qwen3_4b() {
        let literal = extract_generation_prompt_literal(QWEN3_4B_TEMPLATE, true).unwrap();
        assert_eq!(literal, "<|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn extract_generation_literal_qwen3_coder() {
        let literal = extract_generation_prompt_literal(QWEN3_CODER_TEMPLATE, true).unwrap();
        assert_eq!(literal, "<|im_start|>assistant\n");
    }

    #[test]
    fn eot_delimiter_extracted_from_template() {
        let eot = extract_eot_delimiter(QWEN35_TEMPLATE, Some("<|im_end|>"));
        assert_eq!(eot.as_deref(), Some("<|im_end|>"));
    }

    #[test]
    fn eot_delimiter_falls_back_to_eos_token() {
        let eot = extract_eot_delimiter("no matching pattern here", Some("<|endoftext|>"));
        assert_eq!(eot.as_deref(), Some("<|endoftext|>"));
    }

    #[test]
    fn repair_all_assistant_blocks_missing_prefix() {
        let prompt = "<|im_start|>user\nhi<|im_end|>\n\
                       <|im_start|>assistant\nOld answer\n</think>\nvisible<|im_end|>\n\
                       <|im_start|>user\nmore<|im_end|>\n\
                       <|im_start|>assistant\nNew answer<|im_end|>\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), true, prompt)
                .unwrap();
        assert!(
            repaired.contains("assistant\n<think>\nOld answer"),
            "first assistant block should get the scaffold prefix"
        );
        assert!(
            repaired.contains("assistant\n<think>\nNew answer"),
            "last assistant block should also get the scaffold prefix"
        );
    }

    #[test]
    fn repair_skips_blocks_that_already_have_prefix() {
        let prompt = "<|im_start|>assistant\n<think>\nreasoning\n</think>\nanswer<|im_end|>\n\
                       <|im_start|>user\nok<|im_end|>\n\
                       <|im_start|>assistant\nNew response<|im_end|>\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), true, prompt)
                .unwrap();
        assert!(
            repaired.contains("assistant\n<think>\nreasoning"),
            "first block already had prefix, content unchanged"
        );
        assert!(
            repaired.contains("assistant\n<think>\nNew response"),
            "second block should get scaffold prefix"
        );
    }

    #[test]
    fn repair_preserves_escaped_markers_in_prior_turns() {
        let prompt =
            "<|im_start|>assistant\n<think>\nreasoning\n<\u{200C}/think>\nanswer<|im_end|>\n\
                       <|im_start|>user\nok<|im_end|>\n\
                       <|im_start|>assistant\nNew response<|im_end|>\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), true, prompt)
                .unwrap();
        assert!(
            repaired.contains("<\u{200C}/think>"),
            "escaped markers in prior turns must be preserved"
        );
        assert!(
            repaired.contains("assistant\n<think>\nNew response"),
            "last block should get scaffold prefix"
        );
    }

    #[test]
    fn repair_last_block_no_eot_trailing() {
        let prompt = "<|im_start|>user\nhi<|im_end|>\n\
                       <|im_start|>assistant\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), true, prompt)
                .unwrap();
        assert_eq!(
            repaired,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn repair_multi_turn_tool_call_scenario() {
        let prompt = "<|im_start|>user\nhi<|im_end|>\n\
                       <|im_start|>assistant\n<tool_call>...</tool_call><|im_end|>\n\
                       <|im_start|>tool\nresult<|im_end|>\n\
                       <|im_start|>assistant\n";
        let repaired =
            RenderedPromptRepairer::try_repair(QWEN35_TEMPLATE, Some("<|im_end|>"), true, prompt)
                .unwrap();
        assert!(
            repaired.contains("assistant\n<think>\n<tool_call>"),
            "tool-call block gets scaffold prefix (model saw <think>\\n before tool call)"
        );
        assert!(
            repaired.ends_with("assistant\n<think>\n"),
            "last empty block gets scaffold prefix"
        );
    }
}
