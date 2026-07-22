use crate::tools::Tool;
use minijinja::{context, Environment};
use regex::Regex;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub num_images: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn new(role: String, content: String, num_images: usize) -> Self {
        Message {
            role,
            content,
            num_images,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ApplyChatTemplateError {
    #[error("failed to add chat template")]
    AddTemplateError(#[source] minijinja::Error),
    #[error("failed to get chat template")]
    GetTemplateError(#[source] minijinja::Error),
    #[error("failed to render chat template")]
    RenderTemplateError(#[source] minijinja::Error),
}

fn escape_special_tokens_in_text(
    content: &str,
    escape_tokens: &[String],
    preserve_tokens: &[String],
) -> String {
    if escape_tokens.is_empty() || content.is_empty() {
        return content.to_string();
    }

    // Protect model-required markers (e.g. multimodal image markers) from
    // escaping by swapping them to temporary sentinels first.
    let mut protected = content.to_string();
    let mut sentinels: Vec<(String, String)> = Vec::new();
    for (idx, token) in preserve_tokens.iter().enumerate() {
        if token.is_empty() || !protected.contains(token) {
            continue;
        }
        let sentinel = format!("__CANDLE_VLLM_PRESERVE_TOKEN_{}__", idx);
        protected = protected.replace(token, &sentinel);
        sentinels.push((sentinel, token.clone()));
    }

    let mut escaped = protected;
    for token in escape_tokens {
        if token.is_empty() {
            continue;
        }
        // Insert ZWNJ after '<' so textual tags remain visible but cannot be
        // recognized as tokenizer special/added-token spans.
        let escaped_token = if let Some(rest) = token.strip_prefix('<') {
            format!("<\u{200C}{}", rest)
        } else {
            format!("{}\u{200C}", token)
        };
        escaped = escaped.replace(token, &escaped_token);
    }

    for (sentinel, token) in sentinels {
        escaped = escaped.replace(&sentinel, &token);
    }

    escaped
}

fn should_escape_marker(token: &str) -> bool {
    if token.is_empty() || token.len() < 3 {
        return false;
    }
    let Some(first) = token.chars().next() else {
        return false;
    };
    matches!(first, '<' | '[' | '{' | '(') || token.contains('|')
}

fn should_escape_nested_xml_tool_markers(tool_markers: &[&str]) -> bool {
    tool_markers
        .iter()
        .any(|marker| marker.starts_with('<') && marker.contains("tool_call"))
}

fn template_supports_thinking(chat_template: Option<&str>) -> bool {
    let Some(template) = chat_template else {
        return false;
    };
    template.contains("enable_thinking")
        || REASONING_BLOCK_PAIRS
            .iter()
            .any(|(open, close)| template.contains(open) || template.contains(close))
}

fn normalize_template_source(source: &str) -> String {
    static TOJSON_ENSURE_ASCII_RE: OnceLock<Regex> = OnceLock::new();
    let regex = TOJSON_ENSURE_ASCII_RE.get_or_init(|| {
        Regex::new(
            r#"(?x)
            \|
            (?P<ws>\s*)
            tojson
            \(
                \s*ensure_ascii\s*=\s*(?:false|true|False|True)\s*
            \)
        "#,
        )
        .expect("valid tojson ensure_ascii regex")
    });
    regex.replace_all(source, "|${ws}tojson").into_owned()
}

#[derive(Clone, Debug)]
pub struct ChatTemplate {
    system_message: Option<String>,
    chat_template: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    messages: Vec<Message>,
    escape_tokens: Vec<String>,
    preserve_tokens: Vec<String>,
    add_generation_prompt: bool,
    enable_thinking: bool,
}

#[derive(Default, Clone)]
pub enum SeparatorStyle {
    #[default]
    AddColonSingle,
    AddColonTwo,
    AddColonSpaceSingle,
    NoColonSingle,
    NoColonTwo,
    AddNewLineSingle,
    Llama,
    Llama3,
    Phi,
    Qwen,
    Gemma,
    Mistral,
    Yi,
    StableLM,
    ChatGLM,
    ChatML,
    ChatIntern,
    Dolly,
    RWKV,
    Phoenix,
    Robin,
    FalconChat,
    GLM,
}

pub struct DefaultConversationSeparators {
    pub sep: String,
    pub sep2: Option<String>,
}

/// Candle engine adapter around xInfer's canonical `ChatTemplate`.
/// All rendering, escaping and replay-prefix behavior stays in `ChatTemplate`.
#[derive(Clone)]
pub struct DefaultConversation {
    inner: ChatTemplate,
}

const REASONING_BLOCK_PAIRS: [(&str, &str); 5] = [
    ("<think>", "</think>"),
    ("<|think|>", "<|/think|>"),
    ("[THINK]", "[/THINK]"),
    ("<thought>", "</thought>"),
    ("<|channel>", "<channel|>"),
];

/// Extract reasoning content from assistant message content.
///
/// Collects text from ALL `<think>...</think>` blocks and returns the
/// combined reasoning plus the remaining content after the last close tag.
/// Empty blocks (from replay suffix patterns) are skipped.
///
/// Also handles the MiniMax-style pattern where the model only generates a
/// close marker (e.g. `</think>`) without a matching open marker (the open
/// marker was injected by the chat template's generation prompt). In that
/// case everything before the first close marker is treated as reasoning.
///
/// Returns `(reasoning_content, remaining_content)` if any matched pair is found.
pub fn extract_reasoning_content(content: &str) -> Option<(String, String)> {
    for &(open, close) in &REASONING_BLOCK_PAIRS {
        if !content.contains(close) {
            continue;
        }

        // Standard paired extraction: <think>...</think>
        if content.contains(open) {
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

        // Standalone close marker (MiniMax-style): the open marker was part
        // of the generation prompt, so the model output starts directly with
        // reasoning text followed by the close marker.
        let Some(close_idx) = content.find(close) else {
            continue;
        };
        let reasoning = content[..close_idx].trim_matches('\n').to_string();
        let remaining = content[close_idx + close.len()..]
            .trim_start_matches('\n')
            .to_string();
        return Some((reasoning, remaining));
    }
    None
}

fn strip_generation_assistant_header(suffix_text: &str) -> &str {
    let Some((first_line, remainder)) = suffix_text.split_once('\n') else {
        return suffix_text;
    };

    // Standard Qwen/ChatML-style: `<|im_start|>assistant` or minimax `ai` suffix:
    if first_line.ends_with("assistant") || first_line.ends_with("ai") {
        return remainder;
    }

    // MiniMax-style role marker: `]~b]ai` (the `]~b]` token + role name)
    if first_line.contains("]~b]") || first_line.ends_with("ai") {
        return remainder;
    }

    suffix_text
}

impl ChatTemplate {
    pub fn collect_escape_tokens(tokenizer: &Tokenizer, tool_markers: &[&str]) -> Vec<String> {
        let mut tokens = tokenizer
            .get_added_tokens_decoder()
            .into_values()
            .filter_map(|added| {
                let content = added.content;
                if added.special || should_escape_marker(&content) {
                    Some(content)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for marker in tool_markers {
            if should_escape_marker(marker) {
                tokens.push((*marker).to_string());
            }
        }

        if should_escape_nested_xml_tool_markers(tool_markers) {
            tokens.extend(
                ["<function=", "</function>", "<parameter=", "</parameter>"]
                    .into_iter()
                    .map(|s| s.to_string()),
            );
        }

        // Escape longest markers first to avoid partial replacement ordering issues.
        tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        tokens.dedup();
        tokens
    }

    pub fn new(
        system_message: Option<String>,
        chat_template: Option<String>,
        bos_token: Option<String>,
        eos_token: Option<String>,
        prompt: Option<String>,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Self {
        let mut template = ChatTemplate {
            system_message: system_message.clone(),
            chat_template,
            bos_token,
            eos_token,
            messages: Vec::new(),
            escape_tokens: Vec::new(),
            preserve_tokens: Vec::new(),
            add_generation_prompt,
            enable_thinking,
        };
        if system_message.is_some() {
            template.append_message(
                "system".to_string(),
                template.system_message.clone().unwrap_or_default(),
                0,
            );
        }
        if let Some(prompt) = prompt {
            template.append_message("user".to_string(), prompt, 0);
        }
        template
    }

    pub fn append_message(&mut self, role: String, content: String, num_images: usize) {
        self.messages.push(Message {
            role,
            content,
            num_images,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }

    pub fn set_messages(&mut self, messages: &Vec<Message>) {
        self.messages.clear();
        self.messages.extend(messages.clone());
    }

    pub fn set_enable_thinking(&mut self, enable: bool) {
        self.enable_thinking = enable;
    }

    pub fn enable_thinking(&self) -> bool {
        self.enable_thinking
    }

    pub fn set_escape_tokens(&mut self, mut tokens: Vec<String>) {
        tokens.retain(|token| !token.is_empty());
        tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        tokens.dedup();
        self.escape_tokens = tokens;
    }

    pub fn set_preserve_tokens(&mut self, mut tokens: Vec<String>) {
        tokens.retain(|token| !token.is_empty());
        tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        tokens.dedup();
        self.preserve_tokens = tokens;
    }

    pub fn escape_text(&self, content: &str) -> String {
        escape_special_tokens_in_text(content, &self.escape_tokens, &self.preserve_tokens)
    }

    #[allow(dead_code)]
    fn clear_message(&mut self) {
        self.messages.clear()
    }

    fn escaped_messages_for_render(&self) -> Vec<Message> {
        // For assistant messages: extract reasoning from <think>...</think>
        // into `reasoning_content` and strip it from `content`.  This lets
        // templates that support `message.reasoning_content` (e.g. Qwen3)
        // use it directly, while templates that parse <think> from content
        // can still find it if needed.
        //
        // We always strip reasoning markers from content to avoid the
        // "double-think" bug and to prevent escape_text from mangling
        // them (ZWNJ insertion would make them invisible to the template).
        //
        // For thinking models: when an assistant message has tool_calls but
        // no reasoning_content (e.g. because the client didn't send it back),
        // inject a placeholder so the template renders non-empty <think> blocks.
        // Empty <think></think> blocks across multiple turns degrade model
        // quality and cause premature stop in multi-turn tool call sessions.
        let need_escape = !self.escape_tokens.is_empty();
        let is_thinking_model =
            self.enable_thinking && template_supports_thinking(self.chat_template.as_deref());
        self.messages
            .iter()
            .map(|message| {
                let mut escaped = message.clone();
                match escaped.role.as_str() {
                    "system" | "developer" => {}
                    "assistant" | "ai" => {
                        if let Some((reasoning, remaining)) =
                            extract_reasoning_content(&escaped.content)
                        {
                            if escaped.reasoning_content.is_none() {
                                escaped.reasoning_content = Some(reasoning);
                            }
                            escaped.content = remaining;
                        }
                        if is_thinking_model
                            && escaped.reasoning_content.is_none()
                            && escaped.tool_calls.is_some()
                        {
                            escaped.reasoning_content = Some("...".to_string());
                        }
                        if need_escape {
                            escaped.content = self.escape_text(&escaped.content);
                        }
                    }
                    _ => {
                        if need_escape {
                            escaped.content = self.escape_text(&escaped.content);
                        }
                    }
                }
                escaped
            })
            .collect()
    }

    fn render_chat_template(
        &self,
        tools: &Vec<Tool>,
        add_generation_prompt: bool,
        log: bool,
    ) -> Result<String, ApplyChatTemplateError> {
        if self.chat_template.is_none() {
            return Err(ApplyChatTemplateError::GetTemplateError(
                minijinja::Error::new(minijinja::ErrorKind::CannotDeserialize, "Not found!"),
            ));
        }
        let mut env = Environment::new();
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        let template = normalize_template_source(self.chat_template.as_ref().unwrap());
        let mut template = template.replace("[::-1]", "|reverse");
        if template.find("{{ meta }}").is_some() {
            template = template.replace("{%- set meta = message.get(\"metadata\", \"\") %}", "");
            template = template.replace("{{ meta }}", "");
        }
        env.add_template("candle-vllm", template.as_str())
            .map_err(ApplyChatTemplateError::AddTemplateError)?;
        let template = env
            .get_template("candle-vllm")
            .map_err(ApplyChatTemplateError::GetTemplateError)?;

        let render_messages = self.escaped_messages_for_render();
        if log {
            tracing::info!("messages {:?}", render_messages);
        }
        template
            .render(context! {
              messages => render_messages,
              add_generation_prompt => add_generation_prompt,
              bos_token => self.bos_token,
              eos_token => self.eos_token,
              enable_thinking => self.enable_thinking,
              tools => tools,
            })
            .map_err(ApplyChatTemplateError::RenderTemplateError)
    }

    pub fn apply_chat_template(
        &self,
        tools: &Vec<Tool>,
        log: bool,
    ) -> Result<String, ApplyChatTemplateError> {
        self.render_chat_template(tools, self.add_generation_prompt, log)
    }

    pub fn generation_prompt_replay_suffix(
        &self,
        tools: &Vec<Tool>,
        rendered_prompt: &str,
    ) -> Option<String> {
        if !self.add_generation_prompt {
            return None;
        }

        let prompt_without_generation = self.render_chat_template(tools, false, false).ok()?;
        let suffix_text = rendered_prompt
            .strip_prefix(&prompt_without_generation)?
            .to_string();
        let suffix_text = strip_generation_assistant_header(&suffix_text).to_string();
        if suffix_text.is_empty() {
            return None;
        }
        Some(suffix_text)
    }

    /// Get the template string for external use (e.g., validation checks)
    pub fn get_template_string(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }
}

impl DefaultConversation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _name: String,
        chat_template: Option<String>,
        messages: Vec<Message>,
        _sep_style: SeparatorStyle,
        bos_token: Option<String>,
        eos_token: Option<String>,
        _roles: (String, String),
        _seps: DefaultConversationSeparators,
    ) -> Self {
        let mut inner =
            ChatTemplate::new(None, chat_template, bos_token, eos_token, None, true, true);
        inner.set_messages(&messages);
        Self { inner }
    }

    pub fn collect_escape_tokens(tokenizer: &Tokenizer, tool_markers: &[&str]) -> Vec<String> {
        ChatTemplate::collect_escape_tokens(tokenizer, tool_markers)
    }

    pub fn set_escape_tokens(&mut self, tokens: Vec<String>) {
        self.inner.set_escape_tokens(tokens);
    }

    pub fn set_preserve_tokens(&mut self, tokens: Vec<String>) {
        self.inner.set_preserve_tokens(tokens);
    }

    pub fn set_system_message(&mut self, system_message: Option<String>) {
        self.inner.system_message = system_message.clone();
        self.inner
            .messages
            .retain(|message| message.role != "system");
        if let Some(content) = system_message {
            self.inner
                .messages
                .insert(0, Message::new("system".into(), content, 0));
        }
    }

    pub fn append_message(&mut self, role: String, content: String) {
        self.inner.append_message(role, content, 0);
    }

    pub fn append_template_message(&mut self, message: Message) {
        self.inner.messages.push(message);
    }

    pub fn clear_message(&mut self) {
        self.inner.clear_message();
    }

    pub fn apply_chat_template(
        &self,
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: &Vec<Tool>,
    ) -> Result<String, ApplyChatTemplateError> {
        let mut template = self.inner.clone();
        template.add_generation_prompt = add_generation_prompt;
        template.set_enable_thinking(enable_thinking);
        template.apply_chat_template(tools, false)
    }

    pub fn generation_prompt_replay_suffix(
        &self,
        enable_thinking: bool,
        tools: &Vec<Tool>,
        rendered_prompt: &str,
    ) -> Option<String> {
        let mut template = self.inner.clone();
        template.add_generation_prompt = true;
        template.set_enable_thinking(enable_thinking);
        template.generation_prompt_replay_suffix(tools, rendered_prompt)
    }

    pub fn get_prompt(&mut self, enable_thinking: bool, tools: &Vec<Tool>) -> String {
        self.inner.add_generation_prompt = true;
        self.inner.set_enable_thinking(enable_thinking);
        self.inner
            .apply_chat_template(tools, false)
            .expect("candle-vllm chat template rendering failed")
    }

    pub fn template_source(&self) -> Option<&str> {
        self.inner.get_template_string()
    }

    pub fn eos_token(&self) -> Option<&str> {
        self.inner.eos_token.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THINKING_TEMPLATE: &str = r#"
{%- for message in messages %}
    {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
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

    const HEADER_ONLY_TEMPLATE: &str = r#"
{%- for message in messages %}
    {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;

    const ALT_ASSISTANT_HEADER_TEMPLATE: &str = r#"
{%- for message in messages %}
    {{- '<turn>' + message.role + '\n' + message.content + '</turn>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<assistant_start>assistant\n<think>\n' }}
{%- endif %}
"#;

    const MINIMAX_TEMPLATE: &str = r#"
{%- set toolcall_begin_token   = '<minimax:tool_call>'         -%}
{%- set toolcall_end_token     = '</minimax:tool_call>'        -%}
{%- macro render_tool_namespace(namespace_name, tool_list) -%}
{%- for tool in tool_list -%}
<tool>{{ tool.function | tojson(ensure_ascii=False) }}</tool>
{% endfor -%}
{%- endmacro -%}
{%- macro visible_text(content) -%}
    {%- if content is string -%}
        {{ content }}
    {%- elif content is iterable and content is not mapping -%}
        {%- for item in content -%}
            {%- if item is mapping and item.type == 'text' -%}
                {{- item.text }}
            {%- elif item is string -%}
                {{- item }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{- content }}
    {%- endif -%}
{%- endmacro -%}
{{- ']~!b[' ~ ']~b]system' ~ '\n' }}
You are MiniMax.
{%- if tools -%}
    {{- '\n\n' ~ '# Tools' ~ '\n' ~ 'You may call one or more tools to assist with the user query.\nHere are the tools available in JSONSchema format:' ~ '\n' }}
    {{- '\n' ~ '<tools>' ~ '\n' }}
    {{- render_tool_namespace("functions", tools) }}
    {{- '</tools>' ~ '\n\n' }}
{{- 'When making tool calls, use XML format to invoke tools and pass parameters:' ~ '\n' }}
{{- '\n' ~ toolcall_begin_token }}
<invoke name="tool-name-1">
<parameter name="param-key-1">param-value-1</parameter>
</invoke>
{{- '\n' ~ toolcall_end_token }}
{%- endif -%}
{{- '[e~[\n' }}
{%- set last_tool_call = namespace(name=none) -%}
{%- for message in messages -%}
    {%- if message.role == 'assistant' -%}
        {{- ']~b]ai' ~ '\n' }}
        {%- set reasoning_content = '' %}
        {%- set content = visible_text(message.content) %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].strip('\n').split('<think>')[-1].strip('\n') %}
                {%- set content = content.split('</think>')[-1].strip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if reasoning_content -%}
            {{- '<think>' ~ '\n' ~ reasoning_content ~ '\n' ~ '</think>' ~ '\n\n' }}
        {%- endif -%}
        {%- if content -%}
            {{- content }}
        {%- endif -%}
        {%- if message.tool_calls -%}
            {{- '\n' ~ toolcall_begin_token ~ '\n' }}
            {%- for tool_call in message.tool_calls -%}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<invoke name="' + tool_call.name + '">' }}
                {% set _args = tool_call.arguments %}
                {%- for k, v in _args.items() %}
                {{- '<parameter name="' + k + '">' }}
                {{- v | tojson if v is not string else v }}
                {{- '</parameter>' }}
                {% endfor %}
                {{- '</invoke>' ~ '\n' }}
            {%- endfor -%}
            {{- toolcall_end_token}}
            {%- set last_tool_call.name = message.tool_calls[-1].function.name -%}
        {%- else -%}
            {%- set last_tool_call.name = none -%}
        {%- endif -%}
        {{- '[e~[' ~ '\n' }}
    {%- elif message.role == 'tool' -%}
        {{- ']~b]tool' }}
        {{- '\n<response>' }}
        {{- message.content }}
        {{- '</response>' }}
        {{- '[e~[\n' }}
    {%- elif message.role == 'user' -%}
        {{- ']~b]user' ~ '\n' }}
        {{- visible_text(message.content) }}
        {{- '[e~[' ~ '\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- ']~b]ai' ~ '\n' ~ '<think>' ~ '\n' }}
{%- endif -%}
"#;

    fn build_template(source: &str, enable_thinking: bool) -> ChatTemplate {
        ChatTemplate::new(
            None,
            Some(source.to_string()),
            None,
            None,
            None,
            true,
            enable_thinking,
        )
    }

    #[test]
    fn generation_prompt_replay_suffix_extracts_thinking_suffix() {
        let template = build_template(THINKING_TEMPLATE, true);
        let rendered = template.apply_chat_template(&Vec::new(), false).unwrap();
        let replay = template
            .generation_prompt_replay_suffix(&Vec::new(), &rendered)
            .unwrap();
        assert_eq!(replay, "<think>\n");
    }

    #[test]
    fn generation_prompt_replay_suffix_extracts_disabled_thinking_suffix() {
        let template = build_template(THINKING_TEMPLATE, false);
        let rendered = template.apply_chat_template(&Vec::new(), false).unwrap();
        let replay = template
            .generation_prompt_replay_suffix(&Vec::new(), &rendered)
            .unwrap();
        assert_eq!(replay, "<think>\n\n</think>\n\n");
    }

    #[test]
    fn generation_prompt_replay_suffix_extracts_header_only_suffix() {
        let template = build_template(HEADER_ONLY_TEMPLATE, true);
        let rendered = template.apply_chat_template(&Vec::new(), false).unwrap();
        assert!(template
            .generation_prompt_replay_suffix(&Vec::new(), &rendered)
            .is_none());
    }

    #[test]
    fn generation_prompt_replay_suffix_strips_non_qwen_assistant_header() {
        let template = build_template(ALT_ASSISTANT_HEADER_TEMPLATE, true);
        let rendered = template.apply_chat_template(&Vec::new(), false).unwrap();
        let replay = template
            .generation_prompt_replay_suffix(&Vec::new(), &rendered)
            .unwrap();
        assert_eq!(replay, "<think>\n");
    }

    #[test]
    fn strip_generation_assistant_header_only_strips_leading_header_line() {
        let suffix = "<|im_start|>assistant\n<think>\nassistant\n";
        assert_eq!(
            strip_generation_assistant_header(suffix),
            "<think>\nassistant\n"
        );
    }

    #[test]
    fn normalize_template_source_strips_unsupported_ensure_ascii_kwarg() {
        let source = "{{ value | tojson(ensure_ascii=False) }}";

        let mut raw_env = Environment::new();
        raw_env.add_template("raw", source).unwrap();
        let raw_template = raw_env.get_template("raw").unwrap();
        let raw_err = raw_template
            .render(context! { value => "hello" })
            .unwrap_err();
        assert!(
            raw_err
                .to_string()
                .contains("unknown keyword argument 'ensure_ascii'"),
            "unexpected raw error: {raw_err}"
        );

        let normalized = normalize_template_source(source);
        assert_eq!(normalized, "{{ value | tojson }}");

        let mut normalized_env = Environment::new();
        normalized_env
            .add_template("normalized", &normalized)
            .unwrap();
        let normalized_template = normalized_env.get_template("normalized").unwrap();
        let rendered = normalized_template
            .render(context! { value => "hello" })
            .unwrap();
        assert_eq!(rendered, "\"hello\"");
    }

    #[test]
    fn minimax_template_renders_tools_block_and_xml_instruction() {
        let tool = crate::tools::function_tool("search_web", "Search the web")
            .parameters_schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "query_tag": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "query_list": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["query_tag", "query_list"]
            }))
            .build();
        let mut template = build_template(MINIMAX_TEMPLATE, true);
        template.set_messages(&vec![Message {
            role: "user".to_string(),
            content: "Find the latest OpenAI release.".to_string(),
            num_images: 0,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }]);

        let rendered = template.apply_chat_template(&vec![tool], false).unwrap();
        assert!(
            rendered.contains("<tools>"),
            "rendered prompt: {}",
            rendered
        );
        assert!(
            rendered.contains("\"name\":\"search_web\"")
                || rendered.contains("\"name\": \"search_web\""),
            "rendered prompt: {}",
            rendered
        );
        assert!(
            rendered.contains("<minimax:tool_call>"),
            "rendered prompt: {}",
            rendered
        );
        assert!(
            rendered.contains("<invoke name=\"tool-name-1\">"),
            "rendered prompt: {}",
            rendered
        );
    }

    #[test]
    fn reasoning_content_extracted_from_assistant_content() {
        let mut template = build_template(THINKING_TEMPLATE, true);
        template.set_escape_tokens(Vec::new());

        let messages = vec![Message {
            role: "assistant".to_string(),
            content: "<think>cached reasoning</think>\nHello".to_string(),
            num_images: 0,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        template.set_messages(&messages);

        let render_msgs = template.escaped_messages_for_render();
        assert_eq!(
            render_msgs[0].reasoning_content.as_deref(),
            Some("cached reasoning"),
        );
        assert_eq!(render_msgs[0].content, "Hello");
    }

    #[test]
    fn reasoning_content_extracted_with_tool_calls() {
        let mut template = build_template(THINKING_TEMPLATE, true);
        template.set_escape_tokens(Vec::new());

        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "Search for X".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: "<think>I need to search</think>\n".to_string(),
                num_images: 0,
                tool_calls: Some(vec![serde_json::json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": {"q": "X"}}
                })]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: "result: found X".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
            },
        ];
        template.set_messages(&messages);

        let render_msgs = template.escaped_messages_for_render();
        let assistant_msg = &render_msgs[1];
        assert_eq!(
            assistant_msg.reasoning_content.as_deref(),
            Some("I need to search"),
        );
        assert!(
            !assistant_msg.content.contains("<think>"),
            "Reasoning markers should be removed from content after extraction",
        );
    }

    // ---------------------------------------------------------------
    // Qwen3-style template tests (reasoning_content extraction)
    // ---------------------------------------------------------------

    const QWEN3_TEMPLATE: &str = r#"
{%- for message in messages %}
{%- if message.content is string %}
{%- set content = message.content %}
{%- else %}
{%- set content = '' %}
{%- endif %}
{%- if message.role == "user" or message.role == "system" %}
{{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' }}
{%- elif message.role == "assistant" %}
{%- set reasoning_content = '' %}
{%- if message.reasoning_content is string %}
{%- set reasoning_content = message.reasoning_content %}
{%- else %}
{%- if '<think>' in content %}
{%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
{%- set content = content.split('</think>')[-1].lstrip('\n') %}
{%- endif %}
{%- endif %}
{%- if reasoning_content %}
{{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
{%- else %}
{{- '<|im_start|>' + message.role + '\n' + content }}
{%- endif %}
{%- if message.tool_calls %}
{%- for tool_call in message.tool_calls %}
{%- if (loop.first and content) or (not loop.first) %}
{{- '\n' }}
{%- endif %}
{%- if tool_call.function %}
{%- set tool_call = tool_call.function %}
{%- endif %}
{{- '<tool_call>\n{"name": "' }}
{{- tool_call.name }}
{{- '", "arguments": ' }}
{%- if tool_call.arguments is string %}
{{- tool_call.arguments }}
{%- else %}
{{- tool_call.arguments | tojson }}
{%- endif %}
{{- '}\n</tool_call>' }}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\n' }}
{%- elif message.role == "tool" %}
{{- '<|im_start|>user\n<tool_response>\n' + content + '\n</tool_response><|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}
"#;

    #[test]
    fn qwen3_template_no_double_think_with_reasoning_content_extraction() {
        let mut template = build_template(QWEN3_TEMPLATE, true);
        template.set_escape_tokens(Vec::new());

        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: "<think>\n</think>\nHi there!".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: "Do something".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];
        template.set_messages(&messages);

        let rendered = template.apply_chat_template(&Vec::new(), false).unwrap();

        // The generation prompt adds one unclosed <think> at the end - that's
        // expected.  Prior assistant turns must not introduce extra pairs.
        let open_count = rendered.matches("<think>").count();
        let close_count = rendered.matches("</think>").count();
        assert_eq!(
            open_count,
            close_count + 1,
            "Expected exactly one more <think> (generation prompt) than </think>. \
             Got {} opens and {} closes in:\n{}",
            open_count,
            close_count,
            rendered
        );

        let double_pattern = "<think>\n\n</think>\n\n<think>";
        assert!(
            !rendered.contains(double_pattern),
            "Double-think pattern must not appear. Got:\n{}",
            rendered
        );
    }

    #[test]
    fn qwen3_template_reasoning_content_field_prevents_double_think() {
        let mut template = build_template(QWEN3_TEMPLATE, true);
        template.set_escape_tokens(Vec::new());

        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "Search for X".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: "<think>\n</think>\n".to_string(),
                num_images: 0,
                tool_calls: Some(vec![serde_json::json!({
                    "function": {"name": "search", "arguments": {"q": "X"}}
                })]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: "result: found X".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
            },
        ];
        template.set_messages(&messages);

        let rendered = template.apply_chat_template(&Vec::new(), false).unwrap();

        let double_pattern = "<think>\n\n</think>\n\n<think>";
        assert!(
            !rendered.contains(double_pattern),
            "Double-think pattern must not appear when reasoning_content is extracted. Got:\n{}",
            rendered
        );

        assert!(
            rendered.contains("<tool_call>"),
            "Tool call must be present in rendered output. Got:\n{}",
            rendered
        );
    }

    #[test]
    fn qwen3_template_prior_turn_reasoning_stripped_is_accepted() {
        let mut template = build_template(QWEN3_TEMPLATE, true);
        template.set_escape_tokens(Vec::new());

        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: "<think>I need to think</think>\nHi!".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: "Next question".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];
        template.set_messages(&messages);

        let rendered = template.apply_chat_template(&Vec::new(), false).unwrap();

        assert!(
            rendered.contains("Hi!"),
            "Prior assistant turn content must be present. Got:\n{}",
            rendered
        );
        let double_pattern = "<think>\n\n</think>\n\n<think>";
        assert!(
            !rendered.contains(double_pattern),
            "No double-think pattern. Got:\n{}",
            rendered
        );
    }

    #[test]
    fn extract_reasoning_content_basic() {
        let (reasoning, remaining) =
            extract_reasoning_content("<think>hello</think>\nworld").unwrap();
        assert_eq!(reasoning, "hello");
        assert_eq!(remaining, "world");
    }

    #[test]
    fn extract_reasoning_content_empty_think() {
        let (reasoning, remaining) = extract_reasoning_content("<think>\n</think>\nworld").unwrap();
        assert_eq!(reasoning, "");
        assert_eq!(remaining, "world");
    }

    #[test]
    fn extract_reasoning_content_no_markers() {
        assert!(extract_reasoning_content("no markers here").is_none());
    }

    #[test]
    fn extract_reasoning_content_qwen_markers() {
        let (reasoning, remaining) =
            extract_reasoning_content("<|think|>hello<|/think|>\nworld").unwrap();
        assert_eq!(reasoning, "hello");
        assert_eq!(remaining, "world");
    }

    #[test]
    fn extract_reasoning_content_standalone_close_marker() {
        // MiniMax-style: model only generates </think>, the <think> was in the prompt
        let (reasoning, remaining) =
            extract_reasoning_content("Let me check the weather</think>\n\n").unwrap();
        assert_eq!(reasoning, "Let me check the weather");
        assert_eq!(remaining, "");
    }

    #[test]
    fn extract_reasoning_content_standalone_close_with_content_after() {
        let (reasoning, remaining) =
            extract_reasoning_content("reasoning here</think>\n\nHere is the answer").unwrap();
        assert_eq!(reasoning, "reasoning here");
        assert_eq!(remaining, "Here is the answer");
    }

    #[test]
    fn strip_generation_assistant_header_handles_minimax_role() {
        let suffix = "]~b]ai\n<think>\n";
        assert_eq!(strip_generation_assistant_header(suffix), "<think>\n");
    }

    #[test]
    fn strip_generation_assistant_header_handles_minimax_role_marker_token() {
        let suffix = "]~b]ai\n<think>\nassistant\n";
        assert_eq!(
            strip_generation_assistant_header(suffix),
            "<think>\nassistant\n"
        );
    }

    #[test]
    fn minimax_template_multi_turn_tool_call_renders_correctly() {
        let mut template = build_template(MINIMAX_TEMPLATE, true);
        template.set_escape_tokens(Vec::new());

        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "Search for OpenAI release".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: "Let me search</think>\n\n".to_string(),
                num_images: 0,
                tool_calls: Some(vec![serde_json::json!({
                    "function": {
                        "name": "search_web",
                        "arguments": {"query": "OpenAI release"}
                    }
                })]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: "Found: OpenAI released GPT-5".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
            },
        ];
        template.set_messages(&messages);

        let rendered = template.apply_chat_template(&Vec::new(), false).unwrap();

        assert!(
            rendered.contains("<minimax:tool_call>"),
            "Tool call should be rendered. Got:\n{}",
            rendered
        );
        assert!(
            rendered.contains("<invoke name=\"search_web\">"),
            "Invoke block should be present. Got:\n{}",
            rendered
        );
        assert!(
            rendered.contains("<response>"),
            "Tool response should be present. Got:\n{}",
            rendered
        );
        assert!(
            rendered.contains("Found: OpenAI released GPT-5"),
            "Tool result content should be present. Got:\n{}",
            rendered
        );
    }
}
