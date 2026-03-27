use crate::tools::{Tool, ToolCall};

use super::{ApplyChatTemplateError, Message};
use minijinja::{context, value::Kwargs, Environment, Error, ErrorKind, Value as JinjaValue};
use serde::Serialize;
use serde_json::Value as JsonValue;
use tokenizers::Tokenizer;

pub const ROLES: (&str, &str) = ("USER", "ASSISTANT");
pub const DEFAULT_SEP: &str = "\n";

/// Separator style for default conversation.
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

/// A struct for managing prompt templates and conversation history.
#[allow(dead_code)]
#[derive(Clone)]
pub struct DefaultConversation {
    name: String,
    system_message: Option<String>,
    chat_template: Option<String>,
    messages: Vec<Message>,
    escape_tokens: Vec<String>,
    preserve_tokens: Vec<String>,
    sep_style: SeparatorStyle,
    bos_token: Option<String>,
    eos_token: Option<String>,
    roles: (String, String),
    sep: String,
    sep2: Option<String>,
}

/// Default conversion separators
pub struct DefaultConversationSeparators {
    pub sep: String,
    pub sep2: Option<String>,
}

impl DefaultConversation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        chat_template: Option<String>,
        messages: Vec<Message>,
        sep_style: SeparatorStyle,
        bos_token: Option<String>,
        eos_token: Option<String>,
        roles: (String, String),
        seps: DefaultConversationSeparators,
    ) -> Self {
        Self {
            name,
            system_message: None,
            chat_template,
            messages,
            escape_tokens: Vec::new(),
            preserve_tokens: Vec::new(),
            sep_style,
            bos_token,
            eos_token,
            roles,
            sep: seps.sep,
            sep2: seps.sep2,
        }
    }
}

fn tojson(value: JinjaValue, kwargs: Kwargs) -> Result<JinjaValue, Error> {
    if let Ok(indent) = kwargs.get("indent") {
        let mut buf = Vec::new();
        let repeat = b" ".repeat(indent);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(&repeat);
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
        value.serialize(&mut ser).unwrap();
        String::from_utf8(buf).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
    } else {
        serde_json::to_string(&value).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
    }
    .map_err(|err| {
        Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
    })
    .map(|s| {
        // When this filter is used the return value is safe for both HTML and JSON
        let mut rv = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '<' => rv.push_str("\\u003c"),
                '>' => rv.push_str("\\u003e"),
                '&' => rv.push_str("\\u0026"),
                '\'' => rv.push_str("\\u0027"),
                _ => rv.push(c),
            }
        }
        JinjaValue::from_safe_string(rv)
    })
}

fn strftime_now(fmt: String) -> Result<String, minijinja::Error> {
    let date = chrono::Utc::now();
    let date_string = date.format(&fmt).to_string();
    Ok(date_string)
}

fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

fn escape_special_tokens_in_text(
    content: &str,
    escape_tokens: &[String],
    preserve_tokens: &[String],
) -> String {
    if escape_tokens.is_empty() || content.is_empty() {
        return content.to_string();
    }

    let mut protected = content.to_string();
    let mut sentinels = Vec::new();
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

impl DefaultConversation {
    pub fn collect_escape_tokens(tokenizer: &Tokenizer, tool_markers: &[&str]) -> Vec<String> {
        let mut tokens = tokenizer
            .get_added_tokens_decoder()
            .into_values()
            .filter(|added| added.special)
            .map(|added| added.content)
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

        tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        tokens.dedup();
        tokens
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

    fn escape_text(&self, content: &str) -> String {
        escape_special_tokens_in_text(content, &self.escape_tokens, &self.preserve_tokens)
    }

    fn escaped_messages_for_render(&self) -> Vec<Message> {
        if self.escape_tokens.is_empty() {
            return self.messages.clone();
        }

        self.messages
            .iter()
            .map(|message| {
                let mut escaped = message.clone();
                if !matches!(escaped.role.as_str(), "system" | "developer") {
                    escaped.content = self.escape_text(&escaped.content);
                }
                escaped
            })
            .collect()
    }

    /// Set the system message.
    pub fn set_system_message(&mut self, system_message: Option<String>) {
        self.system_message = system_message.clone();
        if let Some(msg) = system_message {
            if let Some(m) = self.messages.iter_mut().find(|m| m.role == "system") {
                m.content = msg;
            } else {
                self.messages.insert(
                    0,
                    Message {
                        role: "system".to_string(),
                        content: msg,
                        num_images: 0,
                        tool_calls: None,
                        tool_call_id: None,
                    },
                );
            }
        }
    }

    pub fn get_system_message(&self) -> Option<String> {
        self.system_message.clone()
    }

    /// Append a new message.
    pub fn append_message(&mut self, role: String, content: String) {
        self.messages.push(Message {
            role,
            content,
            num_images: 0,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    pub fn append_message_with_tool_metadata(
        &mut self,
        role: String,
        content: String,
        tool_calls: Option<Vec<ToolCall>>,
        tool_call_id: Option<String>,
    ) {
        self.messages.push(Message {
            role,
            content,
            num_images: 0,
            tool_calls: tool_calls.map(|calls| {
                calls
                    .iter()
                    .map(Self::to_template_tool_call)
                    .collect::<Vec<_>>()
            }),
            tool_call_id,
        });
    }

    pub fn append_template_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn get_roles(&self) -> &(String, String) {
        &self.roles
    }

    pub fn template_source(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub fn eos_token(&self) -> Option<&str> {
        self.eos_token.as_deref()
    }

    pub fn clear_message(&mut self) {
        self.messages.clear()
    }

    pub fn apply_chat_template(
        &self,
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: &Vec<Tool>,
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
        let template = self.chat_template.as_ref().unwrap();
        let mut template = template.replace("[::-1]", "|reverse");
        if template.contains("{{ meta }}") {
            template = template.replace("{%- set meta = message.get(\"metadata\", \"\") %}", "");
            template = template.replace("{{ meta }}", "");
        }
        env.add_template(self.name.as_str(), template.as_str())
            .map_err(ApplyChatTemplateError::AddTemplateError)?;

        env.add_function("raise_exception", raise_exception);
        env.add_filter("tojson", tojson);
        env.add_function("strftime_now", strftime_now);

        let date = chrono::Utc::now();
        let date_string = date.format("%d, %B, %Y").to_string();

        let template = env
            .get_template(&self.name)
            .map_err(ApplyChatTemplateError::GetTemplateError)?;
        let render_messages = self.escaped_messages_for_render();
        template
            .render(context! {
              messages => render_messages,
              add_generation_prompt => add_generation_prompt,
              bos_token => self.bos_token,
              eos_token => self.eos_token,
              enable_thinking => enable_thinking,
              date_string => date_string,
              tools => tools,
            })
            .map_err(ApplyChatTemplateError::RenderTemplateError)
    }
    /// Convert this conversation to a String prompt
    pub fn get_prompt(&mut self, thinking: bool, tools: &Vec<Tool>) -> String {
        match self.apply_chat_template(true, thinking, tools) {
            Ok(prompt) => prompt,
            Err(e) => {
                if self.chat_template.is_some() {
                    tracing::warn!("apply chat template failed {:?}", e);
                }
                //no chat template exists? using the built-in template
                let system_prompt = self
                    .system_message
                    .as_ref()
                    .map_or("".to_string(), |msg| format!("<|system|>\n {msg}"));
                let render_messages = self.escaped_messages_for_render();

                match self.sep_style {
                    SeparatorStyle::AddColonSingle
                    | SeparatorStyle::AddColonSpaceSingle
                    | SeparatorStyle::AddNewLineSingle => {
                        let mut accum = system_prompt + &self.sep;
                        for message in &render_messages {
                            accum += &format!("{}: {}{}", message.role, message.content, self.sep);
                        }
                        accum
                    }

                    SeparatorStyle::AddColonTwo => {
                        let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                        let mut accum = system_prompt + &self.sep;
                        for (i, message) in render_messages.iter().enumerate() {
                            accum +=
                                &format!("{}: {}{}", message.role, message.content, seps[i % 2]);
                        }
                        accum
                    }

                    SeparatorStyle::NoColonSingle => {
                        let mut accum = system_prompt.clone();
                        for message in &render_messages {
                            accum += &format!("{}: {}{}", message.role, message.content, self.sep);
                        }
                        accum
                    }

                    SeparatorStyle::NoColonTwo => {
                        let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                        let mut accum = system_prompt.clone();
                        for (i, message) in render_messages.iter().enumerate() {
                            accum +=
                                &format!("{}: {}{}", message.role, message.content, seps[i % 2]);
                        }
                        accum
                    }

                    SeparatorStyle::RWKV => {
                        let mut accum = system_prompt.clone() + &self.sep;
                        for message in &render_messages {
                            accum += &format!(
                                "{}: {}\n\n",
                                message.role,
                                message.content.replace("\r\n", "\n").replace("\n\n", "\n")
                            );
                        }
                        accum
                    }

                    SeparatorStyle::Llama | SeparatorStyle::Mistral => {
                        let mut accum = "".to_string();
                        for (i, message) in render_messages.iter().enumerate() {
                            if message.role.clone() == self.roles.0 {
                                accum += &format!("[INST] {} [/INST]", message.content);
                            } else if message.role.clone() == self.roles.1 {
                                //assistant message
                                accum += &format!("{} \n", message.content);
                            } else if i == 0 && !system_prompt.is_empty() {
                                accum += &system_prompt;
                            }
                        }
                        accum
                    }

                    SeparatorStyle::Llama3 => {
                        let mut accum = "<|begin_of_text|>".to_string();
                        for (i, message) in render_messages.iter().enumerate() {
                            if message.role.clone() == self.roles.0 {
                                //user message
                                accum += &format!(
                                    "<|start_header_id|>user<|end_header_id|>\n\n {} <|eot_id|>",
                                    message.content
                                );
                            } else if message.role.clone() == self.roles.1 {
                                //assistant message
                                accum += &format!("<|start_header_id|>assistant<|end_header_id|>\n\n {} <|eot_id|>", message.content);
                            } else if i == 0 && !system_prompt.is_empty() {
                                accum += &system_prompt;
                            }
                        }
                        accum
                    }

                    SeparatorStyle::Phi => {
                        let mut accum = "".to_string();
                        for (i, message) in render_messages.iter().enumerate() {
                            if message.role.clone() == self.roles.0 {
                                //user message
                                accum += &format!("<|user|> {}<|end|>", message.content);
                            } else if message.role.clone() == self.roles.1 {
                                //assistant message
                                accum += &format!("<|assistant|>{}<|end|>", message.content);
                            } else if i == 0 && !system_prompt.is_empty() {
                                accum += &system_prompt;
                            }
                        }
                        accum
                    }

                    SeparatorStyle::Qwen | SeparatorStyle::Yi => {
                        let mut accum = "".to_string();
                        for (i, message) in render_messages.iter().enumerate() {
                            if message.role.clone() == self.roles.0 {
                                //user message
                                accum +=
                                    &format!("<|im_start|>user\n {} <|im_end|>", message.content);
                            } else if message.role.clone() == self.roles.1 {
                                //assistant message
                                accum += &format!(
                                    "<|im_start|>assistant\n {} <|im_end|>",
                                    message.content
                                );
                            } else if i == 0 && !system_prompt.is_empty() {
                                accum += &system_prompt;
                            }
                        }
                        accum
                    }

                    SeparatorStyle::Gemma => {
                        let mut accum = "".to_string();
                        for message in render_messages.iter() {
                            accum += &format!(
                                "<bos><start_of_turn>{}\n {} <end_of_turn>\n",
                                message.role, message.content
                            );
                        }
                        accum += "<start_of_turn>model\n";
                        accum
                    }

                    SeparatorStyle::StableLM => {
                        let mut accum = "".to_string();
                        for (i, message) in render_messages.iter().enumerate() {
                            if message.role.clone() == self.roles.0 {
                                //user message
                                accum +=
                                    &format!("<|user|>user\n {}<|endoftext|>", message.content);
                            } else if message.role.clone() == self.roles.1 {
                                //assistant message
                                accum +=
                                    &format!("<|assistant|>\n {}<|endoftext|>", message.content);
                            } else if i == 0 && !system_prompt.is_empty() {
                                accum += &system_prompt;
                            }
                        }
                        accum
                    }

                    SeparatorStyle::ChatGLM => {
                        let round_add_n = if self.name == "chatglm2" { 1 } else { 0 };

                        let mut accum = if !system_prompt.is_empty() {
                            system_prompt.clone()
                        } else {
                            "".to_string()
                        };

                        for (i, message) in render_messages.iter().enumerate() {
                            if i % 2 == 0 {
                                accum += &format!("[Round {}]{}", i / 2 + round_add_n, self.sep);
                            }
                            accum += &format!("{}: {}{}", message.role, message.content, self.sep);
                        }
                        accum
                    }

                    SeparatorStyle::ChatML => {
                        let mut accum = if !system_prompt.is_empty() {
                            format!("{}{}\n", system_prompt, self.sep)
                        } else {
                            "".to_string()
                        };
                        for message in &render_messages {
                            accum +=
                                &format!("{}\n{}{}\n", message.role, message.content, self.sep);
                        }
                        accum
                    }

                    SeparatorStyle::ChatIntern => {
                        let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                        let mut accum = system_prompt.clone();
                        for (i, message) in render_messages.iter().enumerate() {
                            if i % 2 == 0 {
                                accum += "<s>";
                            }
                            accum +=
                                &format!("{}:{}{}\n", message.role, message.content, seps[i % 2]);
                        }
                        accum
                    }

                    SeparatorStyle::Dolly => {
                        let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                        let mut accum = system_prompt.clone();
                        for (i, message) in render_messages.iter().enumerate() {
                            accum +=
                                &format!("{}:\n{}{}", message.role, message.content, seps[i % 2]);
                            if i % 2 == 1 {
                                accum += "\n\n";
                            }
                        }
                        accum
                    }

                    SeparatorStyle::Phoenix => {
                        let mut accum = system_prompt.clone() + &self.sep;
                        for message in &render_messages {
                            accum += &format!("{}: <s>{}</s>", message.role, message.content);
                        }
                        accum
                    }

                    SeparatorStyle::Robin => {
                        let mut accum = system_prompt.clone() + &self.sep;
                        for message in &render_messages {
                            accum += &format!("{}:\n{}{}", message.role, message.content, self.sep);
                        }
                        accum
                    }

                    SeparatorStyle::FalconChat => {
                        let mut accum = "".to_string();
                        if !system_prompt.is_empty() {
                            accum += &format!("{}{}", system_prompt, self.sep)
                        }
                        for message in &render_messages {
                            accum += &format!("{}: {}{}", message.role, message.content, self.sep);
                        }
                        accum
                    }

                    SeparatorStyle::GLM => {
                        let mut accum = "[gMASK]<sop>".to_string();
                        accum += &system_prompt.clone();
                        for message in render_messages.iter() {
                            if message.role.clone() == self.roles.0 {
                                //user message
                                accum += &format!("<|user|>\n {}", message.content);
                            } else if message.role.clone() == self.roles.1 {
                                //assistant message
                                accum += &format!("<|assistant|>\n {}", message.content);
                            }
                        }
                        accum
                    }
                }
            }
        }
    }
}

impl DefaultConversation {
    fn to_template_tool_call(call: &ToolCall) -> JsonValue {
        let args = Self::parse_template_tool_arguments(&call.function.arguments);
        serde_json::json!({
            "id": call.id,
            "type": call.call_type,
            "function": {
                "name": call.function.name,
                "arguments": args
            }
        })
    }

    fn parse_template_tool_arguments(arguments: &str) -> JsonValue {
        let raw = arguments.trim();
        if raw.is_empty() {
            return serde_json::json!({});
        }

        match serde_json::from_str::<JsonValue>(raw).ok() {
            Some(JsonValue::Object(obj)) => JsonValue::Object(obj),
            Some(JsonValue::String(inner)) => {
                match serde_json::from_str::<JsonValue>(inner.trim()).ok() {
                    Some(JsonValue::Object(obj)) => JsonValue::Object(obj),
                    _ => serde_json::json!({}),
                }
            }
            _ => serde_json::json!({}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_tool_markup_in_non_system_messages() {
        let mut conversation = DefaultConversation::new(
            "test".to_string(),
            Some("{{ messages[0].content }}".to_string()),
            vec![Message {
                role: "user".to_string(),
                content: "<tool_call><function=read></function></tool_call>".to_string(),
                num_images: 0,
                tool_calls: None,
                tool_call_id: None,
            }],
            SeparatorStyle::default(),
            None,
            None,
            ("user".to_string(), "assistant".to_string()),
            DefaultConversationSeparators {
                sep: "\n".to_string(),
                sep2: None,
            },
        );
        conversation.set_escape_tokens(vec![
            "<tool_call>".to_string(),
            "</tool_call>".to_string(),
            "<function=".to_string(),
            "</function>".to_string(),
        ]);

        let prompt = conversation
            .apply_chat_template(true, false, &Vec::new())
            .unwrap();
        assert!(prompt.contains("<\u{200C}tool_call>"));
        assert!(!prompt.contains("<tool_call>"));
    }
}
