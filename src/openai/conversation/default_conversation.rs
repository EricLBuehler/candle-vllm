use super::{ApplyChatTemplateError, Conversation, Message};
use minijinja::{context, value::Kwargs, Environment, Error, ErrorKind, Value};
use serde::Serialize;

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
pub struct DefaultConversation {
    name: String,
    system_message: Option<String>,
    chat_template: Option<String>,
    messages: Vec<Message>,
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
            sep_style,
            bos_token,
            eos_token,
            roles,
            sep: seps.sep,
            sep2: seps.sep2,
        }
    }
}

fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
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
        Value::from_safe_string(rv)
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

impl Conversation for DefaultConversation {
    /// Set the system message.
    fn set_system_message(&mut self, system_message: Option<String>) {
        self.system_message = system_message;
    }

    /// Append a new message.
    fn append_message(&mut self, role: String, content: String) {
        self.messages.push(Message { role, content });
    }

    fn get_roles(&self) -> &(String, String) {
        &self.roles
    }

    fn clear_message(&mut self) {
        self.messages.clear()
    }

    fn apply_chat_template(
        &self,
        add_generation_prompt: bool,
        enable_thinking: bool,
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
        template
            .render(context! {
              messages => self.messages,
              add_generation_prompt => add_generation_prompt,
              bos_token => self.bos_token,
              eos_token => self.eos_token,
              enable_thinking => enable_thinking,
              date_string => date_string,
            })
            .map_err(ApplyChatTemplateError::RenderTemplateError)
    }
    /// Convert this conversation to a String prompt
    fn get_prompt(&mut self, thinking: bool) -> String {
        match self.apply_chat_template(true, thinking) {
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

                match self.sep_style {
                    SeparatorStyle::AddColonSingle
                    | SeparatorStyle::AddColonSpaceSingle
                    | SeparatorStyle::AddNewLineSingle => {
                        let mut accum = system_prompt + &self.sep;
                        for message in &self.messages {
                            accum += &format!("{}: {}{}", message.role, message.content, self.sep);
                        }
                        accum
                    }

                    SeparatorStyle::AddColonTwo => {
                        let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                        let mut accum = system_prompt + &self.sep;
                        for (i, message) in self.messages.iter().enumerate() {
                            accum +=
                                &format!("{}: {}{}", message.role, message.content, seps[i % 2]);
                        }
                        accum
                    }

                    SeparatorStyle::NoColonSingle => {
                        let mut accum = system_prompt.clone();
                        for message in &self.messages {
                            accum += &format!("{}: {}{}", message.role, message.content, self.sep);
                        }
                        accum
                    }

                    SeparatorStyle::NoColonTwo => {
                        let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                        let mut accum = system_prompt.clone();
                        for (i, message) in self.messages.iter().enumerate() {
                            accum +=
                                &format!("{}: {}{}", message.role, message.content, seps[i % 2]);
                        }
                        accum
                    }

                    SeparatorStyle::RWKV => {
                        let mut accum = system_prompt.clone() + &self.sep;
                        for message in &self.messages {
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
                        for (i, message) in self.messages.iter().enumerate() {
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
                        for (i, message) in self.messages.iter().enumerate() {
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
                        for (i, message) in self.messages.iter().enumerate() {
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
                        for (i, message) in self.messages.iter().enumerate() {
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
                        for message in self.messages.iter() {
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
                        for (i, message) in self.messages.iter().enumerate() {
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

                        for (i, message) in self.messages.iter().enumerate() {
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
                        for message in &self.messages {
                            accum +=
                                &format!("{}\n{}{}\n", message.role, message.content, self.sep);
                        }
                        accum
                    }

                    SeparatorStyle::ChatIntern => {
                        let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                        let mut accum = system_prompt.clone();
                        for (i, message) in self.messages.iter().enumerate() {
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
                        for (i, message) in self.messages.iter().enumerate() {
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
                        for message in &self.messages {
                            accum += &format!("{}: <s>{}</s>", message.role, message.content);
                        }
                        accum
                    }

                    SeparatorStyle::Robin => {
                        let mut accum = system_prompt.clone() + &self.sep;
                        for message in &self.messages {
                            accum += &format!("{}:\n{}{}", message.role, message.content, self.sep);
                        }
                        accum
                    }

                    SeparatorStyle::FalconChat => {
                        let mut accum = "".to_string();
                        if !system_prompt.is_empty() {
                            accum += &format!("{}{}", system_prompt, self.sep)
                        }
                        for message in &self.messages {
                            accum += &format!("{}: {}{}", message.role, message.content, self.sep);
                        }
                        accum
                    }

                    SeparatorStyle::GLM => {
                        let mut accum = "[gMASK]<sop>".to_string();
                        accum += &system_prompt.clone();
                        for message in self.messages.iter() {
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
