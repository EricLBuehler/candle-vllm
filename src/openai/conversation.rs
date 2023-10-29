const ROLES: (&str, &str) = ("ROLES", "ASSISTANT");
const SYSTEM_TEMPLATE: &str = "{system_template}";
const DEFAULT_SEP: &str = "\n";

pub struct Message((String, Option<String>));

impl Message {
    pub fn new(message: (String, String)) -> Message {
        Message((message.0, Some(message.1)))
    }
}

#[derive(Default)]
pub enum SeperatorStyle {
    #[default]
    AddColonSingle,
    AddColonTwo,
    AddColonSpaceSingle,
    NoColonSingle,
    NoColonTwo,
    AddNewLineSingle,
    Llama2,
    ChatGLM,
    ChatML,
    ChatIntern,
    Dolly,
    RWKV,
    Phoenix,
    Robin,
    FalconChat,
}

/// A struct for managing prompt templates and conversation history.
#[allow(dead_code)]
pub struct Conversation {
    name: String,
    system_message: String,
    messages: Vec<Message>,
    offset: usize,
    sep_style: SeperatorStyle,
    stop_criteria: String,
    stop_token_ids: Vec<isize>,
    roles: (String, String),
    sep: String,
    sep2: Option<String>,
}

impl Conversation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        system_message: String,
        messages: Vec<Message>,
        offset: usize,
        sep_style: SeperatorStyle,
        stop_criteria: String,
        stop_token_ids: Vec<isize>,
        roles: (String, String),
        sep: String,
        sep2: Option<String>,
    ) -> Self {
        Self {
            name,
            system_message,
            messages,
            offset,
            sep_style,
            stop_criteria,
            stop_token_ids,
            roles,
            sep,
            sep2,
        }
    }

    pub fn default(name: String) -> Self {
        Self::new(
            name,
            SYSTEM_TEMPLATE.to_string(),
            Default::default(),
            0,
            Default::default(),
            Default::default(),
            Default::default(),
            (ROLES.0.to_string(), ROLES.1.to_string()),
            DEFAULT_SEP.to_string(),
            None,
        )
    }

    /// Set the system message.
    pub fn set_system_message(&mut self, system_message: String) {
        self.system_message = system_message;
    }

    /// Append a new message.
    pub fn append_message(&mut self, role: String, message: String) {
        self.messages.push(Message((role, Some(message))));
    }

    /// Set the last message to `None`.
    pub fn update_last_messge(&mut self) {
        self.messages.last_mut().unwrap().0 .1 = None;
    }

    pub fn get_prompt(&mut self) -> String {
        match self.sep_style {
            SeperatorStyle::AddColonSingle => {
                let mut accum = self.system_message.clone() + &self.sep;
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}: {message}{}", self.sep);
                    } else {
                        accum += &format!("{role}:");
                    }
                }
                accum
            }

            SeperatorStyle::AddColonTwo => {
                let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                let mut accum = self.system_message.clone() + &self.sep;
                for (i, message) in self.messages.iter().enumerate() {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}: {message}{}", seps[i % 2]);
                    } else {
                        accum += &format!("{role}:");
                    }
                }
                accum
            }

            SeperatorStyle::AddColonSpaceSingle => {
                let mut accum = self.system_message.clone() + &self.sep;
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}: {message}{}", self.sep);
                    } else {
                        accum += &format!("{role}: "); //must end with space
                    }
                }
                accum
            }

            SeperatorStyle::AddNewLineSingle => {
                let mut accum = if self.system_message.is_empty() {
                    "".to_string()
                } else {
                    self.system_message.clone() + &self.sep
                };
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}\n{message}{}", self.sep);
                    } else {
                        accum += &format!("{role}\n");
                    }
                }
                accum
            }

            SeperatorStyle::NoColonSingle => {
                let mut accum = self.system_message.clone();
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}{message}{}", self.sep);
                    } else {
                        accum += role;
                    }
                }
                accum
            }

            SeperatorStyle::NoColonTwo => {
                let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                let mut accum = self.system_message.clone();
                for (i, message) in self.messages.iter().enumerate() {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}{message}{}", seps[i % 2]);
                    } else {
                        accum += role;
                    }
                }
                accum
            }

            SeperatorStyle::RWKV => {
                let mut accum = self.system_message.clone() + &self.sep;
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!(
                            "{role}: {}\n\n",
                            message.replace("\r\n", "\n").replace("\n\n", "\n")
                        );
                    } else {
                        accum += &format!("{role}:");
                    }
                }
                accum
            }

            SeperatorStyle::Llama2 => {
                let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                let mut accum = if !self.system_message.is_empty() {
                    self.system_message.clone()
                } else {
                    "[INST] ".to_string()
                };
                for (i, message) in self.messages.iter().enumerate() {
                    let Message((_role, message)) = message;

                    let tag = &[self.roles.0.clone(), self.roles.1.clone()][i % 2];

                    if let Some(message) = message {
                        if i == 0 {
                            accum += &format!("{message} ");
                        } else {
                            accum += &format!("{tag} {message}{}", seps[i % 2]);
                        }
                    } else {
                        accum += tag;
                    }
                }
                accum
            }

            SeperatorStyle::ChatGLM => {
                let round_add_n = if self.name == "chatglm2" { 1 } else { 0 };

                let mut accum = if !self.system_message.is_empty() {
                    self.system_message.clone()
                } else {
                    "".to_string()
                };

                for (i, message) in self.messages.iter().enumerate() {
                    if i % 2 == 0 {
                        accum += &format!("[Round {}]{}", i / 2 + round_add_n, self.sep);
                    }
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}: {message}{}", self.sep);
                    } else {
                        accum += &format!("{role}: ");
                    }
                }
                accum
            }

            SeperatorStyle::ChatML => {
                let mut accum = if !self.system_message.is_empty() {
                    format!("{}{}\n", self.system_message.clone(), self.sep)
                } else {
                    "".to_string()
                };
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}\n{message}{}\n", self.sep);
                    } else {
                        accum += &format!("{role}\n");
                    }
                }
                accum
            }

            SeperatorStyle::ChatIntern => {
                let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                let mut accum = self.system_message.clone();
                for (i, message) in self.messages.iter().enumerate() {
                    if i % 2 == 0 {
                        accum += "<s>";
                    }

                    let Message((role, message)) = message;

                    if let Some(message) = message {
                        accum += &format!("{role}:{message}{}\n", seps[i % 2]);
                    } else {
                        accum += &format!("{role}:");
                    }
                }
                accum
            }

            SeperatorStyle::Dolly => {
                let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                let mut accum = self.system_message.clone();
                for (i, message) in self.messages.iter().enumerate() {
                    let Message((role, message)) = message;

                    if let Some(message) = message {
                        accum += &format!("{role}:\n{message}{}", seps[i % 2]);
                        if i % 2 == 1 {
                            accum += "\n\n";
                        }
                    } else {
                        accum += &format!("{role}:\n");
                    }
                }
                accum
            }

            SeperatorStyle::Phoenix => {
                let mut accum = self.system_message.clone() + &self.sep;
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}: <s>{message}</s>");
                    } else {
                        accum += &format!("{role}: <s>");
                    }
                }
                accum
            }

            SeperatorStyle::Robin => {
                let mut accum = self.system_message.clone() + &self.sep;
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}:\n{message}{}", self.sep);
                    } else {
                        accum += &format!("{role}:\n");
                    }
                }
                accum
            }

            SeperatorStyle::FalconChat => {
                let mut accum = "".to_string();
                if !self.system_message.is_empty() {
                    accum += &format!("{}{}", self.system_message, self.sep)
                }
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}: {message}{}", self.sep);
                    } else {
                        accum += &format!("{role}:");
                    }
                }
                accum
            }
        };

        todo!();
    }
}
