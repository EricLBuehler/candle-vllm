pub mod default_conversation;
use serde::Serialize;
/// A trait for using conversation managers with a `ModulePipeline`.
pub trait Conversation {
    fn set_system_message(&mut self, system_message: Option<String>);

    fn append_message(&mut self, role: String, message: String);

    fn get_roles(&self) -> &(String, String);

    fn apply_chat_template(
        &self,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, ApplyChatTemplateError>;

    fn get_prompt(&mut self, thinking: bool) -> String;

    fn clear_message(&mut self);
}

#[derive(Serialize, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
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
