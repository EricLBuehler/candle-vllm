pub mod default_conversation;
use serde::Serialize;
use serde_json::Value;

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
