use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

pub const EMPTY_TOOL_RESULT_ACK: &str = "Tool executed successfully with no textual output.";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ImageUrlContent {
    Url(String),
    Object {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MessageContent {
    #[serde(alias = "input_text", alias = "text")]
    Text { text: String },
    #[serde(alias = "image_url")]
    ImageUrl { image_url: ImageUrlContent },
    #[serde(alias = "image_base64")]
    ImageBase64 { image_base64: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContentType {
    PureText(String),
    Single(MessageContent),
    Multi(Vec<MessageContent>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<MessageContentType>,
    #[serde(default)]
    pub tool_calls: Option<Vec<crate::tools::ToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Messages {
    Chat(Vec<ChatMessage>),
    Map(Vec<HashMap<String, String>>),
    Literal(String),
}

fn extract_text_from_content(content: Option<&MessageContentType>) -> String {
    match content {
        Some(MessageContentType::PureText(text)) => text.clone(),
        Some(MessageContentType::Single(item)) => match item {
            MessageContent::Text { text } => text.clone(),
            _ => String::new(),
        },
        Some(MessageContentType::Multi(items)) => items
            .iter()
            .filter_map(|item| match item {
                MessageContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
        None => String::new(),
    }
}

pub fn normalize_empty_openai_tool_results(messages: &mut [ChatMessage]) {
    for msg in messages {
        if msg.role != "tool" {
            continue;
        }

        let is_empty = extract_text_from_content(msg.content.as_ref())
            .trim()
            .is_empty();
        if is_empty {
            msg.content = Some(MessageContentType::PureText(
                EMPTY_TOOL_RESULT_ACK.to_string(),
            ));
        }
    }
}

pub fn validate_openai_tool_messages(messages: &[ChatMessage]) -> Result<(), String> {
    let mut assistant_tool_call_ids: HashSet<String> = HashSet::new();
    let mut tool_result_ids_seen: HashSet<String> = HashSet::new();
    let mut pending_tool_results: Option<HashSet<String>> = None;

    for (idx, msg) in messages.iter().enumerate() {
        if let Some(expected_results) = pending_tool_results.as_mut() {
            if msg.role != "tool" {
                let mut pending_ids = expected_results.iter().cloned().collect::<Vec<_>>();
                pending_ids.sort();
                return Err(format!(
                    "messages[{idx}] must be role=tool to answer pending assistant tool_calls {:?}",
                    pending_ids
                ));
            }
            if msg.tool_calls.is_some() {
                return Err(format!(
                    "messages[{idx}] role=tool must not include tool_calls"
                ));
            }

            let call_id = msg.tool_call_id.as_deref().unwrap_or("").trim();
            if call_id.is_empty() {
                return Err(format!(
                    "messages[{idx}] role=tool requires a non-empty tool_call_id"
                ));
            }
            if !tool_result_ids_seen.insert(call_id.to_string()) {
                return Err(format!(
                    "messages[{idx}] role=tool has duplicate tool_call_id '{}'",
                    call_id
                ));
            }
            if !expected_results.remove(call_id) {
                let mut pending_ids = expected_results.iter().cloned().collect::<Vec<_>>();
                pending_ids.sort();
                return Err(format!(
                    "messages[{idx}] role=tool references unexpected tool_call_id '{}'. pending ids: {:?}",
                    call_id, pending_ids
                ));
            }

            let text = extract_text_from_content(msg.content.as_ref());
            if text.trim().is_empty() {
                return Err(format!(
                    "messages[{idx}] role=tool requires non-empty content"
                ));
            }
            if expected_results.is_empty() {
                pending_tool_results = None;
            }
            continue;
        }

        match msg.role.as_str() {
            "assistant" => {
                if let Some(tool_calls) = &msg.tool_calls {
                    if tool_calls.is_empty() {
                        continue;
                    }
                    let mut expected_results = HashSet::new();
                    for (tool_idx, call) in tool_calls.iter().enumerate() {
                        let call_id = call.id.trim();
                        if call_id.is_empty() {
                            return Err(format!(
                                "messages[{idx}] assistant tool_calls[{tool_idx}] requires a non-empty id"
                            ));
                        }
                        if !expected_results.insert(call_id.to_string()) {
                            return Err(format!(
                                "messages[{idx}] assistant tool_call id '{}' is duplicated",
                                call_id
                            ));
                        }
                        if !assistant_tool_call_ids.insert(call_id.to_string()) {
                            return Err(format!(
                                "messages[{idx}] assistant tool_call id '{}' is duplicated",
                                call_id
                            ));
                        }
                    }
                    pending_tool_results = Some(expected_results);
                }
            }
            "tool" => {
                let call_id = msg.tool_call_id.as_deref().unwrap_or("").trim();
                if !call_id.is_empty() && tool_result_ids_seen.contains(call_id) {
                    return Err(format!(
                        "messages[{idx}] role=tool has duplicate tool_call_id '{}'",
                        call_id
                    ));
                }
                return Err(format!(
                    "messages[{idx}] role=tool has no preceding assistant tool_calls to answer"
                ));
            }
            _ => {}
        }
    }

    if let Some(pending) = pending_tool_results {
        let mut pending_ids = pending.into_iter().collect::<Vec<_>>();
        pending_ids.sort();
        return Err(format!(
            "Missing role=tool results for assistant tool_call ids: {:?}",
            pending_ids
        ));
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopTokens {
    Multi(Vec<String>),
    Single(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Messages,
    pub temperature: Option<f32>, //0.7
    pub top_p: Option<f32>,       //1.0
    pub min_p: Option<f32>,       //0.0
    #[serde(default)]
    pub n: Option<usize>, //1
    pub max_tokens: Option<usize>, //None
    #[serde(default)]
    pub stop: Option<StopTokens>,
    #[serde(default)]
    pub stream: Option<bool>, //false
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub presence_penalty: Option<f32>, //0.0
    pub repeat_last_n: Option<usize>, //0.0
    #[serde(default)]
    pub frequency_penalty: Option<f32>, //0.0
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>, //None
    #[serde(default)]
    pub user: Option<String>, //None
    pub top_k: Option<isize>,         //-1
    #[serde(default)]
    pub best_of: Option<usize>, //None
    #[serde(default)]
    pub use_beam_search: Option<bool>, //false
    #[serde(default)]
    pub ignore_eos: Option<bool>, //false
    #[serde(default)]
    pub skip_special_tokens: Option<bool>, //false
    #[serde(default)]
    pub stop_token_ids: Option<Vec<usize>>, //[]
    #[serde(default)]
    pub logprobs: Option<bool>, //false
    #[serde(alias = "enable_thinking")]
    pub thinking: Option<bool>, //false
    #[serde(default)]
    pub tools: Option<Vec<crate::tools::Tool>>,
    #[serde(default)]
    pub tool_choice: Option<crate::tools::ToolChoice>,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        Self {
            model: Some("default".to_string()),
            messages: Messages::Literal("".to_string()),
            temperature: None,
            top_p: None,
            min_p: None,
            n: None,
            max_tokens: None,
            stop: None,
            stream: None,
            stream_options: None,
            presence_penalty: None,
            repeat_last_n: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            top_k: None,
            best_of: None,
            use_beam_search: None,
            ignore_eos: None,
            skip_special_tokens: None,
            stop_token_ids: None,
            logprobs: None,
            thinking: None,
            tools: None,
            tool_choice: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    String(String),
    MultiString(Vec<String>),
    Tokens(Vec<u32>),
    MultiTokens(Vec<Vec<u32>>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::String(s) => vec![s],
            EmbeddingInput::MultiString(s) => s,
            EmbeddingInput::Tokens(_) => unimplemented!("Token input not supported yet"),
            EmbeddingInput::MultiTokens(_) => unimplemented!("Token input not supported yet"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingType {
    Last,
    Mean, //default
}

impl Default for EmbeddingType {
    fn default() -> Self {
        Self::Mean
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: EncodingFormat,
    #[serde(default)]
    pub embedding_type: EmbeddingType,
}

impl Default for EncodingFormat {
    fn default() -> Self {
        Self::Float
    }
}

#[cfg(test)]
mod tests {
    use super::{
        validate_openai_tool_messages, ChatCompletionRequest, ChatMessage, MessageContentType,
    };

    #[test]
    fn chat_completion_request_reads_stream_options() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
                "stream_options": {"include_usage": true}
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.stream, Some(true));
        assert_eq!(
            request.stream_options.map(|options| options.include_usage),
            Some(true)
        );
    }

    #[test]
    fn validates_openai_tool_messages_with_known_tool_call_id() {
        let messages = vec![
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(MessageContentType::PureText(String::new())),
                tool_calls: Some(vec![crate::tools::ToolCall {
                    index: None,
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: crate::tools::FunctionCall {
                        name: "test".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "tool".to_string(),
                content: Some(MessageContentType::PureText("{\"ok\":true}".to_string())),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
            },
        ];

        assert!(validate_openai_tool_messages(&messages).is_ok());
    }

    #[test]
    fn rejects_openai_tool_message_with_no_preceding_assistant_tool_calls() {
        let messages = vec![ChatMessage {
            role: "tool".to_string(),
            content: Some(MessageContentType::PureText("{\"ok\":true}".to_string())),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        }];

        let err = validate_openai_tool_messages(&messages).unwrap_err();
        assert!(err.contains("no preceding assistant tool_calls"));
    }
}
