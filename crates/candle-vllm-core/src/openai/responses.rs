use super::requests::{FunctionCall, FunctionCallDelta, ToolCall, ToolCallDelta};
use super::streaming::Streamer;
use crate::openai::sampling_params::Logprobs;
use axum::extract::Json;
use axum::http::{self, StatusCode};
use axum::response::{IntoResponse, Sse};
use derive_more::{Display, Error};
use serde::{Deserialize, Serialize};

#[derive(Debug, Display, Error, Serialize)]
#[display(fmt = "Error: {data}")]
pub struct APIError {
    data: String,
}

impl APIError {
    pub fn new(data: String) -> Self {
        Self { data }
    }

    pub fn new_str(data: &str) -> Self {
        Self {
            data: data.to_string(),
        }
    }

    pub fn from<T: ToString>(value: T) -> Self {
        Self::new(value.to_string())
    }
}

#[macro_export]
macro_rules! try_api {
    ($candle_result:expr) => {
        match $candle_result {
            Ok(v) => v,
            Err(e) => {
                return Err(crate::openai::responses::APIError::from(e));
            }
        }
    };
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionUsageResponse {
    pub request_id: String,
    pub created: u64,
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    pub prompt_time_costs: usize,     //milliseconds
    pub completion_time_costs: usize, //milliseconds
}

// ============================================================================
// Chat Choice Data (Non-streaming response message)
// ============================================================================

/// The message content in a chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoiceData {
    /// The text content of the message (can be null if tool_calls is present)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// The role of the message author (always "assistant" for completions)
    pub role: String,
    /// Tool calls made by the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatChoiceData {
    /// Create a new text response
    pub fn text(content: String) -> Self {
        Self {
            content: Some(content),
            role: "assistant".to_string(),
            tool_calls: None,
        }
    }

    /// Create a response with tool calls only
    pub fn with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            content: None,
            role: "assistant".to_string(),
            tool_calls: Some(tool_calls),
        }
    }

    /// Create a response with both text and tool calls
    pub fn with_content_and_tool_calls(content: String, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            content: Some(content),
            role: "assistant".to_string(),
            tool_calls: Some(tool_calls),
        }
    }

    /// Check if this response contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().map_or(false, |tc| !tc.is_empty())
    }
}

impl Default for ChatChoiceData {
    fn default() -> Self {
        Self {
            content: None,
            role: "assistant".to_string(),
            tool_calls: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WrapperLogprobs {
    pub content: Vec<Logprobs>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub message: ChatChoiceData,
    pub finish_reason: Option<String>,
    pub index: usize,
    pub logprobs: Option<WrapperLogprobs>,
}

impl ChatChoice {
    /// Get the appropriate finish_reason based on content
    pub fn determine_finish_reason(&self) -> &str {
        if self.message.has_tool_calls() {
            "tool_calls"
        } else {
            self.finish_reason.as_deref().unwrap_or("stop")
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatChoice>,
    pub created: u64,
    pub model: String,
    pub object: &'static str,
    pub usage: ChatCompletionUsageResponse,
    /// Conversation ID from the request (if provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
    /// Resource ID from the request (if provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_id: Option<String>,
}

// ============================================================================
// Streaming Response Types (Deltas)
// ============================================================================

/// Delta content for streaming responses
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChoiceData {
    /// Text content delta (incremental text)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Role (only sent in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Tool call deltas for streaming tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    /// Reasoning content delta (for reasoning/thinking models)
    /// This field contains incremental reasoning tokens emitted by models
    /// that support chain-of-thought or thinking capabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

impl ChoiceData {
    /// Create a content delta
    pub fn content(text: String) -> Self {
        Self {
            content: Some(text),
            role: None,
            tool_calls: None,
            reasoning: None,
        }
    }

    /// Create a role-only delta (first chunk)
    pub fn role(role: String) -> Self {
        Self {
            content: None,
            role: Some(role),
            tool_calls: None,
            reasoning: None,
        }
    }

    /// Create a tool call delta
    pub fn tool_call(delta: ToolCallDelta) -> Self {
        Self {
            content: None,
            role: None,
            tool_calls: Some(vec![delta]),
            reasoning: None,
        }
    }

    /// Create a reasoning delta (for reasoning/thinking models)
    pub fn reasoning(text: String) -> Self {
        Self {
            content: None,
            role: None,
            tool_calls: None,
            reasoning: Some(text),
        }
    }

    /// Create an empty delta (for finish chunk)
    pub fn empty() -> Self {
        Self {
            content: None,
            role: None,
            tool_calls: None,
            reasoning: None,
        }
    }

    /// Check if this delta is empty
    pub fn is_empty(&self) -> bool {
        self.content.is_none() && self.role.is_none() && self.tool_calls.is_none() && self.reasoning.is_none()
    }

    /// Check if this delta contains reasoning content
    pub fn has_reasoning(&self) -> bool {
        self.reasoning.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub delta: ChoiceData,
    pub finish_reason: Option<String>,
    pub index: usize,
}

impl Choice {
    /// Create a content chunk
    pub fn content_chunk(index: usize, content: String) -> Self {
        Self {
            delta: ChoiceData::content(content),
            finish_reason: None,
            index,
        }
    }

    /// Create a role chunk (first chunk in stream)
    pub fn role_chunk(index: usize, role: String) -> Self {
        Self {
            delta: ChoiceData::role(role),
            finish_reason: None,
            index,
        }
    }

    /// Create a tool call chunk
    pub fn tool_call_chunk(index: usize, tool_call_delta: ToolCallDelta) -> Self {
        Self {
            delta: ChoiceData::tool_call(tool_call_delta),
            finish_reason: None,
            index,
        }
    }

    /// Create a reasoning chunk (for reasoning/thinking models)
    pub fn reasoning_chunk(index: usize, reasoning: String) -> Self {
        Self {
            delta: ChoiceData::reasoning(reasoning),
            finish_reason: None,
            index,
        }
    }

    /// Create a finish chunk
    pub fn finish_chunk(index: usize, reason: &str) -> Self {
        Self {
            delta: ChoiceData::empty(),
            finish_reason: Some(reason.to_string()),
            index,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub choices: Vec<Choice>,
    pub created: u64,
    pub model: String,
    pub object: &'static str,
    pub system_fingerprint: Option<String>,
    /// Conversation ID from the request (if provided, sent in first chunk only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
    /// Resource ID from the request (if provided, sent in first chunk only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_id: Option<String>,
}

impl ChatCompletionChunk {
    /// Create a new chunk with a single choice
    pub fn new(id: String, choice: Choice, created: u64, model: String) -> Self {
        Self {
            id,
            choices: vec![choice],
            created,
            model,
            object: "chat.completion.chunk",
            system_fingerprint: None,
            conversation_id: None,
            resource_id: None,
        }
    }

    /// Create a content chunk
    pub fn content(id: String, index: usize, content: String, created: u64, model: String) -> Self {
        Self::new(id, Choice::content_chunk(index, content), created, model)
    }

    /// Create a reasoning chunk (for reasoning/thinking models)
    pub fn reasoning(id: String, index: usize, reasoning: String, created: u64, model: String) -> Self {
        Self::new(id, Choice::reasoning_chunk(index, reasoning), created, model)
    }

    /// Create a finish chunk
    pub fn finish(id: String, index: usize, reason: &str, created: u64, model: String) -> Self {
        Self::new(id, Choice::finish_chunk(index, reason), created, model)
    }
}

// ============================================================================
// Error Response Handling
// ============================================================================

trait ErrorToResponse: Serialize {
    fn to_response(&self, code: StatusCode) -> axum::response::Response {
        let mut r = Json(self).into_response();
        *r.status_mut() = code;
        r
    }
}

#[derive(Serialize)]
struct JsonError {
    message: String,
}

impl JsonError {
    fn new(message: String) -> Self {
        Self { message }
    }
}
impl ErrorToResponse for JsonError {}

// ============================================================================
// Chat Responder Enum
// ============================================================================

pub enum ChatResponder {
    Streamer(Sse<Streamer>),
    Completion(ChatCompletionResponse),
    ModelError(APIError),
    InternalError(APIError),
    ValidationError(APIError),
}

impl IntoResponse for ChatResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            ChatResponder::Streamer(s) => s.into_response(),
            ChatResponder::Completion(s) => Json(s).into_response(),
            ChatResponder::InternalError(e) => {
                JsonError::new(e.to_string()).to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            ChatResponder::ValidationError(e) => {
                JsonError::new(e.to_string()).to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
            ChatResponder::ModelError(msg) => {
                JsonError::new(msg.to_string()).to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
}

// ============================================================================
// Helper Functions for Creating Tool Call Responses
// ============================================================================

/// Create a ToolCall from parsed data
pub fn create_tool_call(id: String, name: String, arguments: String) -> ToolCall {
    ToolCall {
        id,
        call_type: "function".to_string(),
        function: FunctionCall { name, arguments },
    }
}

/// Create a ToolCallDelta for the first chunk of a tool call
pub fn create_tool_call_delta_start(index: usize, id: String, name: String) -> ToolCallDelta {
    ToolCallDelta {
        index,
        id: Some(id),
        call_type: Some("function".to_string()),
        function: Some(FunctionCallDelta {
            name: Some(name),
            arguments: Some(String::new()),
        }),
    }
}

/// Create a ToolCallDelta for subsequent chunks (arguments only)
pub fn create_tool_call_delta_arguments(index: usize, arguments: String) -> ToolCallDelta {
    ToolCallDelta {
        index,
        id: None,
        call_type: None,
        function: Some(FunctionCallDelta {
            name: None,
            arguments: Some(arguments),
        }),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_choice_data_text() {
        let data = ChatChoiceData::text("Hello".to_string());
        assert_eq!(data.content, Some("Hello".to_string()));
        assert_eq!(data.role, "assistant");
        assert!(!data.has_tool_calls());
    }

    #[test]
    fn test_chat_choice_data_with_tool_calls() {
        let tool_call = create_tool_call(
            "call_123".to_string(),
            "get_weather".to_string(),
            r#"{"location": "Paris"}"#.to_string(),
        );
        let data = ChatChoiceData::with_tool_calls(vec![tool_call]);

        assert!(data.content.is_none());
        assert!(data.has_tool_calls());
        assert_eq!(data.tool_calls.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_serialize_response_with_tool_calls() {
        let tool_call = create_tool_call(
            "call_abc".to_string(),
            "search".to_string(),
            r#"{"query": "rust"}"#.to_string(),
        );
        let data = ChatChoiceData::with_tool_calls(vec![tool_call]);

        let json = serde_json::to_string(&data).unwrap();
        assert!(json.contains("tool_calls"));
        assert!(json.contains("call_abc"));
        assert!(json.contains("search"));
    }

    #[test]
    fn test_choice_data_delta_empty() {
        let delta = ChoiceData::empty();
        assert!(delta.is_empty());
    }

    #[test]
    fn test_choice_data_delta_content() {
        let delta = ChoiceData::content("Hello".to_string());
        assert!(!delta.is_empty());
        assert_eq!(delta.content, Some("Hello".to_string()));
        assert!(!delta.has_reasoning());
    }

    #[test]
    fn test_choice_data_delta_reasoning() {
        let delta = ChoiceData::reasoning("Let me think about this...".to_string());
        assert!(!delta.is_empty());
        assert!(delta.has_reasoning());
        assert_eq!(delta.reasoning, Some("Let me think about this...".to_string()));
        assert!(delta.content.is_none());
    }

    #[test]
    fn test_reasoning_chunk() {
        let chunk = ChatCompletionChunk::reasoning(
            "cmpl-456".to_string(),
            0,
            "<think>Analyzing the problem...</think>".to_string(),
            1234567890,
            "reasoning-model".to_string(),
        );

        assert_eq!(chunk.id, "cmpl-456");
        assert_eq!(chunk.choices.len(), 1);
        assert!(chunk.choices[0].delta.has_reasoning());
        assert_eq!(
            chunk.choices[0].delta.reasoning,
            Some("<think>Analyzing the problem...</think>".to_string())
        );
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn test_tool_call_delta_serialization() {
        let delta =
            create_tool_call_delta_start(0, "call_123".to_string(), "get_weather".to_string());

        let json = serde_json::to_string(&delta).unwrap();
        assert!(json.contains("call_123"));
        assert!(json.contains("get_weather"));
        assert!(json.contains("function"));
    }

    #[test]
    fn test_chat_completion_chunk() {
        let chunk = ChatCompletionChunk::content(
            "cmpl-123".to_string(),
            0,
            "Hello".to_string(),
            1234567890,
            "test-model".to_string(),
        );

        assert_eq!(chunk.id, "cmpl-123");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }
}
