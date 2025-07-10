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

// impl error::ResponseError for APIError {
//     fn error_response(&self) -> HttpResponse {
//         //pack error to json so that client can handle it
//         HttpResponse::BadRequest()
//             .content_type("application/json")
//             .json(self.data.to_string())
//     }
// }

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
        //panic!("{}", value.to_string());
        Self::new(value.to_string())
    }
}

#[macro_export]
macro_rules! try_api {
    ($candle_result:expr) => {
        match $candle_result {
            Ok(v) => v,
            Err(e) => {
                return Err(APIError::from(e));
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

// tool_calls, function_call not supported!
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoiceData {
    pub content: Option<String>,
    pub role: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatChoice>,
    pub created: u64,
    pub model: String,
    pub object: &'static str,
    pub usage: ChatCompletionUsageResponse,
}

// tool_calls, function_call not supported!
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceData {
    pub content: Option<String>,
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub delta: ChoiceData,
    pub finish_reason: Option<String>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub choices: Vec<Choice>,
    pub created: u64,
    pub model: String,
    pub object: &'static str,
    pub system_fingerprint: Option<String>,
}

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
