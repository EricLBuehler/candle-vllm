use crate::openai::sampling_params::Logprobs;
use actix_web::{error, HttpResponse};
use derive_more::{Display, Error};

use serde::{Deserialize, Serialize};

#[derive(Debug, Display, Error, Serialize)]
#[display(fmt = "Error: {}", data)]
pub struct APIError {
    data: String,
}

impl error::ResponseError for APIError {
    fn error_response(&self) -> HttpResponse {
        //pack error to json so that client can handle it
        HttpResponse::BadRequest()
            .content_type("application/json")
            .json(self.data.to_string())
    }
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
