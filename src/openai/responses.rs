use actix_web::error;
use derive_more::{Display, Error};

use serde::{Deserialize, Serialize};

#[derive(Debug, Display, Error)]
#[display(fmt = "Error: {}", data)]
pub struct APIError {
    data: String,
}

impl error::ResponseError for APIError {}

impl APIError {
    pub fn new(data: String) -> Self {
        Self { data }
    }

    pub fn new_str(data: &str) -> Self {
        Self {
            data: data.to_string(),
        }
    }

    pub fn new_from_io_err(data: std::io::Error) -> Self {
        Self {
            data: data.to_string(),
        }
    }

    pub fn new_from_serde_err(data: serde_json::Error) -> Self {
        Self {
            data: data.to_string(),
        }
    }

    pub fn new_from_candle_err(data: candle_core::Error) -> Self {
        Self {
            data: data.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionUsageResponse {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<String>,
    pub created: usize,
    pub model: String,
    pub object: String,
    pub usage: ChatCompletionUsageResponse,
}
