//! HTTP-specific response handling for the OpenAI-compatible API.
//!
//! This module contains axum-specific response types and implementations
//! for serving chat completion responses over HTTP/SSE.

use axum::extract::Json;
use axum::http::{self, StatusCode};
use axum::response::sse::Event;
use axum::response::{IntoResponse, Sse};
use candle_vllm_openai::streaming::{ChatResponse, StreamingStatus};
use candle_vllm_openai::types::responses::{APIError, ChatCompletionChunk, ChatCompletionResponse};
use flume::Receiver;
use futures::Stream;
use serde::Serialize;
use std::pin::Pin;
use std::task::{Context, Poll};

// ============================================================================
// Error Response Handling
// ============================================================================

/// Trait for converting errors to HTTP responses
trait ErrorToResponse: Serialize {
    fn to_response(&self, code: StatusCode) -> axum::response::Response {
        let mut r = Json(self).into_response();
        *r.status_mut() = code;
        r
    }
}

/// JSON error response format
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
// SSE Streamer
// ============================================================================

/// HTTP SSE streamer for chat completion responses
///
/// This wraps a channel receiver and implements the `Stream` trait
/// for use with axum's SSE response type.
pub struct HttpStreamer {
    /// Receiver for chat responses
    pub rx: Receiver<ChatResponse>,
    /// Current streaming status
    pub status: StreamingStatus,
}

impl HttpStreamer {
    /// Create a new HTTP streamer from a receiver
    pub fn new(rx: Receiver<ChatResponse>) -> Self {
        Self {
            rx,
            status: StreamingStatus::Uninitialized,
        }
    }
}

impl Stream for HttpStreamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.status == StreamingStatus::Stopped {
            return Poll::Ready(None);
        }

        match self.rx.try_recv() {
            Ok(resp) => match resp {
                ChatResponse::InternalError(e) => Poll::Ready(Some(Ok(Event::default().data(e)))),
                ChatResponse::ValidationError(e) => Poll::Ready(Some(Ok(Event::default().data(e)))),
                ChatResponse::ModelError(e) => Poll::Ready(Some(Ok(Event::default().data(e)))),
                ChatResponse::Chunk(response) => {
                    if self.status != StreamingStatus::Started {
                        self.status = StreamingStatus::Started;
                    }
                    Poll::Ready(Some(Event::default().json_data(response)))
                }
                ChatResponse::Done => {
                    self.status = StreamingStatus::Stopped;
                    Poll::Ready(Some(Ok(Event::default().data("[DONE]"))))
                }
            },
            Err(e) => {
                if self.status == StreamingStatus::Started && e == flume::TryRecvError::Disconnected
                {
                    self.status = StreamingStatus::Interrupted;
                    Poll::Ready(None)
                } else {
                    Poll::Pending
                }
            }
        }
    }
}

// Type alias for backward compatibility with core crate
pub type Streamer = HttpStreamer;

// ============================================================================
// Chat Responder Enum
// ============================================================================

/// HTTP response type for chat completion endpoints
///
/// This enum represents all possible responses from the chat completion endpoint,
/// including streaming (SSE), non-streaming, and error responses.
pub enum ChatResponder {
    /// Streaming SSE response
    Streamer(Sse<HttpStreamer>),
    /// Non-streaming completion response
    Completion(ChatCompletionResponse),
    /// Model execution error
    ModelError(APIError),
    /// Internal server error
    InternalError(APIError),
    /// Request validation error
    ValidationError(APIError),
}

impl ChatResponder {
    /// Create a streaming response
    pub fn streaming(streamer: HttpStreamer) -> Self {
        Self::Streamer(Sse::new(streamer))
    }

    /// Create a completion response
    pub fn completion(response: ChatCompletionResponse) -> Self {
        Self::Completion(response)
    }

    /// Create a model error response
    pub fn model_error(error: impl Into<String>) -> Self {
        Self::ModelError(APIError::new(error.into()))
    }

    /// Create an internal error response
    pub fn internal_error(error: impl Into<String>) -> Self {
        Self::InternalError(APIError::new(error.into()))
    }

    /// Create a validation error response
    pub fn validation_error(error: impl Into<String>) -> Self {
        Self::ValidationError(APIError::new(error.into()))
    }
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
// Utility Functions
// ============================================================================

/// Create a streaming chat response from a channel
pub fn create_streaming_response(rx: Receiver<ChatResponse>) -> ChatResponder {
    ChatResponder::streaming(HttpStreamer::new(rx))
}

/// Create a non-streaming chat response
pub fn create_completion_response(response: ChatCompletionResponse) -> ChatResponder {
    ChatResponder::completion(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_error() {
        let error = JsonError::new("test error".to_string());
        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("test error"));
    }

    #[test]
    fn test_chat_responder_creation() {
        let completion = ChatCompletionResponse {
            id: "test".to_string(),
            choices: vec![],
            created: 0,
            model: "test".to_string(),
            object: "chat.completion",
            usage: candle_vllm_openai::types::responses::ChatCompletionUsageResponse {
                request_id: "test".to_string(),
                created: 0,
                completion_tokens: 0,
                prompt_tokens: 0,
                total_tokens: 0,
                prompt_time_costs: 0,
                completion_time_costs: 0,
            },
            conversation_id: None,
            resource_id: None,
        };

        let responder = ChatResponder::completion(completion);
        match responder {
            ChatResponder::Completion(_) => (),
            _ => panic!("Expected Completion variant"),
        }
    }
}

