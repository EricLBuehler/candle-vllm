//! Streaming response types for chat completions.
//!
//! This module provides the core streaming types that are transport-agnostic.
//! For HTTP/SSE-specific streaming, see the server crate.

use candle_vllm_core::openai::responses::ChatCompletionChunk;

/// Status of a streaming response
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingStatus {
    /// Stream has not yet started
    Uninitialized,
    /// Stream is actively producing chunks
    Started,
    /// Stream was interrupted (e.g., client disconnected)
    Interrupted,
    /// Stream completed normally
    Stopped,
}

impl Default for StreamingStatus {
    fn default() -> Self {
        Self::Uninitialized
    }
}

/// Response type for streaming chat completions
///
/// This enum represents the possible responses that can be sent during streaming.
/// It is transport-agnostic and can be used with any streaming mechanism.
#[derive(Debug, Clone)]
pub enum ChatResponse {
    /// Internal server error
    InternalError(String),
    /// Request validation error
    ValidationError(String),
    /// Model execution error
    ModelError(String),
    /// A chunk of the streaming response
    Chunk(ChatCompletionChunk),
    /// Signal that the stream is complete
    Done,
}

impl ChatResponse {
    /// Create an internal error response
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::InternalError(message.into())
    }

    /// Create a validation error response
    pub fn validation_error(message: impl Into<String>) -> Self {
        Self::ValidationError(message.into())
    }

    /// Create a model error response
    pub fn model_error(message: impl Into<String>) -> Self {
        Self::ModelError(message.into())
    }

    /// Create a chunk response
    pub fn chunk(chunk: ChatCompletionChunk) -> Self {
        Self::Chunk(chunk)
    }

    /// Check if this is an error response
    pub fn is_error(&self) -> bool {
        matches!(
            self,
            Self::InternalError(_) | Self::ValidationError(_) | Self::ModelError(_)
        )
    }

    /// Check if this is the done signal
    pub fn is_done(&self) -> bool {
        matches!(self, Self::Done)
    }

    /// Get the error message if this is an error
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::InternalError(msg) | Self::ValidationError(msg) | Self::ModelError(msg) => {
                Some(msg)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_status_default() {
        assert_eq!(StreamingStatus::default(), StreamingStatus::Uninitialized);
    }

    #[test]
    fn test_chat_response_errors() {
        let internal = ChatResponse::internal_error("internal error");
        assert!(internal.is_error());
        assert_eq!(internal.error_message(), Some("internal error"));

        let validation = ChatResponse::validation_error("validation error");
        assert!(validation.is_error());

        let model = ChatResponse::model_error("model error");
        assert!(model.is_error());
    }

    #[test]
    fn test_chat_response_done() {
        let done = ChatResponse::Done;
        assert!(done.is_done());
        assert!(!done.is_error());
        assert!(done.error_message().is_none());
    }
}

