//! OpenAI-compatible adapter for the inference engine.
//!
//! This module provides an adapter that wraps the core `InferenceEngine` and exposes
//! an OpenAI-compatible API for chat completions with tool calling support.

use candle_vllm_core::api::{Error, InferenceEngine, Result};
use candle_vllm_core::openai::openai_server::chat_completions_with_data;
use candle_vllm_core::openai::requests::ChatCompletionRequest;
use candle_vllm_core::openai::responses::{
    ChatCompletionChunk, ChatCompletionResponse, ChatResponder,
};
use candle_vllm_core::openai::sampling_params::GenerationConfig;
use candle_vllm_core::openai::{OpenAIServerData, PipelineConfig};
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Adapter that provides OpenAI-compatible chat completion API.
pub struct OpenAIAdapter {
    engine: Arc<InferenceEngine>,
}

impl OpenAIAdapter {
    /// Create a new OpenAIAdapter wrapping the given InferenceEngine.
    pub fn new(engine: InferenceEngine) -> Self {
        Self {
            engine: Arc::new(engine),
        }
    }

    /// Perform a chat completion request.
    ///
    /// This method converts the OpenAI-style request into internal format,
    /// executes generation, and returns an OpenAI-compatible response.
    pub async fn chat_completion(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        // Convert request to internal format and execute
        let pipeline_config = PipelineConfig {
            max_model_len: self.engine.model_info().max_sequence_length,
            default_max_tokens: request.max_tokens.unwrap_or(128),
            generation_cfg: Some(GenerationConfig {
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: request.top_k,
                min_p: request.min_p,
                frequency_penalty: request.frequency_penalty,
                presence_penalty: request.presence_penalty,
            }),
        };

        let data = OpenAIServerData {
            model: self.engine.engine().clone(),
            pipeline_config,
            record_conversation: false,
            device: self.engine.device().clone(),
        };

        let responder = chat_completions_with_data(Arc::new(data), request).await;

        match responder {
            ChatResponder::Completion(resp) => Ok(resp),
            ChatResponder::ModelError(e) => Err(Error::Generation(e.to_string())),
            ChatResponder::InternalError(e) => Err(Error::Generation(e.to_string())),
            ChatResponder::ValidationError(e) => Err(Error::Config(e.to_string())),
            ChatResponder::Streamer(_) => {
                Err(Error::Generation("unexpected stream response".to_string()))
            }
        }
    }

    /// Perform a streaming chat completion.
    ///
    /// Returns a stream of chat completion chunks.
    pub async fn chat_completion_stream(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Result<
        impl futures::Stream<Item = Result<candle_vllm_core::openai::responses::ChatCompletionChunk>>,
    > {
        // Set stream flag
        let mut stream_request = request;
        stream_request.stream = Some(true);

        let pipeline_config = PipelineConfig {
            max_model_len: self.engine.model_info().max_sequence_length,
            default_max_tokens: stream_request.max_tokens.unwrap_or(128),
            generation_cfg: Some(GenerationConfig {
                temperature: stream_request.temperature,
                top_p: stream_request.top_p,
                top_k: stream_request.top_k,
                min_p: stream_request.min_p,
                frequency_penalty: stream_request.frequency_penalty,
                presence_penalty: stream_request.presence_penalty,
            }),
        };

        let data = OpenAIServerData {
            model: self.engine.engine().clone(),
            pipeline_config,
            record_conversation: false,
            device: self.engine.device().clone(),
        };

        let responder = chat_completions_with_data(Arc::new(data), stream_request).await;

        match responder {
            ChatResponder::Streamer(sse) => {
                // Extract the stream from the SSE wrapper
                // For now, return a placeholder stream that indicates streaming is not fully implemented
                Ok(StreamingPlaceholder {})
            }
            ChatResponder::Completion(_) => Err(Error::Generation(
                "unexpected completion response".to_string(),
            )),
            ChatResponder::ModelError(e) => Err(Error::Generation(e.to_string())),
            ChatResponder::InternalError(e) => Err(Error::Generation(e.to_string())),
            ChatResponder::ValidationError(e) => Err(Error::Config(e.to_string())),
        }
    }
}

/// Placeholder stream implementation for streaming chat completions.
/// TODO: Implement proper stream extraction from SSE wrapper.
struct StreamingPlaceholder {}

impl Stream for StreamingPlaceholder {
    type Item = Result<ChatCompletionChunk>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(None)
    }
}
