//! Task payload and result types for LLM inference jobs.
//!
//! These types implement the prometheus_parking_lot `TaskPayload` requirements
//! and define the contract between the scheduler and inference workers.

use crate::openai::responses::{ChatChoice, ChatCompletionUsageResponse};
use crate::openai::sampling_params::SamplingParams;
use crate::InputMetadata;
use serde::{Deserialize, Serialize};

/// Type alias for job categorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobType {
    /// Non-streaming completion request
    Completion,
    /// Streaming token-by-token request
    Streaming,
}

/// A serializable description of an LLM inference request.
///
/// This struct contains all data needed to process one inference request,
/// whether streaming or non-streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceJob {
    /// Unique request ID for tracing and result correlation
    pub request_id: String,

    /// Input token IDs (prompt)
    pub tokens: Vec<u32>,

    /// Token positions for positional encoding
    pub positions: Vec<usize>,

    /// Whether this is a streaming request
    pub is_streaming: bool,

    /// Sampling parameters for generation
    pub sampling_params: SamplingParams,

    /// Created timestamp (for streaming response metadata)
    pub created: u64,

    /// Maximum context length for this request
    pub max_context_len: usize,

    /// Whether this is a prefill phase
    pub is_prefill: bool,

    /// Maximum sequence length for query
    pub max_seqlen_q: usize,

    /// Maximum sequence length for key
    pub max_seqlen_k: usize,
}

impl InferenceJob {
    /// Create a new inference job for a completion request.
    #[must_use]
    pub fn new_completion(
        request_id: String,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        sampling_params: SamplingParams,
        max_context_len: usize,
    ) -> Self {
        let seq_len = tokens.len();
        Self {
            request_id,
            tokens,
            positions,
            is_streaming: false,
            sampling_params,
            created: crate::openai::utils::get_created_time_secs(),
            max_context_len,
            is_prefill: true,
            max_seqlen_q: seq_len,
            max_seqlen_k: seq_len,
        }
    }

    /// Create a new inference job for a streaming request.
    #[must_use]
    pub fn new_streaming(
        request_id: String,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        sampling_params: SamplingParams,
        created: u64,
        max_context_len: usize,
    ) -> Self {
        let seq_len = tokens.len();
        Self {
            request_id,
            tokens,
            positions,
            is_streaming: true,
            sampling_params,
            created,
            max_context_len,
            is_prefill: true,
            max_seqlen_q: seq_len,
            max_seqlen_k: seq_len,
        }
    }

    /// Get the number of prompt tokens
    #[must_use]
    pub fn prompt_len(&self) -> usize {
        self.tokens.len()
    }

    /// Reconstruct InputMetadata from job parameters.
    ///
    /// This is used by the executor to prepare inputs for the model.
    pub fn to_input_metadata(
        &self,
        device: &candle_core::Device,
    ) -> Result<InputMetadata, candle_core::Error> {
        let seq_len = self.tokens.len();
        let cu_seqlens = candle_core::Tensor::new(&[0u32, seq_len as u32], device)?;

        Ok(InputMetadata {
            is_prefill: self.is_prefill,
            slot_mapping: candle_core::Tensor::zeros(seq_len, candle_core::DType::I64, device)?,
            block_tables: None,
            context_lens: None,
            cu_seqlens_q: Some(cu_seqlens.clone()),
            cu_seqlens_k: Some(cu_seqlens),
            max_seqlen_q: self.max_seqlen_q,
            max_seqlen_k: self.max_seqlen_k,
            max_context_len: self.max_context_len,
        })
    }
}

/// A single token in a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingTokenResult {
    /// The generated token text
    pub text: String,

    /// Token ID
    pub token_id: u32,

    /// Whether this is the final token
    pub is_finished: bool,

    /// Finish reason if finished (e.g., "stop", "length")
    pub finish_reason: Option<String>,

    /// Whether this token is part of reasoning/thinking output
    pub is_reasoning: bool,
}

/// Result of an inference job execution.
///
/// This enum represents either a complete response (for non-streaming)
/// or a channel for receiving streaming tokens.
#[derive(Debug)]
pub enum InferenceResult {
    /// Complete response for non-streaming requests
    Completion {
        /// Chat completion choices
        choices: Vec<ChatChoice>,
        /// Usage statistics
        usage: ChatCompletionUsageResponse,
    },

    /// Streaming response - contains a receiver for tokens
    Streaming {
        /// Request ID for correlation
        request_id: String,
        /// Receiver for streaming tokens
        token_rx: flume::Receiver<Result<StreamingTokenResult, String>>,
    },

    /// Error during inference
    Error {
        /// Error message
        message: String,
    },
}

impl InferenceResult {
    /// Create a completion result
    #[must_use]
    pub fn completion(choices: Vec<ChatChoice>, usage: ChatCompletionUsageResponse) -> Self {
        Self::Completion { choices, usage }
    }

    /// Create a streaming result with the given token receiver
    #[must_use]
    pub fn streaming(
        request_id: String,
        token_rx: flume::Receiver<Result<StreamingTokenResult, String>>,
    ) -> Self {
        Self::Streaming {
            request_id,
            token_rx,
        }
    }

    /// Create an error result
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
        }
    }

    /// Check if this is an error result
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Get the error message if this is an error result
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Error { message } => Some(message),
            _ => None,
        }
    }
}

/// Serializable wrapper for InferenceResult (for mailbox storage and ResourcePool).
///
/// This enum handles both completion and streaming results in a way that can be
/// serialized for the prometheus-parking-lot mailbox system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializableInferenceResult {
    /// Complete response for non-streaming requests
    Completion {
        /// Chat completion choices
        choices: Vec<ChatChoice>,
        /// Usage statistics
        usage: ChatCompletionUsageResponse,
    },

    /// Streaming response - contains a mailbox key to retrieve the channel
    StreamingChannel {
        /// Request ID for correlation
        request_id: String,
        /// Mailbox key to retrieve the streaming channel from StreamingRegistry
        channel_key: String,
    },

    /// Error during inference
    Error {
        /// Error message
        message: String,
    },
}

impl SerializableInferenceResult {
    /// Create a completion result
    #[must_use]
    pub fn completion(choices: Vec<ChatChoice>, usage: ChatCompletionUsageResponse) -> Self {
        Self::Completion { choices, usage }
    }

    /// Create a streaming channel result
    #[must_use]
    pub fn streaming_channel(request_id: String, channel_key: String) -> Self {
        Self::StreamingChannel {
            request_id,
            channel_key,
        }
    }

    /// Create an error result
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
        }
    }

    /// Check if this is an error result
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Get the error message if this is an error result
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Error { message } => Some(message),
            _ => None,
        }
    }
}

impl From<&InferenceResult> for SerializableInferenceResult {
    fn from(result: &InferenceResult) -> Self {
        match result {
            InferenceResult::Completion { choices, usage } => Self::Completion {
                choices: choices.clone(),
                usage: usage.clone(),
            },
            InferenceResult::Streaming { request_id, .. } => {
                // This shouldn't happen - streaming should use the channel_key constructor
                Self::Error {
                    message: format!(
                        "Streaming result for {} cannot be converted to serializable form directly",
                        request_id
                    ),
                }
            }
            InferenceResult::Error { message } => Self::Error {
                message: message.clone(),
            },
        }
    }
}
