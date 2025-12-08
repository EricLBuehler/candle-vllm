//! OpenAI compatibility layer for candle-vllm.
//!
//! This crate provides OpenAI-compatible types and adapters for the candle-vllm inference engine.
//! It is designed for library-first usage, allowing embedding in applications without requiring
//! a full HTTP server.
//!
//! ## Modules
//!
//! - [`conversation`] - Chat template management and multi-turn conversation handling
//! - [`tool_calling`] - Tool call parsing for different model formats
//! - [`streaming`] - Streaming response types
//! - [`adapter`] - High-level adapter wrapping the inference engine
//! - [`model_registry`] - Model configuration and registry
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use candle_vllm_openai::{OpenAIAdapter, ChatCompletionRequest, Message};
//!
//! // Create an adapter wrapping an inference engine
//! let adapter = OpenAIAdapter::new(engine);
//!
//! // Build a request
//! let request = ChatCompletionRequest {
//!     model: "mistral".to_string(),
//!     messages: Messages::Chat(vec![Message::user("Hello!")]),
//!     // ... other options
//! };
//!
//! // Get a completion
//! let response = adapter.chat_completion(request).await?;
//! ```

// Local modules with additional functionality
pub mod conversation;
pub mod model_registry;
pub mod streaming;
pub mod tool_calling;

// Adapter module
pub mod adapter;

// ============================================================================
// Re-export types from candle-vllm-core for convenience
// This allows users to import everything from candle-vllm-openai
// ============================================================================

// Re-export request types from core
#[doc(inline)]
pub use candle_vllm_core::openai::requests::{
    ChatCompletionRequest, ChatMessage, ContentPart, FunctionCall, FunctionCallDelta,
    FunctionDefinition, ImageUrl, MessageContent, Messages, StopTokens, Tool, ToolCall,
    ToolCallDelta, ToolChoice, ToolChoiceFunction, ToolChoiceSpecific,
};

// Type alias for backward compatibility
pub type Message = ChatMessage;

// Type alias for backward compatibility
pub type StopCondition = StopTokens;

// Re-export response types from core
#[doc(inline)]
pub use candle_vllm_core::openai::responses::{
    create_tool_call, create_tool_call_delta_arguments, create_tool_call_delta_start, APIError,
    ChatChoice, ChatChoiceData, ChatCompletionChunk, ChatCompletionResponse,
    ChatCompletionUsageResponse, Choice, ChoiceData, WrapperLogprobs,
};

// Re-export sampling params logprobs type
#[doc(inline)]
pub use candle_vllm_core::openai::sampling_params::Logprobs;

// Type aliases for naming consistency
pub type ChatCompletionChunkChoice = Choice;
pub type ChunkChoiceData = ChoiceData;
pub type TopLogprob = candle_vllm_core::openai::sampling_params::TopLogprob;

// Re-export streaming types
#[doc(inline)]
pub use streaming::{ChatResponse, StreamingStatus};

// Re-export tool parsing
#[doc(inline)]
pub use tool_calling::{
    get_tool_parser, get_tool_parser_by_name, AutoToolParser, JsonToolParser, LlamaToolParser,
    MistralToolParser, ParsedOutput, ParsedToolCall, QwenToolParser, ToolCallParser,
};

// Re-export conversation types
#[doc(inline)]
pub use conversation::{
    format_tool_result_for_model, format_tool_result_json, format_tool_result_llama,
    format_tool_result_mistral, format_tool_result_qwen, format_tool_results_mistral,
    format_tools_for_template, format_tools_mistral, ApplyChatTemplateError, Conversation,
    DefaultConversation, DefaultConversationSeparators, ModelFamily, SeparatorStyle,
    ToolConversationBuilder,
};

#[doc(inline)]
pub use adapter::OpenAIAdapter;

#[doc(inline)]
pub use model_registry::ModelRegistry;
