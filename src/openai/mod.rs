use self::{pipelines::llm_engine::LLMEngine, responses::APIError};
use crate::{
    openai::sampling_params::{GenerationConfig, SamplingParams},
    tools::{Tool, ToolChoice},
};
use candle_core::Device;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::SystemTime;
use tokenizers::{EncodeInput, Encoding, Tokenizer};
#[cfg(feature = "nccl")]
pub mod communicator;
pub mod distributed;
pub mod requests;
pub mod responses;
pub mod sampling_params;
pub mod streaming;
use either::Either;
use serde::{Deserialize, Serialize};
pub trait TokenizerWrapper<'s, E>
where
    E: Into<EncodeInput<'s>>,
{
    fn tokenize(&self, input: E) -> Result<Encoding, APIError>;
    fn detokenize(&self, input: &[u32]) -> Result<String, APIError>;
}

impl<'s, E> TokenizerWrapper<'s, E> for Tokenizer
where
    E: Into<EncodeInput<'s>>,
{
    fn tokenize(&self, input: E) -> Result<Encoding, APIError> {
        self.encode(input, false).map_err(APIError::from)
    }

    fn detokenize(&self, input: &[u32]) -> Result<String, APIError> {
        self.decode(input, false).map_err(APIError::from)
    }
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub max_model_len: usize,
    pub default_max_tokens: usize,
    pub generation_cfg: Option<GenerationConfig>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct MaxModelLen(
    #[serde(with = "either::serde_untagged")] pub Either<Option<usize>, Option<f64>>,
);

#[derive(Clone, Debug, serde::Deserialize)]
#[allow(unused)]
pub struct TokenContent {
    content: Option<String>,
    lstrip: Option<bool>,
    normalized: Option<bool>,
    rstrip: Option<bool>,
    single_word: Option<bool>,
}
#[derive(Deserialize, Debug, Clone)]
pub struct BosEosToken(
    #[serde(with = "either::serde_untagged")] pub Either<Option<String>, Option<TokenContent>>,
);

#[derive(Clone, Debug, serde::Deserialize)]
pub struct TokenizerConfig {
    pub model_max_length: MaxModelLen,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub chat_template: Option<String>,
    pub bos_token: Option<BosEosToken>,
    pub eos_token: Option<BosEosToken>,
}

pub struct OpenAIServerData {
    pub model: Arc<RwLock<LLMEngine>>,
    pub pipeline_config: PipelineConfig,
    pub record_conversation: bool,
    pub device: Device,
    pub mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TaskData {
    pub seq_id: usize,
    pub group_id: usize,
    pub prompt: Vec<u32>,
    pub request_id: String,
    pub created: SystemTime,
    pub sampling_params: SamplingParams,
    pub use_logprobs: bool,
    pub is_embedding: bool,
    pub encoding_format: requests::EncodingFormat,
    pub embedding_type: requests::EmbeddingType,
}

pub mod conversation;
pub mod logits_processor;
pub mod models;
pub mod openai_server;
pub mod pipelines;
pub mod utils;

#[derive(Debug, Clone)]
enum ToolChoiceKind {
    Auto,
    None,
    Function(String),
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ResolvedToolConfig {
    pub tools: Vec<Tool>,
    choice: ToolChoiceKind,
}

fn normalize_tool_choice(choice: &Option<ToolChoice>) -> ToolChoiceKind {
    match choice {
        None => ToolChoiceKind::Auto,
        Some(ToolChoice::Function { function, .. }) => {
            ToolChoiceKind::Function(function.name.clone())
        }
        Some(ToolChoice::Auto(value)) | Some(ToolChoice::None(value)) => match value.as_str() {
            "none" => ToolChoiceKind::None,
            "auto" => ToolChoiceKind::Auto,
            _ => ToolChoiceKind::Auto,
        },
    }
}

pub fn resolve_tools_for_request(
    request_tools: &Option<Vec<Tool>>,
    tool_choice: &Option<ToolChoice>,
    mcp_manager: Option<&Arc<crate::mcp::McpClientManager>>,
) -> Result<ResolvedToolConfig, APIError> {
    let choice = normalize_tool_choice(tool_choice);
    let mut tools = if let Some(req_tools) = request_tools {
        if req_tools.is_empty() {
            Vec::new()
        } else {
            req_tools.clone()
        }
    } else if let Some(manager) = mcp_manager {
        manager.cached_tools()
    } else {
        Vec::new()
    };

    if matches!(choice, ToolChoiceKind::None) {
        tools.clear();
        return Ok(ResolvedToolConfig { tools, choice });
    }

    if let ToolChoiceKind::Function(name) = &choice {
        if tools.is_empty() {
            return Err(APIError::new(format!(
                "tool_choice '{}' requires tools to be provided.",
                name
            )));
        }
        tools.retain(|tool| tool.function.name == *name);
        if tools.is_empty() {
            return Err(APIError::new(format!(
                "tool_choice '{}' not found in tools.",
                name
            )));
        }
    }

    Ok(ResolvedToolConfig { tools, choice })
}
