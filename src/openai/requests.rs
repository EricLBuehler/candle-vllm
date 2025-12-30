use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Messages {
    Map(Vec<HashMap<String, String>>),
    Literal(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopTokens {
    Multi(Vec<String>),
    Single(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Messages,
    pub temperature: Option<f32>, //0.7
    pub top_p: Option<f32>,       //1.0
    pub min_p: Option<f32>,       //0.0
    #[serde(default)]
    pub n: Option<usize>, //1
    pub max_tokens: Option<usize>, //None
    #[serde(default)]
    pub stop: Option<StopTokens>,
    #[serde(default)]
    pub stream: Option<bool>, //false
    #[serde(default)]
    pub presence_penalty: Option<f32>, //0.0
    pub repeat_last_n: Option<usize>, //0.0
    #[serde(default)]
    pub frequency_penalty: Option<f32>, //0.0
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>, //None
    #[serde(default)]
    pub user: Option<String>, //None
    pub top_k: Option<isize>,         //-1
    #[serde(default)]
    pub best_of: Option<usize>, //None
    #[serde(default)]
    pub use_beam_search: Option<bool>, //false
    #[serde(default)]
    pub ignore_eos: Option<bool>, //false
    #[serde(default)]
    pub skip_special_tokens: Option<bool>, //false
    #[serde(default)]
    pub stop_token_ids: Option<Vec<usize>>, //[]
    #[serde(default)]
    pub logprobs: Option<bool>, //false
    pub thinking: Option<bool>,       //false
    #[serde(default)]
    pub tools: Option<Vec<crate::tools::Tool>>,
    #[serde(default)]
    pub tool_choice: Option<crate::tools::ToolChoice>,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        Self {
            model: Some("default".to_string()),
            messages: Messages::Literal("".to_string()),
            temperature: None,
            top_p: None,
            min_p: None,
            n: None,
            max_tokens: None,
            stop: None,
            stream: None,
            presence_penalty: None,
            repeat_last_n: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            top_k: None,
            best_of: None,
            use_beam_search: None,
            ignore_eos: None,
            skip_special_tokens: None,
            stop_token_ids: None,
            logprobs: None,
            thinking: None,
            tools: None,
            tool_choice: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    String(String),
    MultiString(Vec<String>),
    Tokens(Vec<u32>),
    MultiTokens(Vec<Vec<u32>>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::String(s) => vec![s],
            EmbeddingInput::MultiString(s) => s,
            EmbeddingInput::Tokens(_) => unimplemented!("Token input not supported yet"),
            EmbeddingInput::MultiTokens(_) => unimplemented!("Token input not supported yet"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingType {
    Last,
    Mean, //default
}

impl Default for EmbeddingType {
    fn default() -> Self {
        Self::Mean
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: EncodingFormat,
    #[serde(default)]
    pub embedding_type: EmbeddingType,
}

impl Default for EncodingFormat {
    fn default() -> Self {
        Self::Float
    }
}
