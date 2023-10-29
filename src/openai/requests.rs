use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<HashMap<String, String>>, //or String
    pub temperature: Option<f32>,               //0.7
    pub top_p: Option<f32>,                     //1.0
    pub n: Option<i32>,                         //1
    pub max_tokens: Option<i32>,                //None
    pub stop: Option<Vec<String>>,              // or String
    pub stream: Option<bool>,                   //false
    pub presence_penalty: Option<f32>,          //0.0
    pub frequency_penalty: Option<f32>,         //0.0
    pub logit_bias: Option<HashMap<String, f32>>, //None
    pub user: Option<String>,                   //None
}
