use crate::openai::models::Config;
use serde::Deserialize;

fn default_num_channels() -> usize {
    3
}

fn default_activation() -> candle_nn::Activation {
    candle_nn::Activation::Silu
}

#[derive(Deserialize, Debug, Clone)]
pub struct VisionConfig {
    pub hidden_size: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub rope_theta: f64,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub head_dim: Option<usize>,
    pub num_attention_heads: usize,
    #[serde(default = "default_activation")]
    pub hidden_act: candle_nn::Activation,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct Mistral3Config {
    pub image_token_index: usize,
    pub multimodal_projector_bias: bool,
    pub projector_hidden_act: candle_nn::Activation,
    pub spatial_merge_size: usize,
    #[allow(dead_code)]
    pub vision_feature_layer: isize,
    pub text_config: Config,
    pub vision_config: VisionConfig,
}
