use crate::{openai::models::Config, serde_default};
use candle_nn::Activation;
use serde::Deserialize;

serde_default!(usize, default_hidden_size, 768);
serde_default!(usize, default_intermediate_size, 3072);
serde_default!(usize, default_num_hidden_layers, 12);
serde_default!(usize, default_num_attention_heads, 12);
serde_default!(usize, default_num_channels, 3);
serde_default!(usize, default_image_size, 224);
serde_default!(usize, default_patch_size, 16);
serde_default!(Activation, default_hidden_act, Activation::GeluPytorchTanh);
serde_default!(f64, default_layer_norm_eps, 1e-6);
serde_default!(bool, default_has_vision, true);

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma3VLConfig {
    pub text_config: Config,
    pub vision_config: VisionConfig,
    pub image_token_index: usize,
    pub mm_tokens_per_image: usize,
    #[serde(default = "default_has_vision")]
    pub has_vision: bool,
}
