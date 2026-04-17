use serde::Deserialize;

fn deserialize_bool_or_f32<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;
    struct BoolOrF32Visitor;

    impl<'de> de::Visitor<'de> for BoolOrF32Visitor {
        type Value = Option<f32>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a float, integer, or boolean")
        }

        fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
            Ok(if v { Some(4.0) } else { None })
        }

        fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
            Ok(Some(v as f32))
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            Ok(Some(v as f32))
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(v as f32))
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
    }

    deserializer.deserialize_any(BoolOrF32Visitor)
}

fn default_floor_scale() -> Option<f32> {
    Some(8192.)
}
fn default_attn_scale() -> Option<f32> {
    Some(0.1)
}
fn default_attn_temperature_tuning() -> Option<f32> {
    Some(4.)
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f64>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(
        default = "default_floor_scale",
        deserialize_with = "deserialize_bool_or_f32"
    )]
    pub floor_scale: Option<f32>,
    #[serde(
        default = "default_attn_scale",
        deserialize_with = "deserialize_bool_or_f32"
    )]
    pub attn_scale: Option<f32>,
    #[serde(
        default = "default_attn_temperature_tuning",
        deserialize_with = "deserialize_bool_or_f32"
    )]
    pub attn_temperature_tuning: Option<f32>,
    #[serde(default)]
    pub use_qk_norm: bool,
    pub moe_layers: Option<Vec<usize>>,
    #[serde(default = "default_interleave_step")]
    pub interleave_moe_layer_step: usize,
    pub intermediate_size_mlp: Option<usize>,
    #[serde(default)]
    pub num_local_experts: usize,
    #[serde(default = "default_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_chunk_size")]
    pub attention_chunk_size: usize,
    pub hidden_act: Option<String>,
    pub rope_scaling:
        Option<std::collections::HashMap<String, crate::openai::models::ScalingValue>>,
}

fn default_interleave_step() -> usize {
    1
}
fn default_experts_per_tok() -> usize {
    1
}
fn default_chunk_size() -> usize {
    8192
}

impl TextConfig {
    pub fn moe_layers(&self) -> Vec<usize> {
        self.moe_layers.clone().unwrap_or_else(|| {
            if self.interleave_moe_layer_step == 0 {
                return vec![];
            }
            (self.interleave_moe_layer_step - 1..self.num_hidden_layers)
                .step_by(self.interleave_moe_layer_step)
                .collect()
        })
    }

    pub fn mlp_intermediate_size(&self) -> usize {
        self.intermediate_size_mlp.unwrap_or(self.intermediate_size)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub hidden_act: Option<String>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    pub intermediate_size: usize,
    pub vision_output_dim: usize,
    pub image_size: usize,
    pub patch_size: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_pixel_shuffle_ratio")]
    pub pixel_shuffle_ratio: f32,
    pub projector_input_dim: Option<usize>,
    pub projector_output_dim: Option<usize>,
    #[serde(default = "default_vision_feature_layer")]
    pub vision_feature_layer: isize,
    #[serde(default = "default_vision_rope_theta")]
    pub rope_theta: f32,
}

fn default_num_channels() -> usize {
    3
}
fn default_norm_eps() -> f64 {
    1e-5
}
fn default_pixel_shuffle_ratio() -> f32 {
    0.5
}
fn default_vision_feature_layer() -> isize {
    -1
}
fn default_vision_rope_theta() -> f32 {
    10000.0
}

impl VisionConfig {
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2) + 1
    }

    pub fn projector_in_dim(&self) -> usize {
        self.projector_input_dim.unwrap_or(4096)
    }

    pub fn projector_out_dim(&self) -> usize {
        self.projector_output_dim.unwrap_or(4096)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Llama4Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_index: usize,
}
