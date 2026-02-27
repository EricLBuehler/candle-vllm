pub mod deepseek;
pub mod gemma;
pub mod gemma3;
pub mod glm4;
pub mod layers;
pub mod linear;
pub mod llama;
pub mod mistral;
pub mod phi2;
pub mod phi4;
pub mod quantized_glm4;
pub mod quantized_llama;
pub mod quantized_phi3;
pub mod quantized_qwen;
pub mod quantized_qwen3_moe;
pub mod qwen;
pub mod qwen3_5;
pub mod qwen3_5_moe;
pub mod qwen3_moe;
pub mod stable_lm;
pub mod utils;
pub mod yi;
use crate::openai::distributed::Comm;
use crate::{InputMetadata, PagedAttention};
use candle_core::{Device, Result, Tensor};
use either::Either;
pub use layers::{attention, mask, mlp, rotary_emb};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum ScalingValue {
    Single(f64),
    Vec(Vec<f64>),
    String(String),
    Bool(bool),
}

#[derive(Deserialize, Debug, Clone)]
pub struct TokenID(
    #[serde(with = "either::serde_untagged")] pub Either<Option<u32>, Option<Vec<u32>>>,
);

#[derive(Deserialize, PartialEq, Debug, Clone)]
pub struct QuantConfig {
    pub quant_method: String,
    #[serde(default)]
    pub activation_scheme: Option<String>,
    #[serde(default)]
    pub weight_per_tensor: Option<bool>,
    #[serde(default)]
    pub act_per_tensor: Option<bool>,
    #[serde(default)]
    pub modules_to_not_convert: Option<Vec<String>>,
    #[serde(default)]
    pub bits: usize,
    #[serde(default)]
    pub group_size: i32,
    pub sym: Option<bool>,
    pub desc_act: Option<bool>,
    pub checkpoint_format: Option<String>,
    pub weight_block_size: Option<Vec<usize>>,
}

#[derive(Deserialize, Clone, Debug)]
pub enum TopkMethod {
    #[serde(rename = "noaux_tc")]
    NoAuxTc,
    #[serde(rename = "greedy")]
    Greedy,
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
}

#[derive(Deserialize, Clone, Debug)]
pub enum ScoringFunc {
    #[serde(rename = "softmax")]
    Softmax,
    #[serde(rename = "sigmoid")]
    Sigmoid,
}

#[derive(Deserialize, Debug, Clone)]
pub struct DeepSeekMoEConfig {
    pub num_experts_per_tok: Option<usize>,
    pub n_routed_experts: usize,
    pub moe_intermediate_size: usize,
    pub scoring_func: ScoringFunc,
    pub topk_method: TopkMethod,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub n_shared_experts: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_lora_rank: usize,
    pub first_k_dense_replace: usize,
    pub moe_layer_freq: usize,
    pub q_lora_rank: Option<usize>,
    pub rope_scaling: Option<DeepSeekRopeScaling>,
    pub n_group: usize,
    pub topk_group: usize,
    pub num_experts_offload_per_rank: Option<usize>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct QwenMoEConfig {
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: Option<usize>,
    pub num_experts: Option<usize>,
    pub mlp_only_layers: Option<Vec<usize>>,
    pub decoder_sparse_step: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: Option<f64>,
}

#[derive(Deserialize, Debug, Clone)]
pub enum MoEConfig {
    DeepSeekMoE(DeepSeekMoEConfig),
    QwenMoE(QwenMoEConfig),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaledRopeType {
    #[serde(alias = "su")]
    #[serde(alias = "longrope")]
    Su,
    #[serde(alias = "yarn")]
    Yarn,
    #[serde(alias = "dynamic")]
    Dynamic,
    #[serde(alias = "linear")]
    Linear,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DeepSeekRopeScaling {
    Yarn {
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        mscale: f32,
        mscale_all_dim: f32,
        factor: f32,
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
    },
    LinearOrDynamic {
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
        factor: f64,
    },
}

#[derive(Deserialize, Debug, Clone)]
pub struct ModelArch(
    #[serde(with = "either::serde_untagged")] pub Either<Option<String>, Option<Vec<String>>>,
);

#[derive(Deserialize, Debug, Clone)]
pub struct ModelArchConfig {
    pub architectures: Option<ModelArch>,
}

#[derive(Deserialize, Debug, Clone)]
struct MultiModalArchConfig {
    pub architectures: Option<Vec<String>>,
    pub text_config: Option<serde_json::Value>,
    pub vision_config: Option<serde_json::Value>,
}

#[doc(hidden)]
#[macro_export]
macro_rules! serde_default_cfg {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}
serde_default_cfg!(usize, max_seq_len, 8192);
serde_default_cfg!(bool, tie_word_embeddings, false);
serde_default_cfg!(f64, rope_theta, 10_000.0f64);
serde_default_cfg!(bool, qk_layernorm, false);
serde_default_cfg!(usize, intermediate_size, 0);

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub architectures: Option<Vec<String>>,
    pub hidden_size: usize,
    pub head_dim: Option<usize>,
    #[serde(
        default = "intermediate_size",
        alias = "ffn_hidden_size",
        alias = "feed_forward_length"
    )]
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "rope_theta")]
    pub rope_theta: f64,
    pub rope_local_base_freq: Option<f64>,
    pub bos_token_id: Option<TokenID>,
    pub eos_token_id: TokenID,
    #[serde(default = "max_seq_len")]
    pub max_seq_len: usize,
    pub original_max_position_embeddings: Option<usize>,
    pub sliding_window: Option<usize>,
    pub sliding_window_pattern: Option<usize>,
    pub hidden_act: Option<candle_nn::Activation>,
    pub hidden_activation: Option<candle_nn::Activation>,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub rope_scaling: Option<HashMap<String, ScalingValue>>,
    pub max_position_embeddings: Option<usize>,
    pub attention_bias: Option<bool>,
    pub partial_rotary_factor: Option<f32>,
    #[serde(default = "qk_layernorm")]
    pub qk_layernorm: bool,
    #[serde(alias = "qkv_bias")]
    pub use_qkv_bias: Option<bool>,
    pub custom_stop_tokens: Option<Vec<String>>,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub moe_config: Option<MoEConfig>,
    pub quantization_config: Option<QuantConfig>,
    pub isq_quant: Option<String>,
    pub fp8_kvcache: Option<bool>,
    pub extra_config_json: Option<String>,
}

impl Config {
    fn apply_rope_overrides(&mut self) {
        if let Some(scaling) = &self.rope_scaling {
            if let Some(ScalingValue::Single(v)) = scaling.get("rope_theta") {
                self.rope_theta = *v;
            }
            if let Some(ScalingValue::Single(v)) = scaling.get("partial_rotary_factor") {
                self.partial_rotary_factor = Some(*v as f32);
            }
        }
        if let Some(raw) = self.extra_config_json.as_ref() {
            if let Ok(root) = serde_json::from_str::<serde_json::Value>(raw) {
                let cfg_root = root.get("text_config").unwrap_or(&root);
                if let Some(rope_params) = cfg_root.get("rope_parameters") {
                    if let Some(v) = rope_params.get("rope_theta").and_then(|v| v.as_f64()) {
                        self.rope_theta = v;
                    }
                    if let Some(v) = rope_params
                        .get("partial_rotary_factor")
                        .and_then(|v| v.as_f64())
                    {
                        self.partial_rotary_factor = Some(v as f32);
                    }
                }
            }
        }
    }

    pub fn load_config(filename: PathBuf) -> Result<Config> {
        match std::fs::read(filename.clone()) {
            Ok(f) => {
                let raw = String::from_utf8(f.clone()).map_err(candle_core::Error::wrap)?;
                let top_level_quant_config = serde_json::from_slice::<serde_json::Value>(&f)
                    .ok()
                    .and_then(|root| root.get("quantization_config").cloned())
                    .and_then(|v| serde_json::from_value::<QuantConfig>(v).ok());
                let mut config: Config =
                    if let Ok(mm_cfg) = serde_json::from_slice::<MultiModalArchConfig>(&f) {
                        if mm_cfg.text_config.is_some() && mm_cfg.vision_config.is_some() {
                            let mut cfg: Config =
                                serde_json::from_value(mm_cfg.text_config.clone().unwrap())
                                    .map_err(candle_core::Error::wrap)?;
                            cfg.architectures = mm_cfg.architectures.clone().or(cfg.architectures);
                            // For multimodal configs, quantization_config (including modules_to_not_convert)
                            // is usually defined at top level rather than under text_config.
                            if let Some(qcfg) = top_level_quant_config.clone() {
                                cfg.quantization_config = Some(qcfg);
                            }
                            cfg
                        } else {
                            match serde_json::from_slice::<Config>(&f) {
                                Ok(cfg) => cfg,
                                Err(root_err) => {
                                    if let Some(text_cfg) = mm_cfg.text_config {
                                        let mut cfg: Config = serde_json::from_value(text_cfg)
                                            .map_err(candle_core::Error::wrap)?;
                                        cfg.architectures =
                                            mm_cfg.architectures.clone().or(cfg.architectures);
                                        if let Some(qcfg) = top_level_quant_config.clone() {
                                            cfg.quantization_config = Some(qcfg);
                                        }
                                        cfg
                                    } else {
                                        return Err(candle_core::Error::wrap(root_err));
                                    }
                                }
                            }
                        }
                    } else {
                        serde_json::from_slice(&f).map_err(candle_core::Error::wrap)?
                    };
                config.extra_config_json = Some(raw);
                config.apply_rope_overrides();
                Ok(config)
            }
            Err(e) => panic!(
                "Unable to load config file {:?}\n ***Tips: use `--f` to specify GGUF file path!",
                e
            ),
        }
    }

    pub fn get_model_arch(filename: &PathBuf) -> Result<String> {
        match std::fs::read(filename) {
            Ok(f) => {
                let arch_config: ModelArchConfig =
                    serde_json::from_slice(&f).map_err(candle_core::Error::wrap)?;
                match arch_config.architectures {
                    Some(ModelArch(Either::Left(Some(arch)))) => Ok(arch),
                    Some(ModelArch(Either::Right(Some(archs)))) => {
                        if archs.len() > 1 {
                            candle_core::bail!("Multiple architectures found in config file {:?}, which is not supported!", archs);
                        } else if archs.is_empty() {
                            candle_core::bail!("No architectures found in config file {:?}!", filename);
                        }
                        Ok(archs[0].clone())
                    }
                    _=> {
                        candle_core::bail!(
                            "No architectures found in config file {:?}!",
                            filename
                        );
                    }
                }
            }
            Err(e) => panic!(
                "Unable to get model arch from config file {:?}\n ***Tips: use `--f` to specify GGUF file path!",
                e
            ),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Qwen3HybridRawConfig {
    #[serde(alias = "layer_types")]
    pub layers_block_type: Option<Vec<String>>,
    #[serde(alias = "linear_conv_kernel_dim")]
    pub conv_kernel_size: Option<usize>,
    pub full_attention_interval: Option<usize>,
    pub linear_num_heads: Option<usize>,
    #[serde(alias = "linear_num_key_heads")]
    pub linear_num_key_heads: Option<usize>,
    #[serde(alias = "linear_num_value_heads")]
    pub linear_num_value_heads: Option<usize>,
    pub linear_num_key_value_heads: Option<usize>,
    pub linear_key_head_dim: Option<usize>,
    pub linear_value_head_dim: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Qwen3HybridConfig {
    pub layer_types: Vec<String>,
    pub conv_kernel_size: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
}

pub fn is_qwen3_hybrid_arch_name(arch: &str) -> bool {
    matches!(
        arch,
        "Qwen3_5ForCausalLM"
            | "Qwen3_5MoeForCausalLM"
            | "Qwen3NextForCausalLM"
            | "Qwen3_5ForConditionalGeneration"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForConditionalGeneration"
    )
}

fn is_qwen3_hybrid_arch(config: &Config) -> bool {
    let arch = config.architectures.as_ref().and_then(|a| a.first());
    arch.map(|a| is_qwen3_hybrid_arch_name(a)).unwrap_or(false)
}

fn qwen3_hybrid_raw_from_extra_config(config: &Config) -> Option<Qwen3HybridRawConfig> {
    if !is_qwen3_hybrid_arch(config) {
        return None;
    }
    let extra = config.extra_config_json.as_ref()?;
    let root = serde_json::from_str::<serde_json::Value>(extra).ok()?;
    let cfg = root.get("text_config").cloned().unwrap_or(root);
    serde_json::from_value::<Qwen3HybridRawConfig>(cfg).ok()
}

pub fn resolve_qwen3_hybrid_config(config: &Config) -> Qwen3HybridConfig {
    let raw_cfg = qwen3_hybrid_raw_from_extra_config(config).unwrap_or_default();

    let mut layer_types = if let Some(layer_types) = raw_cfg.layers_block_type {
        layer_types
    } else if let Some(interval) = raw_cfg.full_attention_interval {
        if interval > 0 {
            (0..config.num_hidden_layers)
                .map(|idx| {
                    if (idx + 1) % interval == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect::<Vec<_>>()
        } else {
            vec!["full_attention".to_string(); config.num_hidden_layers]
        }
    } else {
        vec!["full_attention".to_string(); config.num_hidden_layers]
    };

    for layer_type in layer_types.iter_mut() {
        if layer_type == "attention" {
            *layer_type = "full_attention".to_string();
        }
    }
    if layer_types.len() != config.num_hidden_layers {
        tracing::warn!(
            "Qwen3 hybrid layer_types length {} != num_hidden_layers {}, fallback to full_attention.",
            layer_types.len(),
            config.num_hidden_layers
        );
        layer_types = vec!["full_attention".to_string(); config.num_hidden_layers];
    }

    let num_v_heads = raw_cfg
        .linear_num_value_heads
        .or(raw_cfg.linear_num_heads)
        .unwrap_or(config.num_attention_heads);
    let num_k_heads = raw_cfg
        .linear_num_key_heads
        .or(raw_cfg.linear_num_key_value_heads)
        .unwrap_or(num_v_heads);
    let key_head_dim = raw_cfg.linear_key_head_dim.unwrap_or(
        config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads),
    );
    let value_head_dim = raw_cfg.linear_value_head_dim.unwrap_or(key_head_dim);
    let conv_kernel_size = raw_cfg.conv_kernel_size.unwrap_or(4);

    Qwen3HybridConfig {
        layer_types,
        conv_kernel_size,
        num_v_heads,
        num_k_heads,
        key_head_dim,
        value_head_dim,
    }
}

pub fn qwen3_hybrid_layer_types(config: &Config) -> Option<Vec<String>> {
    if !is_qwen3_hybrid_arch(config) {
        return None;
    }
    Some(resolve_qwen3_hybrid_config(config).layer_types)
}

impl Config {
    pub fn kv_cache_num_layers(&self) -> usize {
        if let Some(layer_types) = qwen3_hybrid_layer_types(self) {
            return layer_types
                .iter()
                .filter(|lt| lt.as_str() == "full_attention")
                .count();
        }
        self.num_hidden_layers
    }

    pub fn get_head_size(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn q_head_dim(&self) -> usize {
        match &self.moe_config {
            Some(MoEConfig::DeepSeekMoE(cfg)) => cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            _ => self.get_head_size(),
        }
    }

    pub fn k_head_dim(&self) -> usize {
        match &self.moe_config {
            Some(MoEConfig::DeepSeekMoE(cfg)) => {
                //q_head_dim
                cfg.qk_rope_head_dim + cfg.qk_nope_head_dim
            }
            _ => self.get_head_size(),
        }
    }

    pub fn v_head_dim(&self) -> usize {
        match &self.moe_config {
            Some(MoEConfig::DeepSeekMoE(cfg)) => {
                //q_head_dim
                cfg.qk_rope_head_dim + cfg.qk_nope_head_dim
            }
            _ => self.get_head_size(),
        }
    }
}

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

pub struct NaiveAttention {
    kv_cache: RefCell<KvCache>,
    num_kv_groups: usize,
    scale: f64,
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x.to_owned())
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

impl NaiveAttention {
    pub fn new(cfg: &Config, sliding_window: Option<usize>) -> Self {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads.unwrap();
        let num_kv_groups = num_heads / num_kv_heads;
        let scale = 1f64 / f64::sqrt(cfg.head_dim.unwrap() as f64);

        let kv_cache = if let Some(sliding_window) = sliding_window {
            KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(2, sliding_window))
        } else {
            KvCache::Normal(candle_nn::kv_cache::KvCache::new(2, cfg.max_seq_len))
        };
        Self {
            kv_cache: RefCell::new(kv_cache),
            num_kv_groups,
            scale,
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        softcapping: Option<f64>,
    ) -> Result<Tensor> {
        let (b_sz, _, seq_len, _) = q.dims4()?;
        {
            if seq_len > 1 {
                self.clear_kv_cache();
            }
        }
        let mut cache = self.kv_cache.borrow_mut();
        let (k, v) = match &mut *cache {
            KvCache::Normal(c) => c.append(k, v)?,
            KvCache::Rotating(c) => c.append(k, v)?,
        };

        let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(&v, self.num_kv_groups)?.contiguous()?;

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        let attn_weights = match softcapping {
            None => attn_weights,
            Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
        };

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(&mask[0])?,
        };
        let attn_out = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        attn_out
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, ()))
    }

    pub fn clear_kv_cache(&self) {
        let mut cache = self.kv_cache.borrow_mut();
        match &mut *cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }
}

pub enum AttentionSelect {
    Naive(NaiveAttention),
    Paged(PagedAttention),
}

impl AttentionSelect {
    pub fn new(
        cfg: &Config,
        sliding_window: Option<usize>,
        comm: Rc<Comm>,
        device: &Device,
        paged: bool,
    ) -> Self {
        if !paged {
            AttentionSelect::Naive(NaiveAttention::new(cfg, sliding_window))
        } else {
            let head_dim = cfg.head_dim.unwrap();
            let attention_heads = cfg.num_attention_heads / comm.world_size();
            let kv_heads = cfg.num_key_value_heads.unwrap() / comm.world_size();
            AttentionSelect::Paged(
                PagedAttention::new(
                    attention_heads,
                    head_dim,
                    1. / ((head_dim as f32).sqrt()),
                    Some(kv_heads),
                    sliding_window,
                    device.clone(),
                    None,
                    cfg.fp8_kvcache.unwrap_or(false),
                )
                .unwrap(),
            )
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        softcapping: Option<f64>,
    ) -> Result<Tensor> {
        match self {
            AttentionSelect::Naive(att) => att.forward(q, k, v, attention_mask, softcapping),
            AttentionSelect::Paged(pag) => {
                let (b_sz, _, seq_len, _) = q.dims4()?;
                pag.forward(
                    q,
                    k,
                    v,
                    attention_mask,
                    cache.map(|(k_, _)| k_.clone()),
                    cache.map(|(_, v_)| v_.clone()),
                    input_metadata,
                    softcapping,
                )?
                .reshape((b_sz, seq_len, ()))
            }
        }
    }
}
