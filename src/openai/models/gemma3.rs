use super::{Config, QuantConfig};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{
    embedding, Comm, ReplicatedLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
    VarBuilder,
};
use crate::openai::models::RopeScaling;
use crate::openai::models::TokenID;
use crate::paged_attention::input_metadata::InputMetadata;
use crate::SpecificConfig;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_core as candle;
use candle_nn::{Activation, RmsNorm};
use either::Either;
use std::collections::HashMap;
use std::iter::zip;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

#[doc(hidden)]
#[macro_export]
macro_rules! serde_default {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}
#[derive(serde::Deserialize, Debug, Clone)]
pub(crate) struct Gemma3RopeScaling(
    #[serde(with = "either::serde_untagged")] pub Either<f64, String>,
);

#[derive(serde::Deserialize, Debug, Clone)]
pub struct GemmaTextConfig {
    #[serde(default = "vocab_size")]
    pub(crate) vocab_size: usize,
    #[serde(default = "hidden_size")]
    pub(crate) hidden_size: usize,
    #[serde(default = "intermediate_size")]
    pub(crate) intermediate_size: usize,
    #[serde(default = "num_hidden_layers")]
    pub(crate) num_hidden_layers: usize,
    #[serde(default = "num_attention_heads")]
    pub(crate) num_attention_heads: usize,
    #[serde(default = "num_key_value_heads")]
    pub(crate) num_key_value_heads: usize,
    #[serde(default = "head_dim")]
    pub(crate) head_dim: usize,
    #[serde(default = "hidden_activation")]
    pub(crate) hidden_activation: Activation,
    #[serde(default = "max_position_embeddings")]
    pub(crate) max_position_embeddings: usize,
    #[serde(default = "rms_norm_eps")]
    pub(crate) rms_norm_eps: f64,
    pub(crate) eos_token_id: Option<TokenID>,
    pub(crate) bos_token_id: Option<TokenID>,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    #[serde(default = "rope_theta")]
    pub(crate) rope_theta: f64,
    #[serde(default = "attention_bias")]
    pub(crate) attention_bias: bool,
    // #[serde(default = "query_pre_attn_scalar")]
    // pub(crate) query_pre_attn_scalar: usize,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) final_logit_softcapping: Option<f64>,
    pub(crate) attn_logit_softcapping: Option<f64>,
    #[serde(default = "rope_local_base_freq")]
    pub(crate) rope_local_base_freq: f64,
    #[serde(default = "sliding_window_pattern")]
    pub(crate) sliding_window_pattern: usize,
    pub(crate) quantization_config: Option<QuantConfig>,
    pub(crate) rope_scaling: Option<HashMap<String, Gemma3RopeScaling>>,
}

serde_default!(usize, vocab_size, 262208);
serde_default!(usize, hidden_size, 2304);
serde_default!(usize, intermediate_size, 9216);
serde_default!(usize, num_hidden_layers, 26);
serde_default!(usize, num_attention_heads, 8);
serde_default!(usize, num_key_value_heads, 4);
serde_default!(usize, head_dim, 256);
serde_default!(usize, max_position_embeddings, 131072);
serde_default!(f64, rms_norm_eps, 1e-6);
serde_default!(bool, tie_word_embeddings, true);
serde_default!(f64, rope_theta, 1_000_000.0);
serde_default!(bool, attention_bias, false);
// serde_default!(usize, query_pre_attn_scalar, 256);
serde_default!(f64, rope_local_base_freq, 10_000.0);
serde_default!(usize, sliding_window_pattern, 6);
serde_default!(Activation, hidden_activation, Activation::Silu);

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Gemma3Config {
    pub bos_token_id: Option<TokenID>,
    pub eos_token_id: Option<TokenID>,
    pub text_config: GemmaTextConfig,
}

impl Gemma3Config {
    pub fn into_config(
        self,
        use_flash_attn: bool,
        kv_cache_dtype: DType,
        scfg: &SpecificConfig,
    ) -> Config {
        let bos_token_id = self
            .text_config
            .bos_token_id
            .or(self.bos_token_id)
            .unwrap_or(super::TokenID(Either::Left(Some(2))));

        let eos_token_id = self
            .text_config
            .eos_token_id
            .or(self.eos_token_id)
            .unwrap_or(super::TokenID(Either::Left(Some(1))));

        let ropescaling = if self.text_config.rope_scaling.is_some() {
            let mut ropescaling = HashMap::<String, RopeScaling>::new();
            for (key, value) in self.text_config.rope_scaling.as_ref().unwrap() {
                match value {
                    Gemma3RopeScaling(Either::Left(l)) => {
                        ropescaling.insert(key.to_string(), RopeScaling(Either::Left(vec![*l])));
                    }
                    Gemma3RopeScaling(Either::Right(r)) => {
                        ropescaling
                            .insert(key.to_string(), RopeScaling(Either::Right(r.to_string())));
                    }
                }
            }
            Some(ropescaling)
        } else {
            None
        };

        Config {
            hidden_size: self.text_config.hidden_size,
            head_dim: Some(self.text_config.head_dim),
            intermediate_size: self.text_config.intermediate_size,
            vocab_size: self.text_config.vocab_size,
            num_hidden_layers: self.text_config.num_hidden_layers,
            num_attention_heads: self.text_config.num_attention_heads,
            num_key_value_heads: self.text_config.num_key_value_heads,
            rms_norm_eps: self.text_config.rms_norm_eps,
            rope_theta: self.text_config.rope_theta,
            rope_local_base_freq: Some(self.text_config.rope_local_base_freq),
            use_flash_attn,
            bos_token_id,
            eos_token_id,
            max_seq_len: self.text_config.max_position_embeddings,
            sliding_window: self.text_config.sliding_window,
            sliding_window_pattern: Some(self.text_config.sliding_window_pattern),
            hidden_act: Some(self.text_config.hidden_activation),
            tie_word_embeddings: self.text_config.tie_word_embeddings,
            rope_scaling: ropescaling,
            original_max_position_embeddings: None,
            attention_bias: self.text_config.attention_bias,
            partial_rotary_factor: None,
            qk_layer_rms_norm: None,
            kv_cache_dtype,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            specific_config: scfg.clone(),
            attn_logit_softcapping: self.text_config.attn_logit_softcapping,
            final_logit_softcapping: self.text_config.final_logit_softcapping,
            quantization_config: self.text_config.quantization_config,
            moe_config: None,
        }
    }
}

fn rms_norm(dim: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get(dim, "weight")?;
    Ok(RmsNorm::new((weight + 1.0f64)?, eps))
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    sin_sliding: Option<Tensor>,
    cos_sliding: Option<Tensor>,
}

impl RotaryEmbedding {
    pub fn create_cache(
        _dtype: DType,
        local_sliding_window: Option<usize>,
        cfg: &Config,
        dev: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let rope_freq = local_sliding_window
            .and(cfg.rope_local_base_freq)
            .unwrap_or(cfg.rope_theta);

        let dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let max_seq_len = cfg.max_seq_len;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_freq.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let factor = 1.0f64;
        //TODO: scaling does not work for gemma3 models (vLLM also drops the usage of scaling)
        // if local_sliding_window.is_some() && cfg.rope_scaling.is_some() {
        //     match &cfg.rope_scaling.as_ref().unwrap()["rope_type"] {
        //         RopeScaling(Either::Right(tp)) => {
        //             assert!(tp == "linear");
        //         }
        //         _ => {}
        //     }
        //     match &cfg.rope_scaling.as_ref().unwrap()["factor"] {
        //         RopeScaling(Either::Left(factors)) => {
        //             factor= factors[0] as f64;
        //         }
        //         _ => panic!("scaling factor not found for gemma3 model!"),
        //     }
        // }
        let t_len = (max_seq_len as f64 * factor) as u32;
        let t = Tensor::arange(0u32, t_len, dev)?
            .to_dtype(DType::F32)?
            .reshape((t_len as usize, 1))?;
        let t = (t / factor)?;
        let freqs = t.matmul(&inv_freq)?;
        Ok((freqs.sin()?, freqs.cos()?))
    }

    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let (sin, cos) = RotaryEmbedding::create_cache(dtype, None, cfg, dev)?;
        let (sin_sliding, cos_sliding) = if cfg.sliding_window.is_some() {
            let (sin, cos) = RotaryEmbedding::create_cache(dtype, cfg.sliding_window, cfg, dev)?;
            (Some(sin), Some(cos))
        } else {
            (None, None)
        };

        Ok(Self {
            sin,
            cos,
            sin_sliding,
            cos_sliding,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &[Vec<usize>],
        is_sliding: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();
        for (b, seqlen_offset) in zip(0..b_sz, input_positions) {
            let (cos, sin) =
                if is_sliding && self.sin_sliding.is_some() && self.cos_sliding.is_some() {
                    let cos =
                        self.cos_sliding
                            .as_ref()
                            .unwrap()
                            .narrow(0, seqlen_offset[0], seq_len)?;
                    let sin =
                        self.sin_sliding
                            .as_ref()
                            .unwrap()
                            .narrow(0, seqlen_offset[0], seq_len)?;
                    (cos, sin)
                } else {
                    let cos = self.cos.narrow(0, seqlen_offset[0], seq_len)?;
                    let sin = self.sin.narrow(0, seqlen_offset[0], seq_len)?;
                    (cos, sin)
                };

            let x_q = q.narrow(0, b, 1)?;
            let x_k = k.narrow(0, b, 1)?;
            let q_embed = candle_nn::rotary_emb::rope(&x_q, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&x_k, &cos, &sin)?;
            q_embeds.push(q_embed);
            k_embeds.push(k_embed);
        }
        Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
    }
}

struct Mlp {
    gate_proj: TensorParallelColumnLinear,
    up_proj: TensorParallelColumnLinear,
    down_proj: TensorParallelRowLinear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            intermediate_sz,
            false,
            vb.pp("gate_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let up_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            intermediate_sz,
            false,
            vb.pp("up_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let down_proj = TensorParallelRowLinear::load_with_hints(
            intermediate_sz,
            hidden_sz,
            false,
            vb.pp("down_proj"),
            comm,
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act.unwrap(),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.act_fn.forward(&self.gate_proj.forward(xs)?)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(&lhs * &rhs)?)
    }
}

struct Attention {
    q_proj: TensorParallelColumnLinear,
    k_proj: TensorParallelColumnLinear,
    v_proj: TensorParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    attn: super::AttentionSelect,
    local_sliding_window: Option<usize>,
}

impl Attention {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
        rotary_emb: Arc<RotaryEmbedding>,
        local_sliding_window: Option<usize>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim.unwrap();
        let bias = cfg.attention_bias;

        let q_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            num_heads * head_dim,
            bias,
            vb.pp("q_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let k_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            num_kv_heads * head_dim,
            bias,
            vb.pp("k_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let v_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            num_kv_heads * head_dim,
            bias,
            vb.pp("v_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;

        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
            hidden_sz,
            bias,
            vb.pp("o_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let attention_heads = cfg.num_attention_heads / comm.world_size();
        let kv_heads = cfg.num_key_value_heads / comm.world_size();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: attention_heads,
            num_kv_heads: kv_heads,
            head_dim,
            rotary_emb: rotary_emb.clone(),
            local_sliding_window,
            attn: super::AttentionSelect::new(
                cfg,
                local_sliding_window,
                comm.clone(),
                vb.device(),
                true,
            ),
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        input_positions: &[Vec<usize>],
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        softcapping: Option<f64>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let (q, k, v) = if seq_len == 1 {
            //no need transpose for seq_len == 1, change reshape dim
            let q = query_states.reshape((b_sz, self.num_heads, seq_len, self.head_dim))?;
            let k = key_states.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = value_states.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        } else {
            let q = query_states
                .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = key_states
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = value_states
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v.contiguous()?)
        };

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let (q, k) = self.rotary_emb.apply_rotary_emb_qkv(
            &q.to_dtype(DType::F32)?,
            &k.to_dtype(DType::F32)?,
            input_positions,
            self.local_sliding_window.is_some(),
        )?;

        let (q, k) = (q.to_dtype(v.dtype())?, k.to_dtype(v.dtype())?);
        let y = self.attn.forward(
            &q,
            &k,
            &v,
            attention_mask,
            cache,
            input_metadata,
            softcapping,
        )?;

        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    sliding_window: Option<usize>,
}

impl DecoderLayer {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
        rotary_emb: Arc<RotaryEmbedding>,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            cfg,
            vb.pp("self_attn"),
            comm.clone(),
            rotary_emb.clone(),
            sliding_window,
        )?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"), comm.clone())?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;

        let pre_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;

        let post_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;

        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_attention_layernorm,
            sliding_window,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        input_positions: &[Vec<usize>],
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        softcapping: Option<f64>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            input_positions,
            cache,
            input_metadata,
            softcapping,
        )?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }
}

pub struct Gemma3 {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: ReplicatedLinear,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    cfg: Config,
}

impl Gemma3 {
    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let vb_m = vb.pp("language_model.model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let reporter = progress_reporter.clone();
        for layer_idx in 0..cfg.num_hidden_layers {
            let is_sliding_window =
                if cfg.sliding_window.is_some() && cfg.sliding_window_pattern.is_some() {
                    (layer_idx + 1) % cfg.sliding_window_pattern.unwrap() > 0
                } else {
                    false
                };
            let layer = DecoderLayer::new(
                cfg,
                vb_l.pp(layer_idx),
                comm.clone(),
                rotary_emb.clone(),
                is_sliding_window.then_some(cfg.sliding_window.unwrap()),
            )?;
            layers.push(layer);
            reporter.write().unwrap().set_progress(layer_idx + 1);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = ReplicatedLinear::from_weight_bias(embed_tokens.embeddings().clone(), None)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            hidden_size: cfg.hidden_size,
            cfg: cfg.clone(),
        })
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        input_positions: &[Vec<usize>],
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len <= 1 {
            return Ok((None, None));
        }
        //normal mask
        let mask = super::get_attention_casual_mask(
            &self.device,
            self.dtype,
            batch_size,
            seq_len,
            input_positions,
            None,
        );

        let sliding_mask = super::get_attention_casual_mask(
            &self.device,
            self.dtype,
            batch_size,
            seq_len,
            input_positions,
            self.cfg.sliding_window,
        );

        Ok((mask, sliding_mask))
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_positions: &[Vec<usize>],
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(b_size, seq_len, input_positions)?;

        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                let mask = if layer.sliding_window.is_some() {
                    &sliding_attention_mask
                } else {
                    &attention_mask
                };

                xs = layer.forward(
                    &xs,
                    mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                    self.cfg.attn_logit_softcapping,
                )?
            }
        } else {
            for layer in self.layers.iter() {
                let mask = if layer.sliding_window.is_some() {
                    &sliding_attention_mask
                } else {
                    &attention_mask
                };
                xs = layer.forward(
                    &xs,
                    mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                    self.cfg.attn_logit_softcapping,
                )?
            }
        }

        let logits = xs
            .i((.., seq_len - 1, ..))?
            .contiguous()?
            .apply(&self.norm)?;
        let logits = self.lm_head.forward(&logits)?;

        let logits = match self.cfg.final_logit_softcapping {
            None => logits,
            Some(sc) => ((logits / sc)?.tanh()? * sc)?,
        };
        logits.to_dtype(DType::F32)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
