#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use super::{
    Config, DeepSeekRopeScaling, MoEConfig, QuantConfig, ScoringFunc, SpecificConfig, TokenID,
    TopkMethod,
};
use crate::backend::custom_ops::moe::{masked_fill, NonZeroOp, SplitOp, TopKLastDimOp, TopKOutput};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{
    embedding, rms_norm, AllReduce, Comm, ReplicatedLinear, TensorParallelColumnLinear,
    TensorParallelRowLinear, VarBuilder,
};
use crate::paged_attention::input_metadata::InputMetadata;
use crate::paged_attention::PagedAttention;
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_core as candle;
use candle_nn::{Activation, Embedding, Module, RmsNorm};
use serde::Deserialize;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::f32::consts::PI;
use std::iter::{zip, FromIterator};
use std::rc::Rc;
use std::sync::{Arc, RwLock};

#[doc(hidden)]
#[macro_export]
macro_rules! serde_default_fn {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}

serde_default_fn!(f64, routed_scaling_factor, 1.0);
serde_default_fn!(TopkMethod, topk_method, TopkMethod::Greedy);
serde_default_fn!(usize, moe_layer_freq, 1);
serde_default_fn!(usize, first_k_dense_replace, 0);
serde_default_fn!(bool, norm_topk_prob, false);
serde_default_fn!(ScoringFunc, scoring_func, ScoringFunc::Softmax);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
// serde_default_fn!(bool, tie_word_embeddings, false);

#[derive(Deserialize, Clone, Debug)]
pub struct DeepSeekConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: Option<usize>,
    pub(crate) n_shared_experts: Option<usize>,
    pub(crate) n_routed_experts: Option<usize>,
    #[serde(default = "routed_scaling_factor")]
    pub(crate) routed_scaling_factor: f64,
    #[serde(default = "topk_method")]
    topk_method: TopkMethod,
    pub(crate) num_experts_per_tok: Option<usize>,
    #[serde(default = "moe_layer_freq")]
    pub(crate) moe_layer_freq: usize,
    #[serde(default = "first_k_dense_replace")]
    pub(crate) first_k_dense_replace: usize,
    // k dense layers
    #[serde(default = "norm_topk_prob")]
    pub(crate) norm_topk_prob: bool,
    #[serde(default = "scoring_func")]
    scoring_func: ScoringFunc,
    #[serde(default = "hidden_act")]
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    // #[serde(default = "tie_word_embeddings")]
    // pub(crate) tie_word_embeddings: bool,
    pub(crate) rope_theta: f32,
    pub(crate) rope_scaling: Option<DeepSeekRopeScaling>,
    // pub(crate) attention_bias: bool,
    pub(crate) q_lora_rank: Option<usize>,
    pub(crate) qk_rope_head_dim: usize,
    pub(crate) kv_lora_rank: usize,
    pub(crate) v_head_dim: usize,
    pub(crate) qk_nope_head_dim: usize,
    pub(crate) n_group: usize,
    pub(crate) topk_group: usize,
    pub(crate) sliding_window: Option<usize>,
    pub quantization_config: Option<QuantConfig>,
    pub bos_token_id: TokenID,
    pub eos_token_id: TokenID,
}

#[derive(Debug, Clone)]
pub struct DeepSeekV2RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl DeepSeekConfig {
    pub fn into_config(
        self,
        use_flash_attn: bool,
        kv_cache_dtype: DType,
        scfg: &SpecificConfig,
    ) -> Config {
        let moe_config = MoEConfig {
            num_experts_per_tok: self.num_experts_per_tok,
            n_routed_experts: self.n_routed_experts.unwrap_or(0),
            moe_intermediate_size: self.moe_intermediate_size,
            scoring_func: self.scoring_func,
            topk_method: self.topk_method,
            norm_topk_prob: self.norm_topk_prob,
            routed_scaling_factor: self.routed_scaling_factor,
            n_shared_experts: self.n_shared_experts,
            qk_nope_head_dim: self.qk_nope_head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim,
            v_head_dim: self.v_head_dim,
            kv_lora_rank: self.kv_lora_rank,
            first_k_dense_replace: self.first_k_dense_replace,
            moe_layer_freq: self.moe_layer_freq,
            rope_scaling: self.rope_scaling,
            q_lora_rank: self.q_lora_rank,
            n_group: self.n_group,
            topk_group: self.topk_group,
            num_experts_offload_per_rank: scfg.num_experts_offload_per_rank,
        };

        Config {
            hidden_size: self.hidden_size,
            head_dim: Some(self.hidden_size / self.num_attention_heads),
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: f64::from(self.rope_theta),
            rope_local_base_freq: None,
            use_flash_attn,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            max_seq_len: self.max_position_embeddings,
            sliding_window: self.sliding_window,
            sliding_window_pattern: None,
            hidden_act: Some(self.hidden_act),
            tie_word_embeddings: false,
            rope_scaling: None,
            original_max_position_embeddings: None,
            attention_bias: false,
            partial_rotary_factor: None,
            qk_layer_rms_norm: None,
            kv_cache_dtype,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            specific_config: scfg.clone(),
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: self.quantization_config,
            moe_config: Some(moe_config),
        }
    }
}

pub struct DeepSeekV2RopeConfig {
    pub rope_scaling: Option<DeepSeekRopeScaling>,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub qk_rope_head_dim: usize,
}

impl DeepSeekV2RotaryEmbedding {
    fn new_unscaled(cfg: &DeepSeekV2RopeConfig, _dtype: DType, dev: &Device) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.qk_rope_head_dim;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), &Device::Cpu)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let sin = freqs.sin()?.to_device(dev)?;
        let cos = freqs.cos()?.to_device(dev)?;

        Ok(Self { sin, cos })
    }

    fn yarn_find_correction_dim(
        num_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> f32 {
        (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2. * PI)).ln())
            / (2. * base.ln())
    }

    fn yarn_find_correction_range(
        low_rot: f32,
        high_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> (f32, f32) {
        let low =
            Self::yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
        let high =
            Self::yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
        (low.max(0.), high.min(dim as f32 - 1.))
    }

    fn yarn_linear_ramp_mask(min: f32, mut max: f32, dim: usize, dev: &Device) -> Result<Tensor> {
        if min == max {
            max += 0.001;
        }
        let linear_func =
            ((Tensor::arange(0f32, dim as f32, dev)? - min as f64)? / (max as f64 - min as f64))?;
        linear_func.clamp(0., 1.)
    }

    pub(crate) fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
        if scale <= 1. {
            return 1.;
        }
        0.1 * mscale * scale.ln() + 1.
    }

    #[allow(clippy::too_many_arguments)]
    fn new_yarn(
        cfg: &DeepSeekV2RopeConfig,
        _dtype: DType,
        dev: &Device,
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        factor: f32,
        mscale: f32,
        mscale_all_dim: f32,
    ) -> Result<Self> {
        let freq_extra: Vec<_> = (0..cfg.qk_rope_head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / cfg.qk_rope_head_dim as f32))
            .collect();
        let freq_extra_len = freq_extra.len();
        let freq_extra = Tensor::from_vec(freq_extra, freq_extra_len, &Device::Cpu)?;
        let freq_inter: Vec<_> = (0..cfg.qk_rope_head_dim)
            .step_by(2)
            .map(|i| 1f32 / (factor * cfg.rope_theta.powf(i as f32 / cfg.qk_rope_head_dim as f32)))
            .collect();
        let freq_inter_len = freq_inter.len();
        let freq_inter = Tensor::from_vec(freq_inter, (1, freq_inter_len), &Device::Cpu)?;

        let (low, high) = Self::yarn_find_correction_range(
            beta_fast,
            beta_slow,
            cfg.qk_rope_head_dim,
            cfg.rope_theta,
            original_max_position_embeddings,
        );
        let inv_freq_mask =
            (1. - Self::yarn_linear_ramp_mask(low, high, cfg.qk_rope_head_dim / 2, &Device::Cpu)?)?;
        let inv_freq = freq_inter
            .broadcast_mul(&(1. - &inv_freq_mask)?)?
            .broadcast_add(&freq_extra.broadcast_mul(&inv_freq_mask)?)?;

        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let mscale =
            Self::yarn_get_mscale(factor, mscale) / Self::yarn_get_mscale(factor, mscale_all_dim);
        let sin = (freqs.sin()? * mscale as f64)?.to_device(dev)?;
        let cos = (freqs.cos()? * mscale as f64)?.to_device(dev)?;

        Ok(Self { sin, cos })
    }

    pub fn new(cfg: &DeepSeekV2RopeConfig, dtype: DType, dev: &Device) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(DeepSeekRopeScaling::LinearOrDynamic {
                scaling_type: _,
                factor: _,
            }) => candle::bail!("linear and dynamic rope are not implemented yet!"),
            Some(DeepSeekRopeScaling::Yarn {
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                factor,
                mscale,
                mscale_all_dim,
                scaling_type: _,
            }) => Self::new_yarn(
                cfg,
                dtype,
                dev,
                *original_max_position_embeddings,
                *beta_fast,
                *beta_slow,
                *factor,
                *mscale,
                *mscale_all_dim,
            ),
            None => Self::new_unscaled(cfg, dtype, dev),
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &[Vec<usize>],
    ) -> Result<(Tensor, Tensor)> {
        let (batch, _h, seq_len, _n_embd) = q.dims4()?;
        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();
        for (b, seqlen_offset) in zip(0..batch, input_positions) {
            let sin = self.sin.narrow(0, seqlen_offset[0], seq_len)?;
            let cos = self.cos.narrow(0, seqlen_offset[0], seq_len)?;
            let q_cur = q.narrow(0, b, 1)?;
            let k_cur = k.narrow(0, b, 1)?;
            let q_embed = candle_nn::rotary_emb::rope_i(&q_cur.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope_i(&k_cur.contiguous()?, &cos, &sin)?;
            q_embeds.push(q_embed);
            k_embeds.push(k_embed);
        }
        Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
    }
}

impl MoEConfig {
    pub(crate) fn q_head_dim(&self) -> usize {
        self.qk_rope_head_dim + self.qk_nope_head_dim
    }

    fn softmax_scale(&self) -> f32 {
        let mut softmax_scale = 1.0 / (self.q_head_dim() as f32).sqrt();
        if let Some(DeepSeekRopeScaling::Yarn {
            mscale_all_dim,
            factor,
            ..
        }) = self.rope_scaling
        {
            let mscale = DeepSeekV2RotaryEmbedding::yarn_get_mscale(factor, mscale_all_dim);
            softmax_scale = softmax_scale * mscale * mscale;
        }
        softmax_scale
    }
}

enum QProj {
    Plain(TensorParallelColumnLinear),
    Lora {
        a: ReplicatedLinear,
        norm: RmsNorm,
        b: TensorParallelColumnLinear,
    },
}

impl QProj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Lora { a, norm, b } => b.forward(&norm.forward(&a.forward(xs)?)?),
            Self::Plain(lin) => lin.forward(xs),
        }
    }
}

struct Attention {
    q: QProj,
    kv_a_proj_with_mqa: ReplicatedLinear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: TensorParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
    cfg: Config,
    q_head_dim: usize,
    num_attention_heads: usize,
    attn: PagedAttention,
}

impl Attention {
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let q_head_dim = cfg.q_head_dim();
        let moe_cfg = cfg.moe_config.as_ref().unwrap();
        let q = match moe_cfg.q_lora_rank {
            Some(lora_rank) => {
                let a = ReplicatedLinear::load_b(
                    cfg.hidden_size,
                    lora_rank,
                    cfg.attention_bias,
                    vb.pp("q_a_proj"),
                    &cfg.specific_config.quant,
                    &cfg.quantization_config,
                )?;
                let norm = rms_norm(lora_rank, cfg.rms_norm_eps, vb.pp("q_a_layernorm"))?;
                let b = TensorParallelColumnLinear::load_with_hints(
                    lora_rank,
                    cfg.num_attention_heads * q_head_dim,
                    false,
                    vb.pp("q_b_proj"),
                    comm.clone(),
                    &cfg.specific_config.quant,
                    &cfg.quantization_config,
                )?;
                QProj::Lora { a, norm, b }
            }
            None => QProj::Plain(TensorParallelColumnLinear::load_with_hints(
                cfg.hidden_size,
                cfg.num_attention_heads * q_head_dim,
                false,
                vb.pp("q_proj"),
                comm.clone(),
                &cfg.specific_config.quant,
                &cfg.quantization_config,
            )?),
        };

        let kv_a_proj_with_mqa = ReplicatedLinear::load_b(
            cfg.hidden_size,
            moe_cfg.kv_lora_rank + moe_cfg.qk_rope_head_dim,
            cfg.attention_bias,
            vb.pp("kv_a_proj_with_mqa"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let kv_a_layernorm = rms_norm(
            moe_cfg.kv_lora_rank,
            cfg.rms_norm_eps,
            vb.pp("kv_a_layernorm"),
        )?;
        let kv_b_proj = TensorParallelColumnLinear::load_with_hints(
            moe_cfg.kv_lora_rank,
            cfg.num_attention_heads * (q_head_dim - moe_cfg.qk_rope_head_dim + moe_cfg.v_head_dim),
            false,
            vb.pp("kv_b_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;

        let o_proj = TensorParallelRowLinear::load_with_hints(
            cfg.num_attention_heads * moe_cfg.v_head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;

        let num_attention_heads = cfg.num_attention_heads / comm.world_size();
        let num_kv_heads = cfg.num_key_value_heads / comm.world_size();

        Ok(Self {
            q,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            cfg: cfg.clone(),
            q_head_dim,
            num_attention_heads,
            attn: PagedAttention::new(
                num_attention_heads,
                moe_cfg.v_head_dim,
                moe_cfg.softmax_scale(),
                Some(num_kv_heads),
                cfg.sliding_window,
                vb.device().clone(),
                None,
            )?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        input_positions: &[Vec<usize>],
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;
        let moe_cfg = self.cfg.moe_config.as_ref().unwrap();
        let (q_nope, mut q_pe) = {
            let q = self.q.forward(xs)?;
            let q = q.reshape((bs, seq_len, self.num_attention_heads, self.q_head_dim))?;
            let (q_nope, q_pe) = q.split2(
                &[moe_cfg.qk_nope_head_dim, moe_cfg.qk_rope_head_dim],
                D::Minus1,
            )?;
            let q_nope = q_nope.transpose(1, 2)?;
            let q_pe = q_pe.contiguous()?.transpose(1, 2)?;
            (q_nope, q_pe)
        };

        let compressed_kv = self.kv_a_proj_with_mqa.forward(xs)?;
        let (compressed_kv, k_pe) =
            compressed_kv.split2(&[moe_cfg.kv_lora_rank, moe_cfg.qk_rope_head_dim], D::Minus1)?;
        let mut k_pe = k_pe
            .reshape((bs, seq_len, 1, moe_cfg.qk_rope_head_dim))?
            .transpose(1, 2)?;
        let kv = {
            let kv = self
                .kv_b_proj
                .forward(&self.kv_a_layernorm.forward(&compressed_kv)?)?;
            kv.reshape((
                bs,
                seq_len,
                self.num_attention_heads,
                moe_cfg.qk_nope_head_dim + moe_cfg.v_head_dim,
            ))?
            .transpose(1, 2)?
        };

        let (k_nope, v) = kv.split2(&[moe_cfg.qk_nope_head_dim, moe_cfg.v_head_dim], D::Minus1)?;
        let mut v = v.contiguous()?;

        (q_pe, k_pe) = self.rotary_emb.forward(
            &q_pe.to_dtype(DType::F32)?,
            &k_pe.to_dtype(DType::F32)?,
            input_positions,
        )?;
        let (q_pe, k_pe) = (q_pe.to_dtype(v.dtype())?, k_pe.to_dtype(v.dtype())?);

        let q = Tensor::cat(&[q_nope, q_pe], D::Minus1)?.contiguous()?;
        let k_pe = k_pe.repeat((1, q.dim(1)?, 1, 1))?;
        let k = Tensor::cat(&[k_nope, k_pe], D::Minus1)?.contiguous()?;

        if self.q_head_dim != moe_cfg.v_head_dim {
            v = v
                .pad_with_zeros(D::Minus1, 0, self.q_head_dim - moe_cfg.v_head_dim)?
                .contiguous()?;
        }

        let mut y = self.attn.forward(
            &q,
            &k,
            &v,
            attention_mask,
            cache.map(|(k_, _)| k_.clone()),
            cache.map(|(_, v_)| v_.clone()),
            input_metadata,
            None,
        )?;

        if self.q_head_dim != moe_cfg.v_head_dim {
            y = y.narrow(D::Minus1, 0, moe_cfg.v_head_dim)?;
        }

        y = y.reshape((bs, seq_len, ()))?;

        self.o_proj.forward(&y)
    }
}

struct Mlp {
    gate: RefCell<ReplicatedLinear>,
    up: RefCell<ReplicatedLinear>,
    down: RefCell<ReplicatedLinear>,
    act: Activation,
    preloaded: Cell<bool>,
    can_be_offloaded: bool,
}

impl Mlp {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        hidden_size: Option<usize>,
        intermediate_size: Option<usize>,
        offload: bool,
    ) -> Result<Self> {
        let hidden_size = hidden_size.unwrap_or(cfg.hidden_size);
        let intermediate_size = intermediate_size.unwrap_or(cfg.intermediate_size);

        let mut gate = ReplicatedLinear::load_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp("gate_proj"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;

        let mut up = ReplicatedLinear::load_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp("up_proj"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;

        let mut down = ReplicatedLinear::load_no_bias(
            intermediate_size,
            hidden_size,
            vb.pp("down_proj"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;

        if offload {
            gate.offload()?;
            up.offload()?;
            down.offload()?;
        }
        Ok(Self {
            gate: RefCell::new(gate),
            up: RefCell::new(up),
            down: RefCell::new(down),
            preloaded: Cell::new(!offload),
            can_be_offloaded: offload,
            act: cfg.hidden_act.unwrap(),
        })
    }

    pub fn reload(&self) {
        if !self.preloaded.get() {
            let _ = self.gate.borrow_mut().reload();
            let _ = self.up.borrow_mut().reload();
            let _ = self.down.borrow_mut().reload();
            self.preloaded.set(true);
        }
    }

    pub fn offload(&self) {
        if self.can_be_offloaded && self.preloaded.get() {
            let _ = self.gate.borrow_mut().offload();
            let _ = self.up.borrow_mut().offload();
            let _ = self.down.borrow_mut().offload();
            self.preloaded.set(false);
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate.borrow().forward(xs)?.apply(&self.act)?;
        let rhs = self.up.borrow().forward(xs)?;
        self.down.borrow().forward(&(&lhs * &rhs)?)
    }
}

struct MoeGate {
    weight: Tensor,
    cfg: Config,
    top_k: usize,
    n_routed_experts: usize,
    e_score_correction_bias: Option<Tensor>,
}

impl MoeGate {
    fn new(cfg: &Config, vb: VarBuilder, n_routed_experts: usize) -> Result<Self> {
        let moe_cfg = cfg.moe_config.as_ref().unwrap();
        let weight = vb.get((n_routed_experts, cfg.hidden_size), "weight")?;
        let e_score_correction_bias = if matches!(moe_cfg.topk_method, TopkMethod::NoAuxTc) {
            Some(vb.get_with_hints_dtype(
                n_routed_experts,
                "e_score_correction_bias",
                Default::default(),
                DType::F32,
            )?)
        } else {
            None
        };
        Ok(Self {
            weight: weight.to_dtype(DType::F32)?,
            cfg: cfg.clone(),
            top_k: moe_cfg.num_experts_per_tok.unwrap(),
            n_routed_experts,
            e_score_correction_bias,
        })
    }

    /// (topk_idx, topk_weight)
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (bs, seq_len, h) = xs.dims3()?;
        let moe_cfg = self.cfg.moe_config.as_ref().unwrap();
        // Compute gating score
        let xs = xs.reshape(((), h))?;
        let logits = xs
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.weight.t()?)?;
        let scores = match moe_cfg.scoring_func {
            ScoringFunc::Softmax => candle_nn::ops::softmax_last_dim(&logits)?,
            ScoringFunc::Sigmoid => candle_nn::ops::sigmoid(&logits)?,
        };

        // Select top-k experts
        let (mut topk_weight, topk_idx) = match moe_cfg.topk_method {
            TopkMethod::Greedy => {
                let TopKOutput { values, indices } = scores.topk_unsorted(self.top_k)?;
                (values, indices)
            }
            TopkMethod::NoAuxTc => {
                let Some(e_score_correction_bias) = &self.e_score_correction_bias else {
                    candle_core::bail!("Expected e_score_correction_bias")
                };
                let scores_for_choice = scores
                    .reshape((bs * seq_len, ()))?
                    .broadcast_add(&e_score_correction_bias.unsqueeze(0)?)?;
                // (n, n_group)
                let group_scores = scores_for_choice
                    .reshape((bs * seq_len, moe_cfg.n_group, ()))?
                    .topk(2)?
                    .values
                    .sum(D::Minus1)?;
                // (n, topk_group)
                let group_idx = group_scores.topk(moe_cfg.topk_group)?.indices;
                // (n, n_group)
                let mut group_mask = group_scores.zeros_like()?;
                // (n, n_group)
                group_mask = group_mask.scatter_add(
                    &group_idx,
                    &group_idx.ones_like()?.to_dtype(group_mask.dtype())?,
                    1,
                )?;
                // (n, e)
                let score_mask = group_mask
                    .unsqueeze(D::Minus1)?
                    .expand((
                        bs * seq_len,
                        moe_cfg.n_group,
                        self.n_routed_experts / moe_cfg.n_group,
                    ))?
                    .reshape((bs * seq_len, ()))?;
                // (n, e)
                // Invert the mask
                let tmp_scores = scores_for_choice.broadcast_mul(&score_mask)?;
                let topk_idx = tmp_scores.topk(self.top_k)?.indices;
                (scores.gather(&topk_idx, 1)?, topk_idx)
            }
            TopkMethod::GroupLimitedGreedy => {
                // (n, n_group)
                let group_scores = scores
                    .reshape((bs * seq_len, moe_cfg.n_group, ()))?
                    .max(D::Minus1)?;
                // (n, topk_group)
                let group_idx = group_scores.topk_unsorted(moe_cfg.topk_group)?.indices;
                // (n, n_group)
                let mut group_mask = group_scores.zeros_like()?;
                // (n, n_group)
                group_mask = group_mask.scatter_add(
                    &group_idx,
                    &group_idx.ones_like()?.to_dtype(group_mask.dtype())?,
                    1,
                )?;
                // (n, e)
                let score_mask = group_mask
                    .unsqueeze(D::Minus1)?
                    .expand((
                        bs * seq_len,
                        moe_cfg.n_group,
                        self.n_routed_experts / moe_cfg.n_group,
                    ))?
                    .reshape((bs * seq_len, ()))?;
                // (n, e)
                // Invert the mask
                let tmp_scores = masked_fill(&score_mask, &(1. - &score_mask.ne(0.)?)?, 0.)?;
                let TopKOutput { values, indices } = tmp_scores.topk_unsorted(self.top_k)?;
                (values, indices)
            }
        };

        if self.top_k > 1 && moe_cfg.norm_topk_prob {
            let denominator = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weight = topk_weight.broadcast_div(&denominator)?;
        }

        topk_weight = (topk_weight * moe_cfg.routed_scaling_factor)?;
        Ok((topk_idx, topk_weight))
    }
}

struct Moe {
    experts: Vec<Option<Mlp>>,
    shared_experts: Option<Mlp>,
    gate: MoeGate,
    all_reduce: AllReduce,
    experts_start_idx: usize,
    experts_end_idx: usize,
    world_size: usize,
}

impl Moe {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        n_shared_experts: Option<usize>,
        n_routed_experts: usize,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let moe_cfg = cfg.moe_config.as_ref().unwrap();
        let mut experts = Vec::with_capacity(n_routed_experts);
        let n_local_experts = n_routed_experts / comm.world_size();
        let experts_start_idx = comm.rank() * n_local_experts;
        let experts_end_idx = experts_start_idx + n_local_experts;
        let n_local_experts_need_offload = moe_cfg.num_experts_offload_per_rank.unwrap_or(0);
        let mut offloaded_count = 0;
        for i in 0..n_routed_experts {
            if i >= experts_start_idx && i < experts_end_idx {
                let vb_e = vb.pp("experts").pp(i);
                let offload = offloaded_count < n_local_experts_need_offload;
                experts.push(Some(Mlp::new(
                    cfg,
                    vb_e,
                    None,
                    Some(moe_cfg.moe_intermediate_size),
                    offload,
                )?));
                offloaded_count += 1;
            } else {
                experts.push(None);
            }
        }

        let shared_experts = if let Some(n_shared_experts) = n_shared_experts {
            let intermediate_size = moe_cfg.moe_intermediate_size * n_shared_experts;
            Some(Mlp::new(
                cfg,
                vb.pp("shared_experts"),
                None,
                Some(intermediate_size),
                false,
            )?)
        } else {
            None
        };
        let gate = MoeGate::new(cfg, vb.pp("gate"), n_routed_experts)?;
        let word_size = comm.world_size();
        Ok(Self {
            experts,
            shared_experts,
            gate,
            all_reduce: AllReduce::new(comm),
            experts_end_idx,
            experts_start_idx,
            world_size: word_size,
        })
    }

    fn moe_infer(&self, xs: &Tensor, topk_ids: &Tensor, topk_weight: &Tensor) -> Result<Tensor> {
        let mut y = xs.zeros_like()?;
        let topk_weight = if topk_weight.dtype() != xs.dtype() {
            topk_weight.to_dtype(xs.dtype())?
        } else {
            topk_weight.to_owned()
        };
        let unique_ids: HashSet<u32> =
            HashSet::from_iter(topk_ids.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?);
        let mut cur_used_experts = Vec::<u32>::new();
        for i in self.experts_start_idx..self.experts_end_idx {
            if unique_ids.contains(&(i as u32)) {
                cur_used_experts.push(i as u32);
                let expert = self.experts[i]
                    .as_ref()
                    .expect("Expert is not present for this rank.");
                expert.reload(); //make sure the current used expert is loaded on device
            }
        }

        for i in &cur_used_experts {
            let idx_top = topk_ids.eq(*i as u32)?.nonzero()?.t()?.contiguous()?;
            let idx = &idx_top.i(0)?.contiguous()?;
            let top = &idx_top.i(1)?.contiguous()?;
            let expert = self.experts[*i as usize]
                .as_ref()
                .expect("Expert is not present for this rank.");

            y = y.index_add(
                idx,
                &expert.forward(&xs.index_select(idx, 0)?)?.broadcast_mul(
                    &topk_weight
                        .index_select(idx, 0)?
                        .gather(&top.unsqueeze(1)?, 1)?
                        .squeeze(1)?
                        .unsqueeze(D::Minus1)?,
                )?,
                0,
            )?;
        }

        if self.world_size > 1 {
            y = self.all_reduce.apply(&y)?;
        }

        for i in &cur_used_experts {
            let expert = self.experts[*i as usize]
                .as_ref()
                .expect("Expert is not present for this rank.");
            expert.offload();
        }
        Ok(y)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let identity = xs.clone();
        let orig_shape = xs.shape();
        let (topk_idx, topk_weight) = self.gate.forward(xs)?;
        let xs = xs.reshape(((), xs.dim(D::Minus1)?))?;

        let mut y = self
            .moe_infer(&xs, &topk_idx, &topk_weight)?
            .reshape(orig_shape)?;
        if let Some(ref shared_experts) = self.shared_experts {
            y = (y + shared_experts.forward(&identity)?)?;
        }
        Ok(y)
    }
}

enum MoeOrMlp {
    Moe(Moe),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn: Attention,
    moe_or_mlp: MoeOrMlp,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        layer_idx: usize,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let moe_cfg = cfg.moe_config.as_ref().unwrap();
        let attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"), comm.clone())?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let moe_or_mlp = if moe_cfg.n_routed_experts > 0
            && layer_idx >= moe_cfg.first_k_dense_replace
            && layer_idx % moe_cfg.moe_layer_freq == 0
        {
            MoeOrMlp::Moe(Moe::new(
                cfg,
                vb.pp("mlp"),
                moe_cfg.n_shared_experts,
                moe_cfg.n_routed_experts,
                comm.clone(),
            )?)
        } else {
            MoeOrMlp::Mlp(Mlp::new(cfg, vb.pp("mlp"), None, None, false)?)
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attn,
            moe_or_mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        input_positions: &[Vec<usize>],
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .attn
            .forward(&xs, attention_mask, input_positions, cache, input_metadata)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .moe_or_mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

pub struct DeepSeek {
    lm_head: ReplicatedLinear,
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    dtype: DType,
    device: Device,
    cfg: Config,
}

impl DeepSeek {
    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let moe_cfg = cfg.moe_config.as_ref().unwrap();
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let reporter = progress_reporter.clone();
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLinear::load_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp("lm_head"),
                &None,
                &None,
            )?
        } else {
            ReplicatedLinear::from_weight_bias(embed_tokens.embeddings().clone(), None)?
        };
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: moe_cfg.rope_scaling.clone(),
            max_position_embeddings: cfg.max_seq_len,
            rope_theta: cfg.rope_theta as f32,
            qk_rope_head_dim: moe_cfg.qk_rope_head_dim,
        };
        let rotary_emb = Arc::new(DeepSeekV2RotaryEmbedding::new(&rope_cfg, dtype, device)?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                layer_idx,
                comm.clone(),
            )?;
            layers.push(layer);
            reporter.write().unwrap().set_progress(layer_idx + 1);
        }

        Ok(Self {
            lm_head,
            embed_tokens,
            norm,
            layers,
            dtype,
            device: device.clone(),
            cfg: cfg.clone(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        input_positions: &[Vec<usize>],
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (bs, seq_len) = x.dims2()?;
        let mut x = self.embed_tokens.forward(x)?;
        let attention_mask = if seq_len == 1 {
            None
        } else {
            super::get_attention_casual_mask(
                &self.device,
                self.dtype,
                bs,
                seq_len,
                input_positions,
                self.cfg.sliding_window,
            )
        };
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), block) in zip(kv_caches.iter(), &self.layers) {
                x = block.forward(
                    &x,
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
            }
        } else {
            for block in &self.layers {
                x = block.forward(
                    &x,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?;
            }
        }
        let xs = x
            .i((.., seq_len - 1, ..))?
            .contiguous()?
            .apply(&self.norm)?;
        self.lm_head.forward(&xs)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
