use super::{Config, QuantConfig};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{
    embedding, layer_norm, Comm, ReplicatedLinear, TensorParallelColumnLinear,
    TensorParallelRowLinear, VarBuilder,
};
use crate::openai::models::TokenID;
use crate::paged_attention::input_metadata::InputMetadata;
use crate::paged_attention::PagedAttention;
use crate::SpecificConfig;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, LayerNorm};
use std::iter::zip;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct StableLMConfig {
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    pub use_cache: bool,
    pub use_qkv_bias: Option<bool>, // Used in StableLM-2
    pub partial_rotary_factor: Option<f32>,
    pub rope_pct: Option<f32>,
    pub tie_word_embeddings: Option<bool>,
    pub bos_token_id: TokenID,
    pub eos_token_id: TokenID,
    pub sliding_window: Option<usize>,
    pub quantization_config: Option<QuantConfig>,
}

impl StableLMConfig {
    pub fn into_config(
        self,
        use_flash_attn: bool,
        kv_cache_dtype: DType,
        scfg: &SpecificConfig,
    ) -> Config {
        Config {
            hidden_size: self.hidden_size,
            head_dim: Some(self.hidden_size / self.num_attention_heads),
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.norm_eps,
            rope_theta: self.rope_theta,
            rope_local_base_freq: None,
            use_flash_attn,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            max_seq_len: self.max_position_embeddings,
            sliding_window: self.sliding_window,
            sliding_window_pattern: None,
            hidden_act: Some(self.hidden_act),
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
            rope_scaling: None,
            original_max_position_embeddings: None,
            attention_bias: false,
            partial_rotary_factor: Some(
                self.partial_rotary_factor
                    .unwrap_or(self.rope_pct.unwrap_or(0.25)),
            ),
            qk_layer_rms_norm: None,
            kv_cache_dtype,
            use_qkv_bias: Some(self.use_qkv_bias.unwrap_or(false)),
            custom_stop_tokens: None,
            specific_config: scfg.clone(),
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: self.quantization_config,
            moe_config: None,
        }
    }
}

#[derive(Debug)]
pub(crate) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    dim: usize,
}

impl RotaryEmbedding {
    pub(crate) fn new(_dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let dim = (cfg.partial_rotary_factor.unwrap() * head_dim as f32) as usize;
        let max_seq_len = cfg.max_seq_len;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            dim,
        })
    }

    fn apply_rotary_emb(&self, xs: &Tensor, input_positions: &[Vec<usize>]) -> Result<Tensor> {
        let (b_size, _num_heads, seq_len, _headdim) = xs.dims4()?;
        let mut embeds = Vec::new();
        for (b, seqlen_offset) in zip(0..b_size, input_positions) {
            let xs_rot = xs.narrow(3, 0, self.dim)?.contiguous()?;
            let xs_pass = xs.narrow(3, self.dim, _headdim - self.dim)?;
            let c = self.cos.narrow(0, seqlen_offset[0], seq_len)?;
            let s = self.sin.narrow(0, seqlen_offset[0], seq_len)?;
            let xs_rot = xs_rot.narrow(0, b, 1)?;
            let xs_pass = xs_pass.narrow(0, b, 1)?;

            let xs_rot = candle_nn::rotary_emb::rope(&xs_rot, &c, &s)?;
            let embed = Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)?.contiguous()?;
            embeds.push(embed);
        }
        Tensor::cat(&embeds, 0)
    }
}

struct MLP {
    gate_proj: TensorParallelColumnLinear,
    up_proj: TensorParallelColumnLinear,
    down_proj: TensorParallelRowLinear,
    act_fn: Activation,
}

impl MLP {
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
            act_fn: cfg.hidden_act.unwrap_or(Activation::Silu),
        })
    }
}

impl Module for MLP {
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
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    attn: PagedAttention,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden_sz / num_heads;

        let q_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            num_heads * head_dim,
            cfg.use_qkv_bias.unwrap_or(false),
            vb.pp("q_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let k_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            num_kv_heads * head_dim,
            cfg.use_qkv_bias.unwrap_or(false),
            vb.pp("k_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let v_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            num_kv_heads * head_dim,
            cfg.use_qkv_bias.unwrap_or(false),
            vb.pp("v_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;

        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
            hidden_sz,
            false,
            vb.pp("o_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let attention_heads = cfg.num_attention_heads / comm.world_size();
        let kv_heads = cfg.num_key_value_heads / comm.world_size();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: attention_heads,
            num_kv_heads: kv_heads,
            head_dim,
            rotary_emb,
            attn: PagedAttention::new(
                attention_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(kv_heads),
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

        let q = self
            .rotary_emb
            .apply_rotary_emb(&q.to_dtype(DType::F32)?, input_positions)?;

        let k = self
            .rotary_emb
            .apply_rotary_emb(&k.to_dtype(DType::F32)?, input_positions)?;

        let q = q.to_dtype(v.dtype())?;
        let k = k.to_dtype(v.dtype())?;

        let y = self
            .attn
            .forward(
                &q,
                &k,
                &v,
                attention_mask,
                cache.map(|(k_, _)| k_.clone()),
                cache.map(|(_, v_)| v_.clone()),
                input_metadata,
                None,
            )?
            .reshape((b_sz, seq_len, ()))?;

        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"), comm.clone())?;
        let mlp = MLP::new(cfg, vb.pp("mlp"), comm.clone())?;
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
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
        let xs =
            self.self_attn
                .forward(&xs, attention_mask, input_positions, cache, input_metadata)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

pub struct StableLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    lm_head: ReplicatedLinear,
    device: Device,
    dtype: DType,
    cfg: Config,
}

impl StableLM {
    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let reporter = progress_reporter.clone();
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer =
                DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), comm.clone())?;
            layers.push(layer);
            reporter.write().unwrap().set_progress(layer_idx + 1);
        }
        let norm = layer_norm(cfg.hidden_size, cfg.rms_norm_eps, true, vb_m.pp("norm"))?;
        let lm_head = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            vb.pp("lm_head"),
            &None,
            &None,
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            cfg: cfg.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_positions: &[Vec<usize>],
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            super::get_attention_casual_mask(
                &self.device,
                self.dtype,
                b_size,
                seq_len,
                input_positions,
                self.cfg.sliding_window,
            )
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?
            }
        } else {
            for layer in self.layers.iter() {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?
            }
        }
        let xs = xs.i((.., seq_len - 1, ..))?.apply(&self.norm)?;
        self.lm_head.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
