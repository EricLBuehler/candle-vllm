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
use candle_nn::{Activation, Embedding, LayerNorm};
use serde::Deserialize;
use std::iter::zip;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
#[derive(Debug, Clone, Deserialize)]
pub struct Phi2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub partial_rotary_factor: f32,
    pub qk_layernorm: bool,
    pub bos_token_id: TokenID,
    pub eos_token_id: TokenID,
    pub sliding_window: Option<usize>,
    pub original_max_position_embeddings: Option<usize>,
    pub quantization_config: Option<QuantConfig>,
}

impl Phi2Config {
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
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            rms_norm_eps: self.layer_norm_eps,
            rope_theta: self.rope_theta,
            rope_local_base_freq: None,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            max_seq_len: self.max_position_embeddings,
            sliding_window: self.sliding_window,
            sliding_window_pattern: None,
            hidden_act: Some(self.hidden_act),
            tie_word_embeddings: false,
            rope_scaling: None,
            use_flash_attn,
            original_max_position_embeddings: self.original_max_position_embeddings,
            attention_bias: false,
            partial_rotary_factor: Some(self.partial_rotary_factor),
            qk_layer_rms_norm: Some(self.qk_layernorm),
            kv_cache_dtype,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            specific_config: scfg.clone(),
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: self.quantization_config,
            moe_config: None,
        }
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    dim: usize,
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, _dtype: DType, dev: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let dim = (cfg.partial_rotary_factor.unwrap_or(1.0) * head_dim as f32) as usize;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| (1f64 / cfg.rope_theta.powf(i as f64 / dim as f64)) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, cfg.max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            dim,
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb(&self, xs: &Tensor, input_positions: &[Vec<usize>]) -> Result<Tensor> {
        let (b_size, _num_heads, seq_len, _headdim) = xs.dims4()?;
        let mut embeds = Vec::new();
        for (b, seqlen_offset) in zip(0..b_size, input_positions) {
            let cos = self.cos.narrow(0, seqlen_offset[0], seq_len)?;
            let sin = self.sin.narrow(0, seqlen_offset[0], seq_len)?;
            let xs_rot = xs.i((b, .., .., ..self.dim))?.contiguous()?;
            let xs_pass = xs.i((b, .., .., self.dim..))?;
            let xs_rot = candle_nn::rotary_emb::rope(&xs_rot, &cos, &sin).unwrap();
            let embed = Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)?.contiguous()?;
            embeds.push(embed);
        }
        Tensor::cat(&embeds, 0)
    }
}

struct Mlp {
    fc1: TensorParallelColumnLinear,
    fc2: TensorParallelRowLinear,
    act: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let fc1 = TensorParallelColumnLinear::load_with_hints(
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
            vb.pp("gate_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let fc2 = TensorParallelRowLinear::load_with_hints(
            cfg.intermediate_size,
            cfg.hidden_size,
            false,
            vb.pp("down_proj"),
            comm,
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        Ok(Self {
            fc1,
            fc2,
            // This does not match the mixformers implementation where Gelu is used rather than
            // GeluNew.
            act: cfg.hidden_act.unwrap_or(Activation::Silu),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.act.forward(&self.fc1.forward(xs)?)?)
    }
}

struct Attention {
    q_proj: TensorParallelColumnLinear,
    k_proj: TensorParallelColumnLinear,
    v_proj: TensorParallelColumnLinear,
    dense: TensorParallelRowLinear,
    q_layernorm: Option<LayerNorm>,
    k_layernorm: Option<LayerNorm>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn: PagedAttention,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let q_proj = TensorParallelColumnLinear::load_with_hints(
            cfg.hidden_size,
            num_heads * head_dim,
            false,
            vb.pp("q_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let k_proj = TensorParallelColumnLinear::load_with_hints(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
            vb.pp("k_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let v_proj = TensorParallelColumnLinear::load_with_hints(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
            vb.pp("v_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;

        let dense = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
            cfg.hidden_size,
            false,
            vb.pp("o_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        // Alternative rope scalings are not supported.
        let rotary_emb = RotaryEmbedding::new(cfg, vb.dtype(), vb.device())?;
        let (q_layernorm, k_layernorm) = if cfg.qk_layer_rms_norm.unwrap() {
            let q_layernorm = layer_norm(head_dim, cfg.rms_norm_eps, true, vb.pp("q_layernorm"))?;
            let k_layernorm = layer_norm(head_dim, cfg.rms_norm_eps, true, vb.pp("k_layernorm"))?;
            (Some(q_layernorm), Some(k_layernorm))
        } else {
            (None, None)
        };
        let attention_heads = cfg.num_attention_heads / comm.world_size();
        let kv_heads = cfg.num_key_value_heads / comm.world_size();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            q_layernorm,
            k_layernorm,
            rotary_emb,
            num_heads: attention_heads,
            num_kv_heads: kv_heads,
            head_dim,
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
        let (b_size, seq_len, _n_embd) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = match &self.q_layernorm {
            None => query_states,
            Some(ln) => query_states.apply(ln)?,
        };
        let key_states = match &self.k_layernorm {
            None => key_states,
            Some(ln) => key_states.apply(ln)?,
        };
        let dtype = value_states.dtype();
        let (q, k, v) = if seq_len == 1 {
            //no need transpose for seq_len == 1, change reshape dim
            let q = query_states.reshape((b_size, self.num_heads, seq_len, self.head_dim))?;
            let k = key_states.reshape((b_size, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = value_states.reshape((b_size, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        } else {
            let q = query_states
                .reshape((b_size, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = key_states
                .reshape((b_size, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = value_states
                .reshape((b_size, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        };

        let q = self
            .rotary_emb
            .apply_rotary_emb(&q.to_dtype(DType::F32)?, input_positions)?;
        let k = self
            .rotary_emb
            .apply_rotary_emb(&k.to_dtype(DType::F32)?, input_positions)?;
        let v = v.to_dtype(DType::F32)?;

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
            .reshape((b_size, seq_len, ()))?;

        let y = y.to_dtype(dtype)?;
        self.dense.forward(&y)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: LayerNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"), comm.clone())?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"), comm.clone())?;
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb.pp("input_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        input_positions: &[Vec<usize>],
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.input_layernorm)?;
        let attn_outputs =
            self.self_attn
                .forward(&xs, mask, input_positions, cache, input_metadata)?;
        let feed_forward_hidden_states = self.mlp.forward(&xs)?;
        attn_outputs + feed_forward_hidden_states + residual
    }
}

pub struct Phi2 {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    final_layernorm: LayerNorm,
    lm_head: ReplicatedLinear,
    cfg: Config,
    device: Device,
}

impl Phi2 {
    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        _dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let final_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb_m.pp("final_layernorm"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_m = vb_m.pp("layers");
        let reporter = progress_reporter.clone();
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, vb_m.pp(layer_idx), comm.clone())?;
            layers.push(layer);
            reporter.write().unwrap().set_progress(layer_idx + 1);
        }
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
            final_layernorm,
            lm_head,
            cfg: cfg.clone(),
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        input_positions: &[Vec<usize>],
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = xs.dims2()?;
        let mut xs = xs.apply(&self.embed_tokens)?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            super::get_attention_casual_mask(
                &self.device,
                DType::F32,
                b_size,
                seq_len,
                input_positions,
                self.cfg.sliding_window,
            )
        };
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
        let xs = xs
            .apply(&self.final_layernorm)?
            .i((.., seq_len - 1, ..))?
            .contiguous()?;

        self.lm_head.forward(&xs)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
