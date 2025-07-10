// This implementation is based on:
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py
use super::{Config, QuantConfig, RopeScaling};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{
    embedding, rms_norm, Comm, ReplicatedLinear, TensorParallelColumnLinear,
    TensorParallelRowLinear, VarBuilder,
};
use crate::openai::models::TokenID;
use crate::paged_attention::input_metadata::InputMetadata;
use crate::paged_attention::PagedAttention;
use crate::SpecificConfig;
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_core as candle;
use candle_nn::RmsNorm;
use either::Either;
use std::collections::HashMap;
use std::iter::zip;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PhiConfig {
    pub vocab_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: TokenID,
    pub eos_token_id: TokenID,
    pub rope_scaling: Option<HashMap<String, RopeScaling>>,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: Option<usize>,
    pub sliding_window: Option<usize>,
    pub quantization_config: Option<QuantConfig>,
}

impl PhiConfig {
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
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            rope_local_base_freq: None,
            use_flash_attn,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            max_seq_len: self.max_position_embeddings,
            sliding_window: self.sliding_window,
            sliding_window_pattern: None,
            hidden_act: Some(self.hidden_act),
            tie_word_embeddings: false,
            rope_scaling: self.rope_scaling,
            original_max_position_embeddings: self.original_max_position_embeddings,
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
            moe_config: None,
        }
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    sin_long: Option<Tensor>,
    cos_long: Option<Tensor>,
    original_max_position_embeddings: Option<usize>,
}

impl RotaryEmbedding {
    fn new(_dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
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

        if let Some(rope_scaling) = &cfg.rope_scaling {
            match (
                &rope_scaling["short_factor"],
                &rope_scaling["long_factor"],
                &rope_scaling["type"],
            ) {
                (
                    RopeScaling(Either::Left(short_factor)),
                    RopeScaling(Either::Left(long_factor)),
                    RopeScaling(Either::Right(tp)),
                ) => {
                    let scale = cfg.max_seq_len as f64
                        / cfg.original_max_position_embeddings.unwrap() as f64;
                    let scaling_factor = if scale <= 1.0 {
                        1.0
                    } else {
                        match tp.as_str() {
                            "su" | "longrope" => (1.0
                                + scale.ln()
                                    / (cfg.original_max_position_embeddings.unwrap() as f64).ln())
                            .sqrt(),
                            "yarn" => 0.1 * scale.ln() + 1.0,
                            _ => 1.0,
                        }
                    };
                    // Calculate inv freqs for short, long
                    let inv_freq_long = (0..dim)
                        .step_by(2)
                        .enumerate()
                        .map(|(k, i)| {
                            (1f64 / (long_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64)))
                                as f32
                        })
                        .collect::<Vec<_>>();
                    let inv_freq_short = (0..dim)
                        .step_by(2)
                        .enumerate()
                        .map(|(k, i)| {
                            (1f64 / (short_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64)))
                                as f32
                        })
                        .collect::<Vec<_>>();
                    let inv_freq_len = inv_freq_long.len();

                    let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
                        .to_dtype(DType::F32)?
                        .reshape((max_seq_len, 1))?;

                    // Calculate sin,cos for long
                    let inv_freq_long = Tensor::from_vec(inv_freq_long, (1, inv_freq_len), dev)?
                        .to_dtype(DType::F32)?;
                    let freqs_long = t.matmul(&inv_freq_long)?;
                    let long_sin = (freqs_long.sin()? * scaling_factor)?;
                    let long_cos = (freqs_long.cos()? * scaling_factor)?;

                    // Calculate sin,cos for short
                    let inv_freq_short = Tensor::from_vec(inv_freq_short, (1, inv_freq_len), dev)?
                        .to_dtype(DType::F32)?;
                    let freqs_short = t.matmul(&inv_freq_short)?;
                    let short_sin = (freqs_short.sin()? * scaling_factor)?;
                    let short_cos = (freqs_short.cos()? * scaling_factor)?;

                    return Ok(Self {
                        sin: short_sin,
                        cos: short_cos,
                        sin_long: Some(long_sin),
                        cos_long: Some(long_cos),
                        original_max_position_embeddings: cfg.original_max_position_embeddings,
                    });
                }
                _ => {
                    panic!("Unknown config for rope scaling!")
                }
            }
        }

        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            sin_long: None,
            cos_long: None,
            original_max_position_embeddings: None,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &[Vec<usize>],
    ) -> Result<(Tensor, Tensor)> {
        let (b_size, _h, seq_len, _n_embd) = q.dims4()?;

        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();

        for (b, seqlen_offset) in zip(0..b_size, input_positions) {
            let (cos, sin) = if self.sin_long.as_ref().is_some()
                && self.cos_long.as_ref().is_some()
                && self.original_max_position_embeddings.is_some()
                && seqlen_offset[0] > self.original_max_position_embeddings.unwrap()
            {
                let cos = self
                    .cos_long
                    .as_ref()
                    .unwrap()
                    .narrow(0, seqlen_offset[0], seq_len)?;
                let sin = self
                    .sin_long
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

struct Attention {
    qkv_proj: ReplicatedLinear, //split into two devices may cause some errors, therefore, we use replicatedlayer
    o_proj: ReplicatedLinear,
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
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            op_size,
            vb.pp("qkv_proj"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let o_proj = ReplicatedLinear::load_no_bias(
            num_heads * head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        //Fix the attention parallel
        assert!(
            comm.world_size() < 2,
            "Packed qkv_proj is not supported under multi-gpu setting!"
        );
        let attention_heads = cfg.num_attention_heads / comm.world_size();
        let kv_heads = cfg.num_key_value_heads / comm.world_size();
        Ok(Self {
            qkv_proj,
            o_proj,
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
        let (b_sz, seq_len, _) = xs.dims3()?;

        let qkv = self.qkv_proj.forward(xs)?;
        let query_pos = self.num_heads * self.head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos)?;
        let key_states = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let value_states = qkv
            .narrow(
                D::Minus1,
                query_pos + self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            )?
            .contiguous()?;

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

        //preserve the precision with F32 type
        let (q, k) = self.rotary_emb.apply_rotary_emb_qkv(
            &q.to_dtype(DType::F32)?,
            &k.to_dtype(DType::F32)?,
            input_positions,
        )?;
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

struct Mlp {
    gate_up_proj: TensorParallelColumnLinear,
    down_proj: TensorParallelRowLinear,
    act_fn: candle_nn::Activation,
    i_size: usize,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            2 * i_size,
            false,
            vb.pp("gate_up_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let down_proj = TensorParallelRowLinear::load_with_hints(
            i_size,
            hidden_size,
            false,
            vb.pp("down_proj"),
            comm,
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act.unwrap(),
            i_size,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = self.gate_up_proj.forward(xs)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        self.down_proj.forward(&up_states)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"), comm.clone())?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"), comm.clone())?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
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

pub struct Phi {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: ReplicatedLinear,
    device: Device,
    dtype: DType,
    cfg: Config,
}

impl Phi {
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
        let rotary_emb = Arc::new(RotaryEmbedding::new(dtype, cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let reporter = progress_reporter.clone();
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer =
                DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), comm.clone())?;
            layers.push(layer);
            reporter.write().unwrap().set_progress(layer_idx + 1);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
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
