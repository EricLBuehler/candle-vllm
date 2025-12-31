// src/openai/models/phi4.rs
// Adapted from vllm.rs implementation for candle-vllm

use super::{Config, ScalingValue};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{
    embedding, rms_norm, Comm, MergedParallelColumnLinear, ReplicatedLinear,
    TensorParallelRowLinear, VarBuilder,
};
use crate::openai::models::mask::get_attention_causal_mask;
use crate::openai::models::rotary_emb::DefaultRotaryEmbedding;
use crate::{InputMetadata, PagedAttention};
use candle::{DType, Device, Module, Result, Tensor};
use candle_core as candle;
use candle_nn::RmsNorm;
use parking_lot::RwLock;
use std::iter::zip;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

impl Phi4ForCausalLM {
    pub fn load_config(filename: &PathBuf, isq: Option<String>) -> Result<Config> {
        let mut config = Config::load_config(filename.clone())?;
        config.head_dim = Some(
            config
                .head_dim
                .unwrap_or(config.hidden_size / config.num_attention_heads),
        );
        config.num_key_value_heads = Some(
            config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
        );
        config.max_seq_len = config.max_position_embeddings.unwrap_or(config.max_seq_len);
        if config.quantization_config.is_some() {
            config.quant = Some(
                config
                    .quantization_config
                    .as_ref()
                    .unwrap()
                    .quant_method
                    .clone(),
            );
        } else if isq.is_some() {
            config.quant = Some(isq.unwrap().to_string());
        }
        Ok(config)
    }
}

#[derive(Debug, Clone)]
struct Phi4RotaryEmbedding {
    normal_emb: DefaultRotaryEmbedding,
    long_emb: Option<DefaultRotaryEmbedding>,
    original_max_position_embeddings: Option<usize>,
}

impl Phi4RotaryEmbedding {
    fn rope_scaling_array(
        value: &ScalingValue,
        expected_len: usize,
        name: &str,
    ) -> Result<Vec<f64>> {
        match value {
            ScalingValue::Vec(v) => {
                if v.len() != expected_len {
                    candle_core::bail!(
                        "{name} length mismatch: expected {expected_len}, got {}",
                        v.len()
                    );
                }
                Ok(v.clone())
            }
            ScalingValue::Single(v) => Ok(vec![*v; expected_len]),
            _ => candle_core::bail!("{name} must be a number array"),
        }
    }

    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let max_seq_len = cfg.max_position_embeddings.unwrap_or(cfg.max_seq_len);
        let rope_theta = cfg.rope_theta;

        let rotary_dim = cfg
            .partial_rotary_factor
            .map(|factor| (factor * dim as f32) as usize)
            .unwrap_or(dim);

        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();

        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;

        if let Some(rope_scaling) = &cfg.rope_scaling {
            let original_max_position_embeddings = rope_scaling
                .get("original_max_position_embeddings")
                .and_then(|v| match v {
                    ScalingValue::Single(f) => Some(*f),
                    _ => None,
                })
                .unwrap_or(
                    cfg.original_max_position_embeddings
                        .unwrap_or(cfg.max_position_embeddings.unwrap_or(8192))
                        as f64,
                );

            let rope_type = rope_scaling
                .get("type")
                .or_else(|| rope_scaling.get("rope_type"))
                .and_then(|v| match v {
                    ScalingValue::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or("default");

            let short_factor = rope_scaling
                .get("short_factor")
                .ok_or_else(|| candle_core::Error::msg("rope_scaling missing short_factor"))?;
            let long_factor = rope_scaling
                .get("long_factor")
                .ok_or_else(|| candle_core::Error::msg("rope_scaling missing long_factor"))?;

            let short_factor =
                Self::rope_scaling_array(short_factor, inv_freq_len, "short_factor")?;
            let long_factor = Self::rope_scaling_array(long_factor, inv_freq_len, "long_factor")?;

            // Compute scaling factor
            let scale = max_seq_len as f64 / original_max_position_embeddings;
            let scaling_factor = if scale <= 1.0 {
                1.0
            } else {
                match rope_type {
                    "su" | "longrope" => {
                        (1.0 + scale.ln() / original_max_position_embeddings.ln()).sqrt()
                    }
                    "yarn" => 0.1 * scale.ln() + 1.0,
                    _ => 1.0,
                }
            };

            let inv_freq_long = (0..rotary_dim)
                .step_by(2)
                .enumerate()
                .map(|(k, i)| {
                    (1f64 / (long_factor[k] * rope_theta.powf(i as f64 / rotary_dim as f64))) as f32
                })
                .collect::<Vec<_>>();
            let inv_freq_short = (0..rotary_dim)
                .step_by(2)
                .enumerate()
                .map(|(k, i)| {
                    (1f64 / (short_factor[k] * rope_theta.powf(i as f64 / rotary_dim as f64)))
                        as f32
                })
                .collect::<Vec<_>>();

            let inv_freq_long =
                Tensor::from_vec(inv_freq_long, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
            let inv_freq_short =
                Tensor::from_vec(inv_freq_short, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;

            let freqs_long = t.matmul(&inv_freq_long)?;
            let long_sin = (freqs_long.sin()? * scaling_factor)?;
            let long_cos = (freqs_long.cos()? * scaling_factor)?;

            let freqs_short = t.matmul(&inv_freq_short)?;
            let short_sin = (freqs_short.sin()? * scaling_factor)?;
            let short_cos = (freqs_short.cos()? * scaling_factor)?;

            let normal_emb = DefaultRotaryEmbedding {
                sin: short_sin.to_dtype(dtype)?,
                cos: short_cos.to_dtype(dtype)?,
                is_gpt_neox: true,
                rotary_dim: if cfg.partial_rotary_factor.is_some() {
                    Some(rotary_dim)
                } else {
                    None
                },
            };

            let long_emb = DefaultRotaryEmbedding {
                sin: long_sin.to_dtype(dtype)?,
                cos: long_cos.to_dtype(dtype)?,
                is_gpt_neox: true,
                rotary_dim: if cfg.partial_rotary_factor.is_some() {
                    Some(rotary_dim)
                } else {
                    None
                },
            };

            return Ok(Self {
                normal_emb,
                long_emb: Some(long_emb),
                original_max_position_embeddings: Some(original_max_position_embeddings as usize),
            });
        }

        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let freqs = t.matmul(&inv_freq)?;

        let normal_emb = DefaultRotaryEmbedding {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            is_gpt_neox: true,
            rotary_dim: if cfg.partial_rotary_factor.is_some() {
                Some(rotary_dim)
            } else {
                None
            },
        };

        Ok(Self {
            normal_emb,
            long_emb: None,
            original_max_position_embeddings: None,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if let (Some(long_emb), Some(original_max_position_embeddings)) =
            (&self.long_emb, self.original_max_position_embeddings)
        {
            let max_position = input_positions
                .flatten_all()?
                .to_vec1::<i64>()?
                .into_iter()
                .max()
                .unwrap_or(0) // Should probably handle empty gracefully, but unwrap_or 0 is fine for logic
                + 1;

            if max_position >= original_max_position_embeddings as i64 {
                long_emb.apply_rotary_emb(q, k, input_positions)
            } else {
                self.normal_emb.apply_rotary_emb(q, k, input_positions)
            }
        } else {
            self.normal_emb.apply_rotary_emb(q, k, input_positions)
        }
    }
}

// Struct and Impl for Phi4Attention (Corrected version)
struct Phi4Attention {
    qkv_proj: MergedParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<Phi4RotaryEmbedding>,
    attn: PagedAttention,
}

impl Phi4Attention {
    fn new(
        rotary_emb: Arc<Phi4RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads.unwrap();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let qkv_proj = MergedParallelColumnLinear::load_merged_chunks(
            cfg.hidden_size,
            num_heads * head_dim + 2 * num_kv_heads * head_dim,
            0,
            vec![
                num_heads * head_dim,
                num_kv_heads * head_dim,
                num_kv_heads * head_dim,
            ],
            vb.pp("qkv_proj"),
            comm.clone(),
            &cfg.quantization_config,
            &cfg.quant,
            vb.dtype(),
        )?;

        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
            cfg.hidden_size,
            false,
            vb.pp("o_proj"),
            comm.clone(),
            &cfg.quant,
            &cfg.quantization_config,
        )?;

        let attention_heads = cfg.num_attention_heads / comm.world_size();
        let kv_heads = cfg.num_key_value_heads.unwrap() / comm.world_size();
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
                cfg.fp8_kvcache.unwrap_or(false),
            )?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        let qkv = self.qkv_proj.forward(xs)?;
        let query_states = &qkv[0];
        let key_states = &qkv[1];
        let value_states = &qkv[2];

        let q = query_states
            .reshape((1, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = key_states
            .reshape((1, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = value_states
            .reshape((1, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&q, &k, input_positions)?;

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
            .reshape((seq_len, ()))?;

        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }
}

// Local MLP definition
struct Mlp {
    gate_up_proj: MergedParallelColumnLinear,
    down_proj: TensorParallelRowLinear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let gate_up_proj = MergedParallelColumnLinear::load_merged_with_hints(
            cfg.hidden_size,
            cfg.intermediate_size * 2,
            2,
            vb.pp("gate_up_proj"),
            comm.clone(),
            &cfg.quant,
            &cfg.quantization_config,
        )?;

        let down_proj = TensorParallelRowLinear::load_with_hints(
            cfg.intermediate_size,
            cfg.hidden_size,
            false,
            vb.pp("down_proj"),
            comm.clone(),
            &cfg.quant,
            &cfg.quantization_config,
        )?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act.unwrap(),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up_states = self.gate_up_proj.forward(xs)?;
        let up_states = (&gate_up_states[1] * self.act_fn.forward(&gate_up_states[0])?)?;
        self.down_proj.forward(&up_states)
    }
}

struct Phi4DecoderLayer {
    self_attn: Phi4Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Phi4DecoderLayer {
    fn new(
        rotary_emb: Arc<Phi4RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let self_attn = Phi4Attention::new(rotary_emb, cfg, vb.pp("self_attn"), comm.clone())?;
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
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
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

pub struct Phi4ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Phi4DecoderLayer>,
    norm: RmsNorm,
    lm_head: ReplicatedLinear,
    device: Device,
    dtype: DType,
    cfg: Config,
}

impl Phi4ForCausalLM {
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
        let rotary_emb = Arc::new(Phi4RotaryEmbedding::new(dtype, cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let reporter = progress_reporter.clone();
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer =
                Phi4DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), comm.clone())?;
            layers.push(layer);
            reporter.write().set_progress(layer_idx + 1);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            if cfg.tie_word_embeddings {
                vb.pp("model.embed_tokens")
            } else {
                vb.pp("lm_head")
            },
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
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, input_positions, kv_caches, input_metadata, false)
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, input_positions, kv_caches, input_metadata, true)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let seqlens = if input_metadata.cu_seqlens_q.is_some() && !return_hidden {
            input_metadata
                .cu_seqlens_q
                .as_ref()
                .unwrap()
                .to_vec1::<u32>()?[1..]
                .into()
        } else {
            Vec::new()
        };
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            input_positions,
            &seqlens,
            self.cfg.sliding_window,
            input_metadata.is_prefill,
        );
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
        }

        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        if return_hidden {
            Ok(xs)
        } else {
            self.lm_head.forward(&xs)?.to_dtype(DType::F32)
        }
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
