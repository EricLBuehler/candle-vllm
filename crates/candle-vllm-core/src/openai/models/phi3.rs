// This implementation is based on:
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py
use super::{Config, ScalingValue};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{
    embedding, rms_norm, Comm, ReplicatedLinear, TensorParallelColumnLinear,
    TensorParallelRowLinear, VarBuilder,
};
use crate::openai::models::mask::get_attention_causal_mask;
use crate::openai::models::rotary_emb::DefaultRotaryEmbedding;
use crate::{InputMetadata, PagedAttention};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_core as candle;
use candle_nn::RmsNorm;
use parking_lot::RwLock;
use std::iter::zip;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

impl Phi {
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
struct RotaryEmbedding {
    normal_emb: DefaultRotaryEmbedding,
    long_emb: Option<DefaultRotaryEmbedding>,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_seq_len;
        let rope_theta = cfg.rope_theta;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        if let Some(rope_scaling) = &cfg.rope_scaling {
            let mut rope_scaling = rope_scaling.clone();
            if !rope_scaling.contains_key("original_max_position_embeddings") {
                //insert cfg.original_max_position_embeddings
                rope_scaling.insert(
                    "original_max_position_embeddings".to_string(),
                    ScalingValue::Single(
                        cfg.original_max_position_embeddings
                            .unwrap_or(cfg.max_position_embeddings.unwrap_or(8192))
                            as f64,
                    ),
                );
            }
            match (
                &rope_scaling["short_factor"],
                &rope_scaling["long_factor"],
                &rope_scaling["type"],
                &rope_scaling["original_max_position_embeddings"],
            ) {
                (
                    ScalingValue::Vec(short_factor),
                    ScalingValue::Vec(long_factor),
                    ScalingValue::String(tp),
                    ScalingValue::Single(original_max_position_embeddings),
                ) => {
                    let scale = cfg.max_seq_len as f64 / *original_max_position_embeddings;
                    let scaling_factor = if scale <= 1.0 {
                        1.0
                    } else {
                        match tp.as_str() {
                            "su" | "longrope" => (1.0
                                + scale.ln() / (*original_max_position_embeddings as f64).ln())
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
                            (1f64 / (long_factor[k] * rope_theta.powf(i as f64 / dim as f64)))
                                as f32
                        })
                        .collect::<Vec<_>>();
                    let inv_freq_short = (0..dim)
                        .step_by(2)
                        .enumerate()
                        .map(|(k, i)| {
                            (1f64 / (short_factor[k] * rope_theta.powf(i as f64 / dim as f64)))
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

                    let normal_emb = DefaultRotaryEmbedding {
                        cos: short_sin.to_dtype(dtype)?,
                        sin: short_cos.to_dtype(dtype)?,
                        is_gpt_neox: true,
                        rotary_dim: None,
                    };

                    let long_emb = Some(DefaultRotaryEmbedding {
                        cos: long_sin.to_dtype(dtype)?,
                        sin: long_cos.to_dtype(dtype)?,
                        is_gpt_neox: true,
                        rotary_dim: None,
                    });

                    return Ok(Self {
                        normal_emb,
                        long_emb,
                    });
                }
                _ => {
                    panic!("Unknown config for rope scaling!")
                }
            }
        }
        let normal_emb = DefaultRotaryEmbedding {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            is_gpt_neox: true,
            rotary_dim: None,
        };
        Ok(Self {
            normal_emb,
            long_emb: None,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if self.long_emb.is_some() {
            self.long_emb
                .as_ref()
                .unwrap()
                .apply_rotary_emb(q, k, input_positions)
        } else {
            self.normal_emb.apply_rotary_emb(q, k, input_positions)
        }
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
        let num_kv_heads = cfg.num_key_value_heads.unwrap();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            op_size,
            vb.pp("qkv_proj"),
            &cfg.quant,
            &cfg.quantization_config,
        )?;
        let o_proj = ReplicatedLinear::load_no_bias(
            num_heads * head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
            &cfg.quant,
            &cfg.quantization_config,
        )?;
        //Fix the attention parallel
        assert!(
            comm.world_size() < 2,
            "Packed qkv_proj is not supported under multi-gpu setting!"
        );
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
        let query_pos = self.num_heads * self.head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos)?.to_dtype(DType::F32)?;
        let key_states = qkv
            .narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?
            .to_dtype(DType::F32)?;
        let value_states = qkv
            .narrow(
                D::Minus1,
                query_pos + self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            )?
            .contiguous()?;

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

        //preserve the precision with F32 type
        let (q, k) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&q, &k, input_positions)?;
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
            .reshape((seq_len, ()))?;

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
            &cfg.quant,
            &cfg.quantization_config,
        )?;
        let down_proj = TensorParallelRowLinear::load_with_hints(
            i_size,
            hidden_size,
            false,
            vb.pp("down_proj"),
            comm,
            &cfg.quant,
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
        let rotary_emb = Arc::new(RotaryEmbedding::new(DType::F32, cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let reporter = progress_reporter.clone();
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer =
                DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), comm.clone())?;
            layers.push(layer);
            reporter.write().set_progress(layer_idx + 1);
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
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let seqlens = if input_metadata.cu_seqlens_q.is_some() {
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
        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
