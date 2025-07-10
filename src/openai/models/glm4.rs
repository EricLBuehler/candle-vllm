use super::{Config, QuantConfig};
use crate::openai::distributed::{
    embedding, rms_norm, Comm, MergedParallelColumnLinear, ReplicatedLinear,
    TensorParallelColumnLinear, TensorParallelRowLinear, VarBuilder,
};
use crate::paged_attention::input_metadata::InputMetadata;
use crate::SpecificConfig;
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_core as candle;
use candle_nn::{Embedding, Module, RmsNorm};
pub const MAX_SEQ_LEN: usize = 4096;
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::models::TokenID;
use either::Either;
use std::iter::zip;
pub use std::rc::Rc;
use std::sync::{Arc, RwLock};
#[derive(Debug, Clone, serde::Deserialize)]
pub struct GLMConfig {
    pub num_hidden_layers: Option<usize>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub head_dim: Option<usize>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f64>,
    pub partial_rotary_factor: Option<f32>,
    pub hidden_act: candle_nn::Activation,
    pub attention_bias: Option<bool>,
    pub sliding_window: Option<usize>,
    pub eos_token_id: TokenID,
    pub max_position_embeddings: Option<usize>,
    pub quantization_config: Option<QuantConfig>,
}

impl GLMConfig {
    pub fn into_config(
        self,
        use_flash_attn: bool,
        kv_cache_dtype: DType,
        scfg: &SpecificConfig,
    ) -> Config {
        Config {
            hidden_size: self.hidden_size,
            head_dim: Some(
                self.head_dim
                    .unwrap_or(self.hidden_size / self.num_attention_heads),
            ),
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers.unwrap_or(40),
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta.unwrap_or(10_000f64),
            use_flash_attn,
            bos_token_id: super::TokenID(Either::Left(Some(128256))),
            eos_token_id: self.eos_token_id,
            max_seq_len: self.max_position_embeddings.unwrap_or(32768),
            sliding_window: self.sliding_window,
            hidden_act: Some(self.hidden_act),
            tie_word_embeddings: false,
            rope_local_base_freq: None,
            sliding_window_pattern: None,
            rope_scaling: None,
            original_max_position_embeddings: None,
            attention_bias: self.attention_bias.unwrap_or(false),
            partial_rotary_factor: self.partial_rotary_factor,
            qk_layer_rms_norm: None,
            use_qkv_bias: None,
            kv_cache_dtype,
            custom_stop_tokens: None,
            specific_config: scfg.clone(),
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: self.quantization_config,
            moe_config: None,
        }
    }
}

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    // inv_freq: Tensor,
    rotary_dim: usize,
}

// fn repeat_interleave(xs: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
//     let xs = xs.unsqueeze(dim + 1)?;
//     let mut dims = xs.dims().to_vec();
//     dims[dim + 1] = repeats;
//     xs.broadcast_as(dims)?.flatten(dim, dim + 1)
// }

impl RotaryEmbedding {
    pub fn new(cfg: &Config, _dtype: DType, dev: &Device) -> Result<Self> {
        let dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let rotary_dim = if cfg.partial_rotary_factor.is_some() {
            (cfg.partial_rotary_factor.unwrap() * dim as f32) as usize
        } else {
            dim
        };
        let max_seq_len = cfg.max_seq_len;
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
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
            rotary_dim,
            // inv_freq: inv_freq.reshape((1, (), 1))?,
        })
    }

    //TODO(guoqingbao): dynamic rope
    // pub fn update(
    //     &self,
    //     seq_len: usize,
    //     input_positions: &[Vec<usize>],
    //     device: &Device,
    // ) -> Result<(Tensor, Tensor)> {
    //     let mut position_ids = Vec::<Tensor>::new();
    //     let b_size = input_positions.len();
    //     for (b, seqlen_offset) in zip(0..b_size, input_positions) {
    //         let t = Tensor::arange(
    //             seqlen_offset[0] as u32,
    //             seqlen_offset[0] as u32 + seq_len as u32,
    //             device,
    //         )?
    //         .to_dtype(DType::F32)?; //optimize: make a full tensor and chunk from it
    //         position_ids.push(t);
    //     }
    //     let position_ids = Tensor::cat(&position_ids, 0)?.reshape((b_size, (), 1))?;
    //     let inv_freq_expanded =
    //         self.inv_freq
    //             .expand((position_ids.dim(0)?, self.inv_freq.dim(1)?, 1))?;
    //     let freqs = inv_freq_expanded
    //         .matmul(&position_ids.t()?)?
    //         .transpose(1, 2)?
    //         .contiguous()?;
    //     let emb = repeat_interleave(&freqs, 2, 2)?;
    //     let cos = emb.cos()?;
    //     let sin = emb.sin()?;
    //     Ok((cos, sin))
    // }

    pub fn apply_rotary_emb(&self, xs: &Tensor, input_positions: &[Vec<usize>]) -> Result<Tensor> {
        let (b_size, _num_heads, seq_len, _headdim) = xs.dims4()?;
        // let (cos, sin) = self.update(seq_len, &position_ids, &xs.device())?;
        let mut embeds = Vec::new();
        for (b, seqlen_offset) in zip(0..b_size, input_positions) {
            let (s, e) = (seqlen_offset[0], seqlen_offset[0] + seq_len);
            let cos = self.cos.i((s..e, ..))?.contiguous()?;
            let sin = self.sin.i((s..e, ..))?.contiguous()?;
            let xs_rot = xs
                .i((b, .., .., ..self.rotary_dim))?
                .unsqueeze(0)?
                .contiguous()?;
            let xs_pass = xs.i((b, .., .., self.rotary_dim..))?.unsqueeze(0)?;
            let xs_rot = candle_nn::rotary_emb::rope_i(&xs_rot, &cos, &sin)?;
            let embed = Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)?.contiguous()?;
            embeds.push(embed);
        }
        Tensor::cat(&embeds, 0)
    }
}

struct SelfAttention {
    q_proj: TensorParallelColumnLinear,
    k_proj: TensorParallelColumnLinear,
    v_proj: TensorParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    attn: super::AttentionSelect,
}

impl SelfAttention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim.unwrap_or(hidden_sz / num_heads);
        let q_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let k_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;

        let v_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            num_kv_heads * head_dim,
            cfg.attention_bias,
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

        assert!(cfg.num_attention_heads >= comm.world_size());
        assert!(cfg.num_attention_heads % comm.world_size() == 0);

        assert!(cfg.num_key_value_heads >= comm.world_size());
        assert!(cfg.num_key_value_heads % comm.world_size() == 0);

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
            attn: super::AttentionSelect::new(
                cfg,
                cfg.sliding_window,
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
            .forward(&q, &k, &v, attention_mask, cache, input_metadata, None)?;

        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }
}

#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_up_proj: MergedParallelColumnLinear,
    down_proj: TensorParallelRowLinear,
    act_fn: candle_nn::Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let gate_up_proj = MergedParallelColumnLinear::load_merged_with_hints(
            cfg.hidden_size,
            cfg.intermediate_size * 2,
            2,
            vb.pp("gate_up_proj"),
            comm.clone(),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;

        let down_proj = TensorParallelRowLinear::load_with_hints(
            cfg.intermediate_size,
            cfg.hidden_size,
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
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up_states = self.gate_up_proj.forward(xs)?;
        let up_states = (&gate_up_states[1] * self.act_fn.forward(&gate_up_states[0])?)?;
        self.down_proj.forward(&up_states)
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: SelfAttention,
    post_attention_layernorm: RmsNorm,
    post_mlp_layernorm: RmsNorm,
    post_self_attn_layernorm: RmsNorm,
    mlp: MLP,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let post_self_attn_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_self_attn_layernorm"),
        )?;
        let post_mlp_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_mlp_layernorm"),
        )?;
        let self_attn =
            SelfAttention::new(rotary_emb.clone(), cfg, vb.pp("self_attn"), comm.clone())?;
        let mlp = MLP::new(cfg, vb.pp("mlp"), comm.clone())?;
        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            post_self_attn_layernorm,
            post_mlp_layernorm,
            self_attn,
            mlp,
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
        let hidden_states = self.input_layernorm.forward(xs)?;
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            input_positions,
            cache,
            input_metadata,
        )?;
        let hidden_states = self.post_self_attn_layernorm.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = self.post_mlp_layernorm.forward(&hidden_states)?;
        residual + hidden_states
    }
}

pub struct GLM4 {
    embedding: Embedding,
    layers: Vec<DecoderLayer>,
    lm_head: ReplicatedLinear,
    norm: RmsNorm,
    dtype: DType,
    device: Device,
    cfg: Config,
}

impl GLM4 {
    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let reporter = progress_reporter.clone();

        let rotary_emb = Arc::new(RotaryEmbedding::new(cfg, dtype, device)?);
        let embedding = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_index in 0..cfg.num_hidden_layers {
            let layer =
                DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_index), comm.clone())?;
            layers.push(layer);
            reporter.write().unwrap().set_progress(layer_index + 1);
        }

        let lm_head = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            vb.pp("lm_head"),
            &None,
            &None,
        )?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embedding,
            layers,
            lm_head,
            norm,
            dtype,
            device: device.clone(),
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
        let mut xs = self.embedding.forward(input_ids)?;

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
            .i((.., seq_len - 1, ..))?
            .contiguous()?
            .apply(&self.norm)?;
        self.lm_head.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
