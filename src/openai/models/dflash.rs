use crate::openai::distributed::{Comm, ReplicatedLinear, VarBuilder};
use crate::openai::models::layers::mlp::Mlp;
use crate::openai::models::layers::others::{rms_norm, NormX};
use crate::openai::models::Config;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::Module;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Mutex;

/// DFlash2 keeps a bounded projected hidden-state context for each request.
pub const DFLASH_CONTEXT_WINDOW: usize = 512;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DFlashConfig {
    #[serde(default)]
    pub block_size: Option<usize>,
    pub mask_token_id: Option<u32>,
    pub target_layer_ids: Option<Vec<usize>>,
    #[serde(default)]
    pub conv_group_size: Option<usize>,
    #[serde(default)]
    pub conv_kernel_size: Option<usize>,
    #[serde(default)]
    pub selector_rank: Option<usize>,
    #[serde(default)]
    pub selector_top_k: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DFlashModelConfig {
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: Option<f64>,
    pub attention_bias: Option<bool>,
    #[serde(default)]
    pub block_size: Option<usize>,
    pub num_target_layers: usize,
    #[serde(default)]
    pub dflash_config: Option<DFlashConfig>,
    pub hidden_act: Option<String>,
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub is_causal: Option<bool>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
}

impl DFlashModelConfig {
    pub fn is_dflash2(&self) -> bool {
        self.architectures
            .as_ref()
            .and_then(|architectures| architectures.first())
            .is_some_and(|architecture| architecture.contains("DFlash2"))
            || self
                .dflash_config
                .as_ref()
                .is_some_and(|config| config.selector_top_k.is_some())
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
            .or_else(|| self.dflash_config.as_ref().and_then(|c| c.block_size))
            .unwrap_or(1)
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|parameters| parameters.get("rope_theta"))
                    .and_then(serde_json::Value::as_f64)
            })
            .unwrap_or(10000.0)
    }

    pub fn target_layer_ids(&self) -> Vec<usize> {
        self.dflash_config
            .as_ref()
            .and_then(|c| c.target_layer_ids.clone())
            .unwrap_or_else(|| {
                build_target_layer_ids(self.num_target_layers, self.num_hidden_layers)
            })
    }

    pub fn mask_token_id(&self) -> Option<u32> {
        self.dflash_config.as_ref().and_then(|c| c.mask_token_id)
    }

    fn validate(&self) -> Result<()> {
        if !self.is_dflash2() {
            candle_core::bail!("DFlash2 config metadata is missing");
        }
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_target_layers == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
        {
            candle_core::bail!("DFlash2 config contains a zero-sized model dimension");
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.num_attention_heads % self.num_key_value_heads != 0
        {
            candle_core::bail!(
                "DFlash2 attention heads must be non-zero and num_attention_heads must be divisible by num_key_value_heads"
            );
        }
        if self.head_dim() == 0 {
            candle_core::bail!("DFlash2 head_dim must be non-zero");
        }
        let dflash_config = self
            .dflash_config
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("DFlash2 config is missing dflash_config"))?;
        if self.block_size() == 0
            || dflash_config.conv_group_size.unwrap_or(0) == 0
            || dflash_config.conv_kernel_size.unwrap_or(0) == 0
            || dflash_config.selector_rank.unwrap_or(0) == 0
            || dflash_config.selector_top_k.unwrap_or(0) == 0
        {
            candle_core::bail!(
                "DFlash2 config contains a zero-valued kernel or selector parameter"
            );
        }
        if self.hidden_size % dflash_config.conv_group_size.unwrap() != 0 {
            candle_core::bail!(
                "DFlash2 convolution group size {} must divide hidden size {}",
                dflash_config.conv_group_size.unwrap(),
                self.hidden_size
            );
        }
        if dflash_config.selector_top_k.unwrap() > self.vocab_size {
            candle_core::bail!(
                "DFlash2 selector_top_k {} exceeds vocab_size {}",
                dflash_config.selector_top_k.unwrap(),
                self.vocab_size
            );
        }
        if let Some(mask_token_id) = dflash_config.mask_token_id {
            if mask_token_id as usize >= self.vocab_size {
                candle_core::bail!(
                    "DFlash2 mask_token_id {} is outside vocab_size {}",
                    mask_token_id,
                    self.vocab_size
                );
            }
        }
        let target_layer_ids = self.target_layer_ids();
        if target_layer_ids.is_empty()
            || target_layer_ids
                .iter()
                .any(|&layer_id| layer_id >= self.num_target_layers)
        {
            candle_core::bail!(
                "DFlash2 target_layer_ids must be non-empty and smaller than num_target_layers"
            );
        }
        Ok(())
    }

    pub fn to_config(&self) -> Config {
        Config {
            architectures: None,
            head_dim: self.head_dim,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: Some(self.num_key_value_heads),
            max_position_embeddings: Some(self.max_position_embeddings),
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            max_seq_len: self.max_position_embeddings,
            intermediate_size: self.intermediate_size,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: self.vocab_size,
            rope_theta: self.rope_theta(),
            rope_local_base_freq: None,
            attention_bias: self.attention_bias,
            use_qkv_bias: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            tie_word_embeddings: false,
            bos_token_id: None,
            eos_token_id: None,
            sliding_window: self.sliding_window,
            sliding_window_pattern: None,
            original_max_position_embeddings: None,
            partial_rotary_factor: None,
            qk_layernorm: false,
            custom_stop_tokens: None,
            hidden_act: Some(candle_nn::Activation::Silu),
            hidden_activation: None,
            rope_scaling: None,
            moe_config: None,
            quantization_config: None,
            isq_quant: None,
            kvcache_dtype: crate::openai::models::KvCacheDtype::Auto,
            extra_config_json: None,
            is_f16_mode: false,
            mtp_enabled: false,
            mtp_max_verify_tokens: 0,
        }
    }
}

fn build_target_layer_ids(num_target_layers: usize, num_draft_layers: usize) -> Vec<usize> {
    if num_draft_layers == 0 || num_target_layers == 0 {
        return Vec::new();
    }
    if num_draft_layers == 1 {
        return vec![num_target_layers / 2];
    }
    let start = 1usize;
    let end = num_target_layers.saturating_sub(3).max(start);
    let span = end.saturating_sub(start);
    (0..num_draft_layers)
        .map(|i| start + (i * span) / (num_draft_layers - 1))
        .collect()
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let half = last_dim / 2;
    let x1 = xs.narrow(D::Minus1, 0, half)?;
    let x2 = xs.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;

    let q_len = q.dim(2)?;
    let cos_len = cos.dim(2)?;

    let cos_q = if cos_len > q_len {
        cos.narrow(2, cos_len - q_len, q_len)?
    } else {
        cos.clone()
    };
    let sin_q = if cos_len > q_len {
        sin.narrow(2, cos_len - q_len, q_len)?
    } else {
        sin.clone()
    };

    let q_embed = (q.broadcast_mul(&cos_q)? + rotate_half(q)?.broadcast_mul(&sin_q)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;

    Ok((q_embed, k_embed))
}

/// Optional sliding-window bias for draft queries over [ctx | noise].
/// DFlash2 checkpoints set `is_causal=false` (block diffusion / encoder-only), so we
/// do NOT apply a causal triangle — only a local window when configured.
fn build_dflash_attn_bias(
    ctx_len: usize,
    q_len: usize,
    sliding_window: Option<usize>,
    is_causal: bool,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    let kv_len = ctx_len + q_len;
    let window = sliding_window.unwrap_or(usize::MAX);
    // Full attention when the whole sequence fits in the window and we are non-causal.
    if !is_causal && kv_len <= window {
        return Ok(None);
    }
    let mut bias = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        let abs_q = ctx_len + i;
        let oldest = abs_q.saturating_add(1).saturating_sub(window);
        let newest = if is_causal {
            abs_q
        } else {
            (abs_q + window.saturating_sub(1)).min(kv_len.saturating_sub(1))
        };
        let row = i * kv_len;
        for j in 0..kv_len {
            if j < oldest || j > newest || (is_causal && j > abs_q) {
                bias[row + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Some(
        Tensor::from_vec(bias, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)?,
    ))
}

pub struct DFlashGroupedConv {
    base_kernel: Tensor,
    kernel_projection: ReplicatedLinear,
    block_size: usize,
    taps: usize,
    num_groups: usize,
}

impl DFlashGroupedConv {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        group_size: usize,
        taps: usize,
        block_size: usize,
        _dtype: DType,
    ) -> Result<Self> {
        if group_size == 0 || hidden_size % group_size != 0 {
            candle_core::bail!(
                "DFlash2 convolution group size {} must divide hidden size {}",
                group_size,
                hidden_size
            );
        }
        let base_kernel = vb.get((2, taps, hidden_size), "base_kernel")?;
        let num_groups = hidden_size / group_size;
        let kernel_projection = ReplicatedLinear::load_no_bias(
            hidden_size,
            2 * taps * num_groups,
            vb.pp("kernel_projection"),
            &None,
            &None,
        )?;
        Ok(Self {
            base_kernel,
            kernel_projection,
            block_size,
            taps,
            num_groups,
        })
    }

    fn convolve(&self, hidden_states: &Tensor, delta: &Tensor, side: usize) -> Result<Tensor> {
        attention_rs::topk::dflash_grouped_conv(
            hidden_states,
            delta,
            &self.base_kernel,
            self.block_size,
            side,
        )
    }

    pub fn prepare(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        let coefficients = self.kernel_projection.forward(hidden_states)?.reshape((
            hidden_states.dim(0)?,
            2,
            self.taps,
            self.num_groups,
        ))?;
        Ok((
            self.convolve(hidden_states, &coefficients.narrow(1, 0, 1)?.squeeze(1)?, 0)?,
            coefficients.narrow(1, 1, 1)?.squeeze(1)?,
        ))
    }

    pub fn finish(&self, hidden_states: &Tensor, coefficients: &Tensor) -> Result<Tensor> {
        self.convolve(hidden_states, coefficients, 1)
    }
}

pub struct DFlashCandidateSelector {
    predecessor_codebook: Tensor,
    successor_codebook: Tensor,
    hidden_projection: ReplicatedLinear,
    top_k: usize,
}

impl DFlashCandidateSelector {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        vocab_size: usize,
        rank: usize,
        top_k: usize,
        _dtype: DType,
    ) -> Result<Self> {
        let predecessor_codebook = vb.get((vocab_size, rank), "predecessor_codebook")?;
        let successor_codebook = vb.get((vocab_size, rank), "successor_codebook")?;
        let hidden_projection = ReplicatedLinear::load_no_bias(
            hidden_size,
            rank,
            vb.pp("hidden_projection"),
            &None,
            &None,
        )?;
        Ok(Self {
            predecessor_codebook,
            successor_codebook,
            hidden_projection,
            top_k,
        })
    }

    pub fn select(
        &self,
        hidden_states: &Tensor,
        logits: &Tensor,
        anchor_token: u32,
    ) -> Result<Vec<u32>> {
        let logits = logits.contiguous()?.to_dtype(DType::F32)?;
        let (unary_logits, candidate_ids) = attention_rs::topk::topk_select(&logits, self.top_k)?;
        let hidden = self
            .hidden_projection
            .forward(hidden_states)?
            .to_dtype(DType::F32)?;
        let selected = attention_rs::topk::dflash_select_candidates(
            &hidden,
            &unary_logits,
            &candidate_ids,
            &self.predecessor_codebook,
            &self.successor_codebook,
            &Tensor::from_vec(vec![anchor_token], (1,), hidden_states.device())?,
        )?;
        selected.to_vec1::<u32>()
    }
}

pub struct DFlashAttention {
    q_proj: ReplicatedLinear,
    k_proj: ReplicatedLinear,
    v_proj: ReplicatedLinear,
    o_proj: ReplicatedLinear,
    q_norm: NormX,
    k_norm: NormX,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f64,
    sliding_window: Option<usize>,
    is_causal: bool,
    dtype: DType,
    device: Device,
}

impl DFlashAttention {
    pub fn new(vb: VarBuilder, config: &DFlashModelConfig, dtype: DType) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;

        let q_proj = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            num_heads * head_dim,
            vb.pp("q_proj"),
            &None,
            &None,
        )?;
        let k_proj = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            num_kv_heads * head_dim,
            vb.pp("k_proj"),
            &None,
            &None,
        )?;
        let v_proj = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            num_kv_heads * head_dim,
            vb.pp("v_proj"),
            &None,
            &None,
        )?;
        let o_proj = ReplicatedLinear::load_no_bias(
            num_heads * head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
            &None,
            &None,
        )?;

        let q_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            vb.pp("q_norm"),
            DType::F32,
            false,
        )?;
        let k_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            vb.pp("k_norm"),
            DType::F32,
            false,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            sliding_window: config.sliding_window,
            is_causal: config.is_causal.unwrap_or(false),
            dtype,
            device: vb.device().clone(),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        target_hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let q_len = hidden_states.dim(0)?;
        let ctx_len = target_hidden.dim(0)?;
        let num_kv_groups = self.num_heads / self.num_kv_heads;

        let q = self.q_proj.forward(hidden_states)?;
        let q = q.reshape((1, q_len, self.num_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let q = q.transpose(1, 2)?;

        let k_ctx = self.k_proj.forward(target_hidden)?;
        let k_noise = self.k_proj.forward(hidden_states)?;
        let v_ctx = self.v_proj.forward(target_hidden)?;
        let v_noise = self.v_proj.forward(hidden_states)?;

        let k = Tensor::cat(&[&k_ctx, &k_noise], 0)?;
        let v = Tensor::cat(&[&v_ctx, &v_noise], 0)?;
        let kv_len = ctx_len + q_len;
        let k = k.reshape((1, kv_len, self.num_kv_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k)?;
        let k = k.transpose(1, 2)?;
        let v = v
            .reshape((1, kv_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin)?;

        let k = if num_kv_groups > 1 {
            k.unsqueeze(2)?
                .expand((1, self.num_kv_heads, num_kv_groups, kv_len, self.head_dim))?
                .reshape((1, self.num_heads, kv_len, self.head_dim))?
        } else {
            k
        };
        let v = if num_kv_groups > 1 {
            v.unsqueeze(2)?
                .expand((1, self.num_kv_heads, num_kv_groups, kv_len, self.head_dim))?
                .reshape((1, self.num_heads, kv_len, self.head_dim))?
        } else {
            v
        };

        let mut attn_weights = (q.matmul(&k.t()?)? * self.scaling)?;
        if let Some(attn_bias) = build_dflash_attn_bias(
            ctx_len,
            q_len,
            self.sliding_window,
            self.is_causal,
            self.dtype,
            &self.device,
        )? {
            attn_weights = attn_weights.broadcast_add(&attn_bias)?;
        }
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((q_len, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

pub struct DFlashDecoderLayer {
    self_attn: DFlashAttention,
    mlp: Mlp,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    attention_conv: DFlashGroupedConv,
    mlp_conv: DFlashGroupedConv,
}

impl DFlashDecoderLayer {
    pub fn new(
        vb: VarBuilder,
        comm: Rc<Comm>,
        config: &DFlashModelConfig,
        dtype: DType,
    ) -> Result<Self> {
        let self_attn = DFlashAttention::new(vb.pp("self_attn"), config, dtype)?;
        let mlp = Mlp::new(&config.to_config(), vb.pp("mlp"), comm)?;
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
            DType::F32,
            false,
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
            DType::F32,
            false,
        )?;
        let (attention_conv, mlp_conv) = {
            let dflash_config = config.dflash_config.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("DFlash2 config is missing dflash_config".into())
            })?;
            let group_size = dflash_config.conv_group_size.ok_or_else(|| {
                candle_core::Error::Msg("DFlash2 config is missing conv_group_size".into())
            })?;
            let taps = dflash_config.conv_kernel_size.ok_or_else(|| {
                candle_core::Error::Msg("DFlash2 config is missing conv_kernel_size".into())
            })?;
            let block_size = config.block_size();
            (
                DFlashGroupedConv::new(
                    vb.pp("attention_conv"),
                    config.hidden_size,
                    group_size,
                    taps,
                    block_size,
                    dtype,
                )?,
                DFlashGroupedConv::new(
                    vb.pp("mlp_conv"),
                    config.hidden_size,
                    group_size,
                    taps,
                    block_size,
                    dtype,
                )?,
            )
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            attention_conv,
            mlp_conv,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        target_hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let (hidden_states, attention_coefficients) =
            self.attention_conv.prepare(&hidden_states)?;
        let attn_output = self
            .self_attn
            .forward(&hidden_states, target_hidden, cos, sin)?;
        let attn_output = self
            .attention_conv
            .finish(&attn_output, &attention_coefficients)?;
        let hidden_states = (attn_output + residual)?;
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let (hidden_states, mlp_coefficients) = self.mlp_conv.prepare(&hidden_states)?;
        let mlp_output = self.mlp.forward(&hidden_states)?;
        let mlp_output = self.mlp_conv.finish(&mlp_output, &mlp_coefficients)?;
        residual + mlp_output
    }
}

pub struct DFlashRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl DFlashRotaryEmbedding {
    pub fn new(config: &DFlashModelConfig, dtype: DType, device: &Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let rope_theta = config.rope_theta();
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos_half = freqs.cos()?.to_dtype(dtype)?;
        let sin_half = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self {
            cos: Tensor::cat(&[&cos_half, &cos_half], D::Minus1)?,
            sin: Tensor::cat(&[&sin_half, &sin_half], D::Minus1)?,
        })
    }

    pub fn get_cos_sin(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.index_select(positions, 0)?;
        let sin = self.sin.index_select(positions, 0)?;
        Ok((cos.unsqueeze(0)?, sin.unsqueeze(0)?))
    }
}

pub struct DFlashDraftModel {
    fc: ReplicatedLinear,
    hidden_norm: NormX,
    layers: Vec<DFlashDecoderLayer>,
    norm: NormX,
    rotary_emb: DFlashRotaryEmbedding,
    pub config: DFlashModelConfig,
    pub target_layer_ids: Vec<usize>,
    pub block_size: usize,
    pub mask_token_id: Option<u32>,
    device: Device,
    dtype: DType,
    candidate_selector: DFlashCandidateSelector,
}

impl DFlashDraftModel {
    pub fn new(
        vb: VarBuilder,
        comm: Rc<Comm>,
        config: &DFlashModelConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if !config.is_dflash2() {
            candle_core::bail!(
                "DFlashDraftModel requires a DFlash2 checkpoint (selector_top_k / DFlash2 architecture)"
            );
        }
        let target_layer_ids = config.target_layer_ids();
        let fc_in_dim = target_layer_ids.len() * config.hidden_size;

        let fc = ReplicatedLinear::load_no_bias(
            fc_in_dim,
            config.hidden_size,
            vb.pp("fc"),
            &None,
            &None,
        )?;

        let hidden_norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("hidden_norm"),
            DType::F32,
            false,
        )?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = DFlashDecoderLayer::new(
                vb.pp(&format!("layers.{}", i)),
                comm.clone(),
                config,
                dtype,
            )?;
            layers.push(layer);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("norm"),
            DType::F32,
            false,
        )?;

        let rotary_emb = DFlashRotaryEmbedding::new(config, dtype, device)?;
        let dflash_config = config.dflash_config.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("DFlash2 config is missing dflash_config".into())
        })?;
        let candidate_selector = DFlashCandidateSelector::new(
            vb.pp("candidate_selector"),
            config.hidden_size,
            config.vocab_size,
            dflash_config.selector_rank.ok_or_else(|| {
                candle_core::Error::Msg("DFlash2 config is missing selector_rank".into())
            })?,
            dflash_config.selector_top_k.ok_or_else(|| {
                candle_core::Error::Msg("DFlash2 config is missing selector_top_k".into())
            })?,
            dtype,
        )?;

        Ok(Self {
            fc,
            hidden_norm,
            layers,
            norm,
            rotary_emb,
            target_layer_ids,
            block_size: config.block_size(),
            mask_token_id: config.mask_token_id(),
            config: config.clone(),
            device: device.clone(),
            dtype,
            candidate_selector,
        })
    }

    pub fn concat_target_hiddens(&self, all_hidden_states: &[Tensor]) -> Result<Tensor> {
        if all_hidden_states.len() < self.target_layer_ids.len() + 1 {
            candle_core::bail!(
                "DFlash expected at least {} target hidden states, got {}",
                self.target_layer_ids.len() + 1,
                all_hidden_states.len()
            );
        }
        let selected: Vec<Tensor> = (0..self.target_layer_ids.len())
            .map(|i| all_hidden_states[i + 1].clone())
            .collect();
        Tensor::cat(&selected, D::Minus1)?.to_dtype(self.dtype)
    }

    pub fn extract_and_project_hidden(&self, all_hidden_states: &[Tensor]) -> Result<Tensor> {
        let concatenated = self.concat_target_hiddens(all_hidden_states)?;
        let projected = self.fc.forward(&concatenated)?;
        self.hidden_norm.forward(&projected)
    }

    /// Project per-layer verify buffers (no embedding prefix) into draft context rows.
    pub fn project_layer_hiddens(&self, layer_hiddens: &[Tensor]) -> Result<Tensor> {
        if layer_hiddens.len() != self.target_layer_ids.len() {
            candle_core::bail!(
                "DFlash expected {} layer hiddens, got {}",
                self.target_layer_ids.len(),
                layer_hiddens.len()
            );
        }
        let concatenated = Tensor::cat(layer_hiddens, D::Minus1)?.to_dtype(self.dtype)?;
        let projected = self.fc.forward(&concatenated)?;
        self.hidden_norm.forward(&projected)
    }

    pub fn forward(
        &self,
        target_hidden: &Tensor,
        noise_embedding: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let positions_flat = positions.flatten_all()?;
        let (cos, sin) = self.rotary_emb.get_cos_sin(&positions_flat)?;

        let mut hidden_states = noise_embedding.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, target_hidden, &cos, &sin)?;
        }

        self.norm.forward(&hidden_states)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn is_dflash2(&self) -> bool {
        true
    }

    pub fn select_candidates(
        &self,
        hidden_states: &Tensor,
        logits: &Tensor,
        anchor_token: u32,
    ) -> Result<Vec<u32>> {
        self.candidate_selector
            .select(hidden_states, logits, anchor_token)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// External DFlash2 drafter and its per-sequence projected target context.
pub struct DFlashDrafter {
    pub draft_model: DFlashDraftModel,
    target_layer_ids: Vec<usize>,
    pub num_speculative_tokens: usize,
    pub mask_token_id: u32,
    context_window: usize,
    device: Device,
    dtype: DType,
    cached_target_hidden: Mutex<HashMap<usize, Tensor>>,
}

impl DFlashDrafter {
    pub fn new(
        draft_config: &DFlashModelConfig,
        draft_weight_files: &[PathBuf],
        comm: Rc<Comm>,
        dtype: DType,
        device: &Device,
        num_speculative_tokens: Option<usize>,
    ) -> Result<Self> {
        draft_config.validate()?;
        let target_layer_ids = draft_config.target_layer_ids();
        let num_speculative_tokens =
            num_speculative_tokens.unwrap_or_else(|| draft_config.block_size().saturating_sub(1));
        if num_speculative_tokens == 0 {
            candle_core::bail!("DFlash2 requires at least one speculative token");
        }
        let required_positions = num_speculative_tokens
            .checked_add(1)
            .ok_or_else(|| candle_core::Error::msg("DFlash2 speculative token count overflows"))?;
        let max_context = draft_config
            .max_position_embeddings
            .checked_sub(required_positions)
            .ok_or_else(|| {
                candle_core::Error::msg(
                    "DFlash2 max_position_embeddings is too small for the speculative block",
                )
            })?;
        if max_context == 0 {
            candle_core::bail!(
                "DFlash2 max_position_embeddings must leave room for target context and the speculative block"
            );
        }
        if draft_weight_files.is_empty() {
            candle_core::bail!("DFlash2 draft model has no safetensors weight files");
        }

        let draft_vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                draft_weight_files,
                dtype,
                device,
            )?
        };
        let draft_model = DFlashDraftModel::new(draft_vb, comm, draft_config, dtype, device)?;
        let context_window = DFLASH_CONTEXT_WINDOW.min(max_context);
        let mask_token_id = draft_config.mask_token_id().unwrap_or(0);

        tracing::info!(
            "DFlash2 drafter initialized: {} draft layers, {} speculative tokens, target layers {:?}, context window {}",
            draft_config.num_hidden_layers,
            num_speculative_tokens,
            target_layer_ids,
            context_window,
        );

        Ok(Self {
            draft_model,
            target_layer_ids,
            num_speculative_tokens,
            mask_token_id,
            context_window,
            device: device.clone(),
            dtype,
            cached_target_hidden: Mutex::new(HashMap::new()),
        })
    }

    pub fn target_layer_ids(&self) -> &[usize] {
        &self.target_layer_ids
    }

    pub fn project_layer_hiddens(&self, layer_hiddens: &[Tensor]) -> Result<Tensor> {
        self.draft_model.project_layer_hiddens(layer_hiddens)
    }

    pub fn draft_tokens(
        &self,
        target_hidden: &Tensor,
        embed_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        lm_head_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        anchor_token: u32,
    ) -> Result<Vec<u32>> {
        let n = self.num_speculative_tokens;
        let mut block_ids = vec![self.mask_token_id; n + 1];
        block_ids[0] = anchor_token;
        let block_tensor = Tensor::from_vec(
            block_ids.into_iter().map(|token| token as i64).collect(),
            (n + 1,),
            &self.device,
        )?;
        let noise_embedding = embed_fn(&block_tensor)?.to_dtype(self.dtype)?;
        let target_hidden = if target_hidden.rank() == 3 {
            let (batch, context, hidden) = target_hidden.dims3()?;
            if batch != 1 {
                candle_core::bail!("DFlash2 draft context must have batch size 1");
            }
            target_hidden.reshape((context, hidden))?
        } else if target_hidden.rank() == 2 {
            target_hidden.clone()
        } else {
            candle_core::bail!("DFlash2 draft context must be a rank-2 or rank-3 tensor");
        }
        .to_dtype(self.dtype)?;
        let noise_embedding = if noise_embedding.rank() == 3 {
            let (batch, sequence, hidden) = noise_embedding.dims3()?;
            if batch != 1 {
                candle_core::bail!("DFlash2 noise embedding must have batch size 1");
            }
            noise_embedding.reshape((sequence, hidden))?
        } else if noise_embedding.rank() == 2 {
            noise_embedding
        } else {
            candle_core::bail!("DFlash2 noise embedding must be a rank-2 or rank-3 tensor");
        };
        let context_len = target_hidden.dim(0)?;
        let total_len = context_len
            .checked_add(n + 1)
            .ok_or_else(|| candle_core::Error::msg("DFlash2 draft sequence length overflows"))?;
        let positions = Tensor::arange(0i64, total_len as i64, &self.device)?;
        let draft_hidden =
            self.draft_model
                .forward(&target_hidden, &noise_embedding, &positions)?;
        let total_out = draft_hidden.dim(0)?;
        if total_out < n {
            candle_core::bail!(
                "DFlash2 draft returned {} rows, expected at least {}",
                total_out,
                n
            );
        }
        let draft_hidden = draft_hidden.narrow(0, total_out - n, n)?;
        let draft_logits = lm_head_fn(&draft_hidden)?;
        let draft_tokens =
            self.draft_model
                .select_candidates(&draft_hidden, &draft_logits, anchor_token)?;
        if draft_tokens.len() != n {
            candle_core::bail!(
                "DFlash2 candidate selector returned {} tokens, expected {}",
                draft_tokens.len(),
                n
            );
        }
        Ok(draft_tokens)
    }

    pub fn build_draft_context(&self, seq_id: usize) -> Result<Option<Tensor>> {
        Ok(self
            .cached_target_hidden
            .lock()
            .map_err(|_| candle_core::Error::msg("DFlash2 context lock poisoned"))?
            .get(&seq_id)
            .cloned())
    }

    pub fn clear_seq_hidden(&self, seq_id: usize) {
        if let Ok(mut cached) = self.cached_target_hidden.lock() {
            cached.remove(&seq_id);
        }
    }

    pub fn append_context(&self, seq_id: usize, projected: &Tensor) -> Result<()> {
        if projected.rank() != 2 {
            candle_core::bail!("DFlash2 projected context must be rank 2");
        }
        let rows = projected.dim(0)?;
        if rows == 0 {
            return Ok(());
        }
        let mut cached = self
            .cached_target_hidden
            .lock()
            .map_err(|_| candle_core::Error::msg("DFlash2 context lock poisoned"))?;
        let updated = match cached.get(&seq_id) {
            Some(previous) => Tensor::cat(&[previous, projected], 0)?,
            None => projected.clone(),
        };
        let total = updated.dim(0)?;
        let keep = total.min(self.context_window);
        cached.insert(seq_id, updated.narrow(0, total - keep, keep)?);
        Ok(())
    }

    pub fn append_verified_context(
        &self,
        seq_id: usize,
        projected: &Tensor,
        accepted_count: usize,
    ) -> Result<()> {
        let rows = projected.dim(0)?;
        // Match the xInfer DFlash2 state machine: a completely rejected
        // proposal does not contribute projected draft context.  The target
        // anchor is already represented by the context accumulated before the
        // verification step; only accepted draft rows are appended here.
        if accepted_count == 0 || rows == 0 {
            return Ok(());
        }
        let keep = accepted_count.saturating_add(1).min(rows);
        self.append_context(seq_id, &projected.narrow(0, 0, keep)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_layer_ids_handle_small_models() {
        assert!(build_target_layer_ids(0, 2).is_empty());
        assert!(build_target_layer_ids(4, 0).is_empty());
        assert_eq!(build_target_layer_ids(10, 1), vec![5]);
        assert!(build_target_layer_ids(2, 4).iter().all(|&id| id < 2));
    }

    #[test]
    fn dflash_config_validates_required_metadata() {
        let config: DFlashModelConfig = serde_json::from_str(
            r#"
            {
              "architectures": ["DFlash2ForCausalLM"],
              "hidden_size": 8,
              "num_hidden_layers": 2,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "intermediate_size": 16,
              "rms_norm_eps": 0.000001,
              "vocab_size": 32,
              "max_position_embeddings": 64,
              "num_target_layers": 8,
              "dflash_config": {
                "block_size": 5,
                "mask_token_id": 3,
                "target_layer_ids": [1, 4],
                "conv_group_size": 2,
                "conv_kernel_size": 3,
                "selector_rank": 4,
                "selector_top_k": 8
              }
            }
            "#,
        )
        .unwrap();
        assert!(config.is_dflash2());
        assert_eq!(config.target_layer_ids(), vec![1, 4]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn dflash_attention_bias_respects_window() -> Result<()> {
        let device = Device::Cpu;
        assert!(build_dflash_attn_bias(1, 1, Some(4), false, DType::F32, &device)?.is_none());
        let bias = build_dflash_attn_bias(2, 2, Some(2), false, DType::F32, &device)?
            .expect("windowed attention should have a bias");
        assert_eq!(bias.dims4()?, (1, 1, 2, 4));
        let values = bias.flatten_all()?.to_vec1::<f32>()?;
        assert!(values[0].is_infinite());
        assert_eq!(values[1], 0.0);
        assert!(values[5].is_infinite());
        assert_eq!(values[6], 0.0);
        Ok(())
    }
}
