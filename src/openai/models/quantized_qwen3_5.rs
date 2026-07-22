use super::layers::quantized_var_builder::VarBuilder as QVarBuilder;
use super::rotary_emb::ScalingRotaryEmbedding;
use super::{attention::QuantizedAttention, Config, KvCacheDtype, Qwen3HybridConfig};
use crate::backend::progress::{ProgressLike, ProgressReporter};
#[cfg(feature = "nccl")]
use crate::openai::distributed::AllReduce;
use crate::openai::distributed::{Comm, Rc, VocabParallelLinear};
use crate::openai::models::layers::qrmsnorm::QRmsNorm;
use crate::openai::models::mask::get_attention_causal_mask;
use crate::openai::models::utils::{resolve_input_seqlens, resolve_mamba_seq_slots};
use crate::InputMetadata;
use attention_rs::gdn;
use attention_rs::mamba_cache::MambaCache;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use either::Either;
use parking_lot::{RwLock, RwLockWriteGuard};
use std::collections::HashMap;
use std::sync::Arc;

/// Undo the GGUF tiled v-head layout along the leading dimension.
/// GGUF stores v-heads interleaved with k-groups as [num_v_per_k, num_k_heads, head_dim, ...].
/// This restores the canonical [num_v_heads, head_dim, ...] ordering.
pub fn undo_tiled_v_heads_first_dim(
    x: &Tensor,
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if num_k_heads == num_v_heads {
        return Ok(x.clone());
    }
    let num_v_per_k = num_v_heads / num_k_heads;
    let dims = x.dims().to_vec();
    if dims.is_empty() || dims[0] != num_v_heads * head_dim {
        candle_core::bail!(
            "undo_tiled_v_heads_first_dim expects leading dim {}, got {:?}",
            num_v_heads * head_dim,
            x.shape()
        );
    }
    let mut reshaped = vec![num_v_per_k, num_k_heads, head_dim];
    reshaped.extend_from_slice(&dims[1..]);
    x.reshape(reshaped)?
        .transpose(0, 1)?
        .contiguous()?
        .reshape(dims)
}

/// Undo the GGUF tiled v-head layout along the trailing dimension.
pub fn undo_tiled_v_heads_last_dim(
    x: &Tensor,
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if num_k_heads == num_v_heads {
        return Ok(x.clone());
    }
    let num_v_per_k = num_v_heads / num_k_heads;
    let dims = x.dims().to_vec();
    let last = *dims.last().ok_or_else(|| {
        candle_core::Error::Msg("undo_tiled_v_heads_last_dim expects non-empty tensor".into())
    })?;
    if last != num_v_heads * head_dim {
        candle_core::bail!(
            "undo_tiled_v_heads_last_dim expects trailing dim {}, got {:?}",
            num_v_heads * head_dim,
            x.shape()
        );
    }
    let split_dim = dims.len() - 1;
    let mut reshaped = dims[..split_dim].to_vec();
    reshaped.extend_from_slice(&[num_v_per_k, num_k_heads, head_dim]);
    x.reshape(reshaped)?
        .transpose(split_dim, split_dim + 1)?
        .contiguous()?
        .reshape(dims)
}

struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
    #[cfg(feature = "nccl")]
    dtype: DType,
}

impl Mlp {
    #[allow(unused_mut)]
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        let mut y = self
            .feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)?;
        #[cfg(feature = "nccl")]
        if let Some(all_reduce) = &self.all_reduce {
            y = all_reduce.apply(&y.to_dtype(self.dtype)?)?;
            y = y.to_dtype(DType::F32)?;
        }
        Ok(y)
    }
}

pub(crate) struct QuantizedGatedDeltaNet {
    in_proj_qkv: QMatMul,
    in_proj_z: QMatMul,
    in_proj_b: QMatMul,
    in_proj_a: QMatMul,
    out_proj: QMatMul,
    conv_weight: Tensor,
    conv_bias: Option<Tensor>,
    a_log: Tensor,
    dt_bias: Tensor,
    gdn_norm_weight: Tensor,
    gdn_norm_bias: Option<Tensor>,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    kv_group_size: usize,
    gdn_layer_idx: usize,
    rms_norm_eps: f64,
    scale: f64,
}

impl QuantizedGatedDeltaNet {
    pub(crate) fn new(
        vb: &QVarBuilder,
        prefix: &str,
        device: &Device,
        hybrid: &Qwen3HybridConfig,
        gdn_layer_idx: usize,
        rms_norm_eps: f64,
        _rank: usize,
        _world_size: usize,
    ) -> Result<Self> {
        let num_v_heads = hybrid.num_v_heads;
        let num_k_heads = hybrid.num_k_heads;
        let head_k_dim = hybrid.key_head_dim;
        let head_v_dim = hybrid.value_head_dim;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let kv_group_size = num_v_heads / num_k_heads;
        let needs_untile = num_k_heads != num_v_heads;

        let prefix_vb = vb.pp(prefix);

        let in_proj_qkv = if needs_untile {
            let qt = prefix_vb.get_no_shape("attn_qkv.weight")?;
            let orig_dtype = qt.dtype();
            let w = qt.dequantize_f16(device)?;
            drop(qt);
            let q = w.narrow(0, 0, key_dim)?;
            let k = w.narrow(0, key_dim, key_dim)?;
            let v_raw = w.narrow(0, key_dim * 2, value_dim)?;
            let v = undo_tiled_v_heads_first_dim(&v_raw, num_k_heads, num_v_heads, head_v_dim)?;
            let restored = Tensor::cat(&[&q, &k, &v], 0)?.contiguous()?;
            drop(w);
            let last_dim = restored.dim(candle_core::D::Minus1)?;
            let wdtype = if last_dim % orig_dtype.block_size() != 0 {
                candle_core::quantized::GgmlDType::Q8_0
            } else {
                orig_dtype
            };
            QMatMul::from_arc(Arc::new(candle_core::quantized::QTensor::quantize_owned(
                restored, wdtype,
            )?))?
        } else {
            QMatMul::from_arc(prefix_vb.get_no_shape("attn_qkv.weight")?)?
        };
        let in_proj_z = if needs_untile {
            let qt = prefix_vb.get_no_shape("attn_gate.weight")?;
            let orig_dtype = qt.dtype();
            let w = qt.dequantize_f16(device)?;
            drop(qt);
            let w = undo_tiled_v_heads_first_dim(&w, num_k_heads, num_v_heads, head_v_dim)?
                .contiguous()?;
            let last_dim = w.dim(candle_core::D::Minus1)?;
            let wdtype = if last_dim % orig_dtype.block_size() != 0 {
                candle_core::quantized::GgmlDType::Q8_0
            } else {
                orig_dtype
            };
            QMatMul::from_arc(Arc::new(candle_core::quantized::QTensor::quantize_owned(
                w, wdtype,
            )?))?
        } else {
            QMatMul::from_arc(prefix_vb.get_no_shape("attn_gate.weight")?)?
        };
        let in_proj_b = if needs_untile {
            let qt = prefix_vb.get_no_shape("ssm_beta.weight")?;
            let orig_dtype = qt.dtype();
            let w = qt.dequantize_f16(device)?;
            drop(qt);
            let w = undo_tiled_v_heads_first_dim(&w, num_k_heads, num_v_heads, 1)?.contiguous()?;
            let last_dim = w.dim(candle_core::D::Minus1)?;
            let wdtype = if last_dim % orig_dtype.block_size() != 0 {
                candle_core::quantized::GgmlDType::Q8_0
            } else {
                orig_dtype
            };
            QMatMul::from_arc(Arc::new(candle_core::quantized::QTensor::quantize_owned(
                w, wdtype,
            )?))?
        } else {
            QMatMul::from_arc(prefix_vb.get_no_shape("ssm_beta.weight")?)?
        };
        let in_proj_a = if needs_untile {
            let qt = prefix_vb.get_no_shape("ssm_alpha.weight")?;
            let orig_dtype = qt.dtype();
            let w = qt.dequantize_f16(device)?;
            drop(qt);
            let w = undo_tiled_v_heads_first_dim(&w, num_k_heads, num_v_heads, 1)?.contiguous()?;
            let last_dim = w.dim(candle_core::D::Minus1)?;
            let wdtype = if last_dim % orig_dtype.block_size() != 0 {
                candle_core::quantized::GgmlDType::Q8_0
            } else {
                orig_dtype
            };
            QMatMul::from_arc(Arc::new(candle_core::quantized::QTensor::quantize_owned(
                w, wdtype,
            )?))?
        } else {
            QMatMul::from_arc(prefix_vb.get_no_shape("ssm_alpha.weight")?)?
        };
        let out_proj = if needs_untile {
            let qt = prefix_vb.get_no_shape("ssm_out.weight")?;
            let orig_dtype = qt.dtype();
            let w = qt.dequantize_f16(device)?;
            drop(qt);
            let v = undo_tiled_v_heads_last_dim(&w, num_k_heads, num_v_heads, head_v_dim)?
                .contiguous()?;
            drop(w);
            let last_dim = v.dim(candle_core::D::Minus1)?;
            let wdtype = if last_dim % orig_dtype.block_size() != 0 {
                candle_core::quantized::GgmlDType::Q8_0
            } else {
                orig_dtype
            };
            QMatMul::from_arc(Arc::new(candle_core::quantized::QTensor::quantize_owned(
                v, wdtype,
            )?))?
        } else {
            QMatMul::from_arc(prefix_vb.get_no_shape("ssm_out.weight")?)?
        };

        let conv_weight_raw = prefix_vb
            .get_no_shape("ssm_conv1d.weight")?
            .dequantize(device)?;
        let conv_weight = if conv_weight_raw.dims().len() == 2 {
            conv_weight_raw.unsqueeze(1)?
        } else {
            conv_weight_raw
        };
        let conv_weight = if needs_untile {
            let q_w = conv_weight.narrow(0, 0, key_dim)?;
            let k_w = conv_weight.narrow(0, key_dim, key_dim)?;
            let v_w = conv_weight.narrow(0, key_dim * 2, value_dim)?;
            let v_w = undo_tiled_v_heads_first_dim(&v_w, num_k_heads, num_v_heads, head_v_dim)?;
            Tensor::cat(&[&q_w, &k_w, &v_w], 0)?
        } else {
            conv_weight
        };
        let conv_bias = prefix_vb
            .get_no_shape("ssm_conv1d.bias")
            .ok()
            .map(|t| t.dequantize(device))
            .transpose()?;
        let conv_bias = if needs_untile {
            conv_bias
                .map(|cb| {
                    let q_b = cb.narrow(0, 0, key_dim)?;
                    let k_b = cb.narrow(0, key_dim, key_dim)?;
                    let v_b = cb.narrow(0, key_dim * 2, value_dim)?;
                    let v_b =
                        undo_tiled_v_heads_first_dim(&v_b, num_k_heads, num_v_heads, head_v_dim)?;
                    Tensor::cat(&[&q_b, &k_b, &v_b], 0)
                })
                .transpose()?
        } else {
            conv_bias
        };

        let a_raw = prefix_vb
            .get_no_shape("ssm_a")?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        let a_log = a_raw.neg()?.log()?;
        let a_log = if needs_untile {
            undo_tiled_v_heads_first_dim(&a_log, num_k_heads, num_v_heads, 1)?
        } else {
            a_log
        };

        let dt_bias = prefix_vb
            .get_no_shape("ssm_dt.bias")?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        let dt_bias = if needs_untile {
            undo_tiled_v_heads_first_dim(&dt_bias, num_k_heads, num_v_heads, 1)?
        } else {
            dt_bias
        };

        let gdn_norm_weight = prefix_vb
            .get_no_shape("ssm_norm.weight")?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        let gdn_norm_bias = prefix_vb
            .get_no_shape("ssm_norm.bias")
            .ok()
            .map(|t| t.dequantize(device).and_then(|t| t.to_dtype(DType::F32)))
            .transpose()?;

        let scale = 1.0f64 / (head_k_dim as f64).sqrt();

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            out_proj,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            gdn_norm_weight,
            gdn_norm_bias,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            kv_group_size,
            gdn_layer_idx,
            rms_norm_eps,
            scale,
        })
    }

    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        mamba_cache: &mut MambaCache,
        input_metadata: &InputMetadata,
        seq_slots: &Tensor,
    ) -> Result<Tensor> {
        let slot_count = seq_slots.dim(0)?;
        if slot_count == 0 {
            candle_core::bail!("Linear attention requires non-empty sequence slots");
        }
        let (token_count, _hidden) = xs.dims2()?;
        let is_prefill = input_metadata.is_prefill;

        let xs_f32 = xs.to_dtype(DType::F32)?;
        let proj_qkv = self.in_proj_qkv.forward(&xs_f32)?;
        let q = proj_qkv.narrow(1, 0, self.key_dim)?.contiguous()?;
        let k = proj_qkv
            .narrow(1, self.key_dim, self.key_dim)?
            .contiguous()?;
        let v = proj_qkv
            .narrow(1, self.key_dim * 2, self.value_dim)?
            .contiguous()?;
        let z = self.in_proj_z.forward(&xs_f32)?;
        let b = self.in_proj_b.forward(&xs_f32)?;
        let a = self.in_proj_a.forward(&xs_f32)?;

        let mixed_qkv = Tensor::cat(&[&q, &k, &v], 1)?;

        let (kv_conv, prefill_conv_state) = if is_prefill {
            let mut conv_state = mamba_cache.get_batch_conv_state(self.gdn_layer_idx, seq_slots)?;
            let cu_seqlens = input_metadata
                .cu_seqlens_q
                .as_ref()
                .expect("cu_seqlens_q must be present in prefill!");

            let out = gdn::causal_conv1d_fwd(
                &mixed_qkv,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut conv_state,
                None,
                Some(cu_seqlens),
                true,
            )?;
            (out, Some(conv_state))
        } else {
            if token_count != slot_count {
                candle_core::bail!(
                    "Linear attention decode mismatch: {} tokens vs {} sequence slots",
                    token_count,
                    slot_count
                );
            }
            let out = gdn::causal_conv1d_update_slots(
                &mixed_qkv,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                mamba_cache.conv_state_mut(self.gdn_layer_idx),
                seq_slots,
                true,
            )?;
            (out, None)
        };
        if let Some(conv_state) = prefill_conv_state {
            mamba_cache.set_batch_conv_state(self.gdn_layer_idx, seq_slots, &conv_state)?;
        }

        let q_conv = kv_conv.narrow(1, 0, self.key_dim)?;
        let k_conv = kv_conv.narrow(1, self.key_dim, self.key_dim)?;
        let v_conv = kv_conv.narrow(1, self.key_dim * 2, self.value_dim)?;

        let (a_expanded, b_expanded) = (a.unsqueeze(0)?, b.unsqueeze(0)?);
        let (g, beta) =
            gdn::fused_gdn_gating(&self.a_log, &a_expanded, &b_expanded, &self.dt_bias)?;
        let (g, beta) = (g.squeeze(0)?, beta.squeeze(0)?);

        let q = q_conv.reshape((token_count, self.num_k_heads, self.head_k_dim))?;
        let k = k_conv.reshape((token_count, self.num_k_heads, self.head_k_dim))?;
        let v = v_conv.reshape((token_count, self.num_v_heads, self.head_v_dim))?;
        let q = gdn::l2_norm_last_dim(&q, 1e-6)?;
        let k = gdn::l2_norm_last_dim(&k, 1e-6)?;

        let output = if is_prefill {
            let cu_seqlens = input_metadata
                .cu_seqlens_q
                .as_ref()
                .expect("cu_seqlens_q must be present in prefill!");
            let global_state = mamba_cache.recurrent_state_mut(self.gdn_layer_idx);
            let try_flashinfer = self.num_k_heads != self.num_v_heads
                && !input_metadata.is_mtp_verify
                && crate::openai::utils::sm90_lower_precision_gdn_prefill();
            let flashinfer_result = if try_flashinfer {
                #[cfg(all(feature = "cuda", feature = "flashinfer"))]
                {
                    let g_exp = g.exp()?;
                    gdn::gated_delta_rule_prefill_flashinfer_gqa(
                        &q,
                        &k,
                        &v,
                        &g_exp,
                        &beta,
                        global_state,
                        seq_slots,
                        &cu_seqlens,
                        self.scale as f32,
                    )?
                }
                #[cfg(not(all(feature = "cuda", feature = "flashinfer")))]
                {
                    None
                }
            } else {
                None
            };
            if self.num_k_heads != self.num_v_heads {
                if let Some(output) = flashinfer_result {
                    output
                } else {
                    gdn::gated_delta_rule_recurrence_varlen_gqa(
                        &q,
                        &k,
                        &v,
                        &g,
                        &beta,
                        global_state,
                        seq_slots,
                        &cu_seqlens,
                        self.scale as f32,
                        None,
                    )?
                }
            } else {
                let (q, k) = (self.repeat_kv_heads(q)?, self.repeat_kv_heads(k)?);
                let q_scaled = (&q * self.scale)?;
                gdn::gated_delta_rule_recurrence_varlen(
                    &q_scaled,
                    &k,
                    &v,
                    &g,
                    &beta,
                    global_state,
                    seq_slots,
                    cu_seqlens,
                    None,
                )?
            }
        } else {
            let batch = slot_count;
            let v_b = v.reshape((batch, self.num_v_heads, self.head_v_dim))?;
            let g_b = g.reshape((batch, self.num_v_heads))?;
            let beta_b = beta.reshape((batch, self.num_v_heads))?;
            let global_state = mamba_cache.recurrent_state_mut(self.gdn_layer_idx);
            let q_b = q.reshape((batch, self.num_k_heads, self.head_k_dim))?;
            let k_b = k.reshape((batch, self.num_k_heads, self.head_k_dim))?;
            gdn::gated_delta_rule_decode_slots_gqa(
                &q_b,
                &k_b,
                &v_b,
                &g_b,
                &beta_b,
                global_state,
                seq_slots,
                self.scale as f32,
            )?
        };

        let output = output.reshape((token_count, self.value_dim))?;
        let gated_output = gdn::gated_rmsnorm_silu_mul(
            &output,
            &z,
            &self.gdn_norm_weight,
            self.gdn_norm_bias.as_ref(),
            self.rms_norm_eps,
            self.head_v_dim,
        )?;

        self.out_proj.forward(&gated_output)
    }
}

enum AttnType {
    FullAttention(QuantizedAttention),
    LinearAttention(QuantizedGatedDeltaNet),
}

struct LayerWeights {
    attn: AttnType,
    attention_norm: QRmsNorm,
    mlp: Mlp,
    ffn_norm: QRmsNorm,
}

impl LayerWeights {
    #[allow(dead_code)]
    fn is_full_attention(&self) -> bool {
        matches!(self.attn, AttnType::FullAttention(_))
    }
}

pub struct GGUFQWen3_5 {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: VocabParallelLinear,
    mamba_cache: RwLock<MambaCache>,
    cfg: Config,
    dtype: DType,
    device: Device,
}

pub fn parse_gguf_hybrid_config(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
    arch: &str,
    block_count: usize,
) -> Qwen3HybridConfig {
    let md_get = |s: &str| metadata.get(s);

    let layer_types: Vec<String> = if let Some(v) = md_get(&format!("{arch}.layer_types")) {
        if let Ok(arr) = v.to_vec() {
            arr.iter()
                .filter_map(|val| val.to_string().ok().map(|s| s.clone()))
                .collect()
        } else {
            vec!["full_attention".to_string(); block_count]
        }
    } else if let Some(v) = md_get(&format!("{arch}.full_attention_interval")) {
        let interval = v.to_u32().unwrap_or(0) as usize;
        if interval > 0 {
            (0..block_count)
                .map(|idx| {
                    if (idx + 1) % interval == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect()
        } else {
            vec!["full_attention".to_string(); block_count]
        }
    } else {
        vec!["full_attention".to_string(); block_count]
    };

    let layer_types: Vec<String> = layer_types
        .into_iter()
        .map(|lt| {
            if lt == "attention" {
                "full_attention".to_string()
            } else {
                lt
            }
        })
        .collect();

    // GGUF stores hybrid/linear attention params under ssm.* keys:
    //   ssm.group_count   -> num_k_heads
    //   ssm.time_step_rank -> num_v_heads
    //   ssm.state_size    -> key_head_dim
    //   ssm.inner_size    -> used to derive value_head_dim
    //   ssm.conv_kernel   -> conv_kernel_size
    let conv_kernel_size = md_get(&format!("{arch}.ssm.conv_kernel"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(4) as usize;

    let num_k_heads = md_get(&format!("{arch}.ssm.group_count"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;

    let num_v_heads = md_get(&format!("{arch}.ssm.time_step_rank"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;

    let key_head_dim = md_get(&format!("{arch}.ssm.state_size"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(128) as usize;

    let inner_size = md_get(&format!("{arch}.ssm.inner_size"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;

    let value_head_dim = if num_v_heads > 0 && inner_size > 0 && inner_size % num_v_heads == 0 {
        inner_size / num_v_heads
    } else {
        key_head_dim
    };

    Qwen3HybridConfig {
        layer_types,
        conv_kernel_size,
        num_v_heads,
        num_k_heads,
        key_head_dim,
        value_head_dim,
    }
}

impl GGUFQWen3_5 {
    pub fn into_config(
        arch: String,
        embedding_length: usize,
        head_dim: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        rms_eps: f64,
        rope_theta: f64,
        max_seq_len: usize,
        original_max_position_embeddings: Option<usize>,
        partial_rotary_factor: Option<f32>,
        _kv_cache_dtype: DType,
        extra_config_json: Option<String>,
    ) -> Config {
        Config {
            architectures: Some(vec![arch]),
            hidden_size: embedding_length,
            head_dim: Some(head_dim),
            intermediate_size: 0,
            vocab_size: 0,
            num_hidden_layers: block_count,
            num_attention_heads: head_count,
            num_key_value_heads: Some(head_count_kv),
            rms_norm_eps: rms_eps,
            rope_theta,
            rope_local_base_freq: None,
            bos_token_id: Some(super::TokenID(Either::Left(Some(151644)))),
            eos_token_id: Some(super::TokenID(Either::Left(Some(151645)))),
            max_seq_len,
            sliding_window: None,
            sliding_window_pattern: None,
            hidden_act: None,
            hidden_activation: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            max_position_embeddings: Some(max_seq_len),
            original_max_position_embeddings,
            attention_bias: Some(false),
            partial_rotary_factor,
            qk_layernorm: false,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: None,
            moe_config: None,
            isq_quant: None,
            kvcache_dtype: KvCacheDtype::Auto,
            extra_config_json,
            is_f16_mode: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_gguf(
        vb: &QVarBuilder,
        device: &Device,
        dtype: DType,
        kv_cache_dtype: DType,
        yarn_scaling_factor: Option<f64>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
        rank: usize,
        world_size: usize,
        #[allow(unused_variables)] comm: Rc<Comm>,
    ) -> Result<Self> {
        let metadata = vb.first_content_metadata();
        let md_get = |s: &str| match metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };
        let reporter = progress_reporter.clone();
        let arch = md_get("general.architecture")?.to_string()?;

        let head_count =
            md_get(format!("{arch}.attention.head_count").as_str())?.to_u32()? as usize;
        let head_count_kv =
            md_get(format!("{arch}.attention.head_count_kv").as_str())?.to_u32()? as usize;
        let head_dim = md_get(format!("{arch}.attention.key_length").as_str());
        let embedding_length =
            md_get(format!("{arch}.embedding_length").as_str())?.to_u32()? as usize;
        let head_dim = if head_dim.is_ok() {
            head_dim.unwrap().to_u32()? as usize
        } else {
            embedding_length / head_count
        };
        let context_length = md_get(format!("{arch}.context_length").as_str())?.to_u32()? as usize;
        let block_count = md_get(format!("{arch}.block_count").as_str())?.to_u32()? as usize;
        let rms_norm_eps =
            md_get(format!("{arch}.attention.layer_norm_rms_epsilon").as_str())?.to_f32()? as f64;
        let rope_freq_base = md_get(format!("{arch}.rope.freq_base").as_str())
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let original_max_position_embeddings =
            md_get(format!("{arch}.rope.scaling.original_context_length").as_str());
        let original_max_position_embeddings = if original_max_position_embeddings.is_ok() {
            Some(original_max_position_embeddings.unwrap().to_u32()? as usize)
        } else {
            None
        };

        let rope_dim = md_get(format!("{arch}.rope.dimension_count").as_str());
        let partial_rotary_factor = if rope_dim.is_ok() {
            let rope_dim = rope_dim.unwrap().to_u32()? as usize;
            if rope_dim != head_dim {
                Some(rope_dim as f32 / head_dim as f32)
            } else {
                None
            }
        } else {
            None
        };

        let hybrid = parse_gguf_hybrid_config(&metadata, &arch, block_count);

        let extra_config_json = build_extra_config_json(&hybrid, &arch);

        let mut cfg = GGUFQWen3_5::into_config(
            arch.clone(),
            embedding_length,
            head_dim,
            block_count,
            head_count,
            head_count_kv,
            rms_norm_eps,
            rope_freq_base as f64,
            context_length,
            original_max_position_embeddings,
            partial_rotary_factor,
            kv_cache_dtype,
            Some(extra_config_json),
        );
        cfg.apply_runtime_rope_overrides(yarn_scaling_factor);
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, &cfg, device, true)?);

        let tok_embeddings = vb.get_no_shape("token_embd.weight")?;
        let vocab_size = tok_embeddings.shape().dims()[0];
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm =
            QRmsNorm::from_arc_qtensor(vb.get_no_shape("output_norm.weight")?, rms_norm_eps)?;
        let output_tensor_name = if vb.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let output = VocabParallelLinear::load_from_gguf(
            vb,
            output_tensor_name,
            vocab_size,
            comm.clone(),
            dtype,
        )?;

        let layer_types = &hybrid.layer_types;
        let mut layers = Vec::with_capacity(block_count);
        let mut gdn_layer_idx = 0usize;

        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let layer_type = layer_types
                .get(layer_idx)
                .map(String::as_str)
                .unwrap_or("full_attention");

            let attn = if layer_type == "full_attention" {
                AttnType::FullAttention(QuantizedAttention::new(
                    &cfg,
                    vb,
                    &prefix,
                    device,
                    dtype,
                    rotary_emb.clone(),
                    cfg.sliding_window,
                    rank,
                    world_size,
                    comm.clone(),
                )?)
            } else {
                let cur_gdn_idx = gdn_layer_idx;
                gdn_layer_idx += 1;
                AttnType::LinearAttention(QuantizedGatedDeltaNet::new(
                    vb,
                    &prefix,
                    device,
                    &hybrid,
                    cur_gdn_idx,
                    rms_norm_eps,
                    rank,
                    world_size,
                )?)
            };

            let mlp = {
                let prefix_vb = vb.pp(&prefix);
                let feed_forward_w1 =
                    prefix_vb.get_sharded_no_shape("ffn_gate.weight", 0, rank, world_size)?;
                let feed_forward_w2 =
                    prefix_vb.get_sharded_no_shape("ffn_down.weight", 1, rank, world_size)?;
                let feed_forward_w3 =
                    prefix_vb.get_sharded_no_shape("ffn_up.weight", 0, rank, world_size)?;
                Mlp {
                    feed_forward_w1: QMatMul::from_arc(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_arc(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_arc(feed_forward_w3)?,
                    #[cfg(feature = "nccl")]
                    all_reduce: if world_size > 1 {
                        Some(AllReduce::new(comm.clone()))
                    } else {
                        None
                    },
                    #[cfg(feature = "nccl")]
                    dtype,
                }
            };

            let prefix_vb = vb.pp(&prefix);
            let attention_norm = prefix_vb.get_no_shape("attn_norm.weight")?;
            let ffn_norm = prefix_vb.get_no_shape("post_attention_norm.weight")?;

            layers.push(LayerWeights {
                attn,
                attention_norm: QRmsNorm::from_arc_qtensor(attention_norm, rms_norm_eps)?,
                mlp,
                ffn_norm: QRmsNorm::from_arc_qtensor(ffn_norm, rms_norm_eps)?,
            });
            reporter.write().set_progress(layer_idx + 1);
        }

        let num_gdn_layers = gdn_layer_idx;
        let num_v_heads = hybrid.num_v_heads;
        let num_k_heads = hybrid.num_k_heads;
        let d_conv = num_k_heads * hybrid.key_head_dim * 2 + num_v_heads * hybrid.value_head_dim;

        let mamba_cache = if num_gdn_layers > 0 {
            MambaCache::new(
                num_gdn_layers,
                1,
                d_conv,
                hybrid.conv_kernel_size,
                num_v_heads,
                hybrid.key_head_dim,
                hybrid.value_head_dim,
                DType::F32,
                DType::F32,
                device,
            )?
        } else {
            MambaCache::new(0, 1, 1, 2, 1, 1, 1, DType::F32, DType::F32, device)?
        };

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            mamba_cache: RwLock::new(mamba_cache),
            cfg,
            dtype,
            device: device.clone(),
        })
    }

    pub fn embed_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.forward(input_ids)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn forward(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(x, input_positions, kv_caches, input_metadata, false, false)
    }

    pub fn forward_embedding(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(x, input_positions, kv_caches, input_metadata, true, false)
    }

    pub fn forward_with_deepstack(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embedded_inputs: bool,
        _visual_pos_masks: &Option<Tensor>,
        _deepstack_visual_embeds: &Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        self.forward_inner(
            x,
            input_positions,
            kv_caches,
            input_metadata,
            false,
            embedded_inputs,
        )
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        return_hidden: bool,
        embedded_inputs: bool,
    ) -> Result<Tensor> {
        let seqlens = resolve_input_seqlens(input_metadata)?;
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            input_positions,
            &seqlens,
            self.cfg.sliding_window,
            input_metadata.is_prefill,
        );
        let mut xs = if embedded_inputs {
            x.clone()
        } else {
            self.tok_embeddings.forward(x)?
        };

        let mut mamba_cache = self.mamba_cache.write();
        let seq_slots = resolve_mamba_seq_slots(
            "Qwen3.5 GGUF",
            &self.device,
            input_metadata,
            xs.dim(0)?,
            &mut mamba_cache,
        )?;

        let mut kv_cache_idx = 0usize;
        for layer in self.layers.iter() {
            let residual = &xs;
            let x = layer.attention_norm.forward(&xs)?;
            let x = match &layer.attn {
                AttnType::FullAttention(attn) => {
                    let cache = if let Some(kv_caches) = kv_caches {
                        let c = &kv_caches[kv_cache_idx];
                        kv_cache_idx += 1;
                        Some((&c.0, &c.1))
                    } else {
                        None
                    };
                    attn.forward(
                        &x,
                        attention_mask.as_ref(),
                        input_positions,
                        cache,
                        input_metadata,
                    )?
                }
                AttnType::LinearAttention(gdn) => {
                    gdn.forward(&x, &mut mamba_cache, input_metadata, &seq_slots)?
                }
            };
            let x = (x + residual)?;

            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            xs = (x + residual)?;
        }

        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;

        if return_hidden {
            return Ok(xs);
        }
        self.output.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }

    pub fn release_sequence_state(&self, sequence_id: usize) {
        self.mamba_cache.write().free_slot(sequence_id);
    }

    pub fn ensure_mamba_slots_for_sequences(&self, sequence_ids: &[usize]) -> Result<Vec<usize>> {
        self.mamba_cache
            .write()
            .ensure_slots_for_sequences(sequence_ids)
    }

    pub fn get_mamba_slots_for_sequences(&self, sequence_ids: &[usize]) -> Result<Vec<usize>> {
        self.mamba_cache
            .write()
            .get_slots_for_sequences(sequence_ids)
    }

    pub fn has_mamba_slot_for_sequence(&self, sequence_id: usize) -> bool {
        self.mamba_cache.read().get_slot(sequence_id).is_some()
    }

    pub fn lock_mamba_cache_for_graph(&self) -> RwLockWriteGuard<'_, MambaCache> {
        self.mamba_cache.write()
    }

    pub fn preallocate_mamba_cache(&self, max_num_seqs: usize) -> Result<()> {
        self.mamba_cache.write().reserve_capacity(max_num_seqs)
    }

    pub fn set_mamba_prefix_cache_capacity(&self, capacity: usize) {
        self.mamba_cache.write().set_prefix_cache_capacity(capacity);
    }

    pub fn capture_mamba_prefix_state(
        &self,
        seq_id: usize,
        hash: u64,
        preserve: bool,
    ) -> Result<bool> {
        self.mamba_cache
            .write()
            .capture_prefix_state(seq_id, hash, preserve)
    }

    pub fn has_mamba_prefix_state(&self, hash: u64) -> bool {
        self.mamba_cache.write().has_prefix_state(hash)
    }

    pub fn restore_mamba_prefix_state(&self, seq_id: usize, hash: u64) -> Result<bool> {
        self.mamba_cache.write().restore_prefix_state(seq_id, hash)
    }

    pub fn reset_mamba_cache(&self) -> Result<()> {
        self.mamba_cache.write().reset_all()
    }
}

pub fn build_extra_config_json(hybrid: &Qwen3HybridConfig, arch: &str) -> String {
    let layer_types_json: Vec<String> = hybrid
        .layer_types
        .iter()
        .map(|lt| format!("\"{}\"", lt))
        .collect();
    format!(
        r#"{{"architectures":["{}"],"layer_types":[{}],"linear_num_value_heads":{},"linear_num_key_heads":{},"linear_key_head_dim":{},"linear_value_head_dim":{},"linear_conv_kernel_dim":{}}}"#,
        arch,
        layer_types_json.join(","),
        hybrid.num_v_heads,
        hybrid.num_k_heads,
        hybrid.key_head_dim,
        hybrid.value_head_dim,
        hybrid.conv_kernel_size,
    )
}
