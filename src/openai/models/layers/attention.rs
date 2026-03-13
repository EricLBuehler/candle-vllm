use super::rotary_emb::ScalingRotaryEmbedding;
use crate::openai::distributed::{
    rms_norm_x, shard, Comm, MergedParallelColumnLinear, TensorParallelColumnLinear,
    TensorParallelRowLinear, VarBuilder,
};
use crate::openai::models::layers::qrmsnorm::QRmsNorm;
use crate::openai::models::Config;
use crate::{InputMetadata, PagedAttention};
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::RmsNorm;

use std::rc::Rc;
use std::sync::Arc;

enum QkvProjection {
    Separate {
        q_proj: TensorParallelColumnLinear,
        k_proj: TensorParallelColumnLinear,
        v_proj: TensorParallelColumnLinear,
    },
    Packed(MergedParallelColumnLinear),
}

pub struct Attention {
    qkv_proj: QkvProjection,
    o_proj: TensorParallelRowLinear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
    attn: PagedAttention,
    softcapping: Option<f64>,
    dtype: DType,
    attn_output_gate: bool,
    no_per_head_norm: bool,
}

impl Attention {
    fn normalize_sharded_2d(
        t: Tensor,
        shard: candle_nn::var_builder::Shard,
        global_dim0: usize,
        global_dim1: usize,
        name: &str,
    ) -> Result<Tensor> {
        if shard.world_size <= 1 {
            return Ok(t);
        }
        if shard.dim > 1 {
            candle_core::bail!("unexpected shard dim {} for {}", shard.dim, name);
        }
        let (d0, d1) = t.dims2()?;
        if shard.dim == 0 {
            let local = global_dim0 / shard.world_size;
            if d0 == local {
                return Ok(t);
            }
            if d0 == global_dim0 {
                return t.narrow(0, shard.rank * local, local)?.contiguous();
            }
            candle_core::bail!(
                "unexpected {} shape ({}, {}), shard dim 0 expects local {} or global {}",
                name,
                d0,
                d1,
                local,
                global_dim0
            );
        }

        let local = global_dim1 / shard.world_size;
        if d1 == local {
            return Ok(t);
        }
        if d1 == global_dim1 {
            return t.narrow(1, shard.rank * local, local)?.contiguous();
        }
        candle_core::bail!(
            "unexpected {} shape ({}, {}), shard dim 1 expects local {} or global {}",
            name,
            d0,
            d1,
            local,
            global_dim1
        );
    }

    fn normalize_sharded_1d(
        t: Tensor,
        shard: candle_nn::var_builder::Shard,
        global_dim: usize,
        name: &str,
    ) -> Result<Tensor> {
        if shard.world_size <= 1 {
            return Ok(t);
        }
        let d0 = t.dim(0)?;
        let local = global_dim / shard.world_size;
        if d0 == local {
            return Ok(t);
        }
        if d0 == global_dim {
            return t.narrow(0, shard.rank * local, local)?.contiguous();
        }
        candle_core::bail!(
            "unexpected {} shape ({}), expects local {} or global {}",
            name,
            d0,
            local,
            global_dim
        );
    }

    fn load_sharded_bias(
        vb: &VarBuilder,
        out_dim: usize,
        shard: candle_nn::var_builder::Shard,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        let bias = match vb.get_with_hints_dtype((out_dim,), "bias", shard, DType::F32) {
            Ok(bias) => bias,
            Err(_) => return Ok(None),
        };
        let bias = Self::normalize_sharded_1d(bias, shard, out_dim, "bias")?;
        if bias.dtype() != dtype {
            Ok(Some(bias.to_dtype(dtype)?))
        } else {
            Ok(Some(bias))
        }
    }

    fn try_load_sharded_fp8_weight_scale(
        vb: &VarBuilder,
        out_dim: usize,
        in_dim: usize,
        shard: candle_nn::var_builder::Shard,
        block_size: &[usize],
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !vb.contains_tensor("weight_scale") && !vb.contains_tensor("weight_scale_inv") {
            return Ok(None);
        }

        let by = block_size[0];
        let bx = block_size[1];
        let scale_dim0 = out_dim.div_ceil(by);
        let scale_dim1 = in_dim.div_ceil(bx);

        let weight = match vb.get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::U8) {
            Ok(weight) => weight,
            Err(_) => return Ok(None),
        };
        let weight = Self::normalize_sharded_2d(weight, shard, out_dim, in_dim, "weight")?;

        let weight_scale = match vb.get_with_hints_dtype(
            (scale_dim0, scale_dim1),
            "weight_scale",
            shard,
            DType::F32,
        ) {
            Ok(scale) => scale,
            Err(_) => match vb.get_with_hints_dtype(
                (scale_dim0, scale_dim1),
                "weight_scale_inv",
                shard,
                DType::F32,
            ) {
                Ok(scale) => scale,
                Err(_) => return Ok(None),
            },
        };
        let weight_scale = Self::normalize_sharded_2d(
            weight_scale,
            shard,
            scale_dim0,
            scale_dim1,
            "weight_scale",
        )?;

        Ok(Some((weight, weight_scale)))
    }

    fn try_load_packed_qkv(
        vb: &VarBuilder,
        hidden_sz: usize,
        q_out_dim: usize,
        kv_out_dim: usize,
        attention_bias: bool,
        comm: Rc<Comm>,
        dtype: DType,
        quant_cfg: &Option<crate::openai::models::QuantConfig>,
        quant: &Option<String>,
    ) -> Result<Option<QkvProjection>> {
        if quant.is_some() {
            return Ok(None);
        }

        let q_shard = shard(0, comm.rank(), comm.world_size());
        let kv_shard = shard(0, comm.rank(), comm.world_size());
        let q_vb = vb.pp("q_proj");
        let k_vb = vb.pp("k_proj");
        let v_vb = vb.pp("v_proj");

        let is_fp8_quant = quant_cfg
            .as_ref()
            .map(|cfg| cfg.quant_method == "fp8")
            .unwrap_or(false);
        if let Some(cfg) = quant_cfg {
            if cfg.quant_method != "fp8" {
                return Ok(None);
            }
        }

        if is_fp8_quant {
            let Some(block_size) = quant_cfg
                .as_ref()
                .and_then(|cfg| cfg.weight_block_size.clone())
            else {
                candle_core::bail!("LnFp8: weight_block_size must be configured for packed qkv");
            };
            if block_size.len() != 2 {
                candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
            }

            let Some((q_weight, q_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &q_vb,
                q_out_dim,
                hidden_sz,
                q_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };
            let Some((k_weight, k_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &k_vb,
                kv_out_dim,
                hidden_sz,
                kv_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };
            let Some((v_weight, v_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &v_vb,
                kv_out_dim,
                hidden_sz,
                kv_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };

            let local_q = q_weight.dim(0)?;
            let local_k = k_weight.dim(0)?;
            let local_v = v_weight.dim(0)?;
            let by = block_size[0];
            let q_global_start = q_shard.rank * local_q;
            let k_global_start = q_out_dim + kv_shard.rank * local_k;
            let v_global_start = q_out_dim + kv_out_dim + kv_shard.rank * local_v;
            if q_global_start % by != 0 || k_global_start % by != 0 || v_global_start % by != 0 {
                return Ok(None);
            }

            let packed_weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;
            let packed_scale = Tensor::cat(&[&q_scale, &k_scale, &v_scale], 0)?;
            let packed_bias = if attention_bias {
                let q_bias = Self::load_sharded_bias(&q_vb, q_out_dim, q_shard, dtype)?;
                let k_bias = Self::load_sharded_bias(&k_vb, kv_out_dim, kv_shard, dtype)?;
                let v_bias = Self::load_sharded_bias(&v_vb, kv_out_dim, kv_shard, dtype)?;
                match (q_bias, k_bias, v_bias) {
                    (Some(qb), Some(kb), Some(vb)) => Some(Tensor::cat(&[&qb, &kb, &vb], 0)?),
                    (None, None, None) => None,
                    _ => return Ok(None),
                }
            } else {
                None
            };

            #[cfg(feature = "cuda")]
            let sm_version = attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?)
                .unwrap_or(0) as usize;

            #[cfg(not(feature = "cuda"))]
            let sm_version = 0;

            let merged = MergedParallelColumnLinear::from_packed_local_fp8(
                packed_weight,
                packed_scale,
                packed_bias,
                block_size,
                sm_version,
                vec![local_q, local_k, local_v],
            );
            return Ok(Some(QkvProjection::Packed(merged)));
        }

        if quant_cfg.is_some() {
            return Ok(None);
        }

        let q_weight =
            q_vb.get_with_hints_dtype((q_out_dim, hidden_sz), "weight", q_shard, dtype)?;
        let k_weight =
            k_vb.get_with_hints_dtype((kv_out_dim, hidden_sz), "weight", kv_shard, dtype)?;
        let v_weight =
            v_vb.get_with_hints_dtype((kv_out_dim, hidden_sz), "weight", kv_shard, dtype)?;
        let q_weight =
            Self::normalize_sharded_2d(q_weight, q_shard, q_out_dim, hidden_sz, "q weight")?;
        let k_weight =
            Self::normalize_sharded_2d(k_weight, kv_shard, kv_out_dim, hidden_sz, "k weight")?;
        let v_weight =
            Self::normalize_sharded_2d(v_weight, kv_shard, kv_out_dim, hidden_sz, "v weight")?;

        let local_q = q_weight.dim(0)?;
        let local_k = k_weight.dim(0)?;
        let local_v = v_weight.dim(0)?;
        let packed_weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;

        let packed_bias = if attention_bias {
            let q_bias = Self::load_sharded_bias(&q_vb, q_out_dim, q_shard, dtype)?;
            let k_bias = Self::load_sharded_bias(&k_vb, kv_out_dim, kv_shard, dtype)?;
            let v_bias = Self::load_sharded_bias(&v_vb, kv_out_dim, kv_shard, dtype)?;
            match (q_bias, k_bias, v_bias) {
                (Some(qb), Some(kb), Some(vb)) => Some(Tensor::cat(&[&qb, &kb, &vb], 0)?),
                (None, None, None) => None,
                _ => return Ok(None),
            }
        } else {
            None
        };

        let merged = MergedParallelColumnLinear::from_packed_local(
            packed_weight,
            packed_bias,
            vec![local_q, local_k, local_v],
        );
        Ok(Some(QkvProjection::Packed(merged)))
    }

    pub fn new(
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads.unwrap();
        let head_dim = cfg.head_dim.unwrap_or(hidden_sz / num_heads);
        let arch = cfg
            .architectures
            .as_ref()
            .and_then(|a| a.first())
            .cloned()
            .unwrap_or_default();
        let is_gemma = matches!(
            arch.as_str(),
            "Gemma3ForConditionalGeneration" | "Gemma3ForCausalLM"
        );
        let is_qwen35_or_next = matches!(
            arch.as_str(),
            "Qwen3_5ForCausalLM"
                | "Qwen3_5ForConditionalGeneration"
                | "Qwen3_5MoeForCausalLM"
                | "Qwen3_5MoeForConditionalGeneration"
                | "Qwen3NextForCausalLM"
                | "Qwen3NextForConditionalGeneration"
        );
        // Qwen3.5/Qwen3-Next and Gemma q/k norms use Gemma-style +1 weight semantics.
        let qk_norm_add_one = is_gemma || is_qwen35_or_next;
        let attention_bias = if is_qwen35_or_next {
            cfg.use_qkv_bias.or(cfg.attention_bias).unwrap_or(false)
        } else {
            cfg.attention_bias.unwrap_or(false)
        };
        let attn_output_gate = is_qwen35_or_next;
        let q_out_dim = num_heads * head_dim * if attn_output_gate { 2 } else { 1 };
        let qkv_proj = if let Some(packed) = Self::try_load_packed_qkv(
            &vb,
            hidden_sz,
            q_out_dim,
            num_kv_heads * head_dim,
            attention_bias,
            comm.clone(),
            vb.dtype(),
            &cfg.quantization_config,
            &cfg.isq_quant,
        )? {
            packed
        } else {
            let q_proj = TensorParallelColumnLinear::load_with_hints(
                hidden_sz,
                q_out_dim,
                attention_bias,
                vb.pp("q_proj"),
                comm.clone(),
                &cfg.isq_quant,
                &cfg.quantization_config,
            )?;
            let k_proj = TensorParallelColumnLinear::load_with_hints(
                hidden_sz,
                num_kv_heads * head_dim,
                attention_bias,
                vb.pp("k_proj"),
                comm.clone(),
                &cfg.isq_quant,
                &cfg.quantization_config,
            )?;
            let q8_0_quant = Some("q8_0".to_string());
            let v_proj_quant = if cfg.quantization_config.is_some() {
                &cfg.isq_quant
            } else if cfg.isq_quant.is_some()
                && !matches!(
                    cfg.isq_quant.as_ref().unwrap().as_str(),
                    "gptq" | "awq" | "marlin"
                )
            {
                &q8_0_quant
            } else {
                &cfg.isq_quant
            };
            let v_proj = TensorParallelColumnLinear::load_with_hints(
                hidden_sz,
                num_kv_heads * head_dim,
                attention_bias,
                vb.pp("v_proj"),
                comm.clone(),
                v_proj_quant,
                &cfg.quantization_config,
            )?;
            QkvProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            }
        };

        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
            hidden_sz,
            false,
            vb.pp("o_proj"),
            comm.clone(),
            &cfg.isq_quant,
            &cfg.quantization_config,
        )?;

        //we use higher precision for q/k norm
        let q_norm = rms_norm_x(
            head_dim,
            cfg.rms_norm_eps,
            vb.pp("q_norm"),
            DType::F32,
            qk_norm_add_one,
        )
        .ok();
        let k_norm = rms_norm_x(
            head_dim,
            cfg.rms_norm_eps,
            vb.pp("k_norm"),
            DType::F32,
            qk_norm_add_one,
        )
        .ok();

        assert!(cfg.num_attention_heads >= comm.world_size());
        assert!(cfg.num_attention_heads % comm.world_size() == 0);

        assert!(cfg.num_key_value_heads.unwrap() >= comm.world_size());
        assert!(cfg.num_key_value_heads.unwrap() % comm.world_size() == 0);

        let attention_heads = cfg.num_attention_heads / comm.world_size();
        let kv_heads = cfg.num_key_value_heads.unwrap() / comm.world_size();
        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: attention_heads,
            num_kv_heads: kv_heads,
            head_dim,
            rotary_emb,
            attn: PagedAttention::new(
                attention_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(kv_heads),
                sliding_window,
                vb.device().clone(),
                None,
                cfg.fp8_kvcache.unwrap_or(false),
            )?,
            softcapping: cfg.attn_logit_softcapping,
            dtype: vb.dtype(),
            attn_output_gate,
            no_per_head_norm: is_qwen35_or_next,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        let (query_states, key_states, value_states) = match &self.qkv_proj {
            QkvProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => (
                q_proj.forward(xs)?,
                k_proj.forward(xs)?,
                v_proj.forward(xs)?,
            ),
            QkvProjection::Packed(qkv_proj) => {
                let qkv = qkv_proj.forward(xs)?;
                if qkv.len() != 3 {
                    candle_core::bail!(
                        "Expected 3 outputs from packed qkv projection, got {}",
                        qkv.len()
                    );
                }
                (qkv[0].clone(), qkv[1].clone(), qkv[2].clone())
            }
        };

        let local_q_dim = self.num_heads * self.head_dim;
        let (query_states, q_gate) = if self.attn_output_gate {
            let q_dim = query_states.dim(1)?;
            if q_dim != local_q_dim * 2 {
                candle_core::bail!(
                    "q_proj output dim mismatch for gated attention, expected {}, got {}",
                    local_q_dim * 2,
                    q_dim
                );
            }
            let q_gate = query_states.reshape((seq_len, self.num_heads, self.head_dim * 2))?;
            let q = q_gate.narrow(2, 0, self.head_dim)?;
            let gate = q_gate.narrow(2, self.head_dim, self.head_dim)?;
            (
                q.reshape((seq_len, local_q_dim))?,
                Some(gate.reshape((seq_len, local_q_dim))?),
            )
        } else {
            (query_states, None)
        };

        let q = query_states.reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = key_states.reshape((seq_len, self.num_kv_heads, self.head_dim))?;
        let v = value_states.reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        // Q/K norm weights are loaded in F32 for Qwen3.5/Next; cast activations
        // to keep CUDA RMSNorm dtype-consistent.
        let (q, k) = if q.dtype() != DType::F32 {
            (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?)
        } else {
            (q, k)
        };

        let (q, k) = if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            // Per‑head RMSNorm in qwen3
            if self.no_per_head_norm {
                let q = q_norm.forward(&q)?;
                let k = k_norm.forward(&k)?;
                (q, k)
            } else {
                let q_flat = q.flatten(0, 1)?;
                let k_flat = k.flatten(0, 1)?;

                let q_flat = q_norm.forward(&q_flat)?;
                let k_flat = k_norm.forward(&k_flat)?;

                let q = q_flat.reshape((seq_len, self.num_heads, self.head_dim))?;
                let k = k_flat.reshape((seq_len, self.num_kv_heads, self.head_dim))?;

                (q, k)
            }
        } else {
            (q, k)
        };

        let (q, k) = self.rotary_emb.apply_rotary_emb(&q, &k, input_positions)?;
        let (q, k) = (q.to_dtype(self.dtype)?, k.to_dtype(self.dtype)?);
        let v = if v.dtype() != self.dtype {
            v.to_dtype(self.dtype)?
        } else {
            v
        };

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
                self.softcapping,
            )?
            .reshape((seq_len, ()))?;

        let y = if let Some(gate) = q_gate {
            let gate = if gate.dtype() != y.dtype() {
                gate.to_dtype(y.dtype())?
            } else {
                gate
            };
            y.broadcast_mul(&candle_nn::ops::sigmoid(&gate)?)?
        } else {
            y
        };

        let y = self.o_proj.forward(&y.to_dtype(xs.dtype())?)?;
        Ok(y)
    }
}

pub struct QuantizedAttention {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_bq: Option<Tensor>,
    attention_bk: Option<Tensor>,
    attention_bv: Option<Tensor>,
    attention_wo: QMatMul,
    q_norm: Option<QRmsNorm>,
    k_norm: Option<QRmsNorm>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    attn: PagedAttention,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
    dtype: DType,
}

impl QuantizedAttention {
    pub fn new<R: std::io::Seek + std::io::Read>(
        config: &Config,
        ct: &gguf_file::Content,
        reader: &mut R,
        prefix: &str,
        device: &Device,
        dtype: DType,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
        let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
        let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;

        let attention_bq = ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device);
        let attention_bk = ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device);
        let attention_bv = ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device);

        let attention_bq = if attention_bq.is_ok() {
            Some(
                attention_bq
                    .unwrap()
                    .dequantize(device)?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };

        let attention_bk = if attention_bk.is_ok() {
            Some(
                attention_bk
                    .unwrap()
                    .dequantize(device)?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };

        let attention_bv = if attention_bv.is_ok() {
            Some(
                attention_bv
                    .unwrap()
                    .dequantize(device)?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };

        let attention_wo = ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

        let (q_norm, k_norm) = {
            let q_norm = ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device);
            let k_norm = ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device);
            let (q_norm, k_norm) = match (q_norm, k_norm) {
                (Ok(q_norm), Ok(k_norm)) => {
                    let q_norm = QRmsNorm::from_qtensor(q_norm, config.rms_norm_eps)?;
                    let k_norm = QRmsNorm::from_qtensor(k_norm, config.rms_norm_eps)?;
                    (Some(q_norm), Some(k_norm))
                }
                _ => (None, None),
            };
            (q_norm, k_norm)
        };

        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        Ok(QuantizedAttention {
            attention_wq: QMatMul::from_qtensor(attention_wq)?,
            attention_wk: QMatMul::from_qtensor(attention_wk)?,
            attention_wv: QMatMul::from_qtensor(attention_wv)?,
            attention_bq,
            attention_bk,
            attention_bv,
            attention_wo: QMatMul::from_qtensor(attention_wo)?,
            q_norm,
            k_norm,
            n_head: config.num_attention_heads,
            n_kv_head: config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
            head_dim,
            attn: PagedAttention::new(
                config.num_attention_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                config.num_key_value_heads,
                sliding_window,
                device.clone(),
                None,
                config.fp8_kvcache.unwrap_or(false),
            )?,
            rotary_emb: rotary_emb.clone(),
            dtype,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = x.dims2()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = if self.attention_bq.is_some() {
            q.broadcast_add(self.attention_bq.as_ref().unwrap())?
        } else {
            q
        };

        let k = if self.attention_bk.is_some() {
            k.broadcast_add(self.attention_bk.as_ref().unwrap())?
        } else {
            k
        };

        let v = if self.attention_bv.is_some() {
            v.broadcast_add(self.attention_bv.as_ref().unwrap())?
        } else {
            v
        };

        let q = q.reshape((seq_len, self.n_head, self.head_dim))?;
        let k = k.reshape((seq_len, self.n_kv_head, self.head_dim))?;
        let v = v.reshape((seq_len, self.n_kv_head, self.head_dim))?;

        let (q, k) = if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            // Per‑head RMSNorm in qwen3
            let q_flat = q.flatten(0, 1)?;
            let k_flat = k.flatten(0, 1)?;

            // q_norm and k_norm weights stored in f32 format in qwen3 gguf
            let q_flat = q_norm.forward(&q_flat)?;
            let k_flat = k_norm.forward(&k_flat)?;

            let q = q_flat.reshape((seq_len, self.n_head, self.head_dim))?;
            let k = k_flat.reshape((seq_len, self.n_kv_head, self.head_dim))?;

            (q, k)
        } else {
            (q, k)
        };

        let (q, k) = self.rotary_emb.apply_rotary_emb(&q, &k, input_positions)?;
        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        let y = self
            .attn
            .forward(
                &q,
                &k,
                &v,
                mask,
                cache.map(|(k_, _)| k_.clone()),
                cache.map(|(_, v_)| v_.clone()),
                input_metadata,
                None,
            )?
            .reshape((seq_len, ()))?;

        let y = self.attention_wo.forward(&y.to_dtype(x.dtype())?)?;
        Ok(y)
    }
}
