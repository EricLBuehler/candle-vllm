#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use super::layers::indexer::DsaIndexer;
use super::layers::mla_attention::MlaConfig;
use super::layers::qrmsnorm::QRmsNorm;
use super::layers::quantized_var_builder::VarBuilder as QVarBuilder;
use super::rotary_emb::ScalingRotaryEmbedding;
use super::{Config, KvCacheDtype, MoEConfig, QwenMoEConfig, ScalingValue};
use crate::backend::custom_ops::moe::TopKLastDimOp;
use crate::backend::progress::{ProgressLike, ProgressReporter};
#[cfg(feature = "nccl")]
use crate::openai::distributed::AllReduce;
use crate::openai::distributed::{Comm, Rc, VocabParallelLinear};
use crate::openai::models::layers::moe::sort_expert_assignments;
use crate::openai::models::mask::get_attention_causal_mask;
use crate::InputMetadata;
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use either::Either;
use parking_lot::RwLock;
use std::iter::zip;
use std::sync::Arc;

// ── GGUF MLA Attention ──────────────────────────────────────────────────────

struct QuantizedMlaAttention {
    q_a_proj: Option<QMatMul>,
    q_a_layernorm: Option<QRmsNorm>,
    q_b_proj: Option<QMatMul>,
    q_proj: Option<QMatMul>,
    kv_a_proj_with_mqa: QMatMul,
    kv_a_layernorm: QRmsNorm,
    o_proj: QMatMul,
    w_uk: Tensor,
    w_uv_t: Tensor,
    num_heads: usize,
    q_head_dim: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    kv_lora_rank: usize,
    v_head_dim: usize,
    sm_scale: f32,
    rope_scale: f32,
    rope_theta: f32,
    dtype: DType,
    indexer: Option<DsaIndexer>,
}

impl QuantizedMlaAttention {
    fn new(
        vb: &QVarBuilder,
        prefix: &str,
        mla_cfg: &MlaConfig,
        config: &Config,
        dtype: DType,
        layer_idx: usize,
        device: &Device,
    ) -> Result<Self> {
        let _hidden_size = mla_cfg.hidden_size;
        let num_heads = mla_cfg.num_attention_heads;
        let kv_lora_rank = mla_cfg.kv_lora_rank;
        let qk_nope_head_dim = mla_cfg.qk_nope_head_dim;
        let qk_rope_head_dim = mla_cfg.qk_rope_head_dim;
        let v_head_dim = mla_cfg.v_head_dim;
        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let prefix_vb = vb.pp(prefix);

        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) =
            if let Some(_q_lora_rank) = mla_cfg.q_lora_rank {
                let q_a = QMatMul::from_arc(prefix_vb.get_no_shape("attn_q_a.weight")?)?;
                let q_a_ln = QRmsNorm::from_arc_qtensor(
                    prefix_vb.get_no_shape("attn_q_a_norm.weight")?,
                    mla_cfg.rms_norm_eps,
                )?;
                let q_b = QMatMul::from_arc(prefix_vb.get_no_shape("attn_q_b.weight")?)?;
                (Some(q_a), Some(q_a_ln), Some(q_b), None)
            } else {
                let q = QMatMul::from_arc(prefix_vb.get_no_shape("attn_q.weight")?)?;
                (None, None, None, Some(q))
            };

        let kv_a_proj_with_mqa =
            QMatMul::from_arc(prefix_vb.get_no_shape("attn_kv_a_mqa.weight")?)?;
        let kv_a_layernorm = QRmsNorm::from_arc_qtensor(
            prefix_vb.get_no_shape("attn_kv_a_norm.weight")?,
            mla_cfg.rms_norm_eps,
        )?;
        let o_proj = QMatMul::from_arc(prefix_vb.get_no_shape("attn_output.weight")?)?;

        // Load kv_b_proj weight and pre-compute absorbed MLA matrices (w_uk, w_uv_t).
        // GGUF converters may split kv_b into separate attn_k_b / attn_v_b tensors,
        // or keep a single combined tensor. We handle both layouts.
        let has_separate_kv_b = prefix_vb.get_no_shape("attn_k_b.weight").is_ok();

        // attn_k_b layout varies between GGUF converters:
        //   Variant A: [qk_nope_head_dim, kv_lora_rank, num_heads] (common)
        //   Variant B: [num_heads, kv_lora_rank, qk_nope_head_dim]
        // Target: w_uk = [num_heads, qk_nope_head_dim, kv_lora_rank]
        let (w_uk, w_uv_t) = if has_separate_kv_b {
            let k_b_qt = prefix_vb.get_no_shape("attn_k_b.weight")?;
            let k_b_weight = k_b_qt.dequantize(device)?;

            let k_b_dims = k_b_weight.dims();
            let k_b_weight = if k_b_dims.len() == 3 {
                let heads_first = k_b_dims[0] == num_heads && k_b_dims[2] != num_heads;
                if heads_first {
                    // [num_heads, kv_lora_rank, qk_nope_head_dim] -> transpose last two
                    k_b_weight.transpose(1, 2)?.contiguous()?.to_dtype(dtype)?
                } else {
                    // [qk_nope_head_dim, kv_lora_rank, num_heads] -> permute to [num_heads, qk_nope_head_dim, kv_lora_rank]
                    k_b_weight
                        .permute((2, 0, 1))?
                        .contiguous()?
                        .to_dtype(dtype)?
                }
            } else {
                k_b_weight
                    .reshape((qk_nope_head_dim, kv_lora_rank, num_heads))?
                    .permute((2, 0, 1))?
                    .contiguous()?
                    .to_dtype(dtype)?
            };
            let w_uk = k_b_weight;

            // attn_v_b: target w_uv_t = [num_heads, kv_lora_rank, v_head_dim]
            // After candle dequantize (reverses GGUF dims):
            //   [num_heads, v_head_dim, kv_lora_rank] -> transpose(1,2)
            //   [v_head_dim, kv_lora_rank, num_heads] -> permute(2,1,0)
            let v_b_qt = prefix_vb.get_no_shape("attn_v_b.weight")?;
            let v_b_weight = v_b_qt.dequantize(device)?;
            let v_b_dims = v_b_weight.dims();
            let w_uv_t = if v_b_dims.len() == 3 {
                let heads_first = v_b_dims[0] == num_heads;
                if heads_first {
                    // [num_heads, v_head_dim, kv_lora_rank] -> [num_heads, kv_lora_rank, v_head_dim]
                    v_b_weight.transpose(1, 2)?.contiguous()?.to_dtype(dtype)?
                } else {
                    // [v_head_dim, kv_lora_rank, num_heads] -> [num_heads, kv_lora_rank, v_head_dim]
                    v_b_weight
                        .permute((2, 1, 0))?
                        .contiguous()?
                        .to_dtype(dtype)?
                }
            } else {
                v_b_weight
                    .reshape((v_head_dim, kv_lora_rank, num_heads))?
                    .permute((2, 1, 0))?
                    .contiguous()?
                    .to_dtype(dtype)?
            };

            (w_uk, w_uv_t)
        } else {
            let kv_b_qt = prefix_vb
                .get_no_shape("attn_k_b.weight")
                .or_else(|_| prefix_vb.get_no_shape("attn_kv_b.weight"))?;
            let kv_b_weight = kv_b_qt.dequantize(device)?.to_dtype(dtype)?.reshape((
                num_heads,
                qk_nope_head_dim + v_head_dim,
                kv_lora_rank,
            ))?;
            let w_uk = kv_b_weight.narrow(1, 0, qk_nope_head_dim)?.contiguous()?;
            let w_uv = kv_b_weight
                .narrow(1, qk_nope_head_dim, v_head_dim)?
                .contiguous()?;
            let w_uv_t = w_uv.transpose(1, 2)?.contiguous()?;
            (w_uk, w_uv_t)
        };

        let mut sm_scale = 1.0 / (q_head_dim as f32).sqrt();
        let mut rope_scale = 1.0f32;

        if let Some(ref rope_scaling) = config.rope_scaling {
            let is_yarn = rope_scaling.get("type").and_then(|v| {
                if let ScalingValue::String(s) = v {
                    Some(s.as_str())
                } else {
                    None
                }
            }) == Some("yarn");
            if is_yarn {
                let factor = rope_scaling
                    .get("factor")
                    .and_then(|v| match v {
                        ScalingValue::Single(f) => Some(*f),
                        _ => None,
                    })
                    .unwrap_or(1.0) as f32;
                let mscale_all_dim = rope_scaling
                    .get("mscale_all_dim")
                    .and_then(|v| match v {
                        ScalingValue::Single(f) => Some(*f),
                        _ => None,
                    })
                    .unwrap_or(0.0) as f32;
                if mscale_all_dim > 0.0 && factor > 1.0 {
                    let mscale = 0.1 * mscale_all_dim * factor.ln() + 1.0;
                    sm_scale *= mscale * mscale;
                }
                rope_scale = 1.0;
            }
        }

        // DSA indexer - GGUF path (uses safetensors VarBuilder which is not available
        // for GGUF; skip DSA indexer for now)
        let indexer: Option<DsaIndexer> = None;
        let _ = (layer_idx, &mla_cfg.index_head_dim);

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            o_proj,
            w_uk,
            w_uv_t,
            num_heads,
            q_head_dim,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            sm_scale,
            rope_scale,
            rope_theta: config.rope_theta as f32,
            dtype,
            indexer,
        })
    }

    #[cfg(feature = "cuda")]
    fn project_mla_output(
        &self,
        attn_out: &Tensor,
        seq_len: usize,
        xs_dtype: DType,
    ) -> Result<Tensor> {
        let attn_t = attn_out.transpose(0, 1)?.contiguous()?;
        let y = attn_t.matmul(&self.w_uv_t)?;
        let y = y.transpose(0, 1)?.contiguous()?;
        let y = y.reshape((seq_len, self.num_heads * self.v_head_dim))?;
        let y = y.to_dtype(xs_dtype)?;
        self.o_proj.forward(&y)
    }

    #[allow(clippy::too_many_arguments, unused_variables)]
    fn forward(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<ScalingRotaryEmbedding>>,
        _attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;
        let xs = &xs.contiguous()?;

        let (q, q_resid) = if let (Some(q_a), Some(q_a_ln), Some(q_b)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_a_out = q_a.forward(xs)?;
            let q_a_normed = q_a_ln.forward(&q_a_out)?;
            let q_a_normed_c = q_a_normed.contiguous()?;
            let q = q_b.forward(&q_a_normed_c)?;
            (q, Some(q_a_normed_c))
        } else {
            (self.q_proj.as_ref().unwrap().forward(xs)?, None)
        };

        let q = q.reshape((seq_len, self.num_heads, self.q_head_dim))?;
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        let kv_a = self.kv_a_proj_with_mqa.forward(xs)?;
        let ckv = kv_a.narrow(D::Minus1, 0, self.kv_lora_rank)?.contiguous()?;
        let k_pe_raw = kv_a
            .narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?
            .contiguous()?;

        let ckv = self.kv_a_layernorm.forward(&ckv)?;

        let k_pe = k_pe_raw.reshape((seq_len, 1, self.qk_rope_head_dim))?;
        let q_pe_for_rope = q_pe.contiguous()?;

        let (q_pe_for_rope, k_pe) = (
            q_pe_for_rope.to_dtype(DType::F32)?,
            k_pe.to_dtype(DType::F32)?,
        );
        let (q_pe, k_pe) = if let Some(rotary_emb) = &rotary_emb {
            let (q_new, k_new) = rotary_emb.apply_rotary_emb(&q_pe_for_rope, &k_pe, positions)?;
            (q_new, k_new)
        } else {
            (q_pe_for_rope, k_pe)
        };
        let k_pe = k_pe.squeeze(1)?;

        let q_pe = q_pe.to_dtype(self.dtype)?;
        let q_nope = q_nope.contiguous()?.to_dtype(self.dtype)?;
        let ckv = ckv.to_dtype(self.dtype)?;
        let k_pe = k_pe.to_dtype(self.dtype)?;

        #[cfg(feature = "flashinfer")]
        if let Some(fm) = input_metadata.flashinfer_metadata.as_ref() {
            if let Some((ckv_cache, kpe_cache)) = cache {
                attention_rs::mla::concat_and_cache_mla(
                    &ckv,
                    &k_pe,
                    ckv_cache,
                    kpe_cache,
                    &input_metadata.slot_mapping,
                )?;

                let q_nope_t = q_nope.transpose(0, 1)?.contiguous()?;
                let q_nope_absorbed = q_nope_t
                    .matmul(&self.w_uk)?
                    .transpose(0, 1)?
                    .contiguous()?
                    .to_dtype(self.dtype)?;
                let q_pe = q_pe.to_dtype(self.dtype)?;

                let page_size = ckv_cache.dim(1)?;

                if input_metadata.is_prefill {
                    if let (Some(indexer), Some(q_res)) = (&self.indexer, &q_resid) {
                        if let Some(block_tables) = &input_metadata.block_tables {
                            if let Some(context_lens) = &input_metadata.context_lens {
                                if let Some(topk_idxs) =
                                    indexer.forward(xs, q_res, rotary_emb, positions)?
                                {
                                    let ckv_cache_3d = ckv_cache.squeeze(2)?;
                                    let kpe_cache_3d = kpe_cache.squeeze(2)?;
                                    let cu_seqlens_q =
                                        input_metadata.cu_seqlens_q.as_ref().ok_or_else(|| {
                                            candle_core::Error::msg(
                                                "MLA sparse prefill requires cu_seqlens_q",
                                            )
                                        })?;
                                    let attn_out = attention_rs::mla::mla_sparse_paged_prefill(
                                        &q_nope_absorbed,
                                        &q_pe,
                                        &ckv_cache_3d,
                                        &kpe_cache_3d,
                                        block_tables,
                                        context_lens,
                                        cu_seqlens_q,
                                        &topk_idxs,
                                        self.sm_scale,
                                    )?;
                                    return self.project_mla_output(&attn_out, seq_len, xs.dtype());
                                }
                            }
                        }
                    }
                }

                let attn_out = if input_metadata.is_prefill {
                    let plan_info = fm.mla_prefill_plan_info.as_ref().ok_or_else(|| {
                        candle_core::Error::msg("MLA prefill requires mla_prefill_plan_info")
                    })?;
                    attention_rs::mla::mla_prefill_run(
                        &q_nope_absorbed,
                        &q_pe,
                        ckv_cache,
                        kpe_cache,
                        &fm.indices,
                        self.num_heads,
                        page_size,
                        self.sm_scale,
                        plan_info,
                        true,
                    )?
                } else {
                    let plan_info = fm.mla_decode_plan_info.as_ref().ok_or_else(|| {
                        candle_core::Error::msg("MLA decode requires mla_decode_plan_info")
                    })?;
                    attention_rs::mla::mla_decode_run(
                        &q_nope_absorbed,
                        &q_pe,
                        ckv_cache,
                        kpe_cache,
                        &fm.indptr,
                        &fm.indices,
                        &fm.last_len,
                        seq_len,
                        self.num_heads,
                        page_size,
                        self.sm_scale,
                        self.rope_scale,
                        self.rope_theta,
                        plan_info,
                        fm.use_cuda_graph,
                    )?
                };

                let attn_out_t = attn_out.transpose(0, 1)?.contiguous()?;
                let y = attn_out_t
                    .matmul(&self.w_uv_t)?
                    .transpose(0, 1)?
                    .contiguous()?;
                let y = y.reshape((seq_len, self.num_heads * self.v_head_dim))?;
                let y = y.to_dtype(xs.dtype())?;
                return self.o_proj.forward(&y);
            }
        }

        #[cfg(feature = "cuda")]
        if let Some((ckv_cache, kpe_cache)) = cache {
            attention_rs::mla::concat_and_cache_mla(
                &ckv,
                &k_pe,
                ckv_cache,
                kpe_cache,
                &input_metadata.slot_mapping,
            )?;

            let q_nope_t = q_nope.transpose(0, 1)?.contiguous()?;
            let q_absorbed = q_nope_t.matmul(&self.w_uk)?.transpose(0, 1)?.contiguous()?;

            let ckv_cache_3d = ckv_cache.squeeze(2)?;
            let kpe_cache_3d = kpe_cache.squeeze(2)?;

            if let (Some(block_tables), Some(context_lens)) =
                (&input_metadata.block_tables, &input_metadata.context_lens)
            {
                if input_metadata.is_prefill {
                    let cu_seqlens_q = input_metadata.cu_seqlens_q.as_ref().ok_or_else(|| {
                        candle_core::Error::msg("MLA fused prefill requires cu_seqlens_q")
                    })?;

                    if let (Some(indexer), Some(q_res)) = (&self.indexer, &q_resid) {
                        if let Some(topk_idxs) =
                            indexer.forward(xs, q_res, rotary_emb, positions)?
                        {
                            let attn_out = attention_rs::mla::mla_sparse_paged_prefill(
                                &q_absorbed,
                                &q_pe,
                                &ckv_cache_3d,
                                &kpe_cache_3d,
                                block_tables,
                                context_lens,
                                cu_seqlens_q,
                                &topk_idxs,
                                self.sm_scale,
                            )?;
                            return self.project_mla_output(&attn_out, seq_len, xs.dtype());
                        }
                    }

                    let attn_out = attention_rs::mla::mla_paged_prefill(
                        &q_absorbed,
                        &q_pe,
                        &ckv_cache_3d,
                        &kpe_cache_3d,
                        block_tables,
                        context_lens,
                        cu_seqlens_q,
                        self.sm_scale,
                    )?;
                    return self.project_mla_output(&attn_out, seq_len, xs.dtype());
                }

                let attn_out = attention_rs::mla::mla_paged_decode(
                    &q_absorbed,
                    &q_pe,
                    &ckv_cache_3d,
                    &kpe_cache_3d,
                    block_tables,
                    context_lens,
                    self.sm_scale,
                )?;
                return self.project_mla_output(&attn_out, seq_len, xs.dtype());
            }
        }
        candle_core::bail!("MLA attention requires CUDA platform!")
    }
}

// ── GGUF MLP ────────────────────────────────────────────────────────────────

struct Mlp {
    ffn_gate: QMatMul,
    ffn_up: QMatMul,
    ffn_down: QMatMul,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
    #[cfg(feature = "nccl")]
    dtype: DType,
}

impl Mlp {
    #[allow(unused_mut)]
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = &xs.contiguous()?;
        let gate = candle_nn::ops::silu(&self.ffn_gate.forward(xs)?)?;
        let up = self.ffn_up.forward(xs)?;
        let mut y = self.ffn_down.forward(&(gate * up)?)?;
        #[cfg(feature = "nccl")]
        if let Some(all_reduce) = &self.all_reduce {
            y = all_reduce.apply(&y.to_dtype(self.dtype)?)?;
            y = y.to_dtype(DType::F32)?;
        }
        Ok(y)
    }
}

// ── GGUF Fused MoE ──────────────────────────────────────────────────────────

struct FusedMoe {
    gate: QMatMul,
    gate_experts: Arc<QTensor>,
    up_experts: Arc<QTensor>,
    down_experts: Arc<QTensor>,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    e_score_correction_bias: Option<Tensor>,
    #[cfg(feature = "nccl")]
    #[allow(dead_code)]
    all_reduce: Option<AllReduce>,
    dtype: DType,
    #[allow(dead_code)]
    world_size: usize,
}

impl FusedMoe {
    #[allow(unused_mut, unused_variables)]
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

        let xs = xs.contiguous()?;
        let router_logits = self.gate.forward(&xs)?;
        let scores = candle_nn::ops::sigmoid(&router_logits.to_dtype(DType::F32)?)?;

        let scores_for_choice = if let Some(ref bias) = self.e_score_correction_bias {
            scores.broadcast_add(&bias.to_dtype(DType::F32)?)?
        } else {
            scores.clone()
        };
        let topk_result = scores_for_choice.topk(self.num_experts_per_tok)?;
        let mut topk_weights = scores.gather(&topk_result.indices, D::Minus1)?;
        let topk_ids = topk_result.indices.to_dtype(DType::U32)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }
        if let Some(rsf) = self.routed_scaling_factor {
            topk_weights = (topk_weights * rsf)?;
        }

        let (expert_ids, sorted_token_ids) = sort_expert_assignments(&topk_ids, is_prefill)?;
        let mut ys = self.run_moe(
            &xs,
            topk_weights,
            &sorted_token_ids,
            &expert_ids,
            is_prefill,
        )?;
        ys = ys.reshape((num_tokens, (), hidden_dim))?.sum(D::Minus2)?;
        if ys.dtype() != self.dtype {
            ys = ys.to_dtype(self.dtype)?;
        }
        #[cfg(feature = "nccl")]
        if self.world_size > 1 {
            if let Some(all_reduce) = &self.all_reduce {
                ys = all_reduce.apply(&ys)?;
            }
        }
        ys.to_dtype(original_dtype)
    }

    fn run_moe(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        sorted_token_ids: &Tensor,
        expert_ids: &Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let xs = &xs.contiguous()?;
        let gate = attention_rs::moe::moe_gemm_gguf(
            xs,
            &self.gate_experts,
            &None,
            sorted_token_ids,
            expert_ids,
            self.num_experts_per_tok,
            is_prefill,
            self.dtype,
        )?;
        let up = attention_rs::moe::moe_gemm_gguf(
            xs,
            &self.up_experts,
            &None,
            sorted_token_ids,
            expert_ids,
            self.num_experts_per_tok,
            is_prefill,
            self.dtype,
        )?;
        let down_inputs = (up * gate.apply(&self.act)?)?;
        attention_rs::moe::moe_gemm_gguf(
            &down_inputs,
            &self.down_experts,
            &Some(topk_weights),
            sorted_token_ids,
            expert_ids,
            self.num_experts_per_tok,
            is_prefill,
            self.dtype,
        )
    }
}

fn try_load_e_score_correction_bias(
    vb: &QVarBuilder,
    prefix: &str,
    device: &Device,
) -> Option<Tensor> {
    let prefix_vb = vb.pp(prefix);
    prefix_vb
        .get_no_shape("ffn_gate_inp.e_score_correction_bias")
        .ok()
        .and_then(|qt| qt.dequantize(device).ok())
        .or_else(|| {
            prefix_vb
                .get_no_shape("e_score_correction_bias")
                .ok()
                .and_then(|qt| qt.dequantize(device).ok())
        })
        .or_else(|| {
            prefix_vb
                .get_no_shape("exp_probs_b.bias")
                .ok()
                .and_then(|qt| qt.dequantize(device).ok())
        })
        .map(|t| {
            let t = t.to_dtype(DType::F32).unwrap_or_else(|_| t.clone());
            t.flatten_all().unwrap_or(t)
        })
}

// ── Layer Dispatch ──────────────────────────────────────────────────────────

enum MoeOrMlp {
    FusedMoe(FusedMoe),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
        }
    }
}

struct LayerWeights {
    self_attn: QuantizedMlaAttention,
    attention_norm: QRmsNorm,
    mlp: MoeOrMlp,
    shared_expert: Option<Mlp>,
    ffn_norm: QRmsNorm,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
}

// ── Main Model ──────────────────────────────────────────────────────────────

pub struct GGUFDeepSeek {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: VocabParallelLinear,
    cfg: Config,
    dtype: DType,
    device: Device,
}

impl GGUFDeepSeek {
    fn build_config(
        embedding_length: usize,
        head_dim: usize,
        i_size: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        rms_eps: f64,
        rope_theta: f64,
        max_seq_len: usize,
        moe_cfg: &QwenMoEConfig,
        extra_config_json: Option<String>,
    ) -> Config {
        Config {
            architectures: Some(vec!["GlmMoeDsaForCausalLM".to_string()]),
            hidden_size: embedding_length,
            head_dim: Some(head_dim),
            intermediate_size: i_size,
            vocab_size: 0,
            num_hidden_layers: block_count,
            num_attention_heads: head_count,
            num_key_value_heads: Some(head_count_kv),
            rms_norm_eps: rms_eps,
            rope_theta,
            rope_local_base_freq: None,
            bos_token_id: None,
            eos_token_id: Some(super::TokenID(Either::Left(None))),
            max_seq_len,
            sliding_window: None,
            sliding_window_pattern: None,
            hidden_act: Some(candle_nn::Activation::Silu),
            hidden_activation: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            max_position_embeddings: Some(max_seq_len),
            original_max_position_embeddings: None,
            attention_bias: Some(false),
            partial_rotary_factor: None,
            qk_layernorm: false,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: None,
            moe_config: Some(MoEConfig::QwenMoE(moe_cfg.clone())),
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
        _kv_cache_dtype: DType,
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
        let md_opt_usize = |s: &str| -> Option<usize> {
            metadata
                .get(s)
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
        };
        let md_opt_f64 = |s: &str| -> Option<f64> {
            metadata
                .get(s)
                .and_then(|v| v.to_f32().ok())
                .map(|v| v as f64)
        };
        let reporter = progress_reporter.clone();
        let arch = md_get("general.architecture")?.to_string()?;

        let head_count =
            md_get(format!("{arch}.attention.head_count").as_str())?.to_u32()? as usize;
        let head_count_kv =
            md_get(format!("{arch}.attention.head_count_kv").as_str())?.to_u32()? as usize;
        let embedding_length =
            md_get(format!("{arch}.embedding_length").as_str())?.to_u32()? as usize;
        let head_dim = md_get(format!("{arch}.attention.key_length").as_str())
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);
        let context_length = md_get(format!("{arch}.context_length").as_str())?.to_u32()? as usize;
        let block_count = md_get(format!("{arch}.block_count").as_str())?.to_u32()? as usize;
        let rms_norm_eps =
            md_get(format!("{arch}.attention.layer_norm_rms_epsilon").as_str())?.to_f32()? as f64;
        let rope_freq_base = md_get(format!("{arch}.rope.freq_base").as_str())
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let feed_forward_length = md_get(format!("{arch}.feed_forward_length").as_str())
            .and_then(|v| v.to_u32())
            .map(|v| v as usize)
            .unwrap_or(0);

        let expert_feed_forward_length =
            md_get(format!("{arch}.expert_feed_forward_length").as_str())?.to_u32()? as usize;
        let expert_count = md_get(format!("{arch}.expert_count").as_str())?.to_u32()? as usize;
        let expert_used_count =
            md_get(format!("{arch}.expert_used_count").as_str())?.to_u32()? as usize;
        let first_k_dense = md_opt_usize(format!("{arch}.leading_dense_block_count").as_str());
        let expert_shared_feed_forward_length =
            md_opt_usize(format!("{arch}.expert_shared_feed_forward_length").as_str())
                .or(Some(expert_feed_forward_length));
        let expert_weights_scale = md_opt_f64(format!("{arch}.expert_weights_scale").as_str());

        let moe_cfg = QwenMoEConfig {
            moe_intermediate_size: expert_feed_forward_length,
            shared_expert_intermediate_size: expert_shared_feed_forward_length,
            num_experts: Some(expert_count),
            mlp_only_layers: None,
            decoder_sparse_step: Some(1),
            norm_topk_prob: true,
            num_experts_per_tok: expert_used_count,
            routed_scaling_factor: expert_weights_scale,
            first_k_dense_replace: first_k_dense,
            n_shared_experts: Some(1),
            n_group: None,
            topk_group: None,
            scoring_func: None,
            topk_method: None,
        };

        // Build MLA/DSA extra config JSON from GGUF metadata
        let q_lora_rank = md_opt_usize(format!("{arch}.attention.q_lora_rank").as_str());
        let kv_lora_rank = md_opt_usize(format!("{arch}.attention.kv_lora_rank").as_str());
        let key_length_mla = md_opt_usize(format!("{arch}.attention.key_length_mla").as_str());
        let v_head_dim = md_opt_usize(format!("{arch}.attention.value_length_mla").as_str());
        let qk_rope_head_dim = md_opt_usize(format!("{arch}.rope.dimension_count").as_str());
        let qk_nope_head_dim = match (key_length_mla, qk_rope_head_dim) {
            (Some(kl), Some(rd)) => Some(kl - rd),
            _ => key_length_mla,
        };
        let index_head_dim = md_opt_usize(format!("{arch}.attention.indexer.key_length").as_str());
        let index_n_heads = md_opt_usize(format!("{arch}.attention.indexer.head_count").as_str());
        let index_topk = md_opt_usize(format!("{arch}.attention.indexer.top_k").as_str());
        let index_skip_topk_offset = first_k_dense;

        let mut json_obj = serde_json::json!({"architectures": ["GlmMoeDsaForCausalLM"]});
        let obj = json_obj.as_object_mut().unwrap();
        if let Some(v) = q_lora_rank {
            obj.insert("q_lora_rank".into(), serde_json::json!(v));
        }
        if let Some(v) = kv_lora_rank {
            obj.insert("kv_lora_rank".into(), serde_json::json!(v));
        }
        if let Some(v) = qk_nope_head_dim {
            obj.insert("qk_nope_head_dim".into(), serde_json::json!(v));
        }
        if let Some(v) = v_head_dim {
            obj.insert("v_head_dim".into(), serde_json::json!(v));
        }
        if let Some(v) = qk_rope_head_dim {
            obj.insert("qk_rope_head_dim".into(), serde_json::json!(v));
        }
        if let Some(v) = index_head_dim {
            obj.insert("index_head_dim".into(), serde_json::json!(v));
        }
        if let Some(v) = index_n_heads {
            obj.insert("index_n_heads".into(), serde_json::json!(v));
        }
        if let Some(v) = index_topk {
            obj.insert("index_topk".into(), serde_json::json!(v));
        }
        if let Some(v) = index_skip_topk_offset {
            obj.insert("index_skip_topk_offset".into(), serde_json::json!(v));
        }
        if let Some(v) = expert_weights_scale {
            obj.insert("routed_scaling_factor".into(), serde_json::json!(v));
        }
        let extra_config_json = Some(json_obj.to_string());

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

        let mut cfg = Self::build_config(
            embedding_length,
            head_dim,
            feed_forward_length,
            block_count,
            head_count,
            head_count_kv,
            rms_norm_eps,
            rope_freq_base as f64,
            context_length,
            &moe_cfg,
            extra_config_json,
        );
        cfg.apply_runtime_rope_overrides(yarn_scaling_factor);

        let mla_cfg = MlaConfig::from_config(&cfg);
        let mut mla_rope_cfg = cfg.clone();
        mla_rope_cfg.head_dim = Some(mla_cfg.qk_rope_head_dim);
        mla_rope_cfg.partial_rotary_factor = None;
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            DType::F32,
            &mla_rope_cfg,
            device,
            true,
        )?);

        let first_k = first_k_dense.unwrap_or(0);
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let prefix_vb = vb.pp(&prefix);

            let is_moe_layer = layer_idx >= first_k && expert_count > 0;

            let mlp = if is_moe_layer {
                let gate = prefix_vb.get_no_shape("ffn_gate_inp.weight")?;
                let gate_experts =
                    prefix_vb.get_sharded_no_shape("ffn_gate_exps.weight", 1, rank, world_size)?;
                let up_experts =
                    prefix_vb.get_sharded_no_shape("ffn_up_exps.weight", 1, rank, world_size)?;
                let down_experts =
                    prefix_vb.get_sharded_no_shape("ffn_down_exps.weight", 2, rank, world_size)?;
                let bias = try_load_e_score_correction_bias(vb, &prefix, device);
                MoeOrMlp::FusedMoe(FusedMoe {
                    gate: QMatMul::from_arc(gate)?,
                    gate_experts,
                    up_experts,
                    down_experts,
                    act: candle_nn::Activation::Silu,
                    norm_topk_prob: moe_cfg.norm_topk_prob,
                    routed_scaling_factor: moe_cfg.routed_scaling_factor,
                    num_experts_per_tok: expert_used_count,
                    e_score_correction_bias: bias,
                    #[cfg(feature = "nccl")]
                    all_reduce: if world_size > 1 {
                        Some(AllReduce::new(comm.clone()))
                    } else {
                        None
                    },
                    dtype,
                    world_size,
                })
            } else {
                let ffn_gate =
                    prefix_vb.get_sharded_no_shape("ffn_gate.weight", 0, rank, world_size)?;
                let ffn_down =
                    prefix_vb.get_sharded_no_shape("ffn_down.weight", 1, rank, world_size)?;
                let ffn_up =
                    prefix_vb.get_sharded_no_shape("ffn_up.weight", 0, rank, world_size)?;
                MoeOrMlp::Mlp(Mlp {
                    ffn_gate: QMatMul::from_arc(ffn_gate)?,
                    ffn_up: QMatMul::from_arc(ffn_up)?,
                    ffn_down: QMatMul::from_arc(ffn_down)?,
                    #[cfg(feature = "nccl")]
                    all_reduce: if world_size > 1 {
                        Some(AllReduce::new(comm.clone()))
                    } else {
                        None
                    },
                    #[cfg(feature = "nccl")]
                    dtype,
                })
            };

            let shared_expert = if is_moe_layer {
                if let Some(sh_size) = expert_shared_feed_forward_length {
                    if sh_size > 0 {
                        let ffn_gate_shexp = prefix_vb.get_sharded_no_shape(
                            "ffn_gate_shexp.weight",
                            0,
                            rank,
                            world_size,
                        )?;
                        let ffn_down_shexp = prefix_vb.get_sharded_no_shape(
                            "ffn_down_shexp.weight",
                            1,
                            rank,
                            world_size,
                        )?;
                        let ffn_up_shexp = prefix_vb.get_sharded_no_shape(
                            "ffn_up_shexp.weight",
                            0,
                            rank,
                            world_size,
                        )?;
                        Some(Mlp {
                            ffn_gate: QMatMul::from_arc(ffn_gate_shexp)?,
                            ffn_up: QMatMul::from_arc(ffn_up_shexp)?,
                            ffn_down: QMatMul::from_arc(ffn_down_shexp)?,
                            #[cfg(feature = "nccl")]
                            all_reduce: if world_size > 1 {
                                Some(AllReduce::new(comm.clone()))
                            } else {
                                None
                            },
                            #[cfg(feature = "nccl")]
                            dtype,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let attention_norm = prefix_vb.get_no_shape("attn_norm.weight")?;
            let ffn_norm = prefix_vb.get_no_shape("ffn_norm.weight")?;

            let self_attn =
                QuantizedMlaAttention::new(vb, &prefix, &mla_cfg, &cfg, dtype, layer_idx, device)?;

            layers.push(LayerWeights {
                self_attn,
                attention_norm: QRmsNorm::from_arc_qtensor(attention_norm, rms_norm_eps)?,
                mlp,
                shared_expert,
                ffn_norm: QRmsNorm::from_arc_qtensor(ffn_norm, rms_norm_eps)?,
                rotary_emb: rotary_emb.clone(),
            });
            reporter.write().set_progress(layer_idx + 1);
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            cfg,
            dtype,
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(x, input_positions, kv_caches, input_metadata, false)
    }

    pub fn forward_embedding(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(x, input_positions, kv_caches, input_metadata, true)
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        return_hidden: bool,
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

        let mut xs = self.tok_embeddings.forward(x)?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                let residual = &xs;
                let x = layer.attention_norm.forward(&xs)?;
                let rope = layer.rotary_emb.clone();
                let attn = layer.self_attn.forward(
                    &x,
                    &Some(rope),
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
                let x = (attn + residual)?;

                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;
                let shared_output = if let Some(ref se) = layer.shared_expert {
                    Some(se.forward(&x)?)
                } else {
                    None
                };
                let mlp_output = layer.mlp.forward(&x, input_metadata.is_prefill)?;
                xs = if let Some(shared_output) = shared_output {
                    (residual + (mlp_output + shared_output)?)?
                } else {
                    (residual + mlp_output)?
                };
            }
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
}
