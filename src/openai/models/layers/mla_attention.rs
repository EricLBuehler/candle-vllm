use crate::openai::distributed::{shard, Comm, ReplicatedLinear};
use crate::openai::models::layers::indexer::{DsaIndexer, IndexerConfig};
use crate::openai::models::layers::others::{rms_norm, NormX};
use crate::openai::models::layers::rotary_emb::ScalingRotaryEmbedding;
use crate::openai::models::Config;
use attention_rs::InputMetadata;
use candle_core::{DType, Result, Tensor, D};
use std::rc::Rc;
use std::sync::Arc;

use crate::openai::distributed::VarBuilder;

pub struct MlaConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub index_head_dim: Option<usize>,
    pub index_n_heads: Option<usize>,
    pub index_topk: Option<usize>,
    pub index_skip_topk_offset: Option<usize>,
}

impl MlaConfig {
    pub fn from_config(config: &Config) -> Self {
        let extra: serde_json::Value = config
            .extra_config_json
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(serde_json::Value::Null);

        Self {
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            q_lora_rank: extra
                .get("q_lora_rank")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            kv_lora_rank: extra
                .get("kv_lora_rank")
                .and_then(|v| v.as_u64())
                .unwrap_or(512) as usize,
            qk_nope_head_dim: extra
                .get("qk_nope_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(128) as usize,
            qk_rope_head_dim: extra
                .get("qk_rope_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(64) as usize,
            v_head_dim: extra
                .get("v_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(128) as usize,
            rms_norm_eps: config.rms_norm_eps,
            attention_bias: config.attention_bias.unwrap_or(false),
            index_head_dim: extra
                .get("index_head_dim")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            index_n_heads: extra
                .get("index_n_heads")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            index_topk: extra
                .get("index_topk")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            index_skip_topk_offset: extra
                .get("index_skip_topk_offset")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        }
    }
}

#[allow(unused)]
pub struct MlaAttention {
    q_a_proj: Option<ReplicatedLinear>,
    q_a_layernorm: Option<NormX>,
    q_b_proj: Option<ReplicatedLinear>,
    q_proj: Option<ReplicatedLinear>,
    kv_a_proj_with_mqa: ReplicatedLinear,
    kv_a_layernorm: NormX,
    #[allow(dead_code)]
    kv_b_proj: Option<ReplicatedLinear>,
    o_proj: ReplicatedLinear,
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
    promote_qk_to_f32: bool,
    dtype: DType,
    indexer: Option<DsaIndexer>,
}

impl MlaAttention {
    pub fn new(
        vb: VarBuilder,
        _comm: Rc<Comm>,
        mla_cfg: &MlaConfig,
        config: &Config,
        dtype: DType,
        layer_idx: usize,
    ) -> Result<Self> {
        let hidden_size = mla_cfg.hidden_size;
        let num_heads = mla_cfg.num_attention_heads;
        let kv_lora_rank = mla_cfg.kv_lora_rank;
        let qk_nope_head_dim = mla_cfg.qk_nope_head_dim;
        let qk_rope_head_dim = mla_cfg.qk_rope_head_dim;
        let v_head_dim = mla_cfg.v_head_dim;
        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let is_qvar_builder = config.isq_quant.is_some();
        let norm_dtype = if is_qvar_builder || config.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };

        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) =
            if let Some(q_lora_rank) = mla_cfg.q_lora_rank {
                let q_a = ReplicatedLinear::load_b(
                    hidden_size,
                    q_lora_rank,
                    mla_cfg.attention_bias,
                    vb.pp("q_a_proj"),
                    &config.isq_quant,
                    &config.quantization_config,
                )?;
                let q_a_ln = rms_norm(
                    q_lora_rank,
                    mla_cfg.rms_norm_eps,
                    vb.pp("q_a_layernorm"),
                    norm_dtype,
                    false,
                )?;
                let q_b = ReplicatedLinear::load_b(
                    q_lora_rank,
                    num_heads * q_head_dim,
                    false,
                    vb.pp("q_b_proj"),
                    &config.isq_quant,
                    &config.quantization_config,
                )?;
                (Some(q_a), Some(q_a_ln), Some(q_b), None)
            } else {
                let q = ReplicatedLinear::load_b(
                    hidden_size,
                    num_heads * q_head_dim,
                    mla_cfg.attention_bias,
                    vb.pp("q_proj"),
                    &config.isq_quant,
                    &config.quantization_config,
                )?;
                (None, None, None, Some(q))
            };

        let kv_a_proj_with_mqa = ReplicatedLinear::load_b(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            mla_cfg.attention_bias,
            vb.pp("kv_a_proj_with_mqa"),
            &config.isq_quant,
            &config.quantization_config,
        )?;

        let kv_a_layernorm = rms_norm(
            kv_lora_rank,
            mla_cfg.rms_norm_eps,
            vb.pp("kv_a_layernorm"),
            norm_dtype,
            false,
        )?;

        let kv_b_proj = Some(ReplicatedLinear::load_b(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            false,
            vb.pp("kv_b_proj"),
            &config.isq_quant,
            &config.quantization_config,
        )?);

        let o_proj = ReplicatedLinear::load_no_bias(
            num_heads * v_head_dim,
            hidden_size,
            vb.pp("o_proj"),
            &config.isq_quant,
            &config.quantization_config,
        )?;

        // Pre-compute absorbed MLA weights from kv_b_proj
        let kv_b_weight = vb.pp("kv_b_proj").get_with_hints_dtype(
            (num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank),
            "weight",
            shard(0, 0, 1),
            dtype,
        )?;
        let w = kv_b_weight.reshape((num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank))?;
        let w_uk = w.narrow(1, 0, qk_nope_head_dim)?.contiguous()?;
        let w_uv = w.narrow(1, qk_nope_head_dim, v_head_dim)?.contiguous()?;
        let w_uv_t = w_uv.transpose(1, 2)?.contiguous()?;

        let mut sm_scale = 1.0 / (q_head_dim as f32).sqrt();
        let mut rope_scale = 1.0f32;

        if let Some(ref rope_scaling) = config.rope_scaling {
            use crate::openai::models::ScalingValue;
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

        // Match xinfer's DSA layer selection. Layer 0 is kept dense by
        // default, and an indexer is only constructed when its weights are
        // present in this layer. This matters for long prefills: enabling a
        // sparse indexer in a layer without the trained indexer weights
        // changes the attention pattern and compounds across the model.
        let skip_offset = mla_cfg.index_skip_topk_offset.unwrap_or(1);
        let has_indexer = mla_cfg.index_head_dim.is_some()
            && layer_idx >= skip_offset
            && vb.pp("indexer").contains_tensor("wq_b.weight");
        let indexer = if has_indexer {
            let idx_cfg = IndexerConfig {
                index_head_dim: mla_cfg.index_head_dim.unwrap(),
                index_n_heads: mla_cfg.index_n_heads.unwrap_or(4),
                index_topk: mla_cfg.index_topk.unwrap_or(2048),
                index_skip_topk_offset: skip_offset,
                qk_rope_head_dim,
                q_lora_rank: mla_cfg.q_lora_rank.unwrap_or(256),
                hidden_size,
            };
            Some(DsaIndexer::new(vb.pp("indexer"), config, idx_cfg, dtype)?)
        } else {
            None
        };

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
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
            promote_qk_to_f32: is_qvar_builder || config.higher_precision_required(),
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

    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    pub fn forward(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<ScalingRotaryEmbedding>>,
        _attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        // Produce Q and optionally retain q_resid for DSA indexer
        let (q, q_resid) = if let (Some(q_a), Some(q_a_ln), Some(q_b)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_a_out = q_a.forward(xs)?;
            let q_a_normed = q_a_ln.forward(&q_a_out)?;
            let q = q_b.forward(&q_a_normed)?;
            (q, Some(q_a_normed))
        } else {
            (self.q_proj.as_ref().unwrap().forward(xs)?, None)
        };

        let q = q.reshape((seq_len, self.num_heads, self.q_head_dim))?;
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        let kv_a = self.kv_a_proj_with_mqa.forward(xs)?;
        let ckv = kv_a.narrow(D::Minus1, 0, self.kv_lora_rank)?;
        let k_pe_raw = kv_a.narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?;

        let ckv = self.kv_a_layernorm.forward(&ckv)?;

        let k_pe = k_pe_raw.reshape((seq_len, 1, self.qk_rope_head_dim))?;
        let q_pe_for_rope = q_pe.contiguous()?;

        let (q_pe_for_rope, k_pe) = if self.promote_qk_to_f32 {
            (
                q_pe_for_rope.to_dtype(DType::F32)?,
                k_pe.to_dtype(DType::F32)?,
            )
        } else {
            (q_pe_for_rope, k_pe)
        };
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

                // DSA sparse prefill with FlashInfer path
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

                    // DSA sparse prefill with native CUDA path
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

                // DSA is prefill-only: dense MLA decode is faster at all practical context lengths.
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
