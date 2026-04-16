use crate::openai::distributed::{shard, Comm, ReplicatedLinear, TensorParallelRowLinear};
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
    kv_b_proj: ReplicatedLinear,
    o_proj: TensorParallelRowLinear,
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
}

impl MlaAttention {
    pub fn new(
        vb: VarBuilder,
        comm: Rc<Comm>,
        mla_cfg: &MlaConfig,
        config: &Config,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_size = mla_cfg.hidden_size;
        let num_heads = mla_cfg.num_attention_heads;
        let kv_lora_rank = mla_cfg.kv_lora_rank;
        let qk_nope_head_dim = mla_cfg.qk_nope_head_dim;
        let qk_rope_head_dim = mla_cfg.qk_rope_head_dim;
        let v_head_dim = mla_cfg.v_head_dim;
        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;

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
                    dtype,
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
            dtype,
            false,
        )?;

        let kv_b_proj = ReplicatedLinear::load_b(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            false,
            vb.pp("kv_b_proj"),
            &config.isq_quant,
            &config.quantization_config,
        )?;

        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * v_head_dim,
            hidden_size,
            false,
            vb.pp("o_proj"),
            comm,
            &config.isq_quant,
            &config.quantization_config,
        )?;

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
            dtype,
        })
    }

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

        let q = if let (Some(q_a), Some(q_a_ln), Some(q_b)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_a_out = q_a.forward(xs)?;
            let q_a_normed = q_a_ln.forward(&q_a_out)?;
            q_b.forward(&q_a_normed)?
        } else {
            self.q_proj.as_ref().unwrap().forward(xs)?
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
                let block_tables_i32 = block_tables.to_dtype(DType::I64)?.to_dtype(DType::U32)?;
                let context_lens_i32 = context_lens.to_dtype(DType::I64)?.to_dtype(DType::U32)?;

                if input_metadata.is_prefill {
                    let cu_seqlens_q = input_metadata.cu_seqlens_q.as_ref().ok_or_else(|| {
                        candle_core::Error::msg("MLA fused prefill requires cu_seqlens_q")
                    })?;
                    let cu_seqlens_i32 = cu_seqlens_q.to_dtype(DType::I64)?.to_dtype(DType::U32)?;

                    let attn_out = attention_rs::mla::mla_paged_prefill(
                        &q_absorbed,
                        &q_pe,
                        &ckv_cache_3d,
                        &kpe_cache_3d,
                        &block_tables_i32,
                        &context_lens_i32,
                        &cu_seqlens_i32,
                        self.sm_scale,
                    )?;
                    return self.project_mla_output(&attn_out, seq_len, xs.dtype());
                }

                let attn_out = attention_rs::mla::mla_paged_decode(
                    &q_absorbed,
                    &q_pe,
                    &ckv_cache_3d,
                    &kpe_cache_3d,
                    &block_tables_i32,
                    &context_lens_i32,
                    self.sm_scale,
                )?;
                return self.project_mla_output(&attn_out, seq_len, xs.dtype());
            }

            if input_metadata.is_prefill {
                let attn_out =
                    self.mla_sdp_prefill(&q_absorbed, &q_pe, &ckv, &k_pe, input_metadata)?;
                let y = attn_out.to_dtype(xs.dtype())?;
                return self.o_proj.forward(&y);
            }
        }
        candle_core::bail!("MLA attention requires CUDA platform!")
    }

    fn mla_sdp_prefill(
        &self,
        q_absorbed: &Tensor,
        q_pe: &Tensor,
        ckv: &Tensor,
        k_pe: &Tensor,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let cu_seqlens = input_metadata
            .cu_seqlens_q
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("MLA prefill requires cu_seqlens_q"))?
            .to_vec1::<u32>()?;
        let num_seqs = cu_seqlens.len() - 1;

        let mut results = Vec::with_capacity(num_seqs);
        for s in 0..num_seqs {
            let start = cu_seqlens[s] as usize;
            let end = cu_seqlens[s + 1] as usize;
            let slen = end - start;

            let q_abs_s = q_absorbed.narrow(0, start, slen)?.contiguous()?;
            let q_pe_s = q_pe.narrow(0, start, slen)?.contiguous()?;
            let ckv_s = ckv.narrow(0, start, slen)?.contiguous()?;
            let k_pe_s = k_pe.narrow(0, start, slen)?.contiguous()?;

            let q_abs_t = q_abs_s.transpose(0, 1)?.contiguous()?;
            let ckv_kt = ckv_s
                .t()?
                .unsqueeze(0)?
                .broadcast_as((self.num_heads, self.kv_lora_rank, slen))?
                .contiguous()?;
            let nope_scores = q_abs_t.matmul(&ckv_kt)?;

            let q_pe_t = q_pe_s.transpose(0, 1)?.contiguous()?;
            let k_pe_kt = k_pe_s
                .t()?
                .unsqueeze(0)?
                .broadcast_as((self.num_heads, self.qk_rope_head_dim, slen))?
                .contiguous()?;
            let pe_scores = q_pe_t.matmul(&k_pe_kt)?;

            let scores = ((nope_scores + pe_scores)? * f64::from(self.sm_scale))?;

            let dev = scores.device().clone();
            let scores_dtype = scores.dtype();
            let mut mask_data = vec![0.0f32; slen * slen];
            for qi in 0..slen {
                for ki in (qi + 1)..slen {
                    mask_data[qi * slen + ki] = f32::NEG_INFINITY;
                }
            }
            let causal_mask =
                Tensor::from_vec(mask_data, (1, slen, slen), &dev)?.to_dtype(scores_dtype)?;
            let scores = scores.broadcast_add(&causal_mask)?;

            let attn_weights = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
                .to_dtype(self.dtype)?;

            let ckv_v = ckv_s
                .unsqueeze(0)?
                .broadcast_as((self.num_heads, slen, self.kv_lora_rank))?
                .contiguous()?;
            let attn_out = attn_weights.matmul(&ckv_v)?;

            let y = attn_out.matmul(&self.w_uv_t)?;
            let y = y.transpose(0, 1)?.contiguous()?;
            let y = y.reshape((slen, self.num_heads * self.v_head_dim))?;
            results.push(y);
        }

        Tensor::cat(&results, 0)
    }
}
