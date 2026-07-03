use crate::openai::distributed::ReplicatedLinear;
use crate::openai::models::layers::others::{layer_norm, NormX};
use crate::openai::models::layers::rotary_emb::ScalingRotaryEmbedding;
use crate::openai::models::Config;
use candle_core::{DType, Result, Tensor, D};
use std::sync::Arc;

use crate::openai::distributed::VarBuilder;

pub struct IndexerConfig {
    pub index_head_dim: usize,
    pub index_n_heads: usize,
    pub index_topk: usize,
    pub index_skip_topk_offset: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub hidden_size: usize,
}

/// DSA (DeepSeek Sparse Attention) lightning indexer (prefill-only).
///
/// Selects the top-k most relevant tokens for each query position,
/// producing sparse indices used to mask the main MLA attention during prefill.
/// All operations are GPU-only — no CPU↔GPU sync.
pub struct DsaIndexer {
    wq_b: ReplicatedLinear,
    wk: ReplicatedLinear,
    k_norm: NormX,
    weights_proj: ReplicatedLinear,
    cfg: IndexerConfig,
    score_scale: f32,
}

impl DsaIndexer {
    pub fn new(vb: VarBuilder, config: &Config, cfg: IndexerConfig, dtype: DType) -> Result<Self> {
        let wq_b = ReplicatedLinear::load_no_bias(
            cfg.q_lora_rank,
            cfg.index_n_heads * cfg.index_head_dim,
            vb.pp("wq_b"),
            &config.isq_quant,
            &config.quantization_config,
        )?;
        let wk = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.index_head_dim,
            vb.pp("wk"),
            &config.isq_quant,
            &config.quantization_config,
        )?;
        let k_norm = layer_norm(cfg.index_head_dim, 1e-6, true, vb.pp("k_norm"), dtype)?;
        let weights_proj = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.index_n_heads,
            vb.pp("weights_proj"),
            &None,
            &config.quantization_config,
        )?;

        let softmax_scale = 1.0 / (cfg.index_head_dim as f32).sqrt();
        let head_scale = (cfg.index_n_heads as f32).powf(-0.5);
        let score_scale = softmax_scale * head_scale;

        Ok(Self {
            wq_b,
            wk,
            k_norm,
            weights_proj,
            cfg,
            score_scale,
        })
    }

    pub fn index_topk(&self) -> usize {
        self.cfg.index_topk
    }

    #[cfg(feature = "cuda")]
    pub fn forward(
        &self,
        xs: &Tensor,
        q_resid: &Tensor,
        rotary_emb: &Option<Arc<ScalingRotaryEmbedding>>,
        positions: &Tensor,
    ) -> Result<Option<Tensor>> {
        let (seq_len, _) = xs.dims2()?;
        if seq_len <= self.cfg.index_topk {
            return Ok(None);
        }

        let idx_q = self.wq_b.forward(q_resid)?;
        let idx_q = idx_q.reshape((seq_len, self.cfg.index_n_heads, self.cfg.index_head_dim))?;

        let idx_q_rope = idx_q.narrow(D::Minus1, 0, self.cfg.qk_rope_head_dim)?;
        let idx_q_pass = idx_q.narrow(
            D::Minus1,
            self.cfg.qk_rope_head_dim,
            self.cfg.index_head_dim - self.cfg.qk_rope_head_dim,
        )?;

        let idx_k = self.wk.forward(xs)?;
        let idx_k = self.k_norm.forward(&idx_k)?;
        let idx_k = idx_k.unsqueeze(1)?;

        let idx_k_rope = idx_k.narrow(D::Minus1, 0, self.cfg.qk_rope_head_dim)?;
        let idx_k_pass = idx_k.narrow(
            D::Minus1,
            self.cfg.qk_rope_head_dim,
            self.cfg.index_head_dim - self.cfg.qk_rope_head_dim,
        )?;

        let (idx_q_rope, idx_k_rope) = if let Some(re) = rotary_emb {
            let idx_q_rope_c = idx_q_rope.contiguous()?;
            let idx_k_rope_c = idx_k_rope.contiguous()?;
            let (q_new, k_new) = re.apply_rotary_emb(&idx_q_rope_c, &idx_k_rope_c, positions)?;
            (q_new, k_new)
        } else {
            (idx_q_rope.contiguous()?, idx_k_rope.contiguous()?)
        };

        let idx_q = Tensor::cat(&[&idx_q_rope, &idx_q_pass.contiguous()?], D::Minus1)?;
        let idx_k = Tensor::cat(&[&idx_k_rope, &idx_k_pass.contiguous()?], D::Minus1)?;
        let idx_k = idx_k.squeeze(1)?;

        let idx_q = idx_q.to_dtype(DType::BF16)?.contiguous()?;
        let idx_k = idx_k.to_dtype(DType::BF16)?.contiguous()?;

        let weights = self.weights_proj.forward(xs)?;
        let weights = weights.to_dtype(DType::F32)?.contiguous()?;

        let topk = self.cfg.index_topk.min(seq_len);
        let topk_indices = attention_rs::mla::dsa_lightning_indexer_prefill(
            &idx_q,
            &idx_k,
            &weights,
            topk,
            self.score_scale,
        )?;
        Ok(Some(topk_indices))
    }
}
