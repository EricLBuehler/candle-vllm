use super::rotary_emb::ScalingRotaryEmbedding;
use crate::openai::distributed::{
    rms_norm_with_dtype, Comm, TensorParallelColumnLinear, TensorParallelRowLinear, VarBuilder,
};
use crate::openai::models::layers::qrmsnorm::QRmsNorm;
use crate::openai::models::Config;
use crate::{InputMetadata, PagedAttention};
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::RmsNorm;

use std::rc::Rc;
use std::sync::Arc;

pub struct Attention {
    q_proj: TensorParallelColumnLinear,
    k_proj: TensorParallelColumnLinear,
    v_proj: TensorParallelColumnLinear,
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
}

impl Attention {
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
        let attention_bias = cfg.attention_bias.unwrap_or(false);

        let q_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            num_heads * head_dim,
            attention_bias,
            vb.pp("q_proj"),
            comm.clone(),
            &cfg.quant,
            &cfg.quantization_config,
        )?;
        let k_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            num_kv_heads * head_dim,
            attention_bias,
            vb.pp("k_proj"),
            comm.clone(),
            &cfg.quant,
            &cfg.quantization_config,
        )?;
        //v_proj requires higher precision
        let q8_0_quant = Some("q8_0".to_string());
        let v_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            num_kv_heads * head_dim,
            attention_bias,
            vb.pp("v_proj"),
            comm.clone(),
            if cfg.quant.is_some()
                && !matches!(
                    cfg.quant.as_ref().unwrap().as_str(),
                    "gptq" | "awq" | "marlin"
                )
            {
                &q8_0_quant
            } else {
                &cfg.quant
            },
            &cfg.quantization_config,
        )?;

        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
            hidden_sz,
            false,
            vb.pp("o_proj"),
            comm.clone(),
            &cfg.quant,
            &cfg.quantization_config,
        )?;

        //we use higher precision for q/k norm
        let q_norm =
            rms_norm_with_dtype(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"), DType::F32).ok();
        let k_norm =
            rms_norm_with_dtype(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"), DType::F32).ok();

        assert!(cfg.num_attention_heads >= comm.world_size());
        assert!(cfg.num_attention_heads % comm.world_size() == 0);

        assert!(cfg.num_key_value_heads.unwrap() >= comm.world_size());
        assert!(cfg.num_key_value_heads.unwrap() % comm.world_size() == 0);

        let attention_heads = cfg.num_attention_heads / comm.world_size();
        let kv_heads = cfg.num_key_value_heads.unwrap() / comm.world_size();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
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

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

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

        let (q, k) = if q.dtype() != DType::F32 {
            (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?)
        } else {
            (q, k)
        };

        let (q, k) = if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            // Per‑head RMSNorm in qwen3
            let q_flat = q.flatten(0, 2)?; // (B*H, L, D) -> (BHL, D) after transpose later
            let k_flat = k.flatten(0, 2)?;

            // q_norm and k_norm weights stored in f32 format in qwen3 gguf
            let q_flat = q_norm.forward(&q_flat)?;
            let k_flat = k_norm.forward(&k_flat)?;

            let q = q_flat.reshape((1, self.num_heads, seq_len, self.head_dim))?;
            let k = k_flat.reshape((1, self.num_kv_heads, seq_len, self.head_dim))?;

            (q, k)
        } else {
            (q, k)
        };

        let (q, k) = self.rotary_emb.apply_rotary_emb(&q, &k, input_positions)?;
        let (q, k) = (q.to_dtype(self.dtype)?, k.to_dtype(self.dtype)?);

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

        let y = self.o_proj.forward(&y)?;
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

        let q = q
            .reshape((1, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((1, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((1, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            // Per‑head RMSNorm in qwen3
            let q_flat = q.flatten(0, 2)?; // (B*H, L, D) -> (BHL, D) after transpose later
            let k_flat = k.flatten(0, 2)?;

            // q_norm and k_norm weights stored in f32 format in qwen3 gguf
            let q_flat = q_norm.forward(&q_flat)?;
            let k_flat = k_norm.forward(&k_flat)?;

            let q = q_flat.reshape((1, self.n_head, seq_len, self.head_dim))?;
            let k = k_flat.reshape((1, self.n_kv_head, seq_len, self.head_dim))?;

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
