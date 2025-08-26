use super::rotary_emb::ScalingRotaryEmbedding;
use super::{Config, QwenMoEConfig};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::paged_attention::input_metadata::InputMetadata;
use crate::paged_attention::PagedAttention;
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;
use either::Either;
use std::iter::zip;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

struct FusedMoe {
    gate: QMatMul,
    gate_experts: QMatMul,
    up_experts: QMatMul,
    down_experts: QMatMul,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl FusedMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let original_dtype = xs.dtype();
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let router_logits = self.gate.forward(&xs.to_dtype(DType::F32)?)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        //last dim size 128
        let indices = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let mut scores = routing_weights.gather(&indices, D::Minus1)?;

        if self.norm_topk_prob {
            scores = scores.broadcast_div(&scores.sum_keepdim(D::Minus1)?)?;
        }

        let ys = {
            let xs = xs.reshape((num_tokens, 1, hidden_dim))?;
            let gate = self.gate_experts.indexed_moe_forward(&xs, &indices)?;
            let up = self.up_experts.indexed_moe_forward(&xs, &indices)?;
            let xs = self
                .down_experts
                .indexed_moe_forward(&(up * gate.apply(&self.act)?)?, &indices)?;
            xs
        };
        ys.broadcast_mul(&scores.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((batch, seq_len, hidden_dim))?
            .to_dtype(original_dtype)
    }
}

enum MoeOrMlp {
    FusedMoe(FusedMoe),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs),
        }
    }
}

struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_bq: Option<Tensor>,
    attention_bk: Option<Tensor>,
    attention_bv: Option<Tensor>,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    mlp: MoeOrMlp,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    attn: PagedAttention,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
    dtype: DType,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        input_positions: &[Vec<usize>],
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

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

        let (q, k, v) = if seq_len == 1 {
            //no need transpose for seq_len == 1, change reshape dim
            let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k, v)
        } else {
            let q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            (q.contiguous()?, k.contiguous()?, v.contiguous()?)
        };

        let (q, k) = if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            // Per‑head RMSNorm in qwen3
            let q_flat = q.flatten(0, 2)?; // (B*H, L, D) -> (BHL, D) after transpose later
            let k_flat = k.flatten(0, 2)?;

            // q_norm and k_norm weights stored in f32 format in qwen3 gguf
            let q_flat = q_norm.forward(&q_flat)?;
            let k_flat = k_norm.forward(&k_flat)?;

            let q = q_flat.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k_flat.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;

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
            .reshape((b_sz, seq_len, ()))?;

        let y = self.attention_wo.forward(&y.to_dtype(x.dtype())?)?;
        Ok(y)
    }
}

pub struct GGUFQWenMoE {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    cfg: Config,
    dtype: DType,
    device: Device,
}

impl GGUFQWenMoE {
    pub fn into_config(
        arch: String,
        embedding_length: usize,
        head_dim: usize,
        i_size: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        rms_eps: f64,
        rope_theta: f64,
        max_seq_len: usize,
        original_max_position_embeddings: Option<usize>,
        partial_rotary_factor: Option<f32>,
        moe_cfg: &QwenMoEConfig,
    ) -> Config {
        Config {
            architectures: Some(vec![arch]),
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
            bos_token_id: Some(super::TokenID(Either::Left(Some(151644)))),
            eos_token_id: super::TokenID(Either::Left(Some(151645))),
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
            qwen_moe_config: Some(moe_cfg.clone()),
            quant: Some("gguf".to_string()),
        }
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        dtype: DType,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
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

        let moe_cfg = QwenMoEConfig {
            moe_intermediate_size: md_get(format!("{arch}.expert_feed_forward_length").as_str())?
                .to_u32()? as usize,
            num_experts: Some(md_get(format!("{arch}.expert_count").as_str())?.to_u32()? as usize),
            mlp_only_layers: Some(vec![]),
            decoder_sparse_step: Some(1),
            norm_topk_prob: true,
            num_experts_per_tok: md_get(format!("{arch}.expert_used_count").as_str())?.to_u32()?
                as usize,
        };

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(v) => QMatMul::from_qtensor(v)?,
            _ => {
                // use tie_word_embeddings
                QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?
            }
        };

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
        let cfg = GGUFQWenMoE::into_config(
            arch.clone(),
            embedding_length,
            head_dim,
            0,
            block_count,
            head_count,
            head_count_kv,
            rms_norm_eps,
            rope_freq_base as f64,
            context_length,
            original_max_position_embeddings,
            partial_rotary_factor,
            &moe_cfg,
        );
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, &cfg, device, true)?);

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
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

            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let mlp = if !moe_cfg
                .mlp_only_layers
                .as_ref()
                .unwrap()
                .contains(&layer_idx)
                && (moe_cfg.num_experts.unwrap() > 0
                    && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap() == 0)
            {
                let gate = ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let gate_experts =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate_exps.weight"), device)?;
                let up_experts =
                    ct.tensor(reader, &format!("{prefix}.ffn_up_exps.weight"), device)?;
                let down_experts =
                    ct.tensor(reader, &format!("{prefix}.ffn_down_exps.weight"), device)?;
                let moe = FusedMoe {
                    gate: QMatMul::from_qtensor(gate)?,
                    gate_experts: QMatMul::from_qtensor(gate_experts)?,
                    up_experts: QMatMul::from_qtensor(up_experts)?,
                    down_experts: QMatMul::from_qtensor(down_experts)?,
                    act: candle_nn::Activation::Silu,
                    norm_topk_prob: moe_cfg.norm_topk_prob,
                    num_experts_per_tok: moe_cfg.num_experts_per_tok,
                };

                MoeOrMlp::FusedMoe(moe)
            } else {
                let mlp = {
                    let feed_forward_w1 =
                        ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                    let feed_forward_w2 =
                        ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                    let feed_forward_w3 =
                        ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                    Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                    }
                };
                MoeOrMlp::Mlp(mlp)
            };

            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let (q_norm, k_norm) = {
                let q_norm = ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?;
                let k_norm = ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?;
                let q_norm = RmsNorm::from_qtensor(q_norm, rms_norm_eps)?;
                let k_norm = RmsNorm::from_qtensor(k_norm, rms_norm_eps)?;
                (Some(q_norm), Some(k_norm))
            };

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_bq,
                attention_bk,
                attention_bv,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                q_norm,
                k_norm,
                mlp,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                attn: PagedAttention::new(
                    head_count,
                    head_dim,
                    1. / ((head_dim as f32).sqrt()),
                    Some(head_count_kv),
                    None,
                    device.clone(),
                    None,
                )?,
                rotary_emb: rotary_emb.clone(),
                dtype,
            });
            reporter.write().unwrap().set_progress(layer_idx + 1);
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
        input_positions: &[Vec<usize>],
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;
        assert!(
            seq_len < self.cfg.max_seq_len,
            "Input token length exceed maximum context allowed for this model."
        );
        let mask = if seq_len <= 1 {
            None
        } else {
            super::get_attention_casual_mask(
                &self.device,
                self.dtype,
                b_sz,
                seq_len,
                input_positions,
                self.cfg.sliding_window,
            )
        };
        let mut layer_in = self.tok_embeddings.forward(x)?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                let x = layer_in;
                let residual = &x;
                let x = layer.attention_norm.forward(&x)?;
                let attn = layer.forward_attn(
                    &x,
                    mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
                let x = (attn + residual)?;

                // MLP
                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;
                let x = layer.mlp.forward(&x)?;
                let x = (x + residual)?;
                layer_in = x
            }
        } else {
            for layer in self.layers.iter() {
                let x = layer_in;
                let residual = &x;
                let x = layer.attention_norm.forward(&x)?;
                let attn =
                    layer.forward_attn(&x, mask.as_ref(), input_positions, None, input_metadata)?;
                let x = (attn + residual)?;

                // MLP
                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;
                let x = layer.mlp.forward(&x)?;
                let x = (x + residual)?;
                layer_in = x
            }
        }
        let x = layer_in
            .i((.., seq_len - 1, ..))?
            .contiguous()?
            .apply(&self.norm)?;
        self.output.forward(&x)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
