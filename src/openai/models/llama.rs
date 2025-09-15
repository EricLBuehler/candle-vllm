use super::{attention::Attention, mlp::Mlp, rotary_emb::ScalingRotaryEmbedding, Config};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{embedding, rms_norm, Comm, ReplicatedLinear, VarBuilder};
use crate::openai::models::mask::get_attention_casual_mask;
use crate::InputMetadata;
use candle::{DType, Device, Result, Tensor};
use candle_core as candle;
use candle_nn::{Embedding, Module, RmsNorm};
use std::iter::zip;
use std::path::PathBuf;
pub use std::rc::Rc;
use std::sync::{Arc, RwLock};

impl Llama {
    pub fn load_config(filename: &PathBuf, isq: Option<String>) -> Result<Config> {
        let mut config = Config::load_config(filename.clone())?;
        config.head_dim = Some(
            config
                .head_dim
                .unwrap_or(config.hidden_size / config.num_attention_heads),
        );
        config.num_key_value_heads = Some(
            config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
        );
        config.max_seq_len = config.max_position_embeddings.unwrap_or(config.max_seq_len);
        if config.quantization_config.is_some() {
            config.quant = Some(
                config
                    .quantization_config
                    .as_ref()
                    .unwrap()
                    .quant_method
                    .clone(),
            );
        } else if isq.is_some() {
            config.quant = Some(isq.unwrap().to_string());
        }
        Ok(config)
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: Attention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self
            .attn
            .forward(&x, attention_mask, input_positions, cache, input_metadata)?
            + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        comm: Rc<Comm>,
        rotay_emb: Arc<ScalingRotaryEmbedding>,
    ) -> Result<Self> {
        let attn = Attention::new(
            rotay_emb.clone(),
            cfg,
            vb.pp("self_attn"),
            comm.clone(),
            cfg.sliding_window,
        )?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"), comm.clone())?;
        let rms_1 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }
}

pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    lm_head: ReplicatedLinear,
    cfg: Config,
    dtype: DType,
    device: Device,
}

impl Llama {
    pub fn forward(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
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
        let attention_mask = get_attention_casual_mask(
            &self.device,
            self.dtype,
            input_positions,
            &seqlens,
            self.cfg.sliding_window,
            input_metadata.is_prefill,
        );
        let mut xs = self.wte.forward(x)?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), block) in zip(kv_caches.iter(), &self.blocks) {
                xs = block.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
            }
        } else {
            for block in &self.blocks {
                xs = block.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?;
            }
        }
        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            vb.pp("lm_head"),
            &None,
            &None,
        )?;

        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, cfg, device, true)?);
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let reporter = progress_reporter.clone();
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| {
                let b = Block::load(
                    vb.pp(format!("model.layers.{i}")),
                    cfg,
                    comm.clone(),
                    rotary_emb.clone(),
                )
                .unwrap();
                reporter.write().unwrap().set_progress(i + 1);
                b
            })
            .collect();

        Ok(Self {
            wte,
            blocks,
            norm,
            lm_head,
            cfg: cfg.clone(),
            dtype,
            device: device.clone(),
        })
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
