use super::{
    attention::Attention, mlp::Mlp, rotary_emb::ScalingRotaryEmbedding, Config, InputMetadata,
};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{embedding, rms_norm, Comm, ReplicatedLinear, VarBuilder};
use crate::openai::models::layers::deepstack::ApplyDeepStack;
use crate::openai::models::mask::get_attention_causal_mask;
use candle::{DType, Device, Module, Result, Tensor};
use candle_core as candle;
use candle_nn::RmsNorm;
use parking_lot::RwLock;
use std::iter::zip;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

impl Qwen {
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
        config.attention_bias = Some(config.attention_bias.unwrap_or(true));
        config.isq_quant = if config.quantization_config.is_some() {
            None
        } else {
            isq
        };
        Ok(config)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            vb.pp("self_attn"),
            comm.clone(),
            cfg.sliding_window,
        )?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"), comm.clone())?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs =
            self.self_attn
                .forward(&xs, attention_mask, input_positions, cache, input_metadata)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

pub struct Qwen {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: ReplicatedLinear,
    device: Device,
    dtype: DType,
    cfg: Config,
    vocab_size: usize,
}

impl Qwen {
    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        Self::new_with_prefix(vb, cfg, dtype, device, comm, progress_reporter, None)
    }

    pub fn new_with_prefix(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
        prefix: Option<String>,
    ) -> Result<Self> {
        let (vb_m, tie_word_embeddings) = if let Some(prefix) = prefix {
            (vb.pp(prefix.trim_end_matches('.')), cfg.tie_word_embeddings)
        } else if !vb.contains_tensor("model.embed_tokens.weight")
            && vb.contains_tensor("embed_tokens.weight")
        {
            (vb.clone(), true)
        } else {
            (vb.pp("model"), cfg.tie_word_embeddings)
        };

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, cfg, device, true)?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let reporter = progress_reporter.clone();
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer =
                DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), comm.clone())?;
            layers.push(layer);
            reporter.write().set_progress(layer_idx + 1);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            if tie_word_embeddings {
                vb_m.pp("embed_tokens")
            } else {
                vb.pp("lm_head")
            },
            &None,
            &None,
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            cfg: cfg.clone(),
            vocab_size: cfg.vocab_size,
        })
    }

    pub fn embed_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            input_positions,
            kv_caches,
            input_metadata,
            false,
            &None,
            &None,
            false,
        )
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            input_positions,
            kv_caches,
            input_metadata,
            false,
            &None,
            &None,
            true,
        )
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embedded_inputs: bool,
        visual_pos_masks: &Option<Tensor>,
        deepstack_visual_embeds: &Option<Vec<Tensor>>,
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
        let mut xs = if embedded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_forward(input_ids)?
        };

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), (idx, layer)) in
                zip(kv_caches.iter(), self.layers.iter().enumerate())
            {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
                if let (Some(pos_mask), Some(deepstacks)) =
                    (visual_pos_masks, deepstack_visual_embeds)
                {
                    if idx < deepstacks.len() {
                        xs = xs.apply_deep_stack(pos_mask, &deepstacks[idx])?;
                    }
                }
            }
        } else {
            for (idx, layer) in self.layers.iter().enumerate() {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?;
                if let (Some(pos_mask), Some(deepstacks)) =
                    (visual_pos_masks, deepstack_visual_embeds)
                {
                    if idx < deepstacks.len() {
                        xs = xs.apply_deep_stack(pos_mask, &deepstacks[idx])?;
                    }
                }
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
        self.lm_head.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn forward_with_deepstack(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embedded_inputs: bool,
        visual_pos_masks: &Option<Tensor>,
        deepstack_visual_embeds: &Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            input_positions,
            kv_caches,
            input_metadata,
            embedded_inputs,
            visual_pos_masks,
            deepstack_visual_embeds,
            false,
        )
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
