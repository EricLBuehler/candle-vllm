use super::{
    attention::Attention,
    layers::deltanet::GatedDeltaNet,
    resolve_qwen3_hybrid_config,
    rotary_emb::ScalingRotaryEmbedding,
    utils::{
        apply_rms_norm_fp32, resolve_input_seqlens, resolve_mamba_seq_slots, resolve_text_backbone,
    },
    Config, InputMetadata, MoEConfig,
};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{
    embedding, rms_norm_x, Comm, ReplicatedLinear, TensorParallelColumnLinear,
    TensorParallelRowLinear, VarBuilder,
};
use crate::openai::models::layers::moe::{FusedMoe, FusedMoeFp8, FusedMoeISQ};
use crate::openai::models::linear::LinearX as Linear;
use crate::openai::models::mask::get_attention_causal_mask;
use crate::openai::models::QwenMoEConfig;
use attention_rs::mamba_cache::MambaCache;
use candle::{DType, Device, Module, Result, Tensor};
use candle_core as candle;
use candle_nn::RmsNorm;
use parking_lot::{RwLock, RwLockWriteGuard};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

impl Qwen3_5MoE {
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
        config.attention_bias = Some(
            config
                .use_qkv_bias
                .or(config.attention_bias)
                .unwrap_or(false),
        );

        config.isq_quant = if config.quantization_config.is_some() {
            None
        } else {
            isq
        };

        if config.moe_config.is_none() {
            let f = std::fs::read(filename).map_err(candle::Error::wrap)?;
            let mut from_root: Option<QwenMoEConfig> =
                serde_json::from_slice::<QwenMoEConfig>(&f).ok();
            if from_root.is_none() {
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&f) {
                    if let Some(text_config) = v.get("text_config") {
                        from_root =
                            serde_json::from_value::<QwenMoEConfig>(text_config.clone()).ok();
                    }
                }
            }
            if let Some(moe_cfg) = from_root {
                config.moe_config = Some(MoEConfig::QwenMoE(moe_cfg));
            }
        }

        // Upstream behavior: if norm_topk_prob is omitted in Qwen3.5/Qwen3-Next
        // configs, default it to true for stable routing behavior.
        let arch = config
            .architectures
            .as_ref()
            .and_then(|a| a.first())
            .map(|s| s.as_str())
            .unwrap_or("");
        if matches!(
            arch,
            "Qwen3_5MoeForCausalLM"
                | "Qwen3_5MoeForConditionalGeneration"
                | "Qwen3NextForCausalLM"
                | "Qwen3NextForConditionalGeneration"
        ) {
            if let Some(MoEConfig::QwenMoE(moe_cfg)) = config.moe_config.as_mut() {
                if let Some(raw) = config.extra_config_json.as_ref() {
                    if let Ok(root) = serde_json::from_str::<serde_json::Value>(raw) {
                        let cfg_root = root.get("text_config").unwrap_or(&root);
                        if cfg_root.get("norm_topk_prob").is_none() {
                            moe_cfg.norm_topk_prob = true;
                        }
                    }
                }
            }
        }

        Ok(config)
    }
}

struct Mlp {
    gate_proj: TensorParallelColumnLinear,
    up_proj: TensorParallelColumnLinear,
    down_proj: TensorParallelRowLinear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Config, intermediate_size: usize, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;

        let gate_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            intermediate_size,
            false,
            vb.pp("gate_proj"),
            comm.clone(),
            &cfg.isq_quant,
            &cfg.quantization_config,
        )?;
        let up_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            intermediate_size,
            false,
            vb.pp("up_proj"),
            comm.clone(),
            &cfg.isq_quant,
            &cfg.quantization_config,
        )?;
        let down_proj = TensorParallelRowLinear::load_with_hints(
            intermediate_size,
            hidden_sz,
            false,
            vb.pp("down_proj"),
            comm,
            &cfg.isq_quant,
            &cfg.quantization_config,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act.unwrap(),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.act_fn.forward(&self.gate_proj.forward(xs)?)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(&lhs * &rhs)?)
    }
}

enum MoeOrMlp {
    FusedMoe(FusedMoe),
    FusedMoeISQ(FusedMoeISQ),
    FusedMoeFp8(FusedMoeFp8),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
            Self::FusedMoeISQ(m) => m.forward(xs, is_prefill),
            Self::FusedMoeFp8(m) => m.forward(xs, is_prefill),
        }
    }
}

enum AttnType {
    FullAttention(Attention),
    LinearAttention(GatedDeltaNet),
}

struct DecoderLayer {
    attn: AttnType,
    mlp: MoeOrMlp,
    shared_gate: Option<Linear>,
    shared_expert: Option<Mlp>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    dtype: DType,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
        dtype: DType,
        _layer_idx: usize,
        layer_type: &str,
        gdn_layer_idx: usize,
    ) -> Result<Self> {
        let attn = if layer_type == "full_attention" {
            AttnType::FullAttention(Attention::new(
                rotary_emb,
                cfg,
                vb.pp("self_attn"),
                comm.clone(),
                cfg.sliding_window,
            )?)
        } else {
            AttnType::LinearAttention(GatedDeltaNet::new(
                vb.pp("linear_attn"),
                comm.clone(),
                cfg,
                gdn_layer_idx,
            )?)
        };

        let moe_cfg = if let Some(MoEConfig::QwenMoE(moe_cfg)) = &cfg.moe_config {
            moe_cfg.clone()
        } else {
            candle::bail!("Expected QwenMoEConfig")
        };

        let is_fp8_model = if let Some(ref quant_cfg) = cfg.quantization_config {
            quant_cfg.quant_method == "fp8"
        } else {
            false
        };
        // Qwen3.5 MoE / Qwen3-Next are routed-MoE layers; keep model path aligned
        // with upstream and avoid dense-MLP fallback in these architectures.
        let mlp = if is_fp8_model {
            if let Some(ref quant_cfg) = cfg.quantization_config {
                MoeOrMlp::FusedMoeFp8(FusedMoeFp8::new(
                    cfg,
                    vb.pp("mlp").clone(),
                    comm.clone(),
                    dtype,
                    quant_cfg,
                )?)
            } else {
                candle_core::bail!("Missing quantization_config for fp8 model!")
            }
        } else if cfg.isq_quant.is_some() {
            MoeOrMlp::FusedMoeISQ(FusedMoeISQ::new(
                cfg,
                vb.pp("mlp").clone(),
                comm.clone(),
                dtype,
            )?)
        } else {
            MoeOrMlp::FusedMoe(FusedMoe::new(
                cfg,
                vb.pp("mlp").clone(),
                comm.clone(),
                dtype,
            )?)
        };

        let input_layernorm = rms_norm_x(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("input_layernorm"),
            DType::F32,
            true,
        )?;
        let post_attention_layernorm = rms_norm_x(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
            DType::F32,
            true,
        )?;

        let (shared_gate, shared_expert) =
            if let Some(intermediate_size) = moe_cfg.shared_expert_intermediate_size {
                if intermediate_size > 0 {
                    let ws = vb.pp("mlp.shared_expert_gate").get_with_hints_dtype(
                        (1, cfg.hidden_size),
                        "weight",
                        Default::default(),
                        dtype,
                    )?;

                    let shared_gate = Linear::new(ws, None, &None, &None);
                    let shared_expert = Mlp::new(
                        cfg,
                        intermediate_size,
                        vb.pp("mlp.shared_expert").clone(),
                        comm,
                    )?;
                    (Some(shared_gate), Some(shared_expert))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

        Ok(Self {
            attn,
            mlp,
            shared_gate,
            shared_expert,
            input_layernorm,
            post_attention_layernorm,
            dtype,
        })
    }

    fn is_full_attention(&self) -> bool {
        matches!(self.attn, AttnType::FullAttention(_))
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        mamba_cache: &mut MambaCache,
        seq_slots: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = apply_rms_norm_fp32(&self.input_layernorm, xs)?;
        let xs = match &self.attn {
            AttnType::FullAttention(attn) => {
                attn.forward(&xs, attention_mask, input_positions, cache, input_metadata)?
            }
            AttnType::LinearAttention(gdn) => {
                if xs.dtype() != self.dtype {
                    gdn.forward(
                        &xs.to_dtype(self.dtype)?,
                        mamba_cache,
                        input_metadata,
                        seq_slots,
                    )?
                    .to_dtype(xs.dtype())?
                } else {
                    gdn.forward(&xs, mamba_cache, input_metadata, seq_slots)?
                }
            }
        };

        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = apply_rms_norm_fp32(&self.post_attention_layernorm, &xs)?;

        let shared_output = match (&self.shared_gate, &self.shared_expert) {
            (Some(shared_gate), Some(shared_expert)) => {
                let gate = candle_nn::ops::sigmoid(&shared_gate.forward(&xs)?)?;
                let shared_output = shared_expert.forward(&xs)?;
                Some(gate.broadcast_mul(&shared_output)?)
            }
            _ => None,
        };

        let mlp_output = self.mlp.forward(&xs, input_metadata.is_prefill)?;

        if let Some(shared_output) = shared_output {
            residual + (mlp_output + shared_output)?
        } else {
            residual + mlp_output
        }
    }
}

pub struct Qwen3_5MoE {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: ReplicatedLinear,
    mamba_cache: RwLock<MambaCache>,
    device: Device,
    dtype: DType,
    cfg: Config,
}

impl Qwen3_5MoE {
    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let text_backbone = resolve_text_backbone(&vb, cfg.tie_word_embeddings);
        let vb_m = if text_backbone.use_root_builder {
            vb.clone()
        } else {
            vb.pp(text_backbone.prefix)
        };
        let tie_word_embeddings = text_backbone.tie_word_embeddings;
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, cfg, device, true)?);

        let hybrid = resolve_qwen3_hybrid_config(cfg);
        let layer_types = &hybrid.layer_types;

        let mut gdn_layer_idx = 0usize;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let reporter = progress_reporter.clone();

        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_type = layer_types
                .get(layer_idx)
                .map(String::as_str)
                .unwrap_or("full_attention");
            let cur_gdn_idx = if layer_type == "linear_attention" {
                let idx = gdn_layer_idx;
                gdn_layer_idx += 1;
                idx
            } else {
                0
            };

            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                comm.clone(),
                dtype,
                layer_idx,
                layer_type,
                cur_gdn_idx,
            )?;
            layers.push(layer);
            reporter.write().set_progress(layer_idx + 1);
        }

        let norm = rms_norm_x(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_m.pp("norm"),
            DType::F32,
            true,
        )?;
        let lm_head = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            if tie_word_embeddings {
                vb_m.pp("embed_tokens")
            } else if vb_m.contains_tensor("lm_head.weight") {
                vb_m.pp("lm_head")
            } else {
                vb.pp("lm_head")
            },
            &None,
            &None,
        )?;

        let world_size = comm.world_size();
        if hybrid.num_v_heads % world_size != 0 || hybrid.num_k_heads % world_size != 0 {
            candle::bail!(
                "linear attention heads must be divisible by world_size (num_v_heads={}, num_k_heads={}, world_size={})",
                hybrid.num_v_heads,
                hybrid.num_k_heads,
                world_size
            );
        }

        let num_v_heads = hybrid.num_v_heads / world_size;
        let num_k_heads = hybrid.num_k_heads / world_size;
        let d_conv = num_k_heads * hybrid.key_head_dim * 2 + num_v_heads * hybrid.value_head_dim;

        let mamba_cache = if gdn_layer_idx > 0 {
            MambaCache::new(
                gdn_layer_idx,
                1,
                d_conv,
                hybrid.conv_kernel_size,
                num_v_heads,
                hybrid.key_head_dim,
                hybrid.value_head_dim,
                dtype,
                DType::F32,
                device,
            )?
        } else {
            MambaCache::new(0, 1, 1, 2, 1, 1, 1, dtype, DType::F32, device)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            mamba_cache: RwLock::new(mamba_cache),
            device: device.clone(),
            dtype,
            cfg: cfg.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, input_positions, kv_caches, input_metadata, false)
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, input_positions, kv_caches, input_metadata, true)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        return_hidden: bool,
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

        let mut xs = self.embed_tokens.forward(input_ids)?;

        let mut mamba_cache = self.mamba_cache.write();
        let seq_slots = resolve_mamba_seq_slots(
            "Qwen3.5 MoE",
            &self.device,
            input_metadata,
            xs.dim(0)?,
            &mut mamba_cache,
        )?;

        let mut kv_cache_idx = 0usize;
        for layer in self.layers.iter() {
            let cache = if layer.is_full_attention() {
                if let Some(kv_caches) = kv_caches {
                    let c = &kv_caches[kv_cache_idx];
                    kv_cache_idx += 1;
                    Some((&c.0, &c.1))
                } else {
                    None
                }
            } else {
                None
            };

            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                input_positions,
                cache,
                input_metadata,
                &mut mamba_cache,
                &seq_slots,
            )?;
        }

        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }

        let xs = apply_rms_norm_fp32(&self.norm, &xs)?;

        if return_hidden {
            return xs.to_dtype(DType::F32);
        }
        self.lm_head
            .forward(&xs.to_dtype(self.dtype)?)?
            .to_dtype(DType::F32)
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

    pub fn lock_mamba_cache_for_graph(&self) -> RwLockWriteGuard<'_, MambaCache> {
        self.mamba_cache.write()
    }

    pub fn preallocate_mamba_cache(&self, max_num_seqs: usize) -> Result<()> {
        self.mamba_cache.write().reserve_capacity(max_num_seqs)
    }

    pub fn set_mamba_prefix_cache_capacity(&self, capacity: usize) {
        self.mamba_cache.write().set_prefix_cache_capacity(capacity);
    }

    pub fn capture_mamba_prefix_state(&self, seq_id: usize, hash: u64) -> Result<bool> {
        self.mamba_cache.write().capture_prefix_state(seq_id, hash)
    }

    pub fn has_mamba_prefix_state(&self, hash: u64) -> bool {
        self.mamba_cache.read().has_prefix_state(hash)
    }

    pub fn restore_mamba_prefix_state(&self, seq_id: usize, hash: u64) -> Result<bool> {
        self.mamba_cache.write().restore_prefix_state(seq_id, hash)
    }

    pub fn reset_mamba_cache(&self) -> Result<()> {
        self.mamba_cache.write().reset_all()
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
