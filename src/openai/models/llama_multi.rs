use super::{Config, QuantConfig};
use crate::openai::distributed::{
    embedding, linear_no_bias, rms_norm, TensorParallelColumnLinear, TensorParallelRowLinear,
};
use crate::paged_attention::input_metadata::InputMetadata;
use crate::paged_attention::PagedAttention;
use crate::SpecificConfig;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_core as candle;
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{Embedding, Linear, Module, RmsNorm};
pub use cudarc::nccl::safe::{Comm, ReduceOp};
pub const MAX_SEQ_LEN: usize = 4096;
use crate::openai::models::TokenID;
use std::iter::zip;
use std::rc::Rc;
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: TokenID,
    pub eos_token_id: TokenID,
    pub max_position_embeddings: Option<usize>,
    pub quantization_config: Option<QuantConfig>,
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(
        self,
        use_flash_attn: bool,
        kv_cache_dtype: DType,
        scfg: &SpecificConfig,
    ) -> Config {
        Config {
            hidden_size: self.hidden_size,
            head_dim: Some(self.hidden_size / self.num_attention_heads),
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: f64::from(self.rope_theta),
            use_flash_attn,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            max_seq_len: self.max_position_embeddings.unwrap_or(MAX_SEQ_LEN),
            sliding_window: None,
            hidden_act: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            original_max_position_embeddings: None,
            attention_bias: false,
            partial_rotary_factor: None,
            qk_layer_rms_norm: None,
            kv_cache_dtype,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            specific_config: scfg.clone(),
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: self.quantization_config,
            moe_config: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    cos: Tensor,
    sin: Tensor,
}

impl Cache {
    pub fn new(dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let n_elem = config.hidden_size / config.num_attention_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f64 / n_elem as f64) as f32)
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, config.max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }
}

struct CausalSelfAttention {
    qkv_proj: TensorParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    attn: PagedAttention,
    cos_sin_cache: Cache,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, input_positions: &[Vec<usize>]) -> Result<Tensor> {
        let (b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let mut embeds = Vec::new();
        for (b, seqlen_offset) in zip(0..b_sz, input_positions) {
            let cos = self
                .cos_sin_cache
                .cos
                .narrow(0, seqlen_offset[0], seq_len)?;
            let sin = self
                .cos_sin_cache
                .sin
                .narrow(0, seqlen_offset[0], seq_len)?;
            let x_b = x.narrow(0, b, 1)?;
            let embed = candle_nn::rotary_emb::rope(&x_b, &cos, &sin).unwrap();
            embeds.push(embed);
        }
        Tensor::cat(&embeds, 0)
    }

    fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        input_positions: &[Vec<usize>],
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let qkv = self.qkv_proj.forward(x)?;
        let hidden_size = self.num_attention_heads * self.head_dim;

        let q = qkv.i((.., .., ..self.num_attention_heads * self.head_dim))?;
        let k = qkv.i((
            ..,
            ..,
            self.num_attention_heads * self.head_dim
                ..self.num_attention_heads * self.head_dim
                    + self.num_key_value_heads * self.head_dim,
        ))?;
        let v = qkv.i((
            ..,
            ..,
            self.num_attention_heads * self.head_dim + self.num_key_value_heads * self.head_dim..,
        ))?;

        let (q, k, v) = if seq_len == 1 {
            //no need transpose for seq_len == 1, change reshape dim
            let q = q.reshape((b_sz, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            (q, k, v)
        } else {
            let q = q
                .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = v
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v.contiguous()?)
        };

        let q = self.apply_rotary_emb(&q, input_positions)?;
        let k = self.apply_rotary_emb(&k, input_positions)?;

        let y = self.attn.forward(
            &q,
            &k,
            &v,
            attention_mask,
            cache.map(|(k_, _)| k_.clone()),
            cache.map(|(_, v_)| v_.clone()),
            input_metadata,
            None,
        )?;

        let y = if attention_mask.is_some() {
            y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?
        } else {
            y.reshape(&[b_sz, seq_len, hidden_size])?
        };
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let qkv_proj = TensorParallelColumnLinear::load_multi(
            vb.clone(),
            &["q_proj", "k_proj", "v_proj"],
            comm.clone(),
        )?;
        let o_proj = TensorParallelRowLinear::load(vb.pp("o_proj"), comm.clone())?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let attention_heads = cfg.num_attention_heads / comm.world_size();
        let kv_heads = cfg.num_key_value_heads / comm.world_size();
        Ok(Self {
            qkv_proj,
            o_proj,
            num_attention_heads: attention_heads,
            num_key_value_heads: kv_heads,
            head_dim,
            attn: PagedAttention::new(
                attention_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(kv_heads),
                None,
                vb.device().clone(),
                None,
            )?,
            cos_sin_cache: Cache::new(dtype, cfg, device)?,
        })
    }
}

struct Mlp {
    c_fc1: TensorParallelColumnLinear,
    c_fc2: TensorParallelColumnLinear,
    c_proj: TensorParallelRowLinear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let c_fc1 = TensorParallelColumnLinear::load(vb.pp("gate_proj"), comm.clone())?;
        let c_fc2 = TensorParallelColumnLinear::load(vb.pp("up_proj"), comm.clone())?;
        let c_proj = TensorParallelRowLinear::load(vb.pp("down_proj"), comm)?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        input_positions: &[Vec<usize>],
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
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg, dtype, device, comm.clone())?;
        let mlp = Mlp::load(vb.pp("mlp"), comm.clone())?;
        let rms_1 = rms_norm(cfg.hidden_size, 1e-5, vb.pp("input_layernorm"))?;
        let rms_2 = rms_norm(cfg.hidden_size, 1e-5, vb.pp("post_attention_layernorm"))?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }
}

pub struct LlamaMulti {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    cfg: Config,
    dtype: DType,
    device: Device,
}

impl LlamaMulti {
    fn prepare_decoder_attention_mask(&self, b_size: usize, tgt_len: usize) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        mask.expand((b_size, 1, tgt_len, tgt_len))?
            .contiguous()?
            .to_dtype(self.dtype)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        input_positions: &[Vec<usize>],
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(_b_sz, seq_len)?;
            Some(mask)
        };
        let mut x = self.wte.forward(x)?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), block) in zip(kv_caches.iter(), &self.blocks) {
                x = block.forward(
                    &x,
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
            }
        } else {
            for block in &self.blocks {
                x = block.forward(
                    &x,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?;
            }
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let ln_f = rms_norm(cfg.hidden_size, 1e-5, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| {
                Block::load(
                    vb.pp(&format!("model.layers.{i}")),
                    cfg,
                    dtype,
                    device,
                    comm.clone(),
                )
                .unwrap()
            })
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
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
