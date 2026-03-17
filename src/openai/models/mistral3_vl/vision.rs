use super::config::VisionConfig;
use crate::openai::distributed::{Comm, ReplicatedLinear};
use crate::openai::models::layers::{
    others::{conv2d, rms_norm, NormX},
    VarBuilderX,
};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use std::rc::Rc;

struct VisionRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl VisionRotaryEmbedding {
    fn new(cfg: &VisionConfig, device: &Device, dtype: DType) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_patches_per_side = cfg.image_size / cfg.patch_size;
        let theta: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(
            0,
            (max_patches_per_side * max_patches_per_side) as u32,
            device,
        )?
        .to_dtype(DType::F32)?
        .reshape((max_patches_per_side * max_patches_per_side, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self {
            cos: idx_theta.cos()?.to_dtype(dtype)?,
            sin: idx_theta.sin()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        use attention_rs::fused_rope::FusedRope;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        FusedRope::apply_inplace(&q, &k, &self.cos, &self.sin, positions, false)?;
        Ok((q, k))
    }
}

struct VisionAttention {
    q_proj: ReplicatedLinear,
    k_proj: ReplicatedLinear,
    v_proj: ReplicatedLinear,
    o_proj: ReplicatedLinear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilderX) -> Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            q_proj: ReplicatedLinear::load_no_bias(dim, dim, vb.pp("q_proj"), &None, &None)?,
            k_proj: ReplicatedLinear::load_no_bias(dim, dim, vb.pp("k_proj"), &None, &None)?,
            v_proj: ReplicatedLinear::load_no_bias(dim, dim, vb.pp("v_proj"), &None, &None)?,
            o_proj: ReplicatedLinear::load_no_bias(dim, dim, vb.pp("o_proj"), &None, &None)?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        positions: &Tensor,
        rotary: &VisionRotaryEmbedding,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;

        let mut q = q.transpose(1, 2)?.contiguous()?;
        let mut k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let mut q_batches = Vec::with_capacity(bsz);
        let mut k_batches = Vec::with_capacity(bsz);
        for batch_idx in 0..bsz {
            let q_rot = q.i(batch_idx)?.transpose(0, 1)?;
            let k_rot = k.i(batch_idx)?.transpose(0, 1)?;
            let (q_applied, k_applied) = rotary.apply(&q_rot, &k_rot, positions)?;
            q_batches.push(q_applied.transpose(0, 1)?.unsqueeze(0)?);
            k_batches.push(k_applied.transpose(0, 1)?.unsqueeze(0)?);
        }
        q = Tensor::cat(&q_batches.iter().collect::<Vec<_>>(), 0)?.contiguous()?;
        k = Tensor::cat(&k_batches.iter().collect::<Vec<_>>(), 0)?.contiguous()?;

        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = if let Some(mask) = mask {
            attn.broadcast_add(mask)?
        } else {
            attn
        };
        let attn =
            candle_nn::ops::softmax_last_dim(&attn.to_dtype(DType::F32)?)?.to_dtype(xs.dtype())?;
        let y = attn.matmul(&v)?.transpose(1, 2)?.reshape((
            bsz,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&y)
    }
}

struct VisionMlp {
    gate_proj: ReplicatedLinear,
    up_proj: ReplicatedLinear,
    down_proj: ReplicatedLinear,
    act: candle_nn::Activation,
}

impl VisionMlp {
    fn new(cfg: &VisionConfig, vb: VarBuilderX) -> Result<Self> {
        Ok(Self {
            gate_proj: ReplicatedLinear::load_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
                &None,
                &None,
            )?,
            up_proj: ReplicatedLinear::load_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
                &None,
                &None,
            )?,
            down_proj: ReplicatedLinear::load_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
                &None,
                &None,
            )?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.apply(&self.act)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

struct AttentionLayer {
    norm: NormX,
    mlp: VisionMlp,
    attention: VisionAttention,
    post_norm: NormX,
}

impl AttentionLayer {
    fn new(vb: VarBuilderX, cfg: &VisionConfig, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm: rms_norm(cfg.hidden_size, 1e-5, vb.pp("attention_norm"), dtype, false)?,
            mlp: VisionMlp::new(cfg, vb.pp("feed_forward"))?,
            attention: VisionAttention::new(
                cfg.hidden_size,
                cfg.num_attention_heads,
                vb.pp("attention"),
            )?,
            post_norm: rms_norm(cfg.hidden_size, 1e-5, vb.pp("ffn_norm"), dtype, false)?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        positions: &Tensor,
        rotary: &VisionRotaryEmbedding,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self
            .attention
            .forward(&self.norm.forward(xs)?, positions, rotary, mask)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self.mlp.forward(&self.post_norm.forward(&xs)?)?;
        xs + residual
    }
}

struct Transformer {
    layers: Vec<AttentionLayer>,
}

impl Transformer {
    fn new(vb: VarBuilderX, cfg: &VisionConfig, dtype: DType) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(AttentionLayer::new(
                vb.pp(format!("layers.{idx}")),
                cfg,
                dtype,
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(
        &self,
        xs: &Tensor,
        positions: &Tensor,
        rotary: &VisionRotaryEmbedding,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, positions, rotary, mask)?;
        }
        Ok(xs)
    }
}

pub struct VisionModel {
    patch_conv: candle_nn::Conv2d,
    ln_pre: NormX,
    transformer: Transformer,
    rotary: VisionRotaryEmbedding,
    max_image_width: u32,
    patch_size: usize,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig, vb: VarBuilderX, _comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let patch_conv = conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            candle_nn::Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("patch_conv"),
            false,
        )?;
        let ln_pre = rms_norm(cfg.hidden_size, 1e-5, vb.pp("ln_pre"), dtype, false)?;
        let transformer = Transformer::new(vb.pp("transformer"), cfg, dtype)?;
        let rotary = VisionRotaryEmbedding::new(cfg, vb.device(), dtype)?;
        Ok(Self {
            patch_conv,
            ln_pre,
            transformer,
            rotary,
            max_image_width: (cfg.image_size / cfg.patch_size) as u32,
            patch_size: cfg.patch_size,
        })
    }

    fn position_ids_in_meshgrid(
        &self,
        patch_embeds_list: &[Tensor],
        device: &Device,
    ) -> Result<Tensor> {
        let mut positions = Vec::new();
        for patch in patch_embeds_list {
            let (height, width) = (patch.dim(D::Minus2)? as u32, patch.dim(D::Minus1)? as u32);
            let idx = Tensor::arange(0, height, device)?;
            let idy = Tensor::arange(0, width, device)?;
            let mesh = Tensor::meshgrid(&[idx, idy], false)?;
            let ids = (&mesh[0] * (self.max_image_width as f64) + &mesh[1])?.flatten_all()?;
            positions.push(ids);
        }
        Tensor::cat(&positions.iter().collect::<Vec<_>>(), 0)
    }

    fn generate_block_attention_mask(
        &self,
        patch_embeds_list: Vec<usize>,
        patch_embeds: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = patch_embeds.dim(1)?;
        let mut mask = Tensor::ones(
            (seq_len, seq_len),
            patch_embeds.dtype(),
            patch_embeds.device(),
        )?;
        let min_value = f32::MIN as f64;
        mask = (mask * min_value)?;

        let block_end_idx: Vec<usize> = patch_embeds_list.iter().fold(Vec::new(), |mut acc, &x| {
            let new_sum = x + acc.last().copied().unwrap_or(0);
            acc.push(new_sum);
            acc
        });
        let block_start_idx: Vec<usize> = {
            let mut extended = vec![0];
            if !patch_embeds_list.is_empty() {
                extended.extend_from_slice(&patch_embeds_list[..patch_embeds_list.len() - 1]);
            }
            extended.into_iter().fold(Vec::new(), |mut acc, x| {
                let new_sum = x + acc.last().copied().unwrap_or(0);
                acc.push(new_sum);
                acc
            })
        };

        for (start, end) in block_start_idx.into_iter().zip(block_end_idx) {
            mask = mask.slice_assign(
                &[start..end, start..end],
                &Tensor::zeros((end - start, end - start), mask.dtype(), mask.device())?,
            )?;
        }

        mask.reshape((1, 1, seq_len, seq_len))
    }

    pub fn forward(&self, xs: &Tensor, image_sizes: Vec<(usize, usize)>) -> Result<Tensor> {
        let patch_embeds = self.patch_conv.forward(xs)?;
        let patch_embeds_list = image_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let patches_h = size.0 / self.patch_size;
                let patches_w = size.1 / self.patch_size;
                patch_embeds
                    .i(i)?
                    .narrow(D::Minus2, 0, patches_h)?
                    .narrow(D::Minus1, 0, patches_w)
            })
            .collect::<Result<Vec<Tensor>>>()?;
        let patch_embeds = Tensor::cat(
            &patch_embeds_list
                .iter()
                .map(|p| p.flatten_from(1)?.t())
                .collect::<Result<Vec<Tensor>>>()?
                .iter()
                .collect::<Vec<_>>(),
            0,
        )?
        .unsqueeze(0)?;
        let patch_embeds = self.ln_pre.forward(&patch_embeds)?;
        let positions = self.position_ids_in_meshgrid(&patch_embeds_list, patch_embeds.device())?;
        let attention_mask = self.generate_block_attention_mask(
            patch_embeds_list
                .iter()
                .map(|p| Ok(p.dim(D::Minus2)? * p.dim(D::Minus1)?))
                .collect::<Result<Vec<usize>>>()?,
            &patch_embeds,
        )?;
        self.transformer.forward(
            &patch_embeds,
            &positions,
            &self.rotary,
            Some(&attention_mask),
        )
    }
}
