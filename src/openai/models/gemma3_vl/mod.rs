mod config;

use std::rc::Rc;
use std::sync::Arc;

use self::config::{Gemma3VLConfig, VisionConfig};
use crate::backend::progress::ProgressReporter;
use crate::openai::distributed::{Comm, ReplicatedLinear, VarBuilder};
use crate::openai::models::gemma3::Gemma3;
use crate::openai::models::layers::others::{conv2d, layer_norm, rms_norm, AvgPool2d, NormX};
use crate::openai::models::Config;
use crate::openai::multimodal::ImageData;
use crate::InputMetadata;
use attention_rs::ops::NonZeroOp;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, Conv2d, Conv2dConfig, Embedding};
use parking_lot::RwLock;

struct VisionEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: Embedding,
    num_patches: usize,
}

impl VisionEmbeddings {
    fn new(vb: VarBuilder, cfg: &VisionConfig, dtype: DType) -> Result<Self> {
        let patch_embedding = conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
            true,
        )?;
        let num_patches = (cfg.image_size / cfg.patch_size).pow(2);
        let position_embedding = Embedding::new(
            vb.pp("position_embedding")
                .get((num_patches, cfg.hidden_size), "weight")?,
            cfg.hidden_size,
        );
        let _ = dtype;

        Ok(Self {
            patch_embedding,
            position_embedding,
            num_patches,
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;
        let (bsz, _, _, _) = patch_embeds.dims4()?;
        let patch_embeds = patch_embeds.flatten_from(2)?.transpose(1, 2)?;
        let position_ids = Tensor::arange(0u32, self.num_patches as u32, pixel_values.device())?
            .unsqueeze(0)?
            .broadcast_as((bsz, self.num_patches))?;
        let position_embeds = self.position_embedding.forward(&position_ids)?;
        patch_embeds + position_embeds
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
    fn new(vb: VarBuilder, cfg: &VisionConfig) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q_proj: ReplicatedLinear::load_b(
                cfg.hidden_size,
                cfg.hidden_size,
                true,
                vb.pp("q_proj"),
                &None,
                &None,
            )?,
            k_proj: ReplicatedLinear::load_b(
                cfg.hidden_size,
                cfg.hidden_size,
                true,
                vb.pp("k_proj"),
                &None,
                &None,
            )?,
            v_proj: ReplicatedLinear::load_b(
                cfg.hidden_size,
                cfg.hidden_size,
                true,
                vb.pp("v_proj"),
                &None,
                &None,
            )?,
            o_proj: ReplicatedLinear::load_b(
                cfg.hidden_size,
                cfg.hidden_size,
                true,
                vb.pp("out_proj"),
                &None,
                &None,
            )?,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
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

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
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
    fc1: ReplicatedLinear,
    fc2: ReplicatedLinear,
    act: Activation,
}

impl VisionMlp {
    fn new(vb: VarBuilder, cfg: &VisionConfig) -> Result<Self> {
        Ok(Self {
            fc1: ReplicatedLinear::load_b(
                cfg.hidden_size,
                cfg.intermediate_size,
                true,
                vb.pp("fc1"),
                &None,
                &None,
            )?,
            fc2: ReplicatedLinear::load_b(
                cfg.intermediate_size,
                cfg.hidden_size,
                true,
                vb.pp("fc2"),
                &None,
                &None,
            )?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(xs)?.apply(&self.act)?)
    }
}

struct VisionEncoderLayer {
    self_attn: VisionAttention,
    mlp: VisionMlp,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
}

impl VisionEncoderLayer {
    fn new(vb: VarBuilder, cfg: &VisionConfig, dtype: DType) -> Result<Self> {
        Ok(Self {
            self_attn: VisionAttention::new(vb.pp("self_attn"), cfg)?,
            mlp: VisionMlp::new(vb.pp("mlp"), cfg)?,
            input_layernorm: layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                true,
                vb.pp("layer_norm1"),
                dtype,
            )?,
            post_attention_layernorm: layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                true,
                vb.pp("layer_norm2"),
                dtype,
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.self_attn.forward(&self.input_layernorm.forward(xs)?)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        xs + residual
    }
}

struct VisionTransformer {
    embeddings: VisionEmbeddings,
    layers: Vec<VisionEncoderLayer>,
    norm: NormX,
}

impl VisionTransformer {
    fn new(vb: VarBuilder, cfg: &VisionConfig, dtype: DType) -> Result<Self> {
        let embeddings = VisionEmbeddings::new(vb.pp("embeddings"), cfg, dtype)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(VisionEncoderLayer::new(
                vb.pp(format!("encoder.layers.{idx}")),
                cfg,
                dtype,
            )?);
        }
        let norm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            true,
            vb.pp("post_layernorm"),
            dtype,
        )?;
        Ok(Self {
            embeddings,
            layers,
            norm,
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(pixel_values)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        self.norm.forward(&xs)
    }
}

struct MultiModalProjector {
    projection_weight: Tensor,
    norm: NormX,
    avg_pool: AvgPool2d,
    patches_per_side: usize,
    tokens_per_image: usize,
}

impl MultiModalProjector {
    fn new(vb: VarBuilder, cfg: &Gemma3VLConfig, dtype: DType) -> Result<Self> {
        let projection_weight = vb
            .get(
                (cfg.vision_config.hidden_size, cfg.text_config.hidden_size),
                "mm_input_projection_weight",
            )?
            .to_dtype(dtype)?;
        let norm = rms_norm(
            cfg.vision_config.hidden_size,
            cfg.vision_config.layer_norm_eps,
            vb.pp("mm_soft_emb_norm"),
            dtype,
            true,
        )?;
        let patches_per_side = cfg.vision_config.image_size / cfg.vision_config.patch_size;
        let pooled_side = cfg.mm_tokens_per_image.isqrt();
        if pooled_side == 0 || pooled_side * pooled_side != cfg.mm_tokens_per_image {
            candle_core::bail!(
                "mm_tokens_per_image {} is not a perfect square",
                cfg.mm_tokens_per_image
            );
        }
        if patches_per_side % pooled_side != 0 {
            candle_core::bail!(
                "patch grid {} is not divisible by pooled side {}",
                patches_per_side,
                pooled_side
            );
        }
        let kernel_size = patches_per_side / pooled_side;
        Ok(Self {
            projection_weight,
            norm,
            avg_pool: AvgPool2d::new(kernel_size, kernel_size),
            patches_per_side,
            tokens_per_image: cfg.mm_tokens_per_image,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bsz, _, hidden_dim) = xs.dims3()?;
        let mut xs = xs.transpose(1, 2)?;
        xs = xs
            .reshape((
                bsz,
                hidden_dim,
                self.patches_per_side,
                self.patches_per_side,
            ))?
            .contiguous()?;
        xs = self.avg_pool.forward(&xs)?;
        xs = xs.flatten_from(2)?.transpose(1, 2)?;
        xs = self.norm.forward(&xs)?;
        let xs = xs.matmul(&self.projection_weight)?;
        xs.reshape((bsz * self.tokens_per_image, self.projection_weight.dim(1)?))
    }
}

pub struct Gemma3ForConditionalGeneration {
    text_model: Gemma3,
    vision_tower: Option<VisionTransformer>,
    multi_modal_projector: Option<MultiModalProjector>,
    image_token_index: usize,
    cfg: Config,
}

impl Gemma3ForConditionalGeneration {
    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        assert!(
            cfg.extra_config_json.is_some(),
            "Invalid multimodal config file"
        );
        let mut mm_cfg: Gemma3VLConfig =
            serde_json::from_str(cfg.extra_config_json.as_ref().unwrap())
                .map_err(candle_core::Error::wrap)?;
        mm_cfg.text_config = cfg.clone();

        let vision_tower = mm_cfg
            .has_vision
            .then(|| {
                VisionTransformer::new(
                    vb.pp("vision_tower.vision_model"),
                    &mm_cfg.vision_config,
                    dtype,
                )
            })
            .transpose()?;
        let multi_modal_projector = mm_cfg
            .has_vision
            .then(|| MultiModalProjector::new(vb.pp("multi_modal_projector"), &mm_cfg, dtype))
            .transpose()?;
        let text_model = Gemma3::new(vb, cfg, dtype, device, comm, progress_reporter)?;

        Ok(Self {
            text_model,
            vision_tower,
            multi_modal_projector,
            image_token_index: mm_cfg.image_token_index,
            cfg: cfg.clone(),
        })
    }

    fn vision_tower(&self, image_features: &Tensor) -> Result<Tensor> {
        let image_outputs = self
            .vision_tower
            .as_ref()
            .expect("vision tower missing for Gemma3-VL")
            .forward(image_features)?;
        self.multi_modal_projector
            .as_ref()
            .expect("multimodal projector missing for Gemma3-VL")
            .forward(&image_outputs)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let (mut input_embeds, dtype) = (
            self.text_model.embed_forward(input_ids)?,
            self.text_model.dtype(),
        );

        if let Some(images) = images {
            let mut image_tensor = images.to_tensor_f32(&input_ids.device())?.to_dtype(dtype)?;
            let image_mask = input_ids.eq(self.image_token_index as u32)?;
            let image_mask = image_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;

            let indices = image_mask.flatten_all()?.nonzero()?.squeeze(1)?;
            if indices.dim(0)? > 0 {
                let num_images = image_tensor.dim(0)?;
                if images.image_idx > 0 && (images.image_idx as usize) < num_images {
                    image_tensor = image_tensor.narrow(
                        0,
                        images.image_idx as usize,
                        num_images - images.image_idx as usize,
                    )?;
                }

                let image_features = self
                    .vision_tower(&image_tensor)?
                    .to_dtype(input_embeds.dtype())?;

                let hidden = input_embeds.dim(D::Minus1)?;
                let indices_len = indices.dim(0)?;
                if indices_len % hidden != 0 {
                    candle_core::bail!(
                        "image indices length {} not divisible by hidden size {}",
                        indices_len,
                        hidden
                    );
                }

                let tokens_in_chunk = indices_len / hidden;
                let total_tokens = image_features.dim(0)?;
                let start = images.image_token_offset.min(total_tokens);
                let end = start + tokens_in_chunk;
                if end > total_tokens {
                    candle_core::bail!(
                        "image token slice out of range: start {}, len {}, total {}",
                        start,
                        tokens_in_chunk,
                        total_tokens
                    );
                }

                let image_features = if start > 0 || end < total_tokens {
                    image_features.narrow(0, start, tokens_in_chunk)?
                } else {
                    image_features
                };

                let mut x_flat = input_embeds.flatten_all()?;
                let image_flat = image_features.flatten_all()?;
                x_flat = x_flat.scatter_add(
                    &indices,
                    &(image_flat - x_flat.gather(&indices, 0)?)?,
                    0,
                )?;
                input_embeds = x_flat.reshape(input_embeds.shape())?;
            }
        }

        if return_hidden {
            self.text_model.forward_embeds_hidden(
                &input_embeds,
                positions,
                kv_caches,
                input_metadata,
            )
        } else {
            self.text_model
                .forward_embeds(&input_embeds, positions, kv_caches, input_metadata)
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            images,
            false,
        )
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            images,
            true,
        )
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
