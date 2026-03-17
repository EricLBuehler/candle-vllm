use std::rc::Rc;
use std::sync::Arc;

mod config;
mod vision;

use self::config::Mistral3Config;
use self::vision::VisionModel;
use crate::backend::progress::ProgressReporter;
use crate::openai::distributed::{Comm, ReplicatedLinear, VarBuilder};
use crate::openai::models::layers::{
    others::{rms_norm, NormX},
    VarBuilderX,
};
use crate::openai::models::{mistral::Mistral, Config};
use crate::openai::multimodal::ImageData;
use crate::InputMetadata;
use attention_rs::ops::NonZeroOp;
use attention_rs::ops::SplitOp;
use candle_core::{DType, Device, Result, Tensor, D};
use parking_lot::RwLock;

struct PatchMerger {
    merge: ReplicatedLinear,
    spatial_merge_size: usize,
    patch_size: usize,
}

impl PatchMerger {
    fn new(cfg: &Mistral3Config, vb: VarBuilderX) -> Result<Self> {
        Ok(Self {
            merge: ReplicatedLinear::load_no_bias(
                cfg.vision_config.hidden_size * cfg.spatial_merge_size.pow(2),
                cfg.vision_config.hidden_size,
                vb.pp("merging_layer"),
                &None,
                &None,
            )?,
            spatial_merge_size: cfg.spatial_merge_size,
            patch_size: cfg.vision_config.patch_size,
        })
    }

    fn forward(&self, image_features: &Tensor, image_sizes: Vec<(usize, usize)>) -> Result<Tensor> {
        let image_sizes = image_sizes
            .iter()
            .map(|&(h, w)| (h / self.patch_size, w / self.patch_size))
            .collect::<Vec<_>>();
        let tokens_per_image = image_sizes.iter().map(|&(h, w)| h * w).collect::<Vec<_>>();
        let d = image_features.dim(D::Minus1)?;
        let mut permuted_tensor = Vec::new();

        for (image_index, image_tokens) in image_features
            .split(&tokens_per_image, 0)?
            .iter()
            .enumerate()
        {
            let (h, w) = image_sizes[image_index];
            let image_grid = image_tokens
                .reshape((h, w, d))?
                .permute((2, 0, 1))?
                .unsqueeze(0)?;
            let patches = image_grid
                .unfold(2, self.spatial_merge_size, self.spatial_merge_size)?
                .unfold(3, self.spatial_merge_size, self.spatial_merge_size)?
                .permute((0, 1, 4, 5, 2, 3))?;
            let grid = patches
                .contiguous()?
                .reshape((1, d * self.spatial_merge_size * self.spatial_merge_size, ()))?
                .reshape((d * self.spatial_merge_size.pow(2), ()))?
                .t()?;
            permuted_tensor.push(grid);
        }

        self.merge.forward(&Tensor::cat(
            &permuted_tensor.iter().collect::<Vec<_>>(),
            0,
        )?)
    }
}

struct MultiModalProjector {
    norm: NormX,
    linear_1: ReplicatedLinear,
    linear_2: ReplicatedLinear,
    act: candle_nn::Activation,
    patch_merger: PatchMerger,
}

impl MultiModalProjector {
    fn new(cfg: &Mistral3Config, vb: VarBuilderX, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm: rms_norm(
                cfg.vision_config.hidden_size,
                cfg.text_config.rms_norm_eps,
                vb.pp("norm"),
                dtype,
                false,
            )?,
            linear_1: ReplicatedLinear::load_b(
                cfg.vision_config.hidden_size,
                cfg.text_config.hidden_size,
                cfg.multimodal_projector_bias,
                vb.pp("linear_1"),
                &cfg.text_config.isq_quant,
                &cfg.text_config.quantization_config,
            )?,
            linear_2: ReplicatedLinear::load_b(
                cfg.text_config.hidden_size,
                cfg.text_config.hidden_size,
                cfg.multimodal_projector_bias,
                vb.pp("linear_2"),
                &cfg.text_config.isq_quant,
                &cfg.text_config.quantization_config,
            )?,
            act: cfg.projector_hidden_act,
            patch_merger: PatchMerger::new(cfg, vb.pp("patch_merger"))?,
        })
    }

    fn forward(&self, image_features: &Tensor, image_sizes: Vec<(usize, usize)>) -> Result<Tensor> {
        let hidden_states = self
            .patch_merger
            .forward(&self.norm.forward(image_features)?, image_sizes)?;
        self.linear_2
            .forward(&self.linear_1.forward(&hidden_states)?.apply(&self.act)?)
    }
}

pub struct Mistral3ForConditionalGeneration {
    text_model: Mistral,
    vision_model: VisionModel,
    mmproj: MultiModalProjector,
    cfg: Mistral3Config,
}

impl Mistral3ForConditionalGeneration {
    pub fn new(
        vb: VarBuilder,
        config: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        assert!(
            config.extra_config_json.is_some(),
            "Invalid multimodal config file"
        );
        let mut cfg: Mistral3Config =
            serde_json::from_str(config.extra_config_json.as_ref().unwrap())
                .map_err(candle_core::Error::wrap)?;
        cfg.text_config = config.clone();

        let vision_model = VisionModel::new(
            &cfg.vision_config,
            vb.pp("vision_tower"),
            comm.clone(),
            dtype,
        )?;
        let mmproj = MultiModalProjector::new(&cfg, vb.pp("multi_modal_projector"), dtype)?;
        let text_model = Mistral::new_with_prefix(
            vb,
            &cfg.text_config,
            dtype,
            device,
            comm,
            progress_reporter,
            Some("language_model".to_string()),
        )?;

        Ok(Self {
            text_model,
            vision_model,
            mmproj,
            cfg,
        })
    }

    fn vision_tower(
        &self,
        image_features: &Tensor,
        image_sizes: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let image_outputs = self
            .vision_model
            .forward(image_features, image_sizes.clone())?;
        self.mmproj.forward(&image_outputs.squeeze(0)?, image_sizes)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
    ) -> Result<Tensor> {
        let (mut input_embeds, dtype) = (
            self.text_model.embed_forward(input_ids)?,
            self.text_model.dtype(),
        );

        if let Some(images) = images {
            let mut image_tensor = images.to_tensor_f32(&input_ids.device())?.to_dtype(dtype)?;
            let image_mask = input_ids.eq(self.cfg.image_token_index as u32)?;
            let image_mask = image_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;

            let indices = image_mask.flatten_all()?.nonzero()?.squeeze(1)?;
            if indices.dim(0)? > 0 {
                let mut image_sizes = images.patches.clone();
                let num_images = image_tensor.dim(0)?;
                if images.image_idx > 0 && (images.image_idx as usize) < num_images {
                    image_tensor = image_tensor.narrow(
                        0,
                        images.image_idx as usize,
                        num_images - images.image_idx as usize,
                    )?;
                    image_sizes = image_sizes[images.image_idx as usize..].to_vec();
                }

                let image_features = self
                    .vision_tower(&image_tensor, image_sizes)?
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

        self.text_model
            .forward_embeds(&input_embeds, positions, kv_caches, input_metadata)
    }

    pub fn get_config(&self) -> &Config {
        self.text_model.get_config()
    }
}
