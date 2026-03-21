use parking_lot::{RwLock, RwLockWriteGuard};
use std::rc::Rc;
use std::sync::Arc;

pub mod config;
pub mod input;
pub mod vision;

use crate::backend::progress::ProgressReporter;
use crate::openai::distributed::{Comm, VarBuilder};
use crate::openai::models::{
    qwen::Qwen, qwen3_5::Qwen3_5, qwen3_5_moe::Qwen3_5MoE, qwen3_moe::Qwen3MoE, Config,
};
use crate::openai::multimodal::ImageData;
use crate::InputMetadata;
use attention_rs::mamba_cache::MambaCache;
use candle_core::{DType, Device, Result, Tensor, D};
use config::Qwen3VLConfig;
use vision::Qwen3VLVisionModel;

pub enum Qwen3TextModel {
    Dense(Qwen),
    MoE(Qwen3MoE),
    Dense35(Qwen3_5),
    MoE35(Qwen3_5MoE),
}

pub struct Qwen3VLForConditionalGeneration {
    text_model: Qwen3TextModel,
    vision_model: Qwen3VLVisionModel,
    image_token_id: u32,
    cfg: Config,
}

impl Qwen3VLForConditionalGeneration {
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
        let mut mm_cfg: Qwen3VLConfig =
            serde_json::from_str(cfg.extra_config_json.as_ref().unwrap())
                .map_err(candle_core::Error::wrap)?;
        mm_cfg.text_config = cfg.clone();
        if mm_cfg.quantization_config.is_some() {
            mm_cfg.text_config.quantization_config = mm_cfg.quantization_config.clone();
        }

        let vision_model =
            Qwen3VLVisionModel::new(&mm_cfg.vision_config, &vb.pp("model.visual"), dtype, device)?;

        let arch = mm_cfg
            .architectures
            .clone()
            .unwrap_or(vec!["Qwen3VLForConditionalGeneration".to_string()]);
        let arch = arch[0].as_str();
        let next_is_moe = matches!(
            cfg.moe_config.as_ref(),
            Some(crate::openai::models::MoEConfig::QwenMoE(moe_cfg))
                if moe_cfg.num_experts.unwrap_or(0) > 0
        );

        let text_model = match arch {
            "Qwen3VLMoeForConditionalGeneration" => Qwen3TextModel::MoE(Qwen3MoE::new_with_prefix(
                vb.clone(),
                cfg,
                dtype,
                device,
                comm.clone(),
                progress_reporter.clone(),
                Some("model.language_model".to_string()),
            )?),
            "Qwen3_5MoeForConditionalGeneration" => {
                Qwen3TextModel::MoE35(Qwen3_5MoE::new_with_prefix(
                    vb.clone(),
                    cfg,
                    dtype,
                    device,
                    comm.clone(),
                    progress_reporter.clone(),
                    Some("model.language_model".to_string()),
                )?)
            }
            "Qwen3_5ForConditionalGeneration" => Qwen3TextModel::Dense35(Qwen3_5::new_with_prefix(
                vb.clone(),
                cfg,
                dtype,
                device,
                comm.clone(),
                progress_reporter.clone(),
                Some("model.language_model".to_string()),
            )?),
            "Qwen3NextForConditionalGeneration" if next_is_moe => {
                Qwen3TextModel::MoE35(Qwen3_5MoE::new_with_prefix(
                    vb.clone(),
                    cfg,
                    dtype,
                    device,
                    comm.clone(),
                    progress_reporter.clone(),
                    Some("model.language_model".to_string()),
                )?)
            }
            "Qwen3NextForConditionalGeneration" => {
                Qwen3TextModel::Dense35(Qwen3_5::new_with_prefix(
                    vb.clone(),
                    cfg,
                    dtype,
                    device,
                    comm.clone(),
                    progress_reporter.clone(),
                    Some("model.language_model".to_string()),
                )?)
            }
            _ => Qwen3TextModel::Dense(Qwen::new_with_prefix(
                vb,
                cfg,
                dtype,
                device,
                comm,
                progress_reporter,
                Some("model.language_model".to_string()),
            )?),
        };

        Ok(Self {
            text_model,
            vision_model,
            image_token_id: mm_cfg.image_token_id,
            cfg: cfg.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
    ) -> Result<Tensor> {
        let (mut input_embeds, dtype) = match &self.text_model {
            Qwen3TextModel::Dense(m) => (m.embed_forward(input_ids)?, m.dtype()),
            Qwen3TextModel::MoE(m) => (m.embed_forward(input_ids)?, m.dtype()),
            Qwen3TextModel::Dense35(m) => (m.embed_forward(input_ids)?, m.dtype()),
            Qwen3TextModel::MoE35(m) => (m.embed_forward(input_ids)?, m.dtype()),
        };
        let device = input_embeds.device().clone();
        let mut visual_pos_masks: Option<Tensor> = None;
        let mut deepstack_visual_embeds: Option<Vec<Tensor>> = None;

        if let Some(images) = images {
            let mut pixel_values = images.to_tensor_f32(&device)?.to_dtype(dtype)?;
            let mut patches = Vec::new();
            for (h, w) in &images.patches {
                patches.extend(vec![1, *h as u32, *w as u32]);
            }
            let mut image_grid_thw = Tensor::from_vec(patches, (images.patches.len(), 3), &device)?;
            let num_images = pixel_values.dim(0)?;
            if images.image_idx > 0 && (images.image_idx as usize) < num_images {
                pixel_values = pixel_values.narrow(
                    0,
                    images.image_idx as usize,
                    num_images - images.image_idx as usize,
                )?;
                image_grid_thw = image_grid_thw.narrow(
                    0,
                    images.image_idx as usize,
                    num_images - images.image_idx as usize,
                )?;
            }

            if pixel_values.rank() == 3 {
                let dims = pixel_values.dims();
                pixel_values = pixel_values.reshape((dims[0] * dims[1], dims[2]))?;
            }

            let (image_embeds, deepstack_image_embeds) =
                self.vision_model.forward(&pixel_values, &image_grid_thw)?;
            let image_embeds = image_embeds
                .to_device(&device)?
                .to_dtype(input_embeds.dtype())?;
            let deepstack_image_embeds = deepstack_image_embeds
                .into_iter()
                .map(|t| t.to_device(&device)?.to_dtype(input_embeds.dtype()))
                .collect::<Result<Vec<_>>>()?;

            let image_mask = input_ids.eq(self.image_token_id)?;
            visual_pos_masks = Some(image_mask.to_dtype(DType::U8)?);
            let image_mask = image_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;

            use attention_rs::ops::NonZeroOp;
            let indices = image_mask.flatten_all()?.nonzero()?.squeeze(1)?;
            if indices.dim(0)? > 0 {
                let hidden = input_embeds.dim(D::Minus1)?;
                let indices_len = indices.dim(0)?;
                let tokens_in_chunk = indices_len / hidden;
                let total_tokens = image_embeds.dim(0)?;
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

                let image_embeds = if start > 0 || end < total_tokens {
                    image_embeds.narrow(0, start, tokens_in_chunk)?
                } else {
                    image_embeds
                };
                let deepstack_image_embeds = deepstack_image_embeds
                    .into_iter()
                    .map(|t| {
                        if start > 0 || end < total_tokens {
                            t.narrow(0, start, tokens_in_chunk)
                        } else {
                            Ok(t)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                let mut x_flat = input_embeds.flatten_all()?;
                let image_flat = image_embeds.flatten_all()?;
                x_flat = x_flat.scatter_add(
                    &indices,
                    &(image_flat - x_flat.gather(&indices, 0)?)?,
                    0,
                )?;
                input_embeds = x_flat.reshape(input_embeds.shape())?;
                deepstack_visual_embeds = Some(deepstack_image_embeds);
            }
        }

        match &self.text_model {
            Qwen3TextModel::Dense(m) => m.forward_with_deepstack(
                &input_embeds,
                positions,
                kv_caches,
                input_metadata,
                true,
                &visual_pos_masks,
                &deepstack_visual_embeds,
            ),
            Qwen3TextModel::MoE(m) => m.forward_with_deepstack(
                &input_embeds,
                positions,
                kv_caches,
                input_metadata,
                true,
                &visual_pos_masks,
                &deepstack_visual_embeds,
            ),
            Qwen3TextModel::Dense35(m) => m.forward_with_deepstack(
                &input_embeds,
                positions,
                kv_caches,
                input_metadata,
                true,
                &visual_pos_masks,
                &deepstack_visual_embeds,
            ),
            Qwen3TextModel::MoE35(m) => m.forward_with_deepstack(
                &input_embeds,
                positions,
                kv_caches,
                input_metadata,
                true,
                &visual_pos_masks,
                &deepstack_visual_embeds,
            ),
        }
    }

    pub fn uses_hybrid_mamba_text_model(&self) -> bool {
        matches!(
            self.text_model,
            Qwen3TextModel::Dense35(_) | Qwen3TextModel::MoE35(_)
        )
    }

    pub fn release_sequence_state(&self, sequence_id: usize) {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.release_sequence_state(sequence_id),
            Qwen3TextModel::MoE35(m) => m.release_sequence_state(sequence_id),
            _ => {}
        }
    }

    pub fn ensure_mamba_slots_for_sequences(&self, sequence_ids: &[usize]) -> Result<Vec<usize>> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.ensure_mamba_slots_for_sequences(sequence_ids),
            Qwen3TextModel::MoE35(m) => m.ensure_mamba_slots_for_sequences(sequence_ids),
            _ => Ok(vec![]),
        }
    }

    pub fn get_mamba_slots_for_sequences(&self, sequence_ids: &[usize]) -> Result<Vec<usize>> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.get_mamba_slots_for_sequences(sequence_ids),
            Qwen3TextModel::MoE35(m) => m.get_mamba_slots_for_sequences(sequence_ids),
            _ => Ok(vec![]),
        }
    }

    pub fn has_mamba_slot_for_sequence(&self, sequence_id: usize) -> bool {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.has_mamba_slot_for_sequence(sequence_id),
            Qwen3TextModel::MoE35(m) => m.has_mamba_slot_for_sequence(sequence_id),
            _ => false,
        }
    }

    pub fn lock_mamba_cache_for_graph(&self) -> Option<RwLockWriteGuard<'_, MambaCache>> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => Some(m.lock_mamba_cache_for_graph()),
            Qwen3TextModel::MoE35(m) => Some(m.lock_mamba_cache_for_graph()),
            _ => None,
        }
    }

    pub fn preallocate_mamba_cache(&self, max_num_seqs: usize) -> Result<()> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.preallocate_mamba_cache(max_num_seqs),
            Qwen3TextModel::MoE35(m) => m.preallocate_mamba_cache(max_num_seqs),
            _ => Ok(()),
        }
    }

    pub fn set_mamba_prefix_cache_capacity(&self, capacity: usize) {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.set_mamba_prefix_cache_capacity(capacity),
            Qwen3TextModel::MoE35(m) => m.set_mamba_prefix_cache_capacity(capacity),
            _ => {}
        }
    }

    pub fn capture_mamba_prefix_state(
        &self,
        seq_id: usize,
        hash: u64,
        preserve: bool,
    ) -> Result<bool> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.capture_mamba_prefix_state(seq_id, hash, preserve),
            Qwen3TextModel::MoE35(m) => m.capture_mamba_prefix_state(seq_id, hash, preserve),
            _ => Ok(true),
        }
    }

    pub fn has_mamba_prefix_state(&self, hash: u64) -> bool {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.has_mamba_prefix_state(hash),
            Qwen3TextModel::MoE35(m) => m.has_mamba_prefix_state(hash),
            _ => true,
        }
    }

    pub fn restore_mamba_prefix_state(&self, seq_id: usize, hash: u64) -> Result<bool> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.restore_mamba_prefix_state(seq_id, hash),
            Qwen3TextModel::MoE35(m) => m.restore_mamba_prefix_state(seq_id, hash),
            _ => Ok(true),
        }
    }

    pub fn reset_mamba_cache(&self) -> Result<()> {
        match &self.text_model {
            Qwen3TextModel::Dense35(m) => m.reset_mamba_cache(),
            Qwen3TextModel::MoE35(m) => m.reset_mamba_cache(),
            _ => Ok(()),
        }
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
