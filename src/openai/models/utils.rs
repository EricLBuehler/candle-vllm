use attention_rs::mamba_cache::MambaCache;
use candle::{DType, Device, Module, Result, Tensor};
use candle_core as candle;
use candle_nn::RmsNorm;

use crate::openai::distributed::VarBuilder;
use crate::InputMetadata;

pub struct TextBackboneResolution {
    pub prefix: &'static str,
    pub tie_word_embeddings: bool,
    pub use_root_builder: bool,
}

pub fn resolve_text_backbone(vb: &VarBuilder, tie_word_embeddings: bool) -> TextBackboneResolution {
    if vb.contains_tensor("model.embed_tokens.weight") {
        TextBackboneResolution {
            prefix: "model",
            tie_word_embeddings,
            use_root_builder: false,
        }
    } else if vb.contains_tensor("model.language_model.model.embed_tokens.weight") {
        TextBackboneResolution {
            prefix: "model.language_model.model",
            tie_word_embeddings,
            use_root_builder: false,
        }
    } else if vb.contains_tensor("model.language_model.embed_tokens.weight") {
        TextBackboneResolution {
            prefix: "model.language_model",
            tie_word_embeddings,
            use_root_builder: false,
        }
    } else if vb.contains_tensor("model.text_model.model.embed_tokens.weight") {
        TextBackboneResolution {
            prefix: "model.text_model.model",
            tie_word_embeddings,
            use_root_builder: false,
        }
    } else if vb.contains_tensor("model.text_model.embed_tokens.weight") {
        TextBackboneResolution {
            prefix: "model.text_model",
            tie_word_embeddings,
            use_root_builder: false,
        }
    } else if vb.contains_tensor("language_model.model.embed_tokens.weight") {
        TextBackboneResolution {
            prefix: "language_model.model",
            tie_word_embeddings,
            use_root_builder: false,
        }
    } else if vb.contains_tensor("language_model.embed_tokens.weight") {
        TextBackboneResolution {
            prefix: "language_model",
            tie_word_embeddings,
            use_root_builder: false,
        }
    } else if vb.contains_tensor("text_model.model.embed_tokens.weight") {
        TextBackboneResolution {
            prefix: "text_model.model",
            tie_word_embeddings,
            use_root_builder: false,
        }
    } else if vb.contains_tensor("text_model.embed_tokens.weight") {
        TextBackboneResolution {
            prefix: "text_model",
            tie_word_embeddings,
            use_root_builder: false,
        }
    } else if vb.contains_tensor("embed_tokens.weight") {
        TextBackboneResolution {
            prefix: "",
            tie_word_embeddings: true,
            use_root_builder: true,
        }
    } else {
        TextBackboneResolution {
            prefix: "model",
            tie_word_embeddings,
            use_root_builder: false,
        }
    }
}

pub fn resolve_input_seqlens(input_metadata: &InputMetadata) -> Result<Vec<u32>> {
    if let Some(seqlens) = input_metadata.seqlens.as_ref() {
        Ok(seqlens.clone())
    } else if let Some(cu_seqlens) = input_metadata.cu_seqlens_q.as_ref() {
        Ok(cu_seqlens.to_vec1::<u32>()?[1..].to_vec())
    } else {
        Ok(Vec::new())
    }
}

pub fn resolve_mamba_seq_slots(
    model_name: &str,
    device: &Device,
    input_metadata: &InputMetadata,
    token_count: usize,
    mamba_cache: &mut MambaCache,
) -> Result<Tensor> {
    if let Some(slot_mapping) = &input_metadata.mamba_slot_mapping {
        if slot_mapping.dtype() != DType::I64 {
            candle::bail!(
                "{} expects mamba_slot_mapping dtype I64, got {:?}",
                model_name,
                slot_mapping.dtype()
            )
        }
        let slot_count = slot_mapping.dim(0)?;
        if slot_count == 0 {
            candle::bail!("{} received empty mamba_slot_mapping", model_name)
        }
        if !input_metadata.is_prefill && slot_count != token_count {
            candle::bail!(
                "{} decode mamba_slot_mapping length mismatch: slots={} tokens={}",
                model_name,
                slot_count,
                token_count
            )
        }
        return Ok(slot_mapping.clone());
    }

    let sequence_ids = input_metadata.sequence_ids.as_ref().ok_or_else(|| {
        candle::Error::Msg(format!("{model_name} requires input_metadata.sequence_ids"))
    })?;

    if sequence_ids.is_empty() {
        candle::bail!("{} received empty sequence_ids", model_name)
    }

    let slots = if input_metadata.is_prefill {
        mamba_cache.ensure_slots_for_sequences(sequence_ids)?
    } else {
        mamba_cache.get_slots_for_sequences(sequence_ids)?
    };
    let seq_slots = if input_metadata.is_prefill {
        let cu_seqlens = input_metadata
            .cu_seqlens_q
            .as_ref()
            .ok_or_else(|| {
                candle::Error::Msg(format!("{model_name} prefill requires cu_seqlens_q"))
            })?
            .to_vec1::<u32>()?;
        if cu_seqlens.len() != sequence_ids.len() + 1 {
            candle::bail!(
                "{} sequence_ids ({}) and cu_seqlens_q ({}) mismatch",
                model_name,
                sequence_ids.len(),
                cu_seqlens.len()
            )
        }
        // Prefill varlen kernels expect one slot per sequence (batch), aligned with cu_seqlens.
        slots.iter().map(|slot| *slot as i64).collect::<Vec<_>>()
    } else {
        if slots.len() != token_count {
            candle::bail!(
                "{} decode sequence_ids ({}) and token count ({}) mismatch",
                model_name,
                slots.len(),
                token_count
            )
        }
        slots.iter().map(|slot| *slot as i64).collect::<Vec<_>>()
    };

    let slot_count = seq_slots.len();
    Tensor::from_vec(seq_slots, (slot_count,), device)
}

pub fn apply_rms_norm_fp32(norm: &RmsNorm, xs: &Tensor) -> Result<Tensor> {
    if xs.dtype() == DType::F32 {
        norm.forward(xs)
    } else {
        norm.forward(&xs.to_dtype(DType::F32)?)?
            .to_dtype(xs.dtype())
    }
}
