use std::iter::zip;

use candle_core::{DType, Device, IndexOp, Tensor};

use crate::{
    openai::{
        pipelines::{logical_not, logical_or},
        responses::APIError,
    },
    paged_attention::{
        input_metadata::InputMetadata,
        sequence::{SequenceGroupMetadata, SequenceOutput},
    },
};

pub struct SequenceGroupOutput {
    pub samples: Vec<SequenceOutput>,
}

pub type SamplerOutput = Vec<SequenceGroupOutput>;

pub trait Sampler {
    type LogitsProcessor;
    type SamplingMetadata;
    type MutableState;
    const SAMPLING_EPS: f32 = 1e-5;

    fn sample<'a>(
        &self,
        logits: Tensor,
        logits_processor: &mut Self::LogitsProcessor,
        sampling_metadata: Self::SamplingMetadata,
        mutable_state: &mut Self::MutableState,
        seq_group_metadatas: Vec<SequenceGroupMetadata>,
    ) -> Result<SamplerOutput, APIError>;
}
