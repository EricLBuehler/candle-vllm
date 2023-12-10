use std::iter::zip;

use candle_core::{IndexOp, Tensor};
use candle_sampling::logits_processor::LogitsProcessor;

use crate::{openai::responses::APIError, paged_attention::sequence::SequenceGroupMetadata};

use super::sampler::{Sampler, SamplerOutput};

pub struct LlamaSampler {
    repeat_last_n: usize,
    repeat_penalty: f32,
    logits_processor: LogitsProcessor,
    context_size: usize,
}

impl LlamaSampler {
    pub fn new(
        repeat_last_n: usize,
        repeat_penalty: f32,
        logits_processor: LogitsProcessor,
        context_size: usize,
    ) -> Self {
        Self {
            repeat_last_n,
            repeat_penalty,
            logits_processor,
            context_size,
        }
    }
}

impl Sampler for LlamaSampler {
    type LogitsProcessor = LogitsProcessor;
    type SamplingMetadata = ();
    type MutableState = usize;

    fn sample(
        &self,
        logits: Tensor,
        logits_processor: &mut Self::LogitsProcessor,
        sampling_metadata: Self::SamplingMetadata,
        index_pos: &mut Self::MutableState,
        seq_group_metadatas: Vec<SequenceGroupMetadata>,
    ) -> Result<SamplerOutput, APIError> {
        let mut next_toks = Vec::new();

        let (n_seqs, n_toks, _) = logits.dims3().unwrap();
        for (seq_n, seq_group_metadata) in zip(0..n_seqs, seq_group_metadatas) {
            let logits = logits.i((seq_n, n_toks - 1)).map_err(APIError::from)?;
            let seq_data = seq_group_metadata.seq_data.get(&seq_n).unwrap();

            let mut tokens = seq_data.prompt_token_ids.clone();
            tokens.extend(seq_data.output_token_ids);
            let tokens = tokens
                .iter()
                .map(|x| TryInto::<u32>::try_into(*x).unwrap())
                .collect::<Vec<_>>();
            let ctxt = &tokens[tokens.len().saturating_sub(self.context_size)..];

            let logits = if self.repeat_penalty == 1. || tokens.is_empty() {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )
                .map_err(APIError::from)?
            };
            *index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits).map_err(APIError::from)?;
            next_toks.push(next_token);
        }

        

        todo!()
    }
}
