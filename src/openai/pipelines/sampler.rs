use std::iter::zip;

use candle_core::{DType, Device, IndexOp, Tensor};

use crate::{
    openai::{
        pipelines::{logical_not, logical_or},
        responses::APIError,
    },
    paged_attention::sequence::SequenceOutput,
};

use super::sampling_metadata::SamplingMetadata;

const SAMPLING_EPS: f32 = 1e-5;

pub struct SequenceDataOutput {
    pub samples: Vec<SequenceOutput>,
}

pub struct Sampler {
    pub vocab_size: usize,
}

impl Sampler {
    pub fn forward(
        &self,
        logits: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Result<Vec<SequenceDataOutput>, APIError> {
        let (presence_penalties, frequency_penalties, repetition_penalties) =
            self._get_penalties(sampling_metadata);
        assert_eq!(
            presence_penalties.len(),
            *logits.shape().dims().get(0).unwrap()
        );
        assert_eq!(
            frequency_penalties.len(),
            *logits.shape().dims().get(0).unwrap()
        );
        assert_eq!(
            repetition_penalties.len(),
            *logits.shape().dims().get(0).unwrap()
        );
        let mut logits = self._apply_penalties(
            logits,
            sampling_metadata,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
        )?;

        // Apply temperature scaling
        let temperatures = self._get_temperatures(sampling_metadata);
        assert_eq!(temperatures.len(), logits.shape().dims()[0]);
        if temperatures.iter().any(|t| t != &1.0) {
            let t = Tensor::from_vec(temperatures, (temperatures.len(),), logits.device())
                .map_err(APIError::from)?
                .to_dtype(logits.dtype())
                .map_err(APIError::from)?;
            logits = logits
                .div(&t.unsqueeze(1).map_err(APIError::from)?)
                .map_err(APIError::from)?;
        }

        let (top_ps, top_ks, min_ps) =
            self._get_top_p_top_k_min_p(sampling_metadata, self.vocab_size.try_into().unwrap());
        assert_eq!(top_ps.len(), top_ks.len());
        assert_ne!(top_ps.len(), logits.shape().dims()[0]);

        let do_top_p = top_ps.iter().any(|p| *p < 1.0 - SAMPLING_EPS);
        let do_top_k = top_ks
            .iter()
            .any(|k| *k != TryInto::<isize>::try_into(self.vocab_size).unwrap());
        if do_top_p || do_top_k {
            logits = self._apply_top_p_top_k(logits, top_ps, top_ks);
        }

        todo!()
    }

    fn _apply_top_p_top_k(&self, logits: Tensor, top_ps: Vec<f32>, top_ks: Vec<isize>) -> Tensor {
        
        todo!()
    }

    fn _get_top_p_top_k_min_p(
        &self,
        sampling_metadata: SamplingMetadata,
        vocab_size: isize,
    ) -> (Vec<f32>, Vec<isize>, Vec<f32>) {
        let mut top_ps = Vec::new();
        let mut top_ks = Vec::new();
        let mut min_ps = Vec::new();
        for (i, (seq_ids, sampling_params)) in sampling_metadata.seq_groups.iter().enumerate() {
            let top_p = sampling_params.top_p;
            let min_p = sampling_params.min_p;
            let top_k = sampling_params.top_k.min(vocab_size);
            let top_k = if top_k == -1 {
                //top_k=-1 means no truncation
                vocab_size
            } else {
                top_k
            };
            if i < sampling_metadata.prompt_lens.len() && sampling_params.prompt_logprobs.is_some()
            {
                let prompt_len = sampling_metadata.prompt_lens.get(i).unwrap();
                top_ps.extend(vec![top_p].repeat(prompt_len - 1));
                top_ks.extend(vec![top_k].repeat(prompt_len - 1));
                min_ps.extend(vec![min_p].repeat(prompt_len - 1));
            }
            top_ps.extend(vec![top_p].repeat(seq_ids.len()));
            top_ks.extend(vec![top_k].repeat(seq_ids.len()));
            min_ps.extend(vec![min_p].repeat(seq_ids.len()));
        }

        (top_ps, top_ks, min_ps)
    }

    fn _get_temperatures(&self, sampling_metadata: SamplingMetadata) -> Vec<f32> {
        let mut temperatures = Vec::new();
        for (i, (seq_ids, sampling_params)) in sampling_metadata.seq_groups.iter().enumerate() {
            let mut temp = sampling_params.temperature;
            if temp < SAMPLING_EPS {
                // (effective) zero temp means deterministic. Set to 1 to avoid divide by 0.
                temp = 0.;
            }
            if i < sampling_metadata.prompt_lens.len() && sampling_params.prompt_logprobs.is_some()
            {
                let prompt_len = sampling_metadata.prompt_lens.get(i).unwrap();
                temperatures.extend(vec![temp].repeat(prompt_len - 1));
            }
            temperatures.extend(vec![temp].repeat(seq_ids.len()));
        }
        temperatures
    }

    fn _apply_penalties(
        &self,
        logits: Tensor,
        sampling_metadata: SamplingMetadata,
        presence_penalties: Vec<f32>,
        frequency_penalties: Vec<f32>,
        repetition_penalties: Vec<f32>,
    ) -> Result<Tensor, APIError> {
        let (num_seqs, vocab_shape) = { (logits.shape().dims()[0], logits.shape().dims()[1]) };
        let mut exited_with_no_break = true;
        for (p, (f, r)) in zip(
            presence_penalties,
            zip(frequency_penalties, repetition_penalties),
        ) {
            if p.abs() < SAMPLING_EPS && f.abs() < SAMPLING_EPS && (r - 1.).abs() < SAMPLING_EPS {
                continue;
            }
            exited_with_no_break = true;
            break;
        }
        if !exited_with_no_break {
            return Ok(logits);
        }

        let (prompt_tokens, output_tokens) = self._get_prompt_and_output_tokens(sampling_metadata);
        assert_eq!(prompt_tokens.len(), logits.shape().dims()[0]);
        assert_eq!(output_tokens.len(), logits.shape().dims()[0]);

        let (prompt_bin_counts, prompt_mask) =
            self._get_bin_counts_and_mask(logits, prompt_tokens, vocab_shape, num_seqs)?;
        let (output_bin_counts, output_mask) =
            self._get_bin_counts_and_mask(logits, output_tokens, vocab_shape, num_seqs)?;

        let repetition_penalties = Tensor::from_vec(
            repetition_penalties,
            (repetition_penalties.len(),),
            &Device::new_cuda(0).map_err(APIError::from)?,
        )
        .map_err(APIError::from)?
        .to_dtype(logits.dtype())
        .map_err(APIError::from)?;
        let frequency_penalties = Tensor::from_vec(
            frequency_penalties,
            (frequency_penalties.len(),),
            &Device::new_cuda(0).map_err(APIError::from)?,
        )
        .map_err(APIError::from)?
        .to_dtype(logits.dtype())
        .map_err(APIError::from)?;
        let presence_penalties = Tensor::from_vec(
            presence_penalties,
            (presence_penalties.len(),),
            &Device::new_cuda(0).map_err(APIError::from)?,
        )
        .map_err(APIError::from)?
        .to_dtype(logits.dtype())
        .map_err(APIError::from)?;

        let mut repetition_penalties = repetition_penalties
            .unsqueeze(1)
            .map_err(APIError::from)?
            .repeat((1, vocab_shape))
            .map_err(APIError::from)?;
        let idx = logical_not(&logical_or(&prompt_mask, &output_mask)?)?;

        assert_eq!(idx.shape().dims().len(), 2);
        // Following few lines equivalent to Pytorch a[b]=scalar
        for to_select in idx.to_vec2::<i64>().map_err(APIError::from)? {
            let mut ranges = Vec::new();
            for idx in to_select {
                ranges.push(idx as usize..idx as usize + 1);
            }
            repetition_penalties = repetition_penalties
                .slice_assign(
                    &ranges[..],
                    &Tensor::new(1i64, repetition_penalties.device())
                        .map_err(APIError::from)?
                        .to_dtype(repetition_penalties.dtype())
                        .map_err(APIError::from)?,
                )
                .map_err(APIError::from)?;
        }

        let positive_logits = logits.gt(0i64).map_err(APIError::from)?; //true=1, false=0
        let mut logits = positive_logits
            .where_cond(
                &(&logits / &repetition_penalties).map_err(APIError::from)?,
                &(&logits * &repetition_penalties).map_err(APIError::from)?,
            )
            .map_err(APIError::from)?;

        // Follow definition in OpenAI API: https://platform.openai.com/docs/api-reference/parameter-details
        logits = (logits
            - (frequency_penalties.unsqueeze(1).map_err(APIError::from)? * output_bin_counts)
                .map_err(APIError::from)?)
        .map_err(APIError::from)?;
        logits = (logits
            - (presence_penalties.unsqueeze(1).map_err(APIError::from)? * output_mask)
                .map_err(APIError::from)?)
        .map_err(APIError::from)?;

        Ok(logits)
    }

    fn _get_bin_counts_and_mask(
        &self,
        logits: Tensor,
        tokens: Vec<Vec<usize>>,
        vocab_size: usize,
        num_seqs: usize,
    ) -> Result<(Tensor, Tensor), APIError> {
        let max_len = tokens.iter().map(|toks| toks.len()).max().unwrap();
        let mut padded_tokens = tokens
            .into_iter()
            .map(|mut toks| {
                toks.extend(vec![vocab_size].repeat(max_len - toks.len()));
                toks
            })
            .collect::<Vec<_>>();
        let mut tensors = Vec::new();
        for vec in padded_tokens {
            tensors.push(
                Tensor::from_vec(
                    vec.iter()
                        .map(|x| (*x).try_into().unwrap())
                        .collect::<Vec<u32>>(),
                    (vec.len(),),
                    &Device::new_cuda(0).map_err(APIError::from)?,
                )
                .map_err(APIError::from)?
                .to_dtype(DType::I64)
                .map_err(APIError::from)?,
            )
        }
        let tokens_tensor = Tensor::cat(&tensors[..], 0)
            .map_err(APIError::from)?
            .to_dtype(DType::I64)
            .map_err(APIError::from)?;

        let bin_counts = Tensor::zeros(
            (num_seqs, vocab_size + 1),
            DType::I64,
            &Device::new_cuda(0).map_err(APIError::from)?,
        )
        .map_err(APIError::from)?;
        bin_counts.scatter_add(
            &tokens_tensor.ones_like().map_err(APIError::from)?,
            &tokens_tensor,
            1,
        );
        let bin_counts = bin_counts.i((.., ..vocab_size)).map_err(APIError::from)?;

        let mask = bin_counts.gt(0i64).map_err(APIError::from)?;

        Ok((bin_counts, mask))
    }

    fn _get_prompt_and_output_tokens(
        &self,
        sampling_metadata: SamplingMetadata,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut prompt_tokens = Vec::new();
        let mut output_tokens = Vec::new();
        for (i, (seq_ids, sampling_params)) in sampling_metadata.seq_groups.iter().enumerate() {
            if i < sampling_metadata.prompt_lens.len() && sampling_params.prompt_logprobs.is_some()
            {
                // NOTE: prompt token positions do not need output toks to compute penalties
                let prompt_len = sampling_metadata.prompt_lens.get(i).unwrap();
                prompt_tokens.extend(vec![vec![].repeat(prompt_len - 1)]);
                output_tokens.extend(vec![vec![].repeat(prompt_len - 1)]);
            } else {
                let seq_data = sampling_metadata.seq_data.get(&i).unwrap();
                prompt_tokens.push(seq_data.prompt_token_ids);
                output_tokens.push(seq_data.output_token_ids);
            }
        }
        (prompt_tokens, output_tokens)
    }

    fn _get_penalties(
        &self,
        sampling_metadata: SamplingMetadata,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut presence_penalties = Vec::new();
        let mut frequency_penalties = Vec::new();
        let mut repetition_penalties = Vec::new();
        for (i, (seq_ids, sampling_params)) in sampling_metadata.seq_groups.iter().enumerate() {
            let p = sampling_params.presence_penalty;
            let f = sampling_params.frequency_penalty;
            let r = sampling_params.repetition_penalty;
            if i < sampling_metadata.prompt_lens.len() && sampling_params.prompt_logprobs.is_some()
            {
                let prompt_len = sampling_metadata.prompt_lens.get(i).unwrap();
                presence_penalties.extend(vec![0.].repeat(prompt_len - 1));
                frequency_penalties.extend([0.].repeat(prompt_len - 1));
                repetition_penalties.extend([0.].repeat(prompt_len - 1));
            }
            presence_penalties.extend([p].repeat(seq_ids.len()));
            frequency_penalties.extend([f].repeat(seq_ids.len()));
            repetition_penalties.extend([r].repeat(seq_ids.len()));
        }
        (
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
        )
    }
}
