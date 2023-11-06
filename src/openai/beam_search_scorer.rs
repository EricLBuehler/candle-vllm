// https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L123

use std::{borrow::BorrowMut, collections::HashMap, error::Error, fmt::Display};

use candle_core::{Device, Tensor};

use super::sampling_params::EarlyStoppingCondition;

#[derive(Debug)]
struct CustomError(String);

impl Display for CustomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for CustomError {}

#[derive(Clone, Debug)]
struct BeamSearchScorer {
    num_beams: usize,
    device: Device,
    length_penalty: f64,
    early_stopping: EarlyStoppingCondition,
    num_beam_hyps_to_keep: usize,
    num_beam_groups: usize,
    group_size: usize,
    beam_hyps: Vec<BeamHypotheses>,
    done: Vec<bool>,
}

#[derive(Clone, Debug)]
enum EosTokenID {
    Single(usize),
    Multi(Vec<usize>),
}

#[derive(Clone, Debug)]
struct ProcessResult {
    next_beam_scores: Tensor,
    next_beam_tokens: Tensor,
    next_beam_indices: Tensor,
}

impl BeamSearchScorer {
    fn new(
        batch_size: usize,
        num_beams: usize,
        device: Device,
        length_penalty: Option<f64>,
        early_stopping: EarlyStoppingCondition,
        num_beam_hyps_to_keep: Option<usize>,
        num_beam_groups: usize,
    ) -> Self {
        let mut beam_hyps = Vec::new();
        for _ in 0..batch_size * num_beam_groups {
            beam_hyps.push(BeamHypotheses::new(
                length_penalty.unwrap_or(0.0),
                early_stopping.clone(),
                num_beams,
            ));
        }
        Self {
            num_beams,
            device,
            length_penalty: length_penalty.unwrap_or(0.0),
            early_stopping: early_stopping,
            num_beam_hyps_to_keep: num_beam_hyps_to_keep.unwrap_or(1),
            num_beam_groups,
            group_size: num_beams / num_beam_groups,
            beam_hyps,
            done: vec![false].repeat(batch_size * num_beam_groups),
        }
    }

    fn process(
        &self,
        input_ids: Tensor,
        next_scores: Tensor,
        next_tokens: Tensor,
        next_indices: Tensor,
        pad_token_id: Option<usize>,
        eos_token_id: Option<EosTokenID>,
        beam_indices: Option<Tensor>,
        group_index: Option<usize>,
    ) -> Result<ProcessResult, Box<dyn Error>> {
        let cur_len = input_ids.dims().last().ok_or("No dims")? + 1; // Length where the next_scores is calculated on
        let batch_size = self.beam_hyps.len() / self.num_beam_groups;

        if batch_size != (input_ids.dims().first().ok_or("No dims")? / self.group_size) {
            if self.num_beam_groups > 1 {
                return Err(Box::new(CustomError(format!("A group beam size of {} is used as the input, but a group beam size of {} is expected by the beam scorer.", input_ids.dims().first().ok_or("No dims")?, self.group_size))));
            } else {
                return Err(Box::new(CustomError(format!("A beam size of {} is used as the input, but a beam size of {} is expected by the beam scorer.", input_ids.dims().first().ok_or("No dims")?, self.group_size))));
            }
        }

        let next_beam_scores = Tensor::zeros(
            (batch_size, self.group_size),
            next_scores.dtype(),
            input_ids.device(),
        )?;
        let next_beam_tokens = Tensor::zeros(
            (batch_size, self.group_size),
            next_tokens.dtype(),
            input_ids.device(),
        )?;
        let next_beam_indices = Tensor::zeros(
            (batch_size, self.group_size),
            next_indices.dtype(),
            input_ids.device(),
        )?;

        for batch_idx in 0..batch_size {
            let batch_group_idx = batch_idx * self.num_beam_groups + group_index.unwrap_or(0);
            if *self.done.get(batch_group_idx).unwrap() {
                if self.num_beams < self.beam_hyps.get(batch_group_idx).unwrap().len() {
                    return Err(Box::new(CustomError(format!(
                        "Batch can only be done if at least {} beams have been generated.",
                        self.num_beams
                    ))));
                }
                if eos_token_id.is_none() || pad_token_id.is_none() {
                    return Err(Box::new(CustomError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined".to_owned())));
                }
                //pad tokens

                todo!() //next_beam_scores[batch_idx, :] = 0;
            }
        }

        todo!()
    }
}

#[derive(Clone, Debug)]
struct BeamHyp {
    score: f64,
    hyp: Tensor,
    beam_indices: Option<Tensor>,
}

#[derive(Clone, Debug)]
struct BeamHypotheses {
    length_penalty: f64,
    early_stopping: EarlyStoppingCondition,
    num_beams: usize,
    worst_score: f64,
    beams: Vec<BeamHyp>,
}

impl BeamHypotheses {
    fn new(length_penalty: f64, early_stopping: EarlyStoppingCondition, num_beams: usize) -> Self {
        Self {
            length_penalty,
            early_stopping,
            num_beams,
            worst_score: f64::MIN,
            beams: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.beams.len()
    }

    fn add(&mut self, hyp: Tensor, sum_logprobs: f64, beam_indices: Option<Tensor>) {
        let score = sum_logprobs / (*hyp.dims().last().unwrap() as f64).powf(self.length_penalty);
        if self.len() < self.num_beams || score > self.worst_score {
            self.beams.push(BeamHyp {
                score,
                hyp,
                beam_indices,
            });
            if self.len() > self.num_beams {
                let mut sorted_next_scores = self
                    .beams
                    .iter()
                    .enumerate()
                    .map(|x| (x.1.score, x.0))
                    .collect::<Vec<_>>();

                sorted_next_scores.sort_by(|(a_score, _a_idx), (b_score, _b_idx)| {
                    a_score.partial_cmp(b_score).expect("Comparison failed")
                });

                // Remove worst scoring
                self.beams.remove(sorted_next_scores.first().unwrap().1);

                // Update worst score
                self.worst_score = sorted_next_scores.get(1).unwrap().0;
            } else {
                self.worst_score = score.min(self.worst_score);
            }
        }
    }
}
