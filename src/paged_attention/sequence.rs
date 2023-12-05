use std::collections::HashMap;

use crate::openai::sampling_params::{self, SamplingParams};

#[derive(Clone)]
pub struct Sequence {
    seq_id: usize,
    prompt: String,
    prompt_token_ids: Vec<usize>,
    output_token_ids: Vec<usize>,
    block_size: usize,
}

impl Sequence {
    pub fn new(
        seq_id: usize,
        prompt: String,
        prompt_token_ids: Vec<usize>,
        block_size: usize,
    ) -> Self {
        Self {
            seq_id,
            prompt,
            prompt_token_ids,
            output_token_ids: Vec::new(),
            block_size,
        }
    }
}

pub struct SequenceGroup {
    request_id: String,
    seqs_dict: HashMap<usize, Sequence>,
    sampling_params: SamplingParams,
    arrival_time: f64,
}

impl SequenceGroup {
    pub fn new(
        request_id: String,
        seqs: Vec<Sequence>,
        sampling_params: SamplingParams,
        arrival_time: f64,
    ) -> Self {
        Self {
            request_id,
            seqs_dict: seqs
                .iter()
                .map(|seq| (seq.seq_id, seq.clone()))
                .collect::<HashMap<_, _>>(),
            sampling_params,
            arrival_time,
        }
    }

    pub fn get_seqs(&self) -> Vec<&Sequence> {
        self.seqs_dict
            .iter()
            .map(|(_, seq)| seq)
            .collect::<Vec<_>>()
    }
}
