use std::cell::RefCell;

pub struct SequenceData {
    prompt_token_ids: Vec<usize>,
    output_token_ids: Vec<usize>,
    cumulative_logprob: f32,
}

impl SequenceData {
    pub fn new(prompt_token_ids: Vec<usize>) -> Self {
        Self {
            prompt_token_ids,
            output_token_ids: Vec::new(),
            cumulative_logprob: 0.,
        }
    }

    pub fn append_token_id(&mut self, token_id: usize, logprob: f32) {
        self.output_token_ids.push(token_id);
        self.cumulative_logprob += logprob;
    }
}

pub struct Sequence {
    data: RefCell<SequenceData>,
    seq_id: usize,
}

impl Sequence {
    pub fn new(prompt_token_ids: Vec<usize>, seq_id: usize) -> Self {
        Self {
            data: RefCell::new(SequenceData::new(prompt_token_ids)),
            seq_id,
        }
    }
}
