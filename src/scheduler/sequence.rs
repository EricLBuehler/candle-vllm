use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    rc::Rc,
};

use super::block_engine::LogicalTokenBlock;

pub enum SequenceStatus {
    FinishedIgnored,
    Waiting,
    Running,
    Swapped,
}

pub struct SequenceData {
    prompt_token_ids: Vec<usize>,
    output_token_ids: Vec<usize>,
    cumulative_logprob: f32,
    status: SequenceStatus,
}

impl SequenceData {
    pub fn new(prompt_token_ids: Vec<usize>) -> Self {
        Self {
            prompt_token_ids,
            output_token_ids: Vec::new(),
            cumulative_logprob: 0.,
            status: SequenceStatus::Waiting,
        }
    }

    pub fn append_token_id(&mut self, token_id: usize, logprob: f32) {
        self.output_token_ids.push(token_id);
        self.cumulative_logprob += logprob;
    }

    pub fn set_status(&mut self, status: SequenceStatus) {
        self.status = status;
    }
}

pub struct Sequence {
    data: RefCell<SequenceData>,
    seq_id: usize,
    logical_token_blocks: Vec<LogicalTokenBlock>,
    block_size: usize,
}

impl Sequence {
    pub fn new(prompt_token_ids: Vec<usize>, seq_id: usize, block_size: usize) -> Self {
        let mut this = Self {
            data: RefCell::new(SequenceData::new(prompt_token_ids)),
            seq_id,
            logical_token_blocks: Vec::new(),
            block_size,
        };
        this.append_tokens_to_blocks(&this.deref().prompt_token_ids[..]);
        this
    }

    pub fn add_token(&mut self, token: usize, logprob: f32) {
        self.deref_mut().append_token_id(token, logprob);
        self.append_token_to_blocks(token);
    }

    pub fn blocks_to_add_new_tok(&mut self) -> usize {
        let last = self.logical_token_blocks.last_mut();
        if !last.is_some_and(|last| last.is_full()) {
            // If we have space
            0
        } else {
            1
        }
    }

    pub fn get_logical_token_blocks(&self) -> usize {
        self.logical_token_blocks.len()
    }

    pub fn get_len(&self) -> usize {
        self.deref().prompt_token_ids.len()
    }

    pub fn get_id(&self) -> usize {
        self.seq_id
    }

    fn append_tokens_to_blocks(&mut self, tokens: &[usize]) {
        for tok in tokens {
            self.append_token_to_blocks(*tok);
        }
    }

    fn append_token_to_blocks(&mut self, token: usize) {
        let last = self.logical_token_blocks.last_mut();
        if !last.is_some_and(|last| last.is_full()) {
            // If we have space
            let last = last.unwrap();
            last.append_token_id(token);
        } else {
            self.logical_token_blocks.push(LogicalTokenBlock::new(
                self.logical_token_blocks.len(),
                self.block_size,
            ));
            self.logical_token_blocks
                .last_mut()
                .unwrap()
                .append_token_id(token);
        }
    }
}

impl Sequence {
    pub fn deref(&self) -> Ref<'_, SequenceData> {
        loop {
            if let Ok(res) = self.data.try_borrow() {
                return res;
            }
        }
    }

    pub fn deref_mut(&self) -> RefMut<'_, SequenceData> {
        loop {
            if let Ok(res) = self.data.try_borrow_mut() {
                return res;
            }
        }
    }
}

type SeqID = usize;

pub struct SequenceGroup {
    seqs: HashMap<SeqID, Rc<Sequence>>,
    arrival_time: u64,
}

impl SequenceGroup {
    pub fn new(seqs: &[Rc<Sequence>], arrival_time: u64) -> Self {
        let mut seq_map = HashMap::new();
        for seq in seqs {
            seq_map.insert(seq.get_id(), seq.clone());
        }
        Self {
            seqs: seq_map,
            arrival_time,
        }
    }

    pub fn set_status(&mut self, status: SequenceStatus) {
        for (_, seq) in self.seqs {
            seq.deref_mut().set_status(status);
        }
    }

    /// Blocks to add one new token to each sequence
    pub fn total_blocks_to_add_new_tok(&self) -> usize {
        self.seqs
            .iter()
            .map(|(_, seq)| seq.blocks_to_add_new_tok())
            .sum()
    }

    pub fn get_prompt_len(&self) -> usize {
        self.seqs.len()
    }

    pub fn get_total_logical_token_blocks(&self) -> usize {
        self.seqs
            .iter()
            .map(|(_, seq)| seq.get_logical_token_blocks())
            .sum()
    }

    pub fn get_seqs(&self) -> &HashMap<SeqID, Rc<Sequence>> {
        &self.seqs
    }
}
