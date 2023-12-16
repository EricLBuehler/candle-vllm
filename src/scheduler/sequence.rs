use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use super::block_engine::LogicalTokenBlock;

#[derive(Clone, Copy)]
pub enum SequenceStatus {
    FinishedIgnored,
    Waiting,
    Running,
    Swapped,
    FinishedAborted,
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

    fn get_cumulative_logprob(&self) -> f32 {
        self.cumulative_logprob
    }
}

/// A Sequence holds information about the data it contains (the tokens), and the logical token blocks
/// to which it is mapped.
pub struct _Sequence {
    data: Mutex<SequenceData>,
    seq_id: usize,
    logical_token_blocks: Vec<LogicalTokenBlock>,
    block_size: usize,
    finish_reason: Option<String>,
}

impl _Sequence {
    pub fn new(prompt_token_ids: Vec<usize>, seq_id: usize, block_size: usize) -> Self {
        let mut this = Self {
            data: Mutex::new(SequenceData::new(prompt_token_ids.clone())),
            seq_id,
            logical_token_blocks: Vec::new(),
            block_size,
            finish_reason: None,
        };
        this.append_tokens_to_blocks(prompt_token_ids);
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

    pub fn get_id(&self) -> usize {
        self.seq_id
    }

    pub fn is_prompt(&self) -> bool {
        self.deref().output_token_ids.is_empty()
    }

    pub fn get_prompt_len(&self) -> usize {
        self.deref().prompt_token_ids.len()
    }

    pub fn get_len(&self) -> usize {
        self.deref().prompt_token_ids.len() + self.deref().output_token_ids.len()
    }

    pub fn get_token_ids(&self) -> Vec<usize> {
        let mut res = self.deref().prompt_token_ids.clone();
        res.extend(self.deref().output_token_ids.clone());
        res
    }

    pub fn get_last_token_id(&self) -> usize {
        if self.deref().output_token_ids.is_empty() {
            *self.deref().prompt_token_ids.last().unwrap()
        } else {
            *self.deref().output_token_ids.last().unwrap()
        }
    }

    pub fn is_finished(&self) -> bool {
        matches!(
            self.deref().status,
            SequenceStatus::FinishedAborted | SequenceStatus::FinishedIgnored
        )
    }

    pub fn get_cumulative_logprob(&self) -> f32 {
        self.deref().get_cumulative_logprob()
    }

    pub fn set_finish_reason(&mut self, finish_reason: String) {
        self.finish_reason = Some(finish_reason);
    }

    pub fn get_finish_reason(&self) -> &String {
        self.finish_reason.as_ref().unwrap()
    }

    fn append_tokens_to_blocks(&mut self, tokens: Vec<usize>) {
        for tok in tokens {
            self.append_token_to_blocks(tok);
        }
    }

    fn append_token_to_blocks(&mut self, token: usize) {
        let last = self.logical_token_blocks.last_mut();
        if !last.as_ref().is_some_and(|last| last.is_full()) {
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

impl _Sequence {
    pub fn deref(&self) -> MutexGuard<'_, SequenceData> {
        loop {
            if let Ok(res) = self.data.try_lock() {
                return res;
            }
        }
    }

    pub fn deref_mut(&self) -> MutexGuard<'_, SequenceData> {
        loop {
            if let Ok(res) = self.data.try_lock() {
                return res;
            }
        }
    }
}

pub struct Sequence(pub Mutex<_Sequence>);

impl Sequence {
    pub fn deref_mut(&self) -> MutexGuard<'_, _Sequence> {
        loop {
            if let Ok(v) = self.0.try_lock() {
                return v;
            }
        }
    }
}

type SeqID = usize;

/// A SequenceGroup holds the `n` (see SamplingParams) sequences generated from a single prompt.
/// A SequenceGroup contains only sequences with the same prompt. They will always be scheduled together.
pub struct SequenceGroup {
    seqs: HashMap<SeqID, Arc<Sequence>>,
    arrival_time: u64,
    group_id: usize,
    request_id: String,
    created: u64,
}

impl SequenceGroup {
    pub fn new(
        seqs: &[Arc<Sequence>],
        arrival_time: u64,
        group_id: usize,
        request_id: String,
        created: u64,
    ) -> Self {
        let mut seq_map = HashMap::new();
        for seq in seqs {
            seq_map.insert(seq.deref_mut().get_id(), seq.clone());
        }
        Self {
            seqs: seq_map,
            arrival_time,
            group_id,
            request_id,
            created,
        }
    }

    pub fn set_status(&self, status: SequenceStatus) {
        for seq in self.seqs.values() {
            seq.deref_mut().deref().set_status(status);
        }
    }

    /// Blocks to add one new token to each sequence
    pub fn total_blocks_to_add_new_tok(&self) -> usize {
        self.seqs
            .values()
            .map(|seq| seq.deref_mut().blocks_to_add_new_tok())
            .sum()
    }

    pub fn get_prompt_len(&self) -> usize {
        self.seqs.len()
    }

    pub fn get_total_logical_token_blocks(&self) -> usize {
        self.seqs
            .values()
            .map(|seq| seq.deref_mut().get_logical_token_blocks())
            .sum()
    }

    pub fn get_seqs(&self) -> &HashMap<SeqID, Arc<Sequence>> {
        &self.seqs
    }

    pub fn arrival_time(&self) -> u64 {
        self.arrival_time
    }

    pub fn get_id(&self) -> &usize {
        &self.group_id
    }

    pub fn is_finished(&self) -> bool {
        self.seqs.iter().all(|(_, x)| x.deref_mut().is_finished())
    }

    pub fn get_request_id(&self) -> &String {
        &self.request_id
    }

    pub fn get_created_time(&self) -> u64 {
        self.created
    }
}
