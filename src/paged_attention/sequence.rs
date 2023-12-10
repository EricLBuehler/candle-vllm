use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    ops::Deref,
    sync::Arc,
};

use crate::openai::sampling_params::SamplingParams;

use super::block::LogicalTokenBlock;

#[derive(Clone, PartialEq, PartialOrd)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Swapped,
    FinishedStopped,
    FinishedLengthCapped,
    FinishedAborted,
    FinishedIgnored,
}

impl SequenceStatus {
    pub fn is_finished(&self) -> bool {
        match *self {
            Self::FinishedStopped
            | Self::FinishedLengthCapped
            | Self::FinishedAborted
            | Self::FinishedIgnored => true,
            _ => false,
        }
    }
}

pub struct SequenceData {
    pub prompt_token_ids: Vec<usize>,
    pub output_token_ids: Vec<usize>,
}

impl SequenceData {
    pub fn get_last_token_id(&self) -> &usize {
        if self.output_token_ids.is_empty() {
            self.prompt_token_ids.last().unwrap()
        } else {
            self.output_token_ids.last().unwrap()
        }
    }
}

pub struct SequenceOutput {
    pub parent_seq_id: usize,
    pub output_token: usize,
    pub logprobs: HashMap<usize, f64>,
}

pub struct SequenceGroupOutput {
    pub samples: Vec<SequenceOutput>,
}

pub struct Sequence {
    pub seq_id: usize,
    pub data: Arc<SequenceData>,
    prompt: String,
    block_size: usize,
    pub status: SequenceStatus,
    pub logical_token_blocks: Vec<LogicalTokenBlock>,
    pub output_data: String,
}

impl Sequence {
    pub fn new(
        seq_id: usize,
        prompt: String,
        prompt_token_ids: Vec<usize>,
        block_size: usize,
    ) -> Self {
        let mut this = Self {
            seq_id,
            prompt,
            data: Arc::new(SequenceData {
                prompt_token_ids: prompt_token_ids.clone(),
                output_token_ids: Vec::new(),
            }),
            block_size,
            status: SequenceStatus::Waiting,
            logical_token_blocks: Vec::new(),
            output_data: String::new(),
        };
        Sequence::_append_tokens_to_blocks(&mut this, prompt_token_ids);
        this
    }

    fn _append_logical_block(&mut self) {
        self.logical_token_blocks.push(LogicalTokenBlock::new(
            self.logical_token_blocks.len(),
            self.block_size,
        ));
    }

    fn _append_tokens_to_blocks(&mut self, token_ids: Vec<usize>) {
        let mut cursor = 0;
        while cursor < token_ids.len() {
            if self.logical_token_blocks.is_empty() {
                self._append_logical_block();
            }

            let mut last_block = self.logical_token_blocks.last_mut().unwrap();
            if last_block.is_full() {
                self._append_logical_block();
                last_block = self.logical_token_blocks.last_mut().unwrap();
            }

            let num_empty_slots = last_block.get_num_empty_slots();
            last_block.append_tokens(token_ids[cursor..cursor + num_empty_slots].to_vec());
            cursor += num_empty_slots;
        }
    }

    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }

    pub fn len(&self) -> usize {
        self.data.prompt_token_ids.len() + self.data.output_token_ids.len()
    }

    pub fn set_status(&mut self, status: SequenceStatus) {
        self.status = status
    }
}

pub struct SequenceGroupInner {
    pub request_id: String,
    seqs_dict: HashMap<usize, Sequence>,
    pub sampling_params: SamplingParams,
    pub arrival_time: u64,
}

impl SequenceGroupInner {
    pub fn new(
        request_id: String,
        seqs: Vec<Sequence>,
        sampling_params: SamplingParams,
        arrival_time: u64,
    ) -> Self {
        Self {
            request_id,
            seqs_dict: seqs
                .into_iter()
                .map(|seq| (seq.seq_id, seq))
                .collect::<HashMap<_, _>>(),
            sampling_params,
            arrival_time,
        }
    }

    pub fn get_max_num_running_seqs(&self) -> usize {
        if self.sampling_params.best_of > self.num_seqs(None) {
            self.sampling_params.best_of
        } else {
            // Get number of unfinished seqs
            self.seqs_dict
                .iter()
                .filter(|(_, seq)| seq.is_finished())
                .count()
        }
    }

    pub fn get_seqs(&self, status: Option<SequenceStatus>) -> Vec<&Sequence> {
        if let Some(status) = status {
            self.seqs_dict
                .values()
                .filter(|seq| seq.status == status)
                .collect::<Vec<_>>()
        } else {
            self.seqs_dict.values().collect::<Vec<_>>()
        }
    }

    pub fn get_mut_seqs(&mut self, status: Option<SequenceStatus>) -> Vec<&mut Sequence> {
        if let Some(status) = status {
            self.seqs_dict
                .values_mut()
                .filter(|seq| seq.status == status)
                .collect::<Vec<_>>()
        } else {
            self.seqs_dict.values_mut().collect::<Vec<_>>()
        }
    }

    pub fn num_seqs(&self, status: Option<SequenceStatus>) -> usize {
        self.get_seqs(status).len()
    }

    pub fn is_finished(&self) -> bool {
        self.get_seqs(None).iter().all(|seq| seq.is_finished())
    }

    pub fn get_prompt(&self) -> String {
        self.seqs_dict.values().nth(0).unwrap().prompt
    }
}

pub struct SequenceGroup {
    inner: RefCell<SequenceGroupInner>,
}

impl SequenceGroup {
    pub fn new(
        request_id: String,
        seqs: Vec<Sequence>,
        sampling_params: SamplingParams,
        arrival_time: u64,
    ) -> Self {
        Self {
            inner: RefCell::new(SequenceGroupInner::new(
                request_id,
                seqs,
                sampling_params,
                arrival_time,
            )),
        }
    }

    pub fn deref_mut(&self) -> RefMut<'_, SequenceGroupInner> {
        loop {
            if let Ok(reference) = self.inner.try_borrow_mut() {
                return reference;
            }
        }
    }

    pub fn deref(&self) -> Ref<'_, SequenceGroupInner> {
        loop {
            if let Ok(reference) = self.inner.try_borrow() {
                return reference;
            }
        }
    }
}

pub struct SequenceGroupMetadata {
    pub request_id: String,
    pub is_prompt: bool,
    pub seq_data: HashMap<usize, Arc<SequenceData>>,
    pub sampling_params: SamplingParams,
    pub block_tables: Option<HashMap<usize, Vec<usize>>>,
}
