use crate::{
    openai::responses::{ChatChoice, ChatChoiceData},
    paged_attention::sequence::{SequenceGroup, SequenceStatus},
};

pub struct RequestOutput {
    pub request_id: String,
    pub prompt: String,
    pub outputs: Vec<ChatChoice>,
    pub finished: bool,
}

impl RequestOutput {
    pub fn new(
        request_id: String,
        prompt: String,
        outputs: Vec<ChatChoice>,
        finished: bool,
    ) -> Self {
        Self {
            request_id,
            prompt,
            outputs,
            finished,
        }
    }

    pub fn from_seq_group(seq_group: SequenceGroup, role: String) -> Self {
        let n = seq_group.deref().sampling_params.n;
        let seqs = seq_group.deref().get_seqs(None);
        //TODO(EricLBuehler): do the logprob sorting!
        let top_n_seqs = seqs;

        let mut outputs = Vec::new();
        for (index, seq) in top_n_seqs.iter().enumerate() {
            let finished_reason = match seq.status {
                SequenceStatus::FinishedStopped => Some("stop"),
                SequenceStatus::FinishedLengthCapped => Some("length"),
                SequenceStatus::FinishedAborted => Some("abort"),
                SequenceStatus::FinishedIgnored => Some("length"), //should be length because those are the sequences that are ignored
                _ => None,
            };
            let output = ChatChoice {
                message: ChatChoiceData {
                    content: Some(seq.output_data),
                    role,
                },
                finish_reason: finished_reason.map(|x| x.to_string()),
                index,
            };
            outputs.push(output);
        }

        Self::new(
            seq_group.deref().request_id,
            seq_group.deref().get_prompt(),
            outputs,
            seq_group.deref().is_finished(),
        )
    }
}
