use std::{
    collections::{HashSet, VecDeque},
    sync::Arc,
};

use crate::scheduler::sequence::Sequence;

use super::{sequence::SequenceGroup, Scheduler};

#[derive(Default)]
pub struct MambaState {
    restored_prefix_sequences: HashSet<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MambaPrefixCapture {
    pub seq_id: usize,
    pub hash: u64,
    pub block_id: usize,
}

#[derive(Clone)]
pub struct MambaRestorePlan {
    pub sequence: Arc<Sequence>,
    pub seq_id: usize,
    pub cached_tokens: usize,
    pub hash: u64,
}

pub struct FinishedMambaSync {
    pub captures: Vec<MambaPrefixCapture>,
    pub released_ids: Vec<usize>,
}

impl Scheduler {
    pub fn reset_mamba_state(&mut self) {
        self.mamba_state.restored_prefix_sequences.clear();
    }

    pub fn collect_prefill_mamba_captures(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        chunk_size: usize,
    ) -> Vec<MambaPrefixCapture> {
        if groups.is_empty() || chunk_size == 0 {
            return Vec::new();
        }

        let mut captures = Vec::new();
        for group in groups {
            for seq in Self::ordered_group_sequences(group) {
                let seq_id = seq.deref().get_id();
                let prompt_len = seq.deref().get_prompt_len();
                let num_cached_tokens = seq.deref().get_num_cached_tokens();
                if prompt_len == 0 || num_cached_tokens >= prompt_len {
                    continue;
                }

                // Capture every chunk-prefill boundary so future requests can restore
                // mamba state at the nearest shared prefix length, not just the final
                // prompt boundary.
                let processed_tokens = (num_cached_tokens + chunk_size).min(prompt_len);
                if processed_tokens <= num_cached_tokens {
                    continue;
                }
                let full_blocks = processed_tokens / self.block_engine.get_block_size();
                if full_blocks == 0 {
                    continue;
                }

                if let Some(hash) = self
                    .block_engine
                    .prefix_hash_for_sequence(&seq, processed_tokens)
                {
                    let Some(block_id) = self
                        .block_engine
                        .prefix_block_id_for_sequence(&seq, full_blocks)
                    else {
                        continue;
                    };
                    captures.push(MambaPrefixCapture {
                        seq_id,
                        hash,
                        block_id,
                    });
                }
            }
        }
        captures
    }

    pub fn collect_decode_mamba_captures(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
    ) -> Vec<MambaPrefixCapture> {
        if groups.is_empty() {
            return Vec::new();
        }

        let block_size = self.block_engine.get_block_size();
        let stride_blocks = crate::mamba_snapshot_block_stride_blocks();
        let mut captures = Vec::new();
        for group in groups {
            for seq in Self::ordered_group_sequences(group) {
                if seq.deref().is_finished() {
                    continue;
                }
                let seq_id = seq.deref().get_id();
                let seq_len = seq.deref().get_len();
                if seq_len < block_size || seq_len % block_size != 0 {
                    continue;
                }
                let full_blocks = seq_len / block_size;
                if stride_blocks > 1 && full_blocks % stride_blocks != 0 {
                    continue;
                }
                if let Some(hash) = self.block_engine.prefix_hash_for_sequence(&seq, seq_len) {
                    let Some(block_id) = self
                        .block_engine
                        .prefix_block_id_for_sequence(&seq, full_blocks)
                    else {
                        continue;
                    };
                    captures.push(MambaPrefixCapture {
                        seq_id,
                        hash,
                        block_id,
                    });
                }
            }
        }
        captures
    }

    pub fn prepare_prompt_mamba_restores(
        &mut self,
        groups: &VecDeque<Arc<SequenceGroup>>,
    ) -> Vec<MambaRestorePlan> {
        if groups.is_empty() {
            return Vec::new();
        }

        let mut plans = Vec::new();
        for group in groups {
            for seq in Self::ordered_group_sequences(group) {
                let seq_id = seq.deref().get_id();
                let cached_tokens = seq.deref().get_num_cached_tokens();
                if cached_tokens == 0
                    || self.mamba_state.restored_prefix_sequences.contains(&seq_id)
                {
                    continue;
                }

                let Some(hash) = self
                    .block_engine
                    .prefix_hash_for_sequence(&seq, cached_tokens)
                else {
                    tracing::warn!(
                        "Seq {} has {} cached tokens but no prefix hash; fallback to full prefill",
                        seq_id,
                        cached_tokens
                    );
                    self.fallback_sequence_to_full_prefill(
                        &seq,
                        seq_id,
                        cached_tokens,
                        "has no prefix hash",
                    );
                    continue;
                };

                plans.push(MambaRestorePlan {
                    sequence: seq,
                    seq_id,
                    cached_tokens,
                    hash,
                });
            }
        }
        plans
    }

    pub fn mark_mamba_restored(&mut self, seq_id: usize) {
        self.mamba_state.restored_prefix_sequences.insert(seq_id);
    }

    pub fn handle_missing_mamba_snapshot(&mut self, restore: &MambaRestorePlan) {
        self.block_engine.invalidate_mamba_prefix_hash(restore.hash);
        self.fallback_sequence_to_full_prefill(
            &restore.sequence,
            restore.seq_id,
            restore.cached_tokens,
            &format!("missing mamba snapshot for hash {}", restore.hash),
        );
    }

    pub fn handle_failed_mamba_restore(&mut self, restore: &MambaRestorePlan) {
        self.fallback_sequence_to_full_prefill(
            &restore.sequence,
            restore.seq_id,
            restore.cached_tokens,
            &format!("failed to restore mamba snapshot for hash {}", restore.hash),
        );
    }

    pub fn record_mamba_prefix_captures<I>(&mut self, captures: I)
    where
        I: IntoIterator<Item = MambaPrefixCapture>,
    {
        for capture in captures {
            self.block_engine
                .record_mamba_prefix_capture(capture.hash, capture.block_id);
        }
    }

    pub fn free_finished_sequence_groups_and_collect_mamba(&mut self) -> FinishedMambaSync {
        let mut captures = Vec::new();
        let released_ids = self.free_finished_sequence_groups_with(|seq_id, hash, block_id| {
            if let (Some(hash), Some(block_id)) = (hash, block_id) {
                captures.push(MambaPrefixCapture {
                    seq_id,
                    hash,
                    block_id,
                });
            }
        });

        for seq_id in &released_ids {
            self.mamba_state.restored_prefix_sequences.remove(seq_id);
        }

        FinishedMambaSync {
            captures,
            released_ids,
        }
    }

    pub fn fallback_sequence_to_full_prefill(
        &mut self,
        sequence: &Sequence,
        seq_id: usize,
        cached_tokens: usize,
        reason: &str,
    ) {
        self.mamba_state.restored_prefix_sequences.remove(&seq_id);
        let rebuilt = self
            .block_engine
            .fallback_sequence_to_full_prefill(sequence);
        if rebuilt {
            tracing::warn!(
                "Seq {} {} (cached {} tokens); rebuilt block table and falling back to full prefill",
                seq_id,
                reason,
                cached_tokens
            );
        } else {
            tracing::warn!(
                "Seq {} {} (cached {} tokens); unable to rebuild block table due memory pressure, keeping cached prefill",
                seq_id,
                reason,
                cached_tokens
            );
        }
    }

    fn ordered_group_sequences(group: &Arc<SequenceGroup>) -> Vec<Arc<Sequence>> {
        let mut seqs = group
            .get_seqs()
            .iter()
            .map(|(seq_id, seq)| (*seq_id, Arc::clone(seq)))
            .collect::<Vec<_>>();
        seqs.sort_unstable_by_key(|(seq_id, _)| *seq_id);
        seqs.into_iter().map(|(_, seq)| seq).collect()
    }
}
