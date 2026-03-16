use std::{collections::VecDeque, sync::Arc};

use candle_core::{Device, Result, Tensor};

#[cfg(feature = "flashinfer")]
use attention_rs::FlashInferMetadata;

use super::{LLMEngine, PreparedInputs, Sequence, SequenceGroup, PREFILL_CHUNK_SIZE, _PAD_SLOT_ID};
use crate::InputMetadata;

impl LLMEngine {
    pub fn prepare_block_tables(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        device: &Device,
    ) -> Result<Tensor> {
        let mut ordered_sequences = Vec::<Arc<Sequence>>::new();
        for group in groups {
            ordered_sequences.extend(Self::ordered_group_sequences(group));
        }

        let mut max_len = 0;
        for seq in &ordered_sequences {
            let len = self
                .scheduler
                .block_engine
                .block_tables
                .get(&seq.deref().get_id())
                .unwrap()
                .len();
            if len > max_len {
                max_len = len;
            }
        }
        let mut flat: Vec<u32> = Vec::with_capacity(ordered_sequences.len() * max_len);

        for seq in &ordered_sequences {
            let table = self
                .scheduler
                .block_engine
                .block_tables
                .get(&seq.deref().get_id())
                .unwrap();
            let table = table
                .iter()
                .map(|block| block.deref_mut().block_id as u32)
                .collect::<Vec<_>>();

            let bt = if let Some(sliding_window) = self.config.sliding_window {
                let sliding_window_blocks = sliding_window / self.cache_config.block_size;
                let slide_idx = if table.len() > sliding_window_blocks {
                    table.len() - sliding_window_blocks
                } else {
                    0
                };
                table.get(slide_idx..).unwrap().to_vec()
            } else {
                table
            };

            flat.extend_from_slice(bt.as_slice());
            flat.extend(std::iter::repeat(0).take(max_len - bt.len()));
        }

        Tensor::from_vec(flat, (ordered_sequences.len(), max_len), device)
    }

    pub fn prepare_mamba_slot_mapping(
        &self,
        sequence_ids: &[usize],
        is_prefill: bool,
        rank: usize,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        let (pipeline, _) = self.get_pipeline(rank).unwrap();
        let slots = if is_prefill {
            pipeline.ensure_mamba_slots_for_sequences(sequence_ids)?
        } else {
            pipeline.get_mamba_slots_for_sequences(sequence_ids)?
        };

        if slots.is_empty() {
            return Ok(None);
        }

        let slots_i64 = slots.into_iter().map(|s| s as i64).collect::<Vec<_>>();
        let len = slots_i64.len();
        Ok(Some(Tensor::from_vec(slots_i64, (len,), device)?))
    }

    pub fn prepare_prompt(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        device: &Device,
        rank: usize,
    ) -> Result<PreparedInputs> {
        let mut context_lens = Vec::new();
        let mut input_ids: Vec<u32> = Vec::new();
        let mut positions = Vec::new();
        let mut cu_seqlens_q = vec![0];
        let mut cu_seqlens_k = vec![0];
        let mut sequence_ids = Vec::new();
        let mut max_seqlen_q = 0;
        let mut max_seqlen_k = 0;
        let mut slot_mapping = Vec::new();
        let chunk_size = self.prefill_chunk_size.unwrap_or(PREFILL_CHUNK_SIZE);
        let mut max_context_len = 0;
        #[cfg(feature = "flashinfer")]
        let mut ordered_sequences = Vec::<Arc<Sequence>>::new();
        #[cfg(feature = "flashinfer")]
        let mut prefill_tokens = Vec::new();
        #[cfg(feature = "flashinfer")]
        let mut batch_indices_vec = Vec::<u32>::new();
        #[cfg(feature = "flashinfer")]
        let mut positions_vec = Vec::<u32>::new();
        for group in groups {
            for seq in Self::ordered_group_sequences(group) {
                #[cfg(feature = "flashinfer")]
                ordered_sequences.push(Arc::clone(&seq));
                let prompt_ids = seq.deref_mut().get_token_ids();
                sequence_ids.push(seq.deref().get_id());
                let seq_len = prompt_ids.len();
                if seq_len > max_context_len {
                    max_context_len = seq_len + self.cache_config.block_size;
                }
                let num_cached_tokens = seq.deref().get_num_cached_tokens();
                let num_tokens = if chunk_size > 0 {
                    std::cmp::min(chunk_size, seq_len - num_cached_tokens)
                } else {
                    seq_len - num_cached_tokens
                };
                #[cfg(feature = "flashinfer")]
                prefill_tokens.push(num_tokens);

                context_lens.push(seq_len as u32);

                let seqlen_q = num_tokens;
                let use_cached_kv = num_cached_tokens > 0
                    && ((cfg!(feature = "flashattn") || cfg!(feature = "flashinfer"))
                        || self.scheduler.prefix_cache_enabled());
                let seqlen_k = if use_cached_kv {
                    num_cached_tokens + num_tokens
                } else {
                    num_tokens
                };

                cu_seqlens_q.push(cu_seqlens_q.last().unwrap() + seqlen_q as u32);
                cu_seqlens_k.push(cu_seqlens_k.last().unwrap() + seqlen_k as u32);
                max_seqlen_q = std::cmp::max(max_seqlen_q, seqlen_q);
                max_seqlen_k = std::cmp::max(max_seqlen_k, seqlen_k);

                input_ids
                    .extend(prompt_ids[num_cached_tokens..num_cached_tokens + num_tokens].to_vec());
                positions.extend(
                    (num_cached_tokens as i64..(num_cached_tokens + num_tokens) as i64)
                        .collect::<Vec<_>>(),
                );
                #[cfg(feature = "flashinfer")]
                {
                    let batch_idx = (sequence_ids.len() - 1) as u32;
                    batch_indices_vec.extend(std::iter::repeat(batch_idx).take(num_tokens));
                    positions_vec.extend(
                        (num_cached_tokens as u32..(num_cached_tokens + num_tokens) as u32)
                            .collect::<Vec<_>>(),
                    );
                }
                let table = self
                    .scheduler
                    .block_engine
                    .block_tables
                    .get(&seq.deref().get_id());
                if table.is_none() {
                    slot_mapping.extend([_PAD_SLOT_ID].repeat(num_tokens));
                    continue;
                }
                let table = table
                    .unwrap()
                    .iter()
                    .map(|block| block.deref_mut().block_id)
                    .collect::<Vec<_>>();

                let start_idx = if let Some(sliding_window) = self.config.sliding_window {
                    if seq_len > sliding_window {
                        0.min(seq_len - sliding_window)
                    } else {
                        0
                    }
                } else {
                    0
                };

                for i in num_cached_tokens..num_cached_tokens + num_tokens {
                    if i < start_idx {
                        slot_mapping.push(_PAD_SLOT_ID);
                        continue;
                    }

                    let block_number = if i / self.cache_config.block_size >= table.len() {
                        candle_core::bail!(
                            "Block table is too small (prompt)! i={} block_size={} table_len={}",
                            i,
                            self.cache_config.block_size,
                            table.len()
                        );
                    } else {
                        table.get(i / self.cache_config.block_size).unwrap()
                    };
                    let block_offset = i % self.cache_config.block_size;
                    let slot = block_number * self.cache_config.block_size + block_offset;
                    slot_mapping.push(slot as i64);
                }
            }
        }

        assert!(
            input_ids.len() > 0 && positions.len() > 0 && slot_mapping.len() > 0,
            "Invalid inputs!"
        );
        if input_ids.len() != slot_mapping.len() {
            candle_core::bail!(
                "input_ids and slot_mapping must have same length: {}, {}",
                input_ids.len(),
                slot_mapping.len()
            );
        }
        if input_ids.len() != *cu_seqlens_q.last().unwrap() as usize {
            candle_core::bail!("input_ids length must match last cu_seqlens_q");
        }

        let length = input_ids.len();
        let input_ids = Tensor::from_vec(input_ids, (length,), device)?;
        let positions = Tensor::from_vec(positions, (length,), device)?;
        let q_len = cu_seqlens_q.len();
        let k_len = cu_seqlens_k.len();
        let s_len = slot_mapping.len();

        let slot_mapping = Tensor::from_vec(slot_mapping, (s_len,), device)?;

        let (context_lens, block_tables) = if cu_seqlens_k.last() > cu_seqlens_q.last() {
            let len = context_lens.len();
            let context_lens_t = Tensor::from_vec(context_lens, len, device)?;
            let block_tables_t = self.prepare_block_tables(groups, device)?;
            (Some(context_lens_t), Some(block_tables_t))
        } else {
            (None, None)
        };
        let cu_seqlens_q_vec = cu_seqlens_q.clone();
        let cu_seqlens_q = Tensor::from_vec(cu_seqlens_q, (q_len,), device)?;
        let cu_seqlens_k = Tensor::from_vec(cu_seqlens_k, (k_len,), device)?;
        let mamba_slot_mapping =
            self.prepare_mamba_slot_mapping(&sequence_ids, true, rank, device)?;
        #[cfg(feature = "flashinfer")]
        let flashinfer_metadata = if self.flashinfer_kv_params_for_rank(rank)?.is_some() {
            let mut indptr = vec![0u32];
            let mut indices = Vec::new();
            let mut last_len = Vec::new();
            for (seq, &num_tokens) in ordered_sequences.iter().zip(prefill_tokens.iter()) {
                let effective_len = seq.deref().get_num_cached_tokens() + num_tokens;
                let Some(table) = self
                    .scheduler
                    .block_engine
                    .block_tables
                    .get(&seq.deref().get_id())
                else {
                    indptr.push(indices.len() as u32);
                    last_len.push(0);
                    continue;
                };
                let table = table
                    .iter()
                    .map(|block| block.deref_mut().block_id as u32)
                    .collect::<Vec<_>>();
                let max_blocks = table.len();
                let num_blocks = if effective_len == 0 {
                    0
                } else {
                    (effective_len + self.cache_config.block_size - 1)
                        / self.cache_config.block_size
                };
                let num_blocks = std::cmp::min(num_blocks, max_blocks);
                indices.extend(table[..num_blocks].iter().copied());
                indptr.push(indices.len() as u32);
                let last = if effective_len == 0 {
                    0
                } else {
                    (effective_len - 1) % self.cache_config.block_size + 1
                };
                last_len.push(last as u32);
            }
            if let Some(limit) = self.cache_config.num_gpu_blocks {
                if let Some((pos, &bad_idx)) = indices
                    .iter()
                    .enumerate()
                    .find(|(_, idx)| **idx as usize >= limit)
                {
                    candle_core::bail!(
                        "flashinfer prefill block index out of range: indices[{}]={} >= num_gpu_blocks ({})",
                        pos,
                        bad_idx,
                        limit
                    );
                }
            }
            let indptr_host = indptr.clone();
            let last_len_host = last_len.clone();
            let mut kv_len_arr_host = Vec::with_capacity(last_len_host.len());
            for i in 0..last_len_host.len() {
                let num_pages = indptr_host[i + 1] - indptr_host[i];
                if num_pages == 0 {
                    kv_len_arr_host.push(0);
                } else {
                    let full = (num_pages - 1) * self.cache_config.block_size as u32;
                    kv_len_arr_host.push(full + last_len_host[i]);
                }
            }
            Some(FlashInferMetadata {
                indptr: Tensor::from_vec(indptr, (indptr_host.len(),), device)?,
                indptr_host,
                indices: Tensor::from_vec(indices.clone(), (indices.len(),), device)?,
                last_len: Tensor::from_vec(last_len, (last_len_host.len(),), device)?,
                last_len_host: Some(last_len_host),
                kv_len_arr_host: Some(kv_len_arr_host),
                cu_seqlens_q_host: Some(cu_seqlens_q_vec.clone()),
                total_num_rows: Some(*cu_seqlens_q_vec.last().unwrap()),
                batch_indices: Some(Tensor::from_vec(batch_indices_vec, (length,), device)?),
                positions: Some(Tensor::from_vec(positions_vec, (length,), device)?),
                use_cuda_graph: false,
                decode_plan_info: None,
            })
        } else {
            None
        };
        #[cfg(not(feature = "flashinfer"))]
        let flashinfer_metadata = None;

        let input_metadata = InputMetadata {
            is_prefill: true,
            sequence_ids: Some(sequence_ids),
            mamba_slot_mapping,
            slot_mapping,
            block_tables,
            context_lens,
            cu_seqlens_q: Some(cu_seqlens_q),
            cu_seqlens_k: Some(cu_seqlens_k),
            max_seqlen_q,
            max_seqlen_k,
            max_context_len,
            disable_flash_attn: None,
            seqlens: Some(cu_seqlens_q_vec[1..].to_vec()),
            flashinfer_metadata,
        };

        Ok(PreparedInputs {
            tokens: input_ids,
            positions,
            metadata: input_metadata,
        })
    }

    pub fn prepare_decode(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        device: &Device,
        rank: usize,
    ) -> Result<PreparedInputs> {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut sequence_ids = Vec::new();
        let mut context_lens = Vec::new();
        let mut block_tables = Vec::new();
        for group in groups {
            for seq in Self::ordered_group_sequences(group) {
                sequence_ids.push(seq.deref().get_id());
                let last_token_id = seq.deref_mut().get_last_token_id();
                input_ids.push(last_token_id);
                let position = seq.deref_mut().get_len() - 1;
                positions.push(position as i64);

                let context_len = if let Some(sliding_window) = self.config.sliding_window {
                    seq.deref_mut().get_len().min(sliding_window)
                } else {
                    seq.deref_mut().get_len()
                };
                context_lens.push(context_len as u32);

                let table = self
                    .scheduler
                    .block_engine
                    .block_tables
                    .get(&seq.deref().get_id())
                    .unwrap();
                let table = table
                    .iter()
                    .map(|block| block.deref_mut().block_id)
                    .collect::<Vec<_>>();

                let block_number = if position / self.cache_config.block_size >= table.len() {
                    candle_core::bail!(
                        "Block table is too small (completion)! start_pos={} block_size={} table_len={}",
                        position,
                        self.cache_config.block_size,
                        table.len()
                    );
                } else {
                    table.get(position / self.cache_config.block_size).unwrap()
                };
                let block_offset = position % self.cache_config.block_size;
                let slot = block_number * self.cache_config.block_size + block_offset;
                let slot: i64 = slot.try_into().unwrap();
                slot_mapping.push(slot);

                if let Some(sliding_window) = self.config.sliding_window {
                    let sliding_window_blocks = sliding_window / self.cache_config.block_size;
                    let slide_idx = if table.len() > sliding_window_blocks {
                        table.len() - sliding_window_blocks
                    } else {
                        0
                    };
                    block_tables.push(table.get(slide_idx..).unwrap().to_vec());
                } else {
                    block_tables.push(table);
                }
            }
        }

        let length = input_ids.len();
        let input_ids = Tensor::from_vec(input_ids, (length,), device)?;
        let positions = Tensor::from_vec(positions, (length,), device)?;
        let slot_mapping = Tensor::from_vec(slot_mapping, (length,), device)?;

        let max_context_len = context_lens.clone().into_iter().max().unwrap();
        let context_lens = Tensor::from_vec(context_lens, (length,), device)?;

        let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
        let block_tables = super::super::_make_tensor_with_pad(
            block_tables
                .iter()
                .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            max_block_table_len,
            0,
            device,
        )?;
        let block_tables = block_tables.reshape(((), max_block_table_len))?;
        let mamba_slot_mapping =
            self.prepare_mamba_slot_mapping(&sequence_ids, false, rank, device)?;
        #[cfg(feature = "flashinfer")]
        let flashinfer_metadata = if self.flashinfer_kv_params_for_rank(rank)?.is_some() {
            #[cfg(all(feature = "cuda", feature = "graph"))]
            let use_cuda_graph = {
                let (pipeline, _) = self.get_pipeline(rank).ok_or_else(|| {
                    candle_core::Error::msg(format!("missing pipeline for rank {rank}"))
                })?;
                pipeline.capturer.is_exact_captured(length)
            };
            #[cfg(not(all(feature = "cuda", feature = "graph")))]
            let use_cuda_graph = false;

            let mut indptr = vec![0u32];
            let mut indices = Vec::new();
            let mut last_len = Vec::new();
            for group in groups {
                for seq in Self::ordered_group_sequences(group) {
                    let table = self
                        .scheduler
                        .block_engine
                        .block_tables
                        .get(&seq.deref().get_id())
                        .unwrap()
                        .iter()
                        .map(|block| block.deref_mut().block_id as u32)
                        .collect::<Vec<_>>();
                    indices.extend(table);
                    indptr.push(indices.len() as u32);
                    let len = seq.deref().get_len();
                    let last = if len == 0 {
                        0
                    } else {
                        (len - 1) % self.cache_config.block_size + 1
                    };
                    last_len.push(last as u32);
                }
            }
            if let Some(limit) = self.cache_config.num_gpu_blocks {
                if let Some((pos, &bad_idx)) = indices
                    .iter()
                    .enumerate()
                    .find(|(_, idx)| **idx as usize >= limit)
                {
                    candle_core::bail!(
                        "flashinfer decode block index out of range: indices[{}]={} >= num_gpu_blocks ({})",
                        pos,
                        bad_idx,
                        limit
                    );
                }
            }
            let indptr_host = indptr.clone();
            let last_len_host = last_len.clone();
            let mut kv_len_arr_host = Vec::with_capacity(last_len_host.len());
            for i in 0..last_len_host.len() {
                let num_pages = indptr_host[i + 1] - indptr_host[i];
                if num_pages == 0 {
                    kv_len_arr_host.push(0);
                } else {
                    let full = (num_pages - 1) * self.cache_config.block_size as u32;
                    kv_len_arr_host.push(full + last_len_host[i]);
                }
            }
            Some(FlashInferMetadata {
                indptr: Tensor::from_vec(indptr, (length + 1,), device)?,
                indptr_host,
                indices: Tensor::from_vec(indices.clone(), (indices.len(),), device)?,
                last_len: Tensor::from_vec(last_len, (length,), device)?,
                last_len_host: Some(last_len_host),
                kv_len_arr_host: Some(kv_len_arr_host),
                cu_seqlens_q_host: None,
                total_num_rows: None,
                batch_indices: None,
                positions: None,
                use_cuda_graph,
                decode_plan_info: None,
            })
        } else {
            None
        };
        #[cfg(not(feature = "flashinfer"))]
        let flashinfer_metadata = None;
        let input_metadata = InputMetadata {
            is_prefill: false,
            sequence_ids: Some(sequence_ids),
            mamba_slot_mapping,
            slot_mapping,
            block_tables: Some(block_tables),
            context_lens: Some(context_lens),
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            max_context_len: max_context_len as usize,
            disable_flash_attn: None,
            seqlens: None,
            flashinfer_metadata,
        };

        Ok(PreparedInputs {
            tokens: input_ids,
            positions,
            metadata: input_metadata,
        })
    }
}
