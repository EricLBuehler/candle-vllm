use std::{collections::HashMap, sync::Arc, time::SystemTime};

use candle_core::{Device, Result, Tensor};
use parking_lot::RwLock;

use crate::openai::communicator::{FlashInferHostData, ForwardPayload, MessageType};
use crate::openai::multimodal::ImageData;
use crate::InputMetadata;

use super::{ChatChoice, ChatCompletionUsageResponse, LLMEngine, PreparedInputs, SequenceGroup};
use std::collections::VecDeque;

impl LLMEngine {
    #[allow(dead_code)]
    fn extract_forward_payload(prepared: &PreparedInputs) -> Result<ForwardPayload> {
        let input_ids = prepared.tokens.to_vec1::<u32>()?;
        let positions = prepared.positions.to_vec1::<i64>()?;
        let slot_mapping = prepared.metadata.slot_mapping.to_vec1::<i64>()?;

        let (block_tables_flat, block_tables_rows, block_tables_cols) =
            if let Some(bt) = &prepared.metadata.block_tables {
                let shape = bt.dims();
                let (rows, cols) = if shape.len() == 2 {
                    (shape[0], shape[1])
                } else {
                    (0, 0)
                };
                (bt.flatten_all()?.to_vec1::<u32>()?, rows, cols)
            } else {
                (vec![], 0, 0)
            };

        let context_lens = prepared
            .metadata
            .context_lens
            .as_ref()
            .map(|t| t.to_vec1::<u32>())
            .transpose()?;

        let cu_seqlens_q = prepared
            .metadata
            .cu_seqlens_q
            .as_ref()
            .map(|t| t.to_vec1::<u32>())
            .transpose()?;

        let cu_seqlens_k = prepared
            .metadata
            .cu_seqlens_k
            .as_ref()
            .map(|t| t.to_vec1::<u32>())
            .transpose()?;

        #[cfg(feature = "flashinfer")]
        let flashinfer_host =
            prepared
                .metadata
                .flashinfer_metadata
                .as_ref()
                .map(|fm| FlashInferHostData {
                    indptr: fm.indptr_host.clone(),
                    indices: fm.indices.to_vec1::<u32>().unwrap_or_default(),
                    last_len: fm.last_len_host.clone().unwrap_or_default(),
                    kv_len_arr: fm.kv_len_arr_host.clone().unwrap_or_default(),
                    use_cuda_graph: fm.use_cuda_graph,
                });
        #[cfg(not(feature = "flashinfer"))]
        let flashinfer_host: Option<FlashInferHostData> = None;

        Ok(ForwardPayload {
            input_ids,
            positions,
            slot_mapping,
            block_tables_flat,
            block_tables_rows,
            block_tables_cols,
            context_lens,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q: prepared.metadata.max_seqlen_q,
            max_seqlen_k: prepared.metadata.max_seqlen_k,
            max_context_len: prepared.metadata.max_context_len,
            sequence_ids: prepared.metadata.sequence_ids.clone().unwrap_or_default(),
            seqlens: prepared.metadata.seqlens.clone(),
            is_prefill: prepared.metadata.is_prefill,
            is_mla: prepared.metadata.is_mla,
            flashinfer_host,
        })
    }

    fn build_forward_payload_from_scheduler(
        engine: &Self,
        scheduled: &VecDeque<Arc<SequenceGroup>>,
        prepared: &PreparedInputs,
    ) -> Result<ForwardPayload> {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut slot_mapping_vec = Vec::new();
        let mut sequence_ids = Vec::new();
        let mut context_lens_vec = Vec::new();
        let mut block_tables_vecs: Vec<Vec<u32>> = Vec::new();
        let is_prefill = prepared.metadata.is_prefill;

        for group in scheduled {
            for seq in Self::ordered_group_sequences(group) {
                let seq_id = seq.deref().get_id();
                sequence_ids.push(seq_id);

                if is_prefill {
                    let prompt_ids = seq.deref_mut().get_token_ids();
                    let seq_len = prompt_ids.len();
                    let num_cached_tokens = seq.deref().get_num_cached_tokens();
                    let chunk_size = engine
                        .prefill_chunk_size
                        .unwrap_or(super::PREFILL_CHUNK_SIZE);
                    let num_tokens = if chunk_size > 0 {
                        std::cmp::min(chunk_size, seq_len - num_cached_tokens)
                    } else {
                        seq_len - num_cached_tokens
                    };
                    context_lens_vec.push((num_cached_tokens + num_tokens) as u32);
                    input_ids.extend(
                        prompt_ids[num_cached_tokens..num_cached_tokens + num_tokens].to_vec(),
                    );
                    positions.extend(
                        (num_cached_tokens as i64..(num_cached_tokens + num_tokens) as i64)
                            .collect::<Vec<_>>(),
                    );
                    let table = engine.scheduler.block_engine.block_tables.get(&seq_id);
                    if let Some(table) = table {
                        let table: Vec<usize> = table
                            .iter()
                            .map(|block| block.deref_mut().block_id)
                            .collect();
                        for i in num_cached_tokens..num_cached_tokens + num_tokens {
                            let start_idx = if let Some(sw) = engine.config.sliding_window {
                                if seq_len > sw {
                                    0.min(seq_len - sw)
                                } else {
                                    0
                                }
                            } else {
                                0
                            };
                            if i < start_idx {
                                slot_mapping_vec.push(super::_PAD_SLOT_ID);
                            } else if i / engine.cache_config.block_size >= table.len() {
                                slot_mapping_vec.push(super::_PAD_SLOT_ID);
                            } else {
                                let block_number = table[i / engine.cache_config.block_size];
                                let block_offset = i % engine.cache_config.block_size;
                                let slot =
                                    block_number * engine.cache_config.block_size + block_offset;
                                slot_mapping_vec.push(slot as i64);
                            }
                        }
                    } else {
                        slot_mapping_vec.extend([super::_PAD_SLOT_ID].repeat(num_tokens));
                    }
                } else {
                    let last_token_id = seq.deref_mut().get_last_token_id();
                    input_ids.push(last_token_id);
                    let position = seq.deref_mut().get_len() - 1;
                    positions.push(position as i64);
                    let context_len = if let Some(sliding_window) = engine.config.sliding_window {
                        seq.deref_mut().get_len().min(sliding_window)
                    } else {
                        seq.deref_mut().get_len()
                    };
                    context_lens_vec.push(context_len as u32);

                    let table = engine
                        .scheduler
                        .block_engine
                        .block_tables
                        .get(&seq_id)
                        .unwrap();
                    let table: Vec<usize> = table
                        .iter()
                        .map(|block| block.deref_mut().block_id)
                        .collect();
                    let block_number = table[position / engine.cache_config.block_size];
                    let block_offset = position % engine.cache_config.block_size;
                    let slot = block_number * engine.cache_config.block_size + block_offset;
                    slot_mapping_vec.push(slot as i64);

                    let used_blocks = Self::used_blocks_for_len(
                        seq.deref().get_len(),
                        engine.cache_config.block_size,
                        table.len(),
                    );
                    let bt = table[..used_blocks]
                        .iter()
                        .map(|&x| x as u32)
                        .collect::<Vec<_>>();
                    if let Some(sliding_window) = engine.config.sliding_window {
                        let swb = sliding_window / engine.cache_config.block_size;
                        let slide_idx = if bt.len() > swb { bt.len() - swb } else { 0 };
                        block_tables_vecs.push(bt[slide_idx..].to_vec());
                    } else {
                        block_tables_vecs.push(bt);
                    }
                }
            }
        }

        let (block_tables_flat, block_tables_rows, block_tables_cols) =
            if !block_tables_vecs.is_empty() {
                let max_len = block_tables_vecs.iter().map(|v| v.len()).max().unwrap_or(0);
                let rows = block_tables_vecs.len();
                let mut flat = Vec::with_capacity(rows * max_len);
                for bt in &block_tables_vecs {
                    flat.extend_from_slice(bt);
                    flat.extend(std::iter::repeat(0u32).take(max_len - bt.len()));
                }
                (flat, rows, max_len)
            } else if is_prefill {
                let bt_from_metadata = prepared.metadata.block_tables.as_ref();
                if let Some(bt) = bt_from_metadata {
                    let shape = bt.dims();
                    if shape.len() == 2 {
                        (bt.flatten_all()?.to_vec1::<u32>()?, shape[0], shape[1])
                    } else {
                        (vec![], 0, 0)
                    }
                } else {
                    (vec![], 0, 0)
                }
            } else {
                (vec![], 0, 0)
            };

        let cu_seqlens_q = prepared
            .metadata
            .cu_seqlens_q
            .as_ref()
            .map(|t| t.to_vec1::<u32>())
            .transpose()?;
        let cu_seqlens_k = prepared
            .metadata
            .cu_seqlens_k
            .as_ref()
            .map(|t| t.to_vec1::<u32>())
            .transpose()?;

        #[cfg(feature = "flashinfer")]
        let flashinfer_host =
            prepared
                .metadata
                .flashinfer_metadata
                .as_ref()
                .map(|fm| FlashInferHostData {
                    indptr: fm.indptr_host.clone(),
                    indices: fm.indices.to_vec1::<u32>().unwrap_or_default(),
                    last_len: fm.last_len_host.clone().unwrap_or_default(),
                    kv_len_arr: fm.kv_len_arr_host.clone().unwrap_or_default(),
                    use_cuda_graph: fm.use_cuda_graph,
                });
        #[cfg(not(feature = "flashinfer"))]
        let flashinfer_host: Option<FlashInferHostData> = None;

        Ok(ForwardPayload {
            input_ids,
            positions,
            slot_mapping: slot_mapping_vec,
            block_tables_flat,
            block_tables_rows,
            block_tables_cols,
            context_lens: Some(context_lens_vec),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q: prepared.metadata.max_seqlen_q,
            max_seqlen_k: prepared.metadata.max_seqlen_k,
            max_context_len: prepared.metadata.max_context_len,
            sequence_ids,
            seqlens: prepared.metadata.seqlens.clone(),
            is_prefill,
            is_mla: prepared.metadata.is_mla,
            flashinfer_host,
        })
    }

    fn rebuild_inputs_from_payload(
        payload: &ForwardPayload,
        device: &Device,
    ) -> Result<PreparedInputs> {
        let length = payload.input_ids.len();
        let tokens = Tensor::from_vec(payload.input_ids.clone(), (length,), device)?;
        let positions = Tensor::from_vec(payload.positions.clone(), (length,), device)?;
        let s_len = payload.slot_mapping.len();
        let slot_mapping = Tensor::from_vec(payload.slot_mapping.clone(), (s_len,), device)?;

        let block_tables = if payload.block_tables_rows > 0 {
            Some(Tensor::from_vec(
                payload.block_tables_flat.clone(),
                (payload.block_tables_rows, payload.block_tables_cols),
                device,
            )?)
        } else {
            None
        };

        let context_lens = payload
            .context_lens
            .as_ref()
            .map(|v| {
                let len = v.len();
                Tensor::from_vec(v.clone(), (len,), device)
            })
            .transpose()?;

        let cu_seqlens_q = payload
            .cu_seqlens_q
            .as_ref()
            .map(|v| {
                let len = v.len();
                Tensor::from_vec(v.clone(), (len,), device)
            })
            .transpose()?;

        let cu_seqlens_k = payload
            .cu_seqlens_k
            .as_ref()
            .map(|v| {
                let len = v.len();
                Tensor::from_vec(v.clone(), (len,), device)
            })
            .transpose()?;

        let metadata = InputMetadata {
            is_prefill: payload.is_prefill,
            is_mla: payload.is_mla,
            sequence_ids: Some(payload.sequence_ids.clone()),
            mamba_slot_mapping: None,
            slot_mapping,
            block_tables,
            context_lens,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q: payload.max_seqlen_q,
            max_seqlen_k: payload.max_seqlen_k,
            max_context_len: payload.max_context_len,
            seqlens: payload.seqlens.clone(),
            flashinfer_metadata: None,
            is_mtp_verify: false,
        };

        Ok(PreparedInputs {
            tokens,
            positions,
            metadata,
        })
    }

    fn daemon_run_forward(engine: &Arc<RwLock<Self>>, payload: &ForwardPayload) -> Result<()> {
        let (pipeline_entry, prepared) = {
            let mut guard = engine.write();
            let (pipeline, _) = guard.get_pipeline(0).unwrap();
            let device = pipeline.device().clone();

            let mut prepared = Self::rebuild_inputs_from_payload(payload, &device)?;

            prepared.metadata.mamba_slot_mapping = guard.prepare_mamba_slot_mapping(
                &payload.sequence_ids,
                payload.is_prefill,
                0,
                &device,
            )?;

            #[cfg(feature = "flashinfer")]
            {
                Self::build_daemon_flashinfer_metadata(
                    &guard,
                    payload,
                    &device,
                    &mut prepared.metadata,
                )?;
            }

            #[cfg(feature = "flashinfer")]
            if !prepared.metadata.is_prefill {
                let use_cuda_graph = prepared
                    .metadata
                    .flashinfer_metadata
                    .as_ref()
                    .map(|fm| fm.use_cuda_graph)
                    .unwrap_or(false);
                if !use_cuda_graph {
                    guard.ensure_flashinfer_decode_plan(
                        0,
                        &device,
                        prepared.tokens.dim(0)?,
                        &mut prepared.metadata,
                    )?;
                }
            }

            let pipeline_entry = guard
                .pipelines
                .remove(&0)
                .ok_or_else(|| candle_core::Error::msg("missing pipeline for daemon rank 0"))?;
            (pipeline_entry, prepared)
        };

        let (pipeline, cache_engine) = (&pipeline_entry.0, &pipeline_entry.1);
        let _logits = pipeline.forward(
            prepared.tokens,
            &prepared.positions,
            Some(&cache_engine.get_kv_cache()),
            &prepared.metadata,
            None,
        );

        let mut guard = engine.write();
        if guard.pipelines.insert(0, pipeline_entry).is_some() {
            candle_core::bail!("pipeline for daemon rank 0 was replaced while detached");
        }
        let _logits = _logits?;
        Ok(())
    }

    #[cfg(feature = "flashinfer")]
    fn build_daemon_flashinfer_metadata(
        engine: &Self,
        payload: &ForwardPayload,
        device: &Device,
        metadata: &mut InputMetadata,
    ) -> Result<()> {
        use attention_rs::FlashInferMetadata;

        let host = match &payload.flashinfer_host {
            Some(h) => h,
            None => return Ok(()),
        };

        let params = match engine.flashinfer_kv_params_for_rank(0)? {
            Some(p) => p,
            None => return Ok(()),
        };

        let indptr_host = host.indptr.clone();
        let last_len_host = host.last_len.clone();
        let kv_len_arr_host = host.kv_len_arr.clone();
        let num_seqs = payload.sequence_ids.len();
        let length = payload.input_ids.len();

        #[cfg(all(feature = "cuda", feature = "graph"))]
        let use_cuda_graph = if !payload.is_prefill {
            let (pipeline, _) = engine.get_pipeline(0).unwrap();
            let require_exact_graph = metadata.mamba_slot_mapping.is_some();
            if require_exact_graph {
                pipeline.capturer.is_exact_captured(length)
            } else {
                pipeline.capturer.is_captured(length)
            }
        } else {
            false
        };
        #[cfg(not(all(feature = "cuda", feature = "graph")))]
        let use_cuda_graph = host.use_cuda_graph;

        let (batch_indices, positions_fi) = if payload.is_prefill {
            let mut bi = Vec::new();
            let mut pv = Vec::new();
            if let Some(ref q_vals) = payload.cu_seqlens_q {
                for seq_idx in 0..num_seqs {
                    let start = q_vals[seq_idx] as usize;
                    let end = q_vals[seq_idx + 1] as usize;
                    let num_tokens = end - start;
                    bi.extend(std::iter::repeat(seq_idx as u32).take(num_tokens));
                    let cached_start = if let Some(ref cl) = payload.context_lens {
                        (cl[seq_idx] as usize).saturating_sub(num_tokens)
                    } else {
                        0
                    };
                    pv.extend(
                        (cached_start as u32..(cached_start + num_tokens) as u32)
                            .collect::<Vec<_>>(),
                    );
                }
            }
            (
                Some(Tensor::from_vec(bi.clone(), (bi.len(),), device)?),
                Some(Tensor::from_vec(pv.clone(), (pv.len(),), device)?),
            )
        } else {
            (None, None)
        };

        let mut prefill_plan_info = None;
        let mut mla_prefill_plan_info = None;

        if payload.is_prefill {
            let cu_seqlens_q_vec = payload.cu_seqlens_q.as_ref().unwrap();
            let total_num_rows = *cu_seqlens_q_vec.last().unwrap();

            if payload.is_mla {
                mla_prefill_plan_info = Some(attention_rs::mla::mla_prefill_plan(
                    device,
                    cu_seqlens_q_vec,
                    &indptr_host,
                    &kv_len_arr_host,
                    last_len_host.len(),
                    params.num_qo_heads,
                    params.head_dim,
                    true,
                )?);
            } else {
                prefill_plan_info = Some(attention_rs::flashinfer::prefill_plan(
                    device,
                    cu_seqlens_q_vec,
                    &indptr_host,
                    &kv_len_arr_host,
                    total_num_rows,
                    last_len_host.len(),
                    params.num_qo_heads,
                    params.num_kv_heads,
                    params.head_dim,
                    params.page_size,
                    params.out_dtype,
                    None,
                    Some(params.kv_dtype),
                    false,
                )?);
            }
        }

        let indices = &host.indices;
        let last_len = &host.last_len;

        metadata.flashinfer_metadata = Some(FlashInferMetadata {
            indptr: Tensor::from_vec(indptr_host.clone(), (indptr_host.len(),), device)?,
            indptr_host,
            indices: Tensor::from_vec(indices.clone(), (indices.len(),), device)?,
            last_len: Tensor::from_vec(last_len.clone(), (last_len_host.len(),), device)?,
            last_len_host: Some(last_len_host),
            kv_len_arr_host: Some(kv_len_arr_host),
            total_num_rows: if payload.is_prefill {
                payload.cu_seqlens_q.as_ref().map(|v| *v.last().unwrap())
            } else {
                None
            },
            batch_indices,
            positions: positions_fi,
            use_cuda_graph,
            decode_plan_info: None,
            prefill_plan_info,
            mla_decode_plan_info: None,
            mla_prefill_plan_info,
        });

        Ok(())
    }

    fn broadcast_forward(&self, payload: &ForwardPayload) {
        let mut dm = self.daemon_manager.write();
        let _ = dm
            .as_mut()
            .unwrap()
            .send_message(&MessageType::RunForward(payload.clone()));
    }

    fn broadcast_finish_sequences(&self, seq_ids: &[usize]) {
        if seq_ids.is_empty() {
            return;
        }
        let mut dm = self.daemon_manager.write();
        let _ = dm
            .as_mut()
            .unwrap()
            .send_message(&MessageType::FinishSequences(seq_ids.to_vec()));
    }

    pub(crate) fn broadcast_shutdown(&self) {
        let mut dm = self.daemon_manager.write();
        let _ = dm.as_mut().unwrap().send_message(&MessageType::Shutdown);
    }

    fn daemon_finish_sequences(engine: &Arc<RwLock<Self>>, seq_ids: &[usize]) {
        let guard = engine.read();
        let (pipeline, _) = match guard.get_pipeline(0) {
            Some(p) => p,
            None => return,
        };
        for &id in seq_ids {
            pipeline.release_sequence_state(id);
        }
    }

    fn daemon_capture_mamba_prefix(
        engine: &Arc<RwLock<Self>>,
        seq_id: usize,
        hash: u64,
        preserve: bool,
    ) -> bool {
        let guard = engine.read();
        let (pipeline, _) = match guard.get_pipeline(0) {
            Some(p) => p,
            None => return false,
        };
        pipeline
            .capture_mamba_prefix_state(seq_id, hash, preserve)
            .unwrap_or(false)
    }

    fn daemon_has_mamba_prefix(engine: &Arc<RwLock<Self>>, hash: u64) -> bool {
        let guard = engine.read();
        let (pipeline, _) = match guard.get_pipeline(0) {
            Some(p) => p,
            None => return false,
        };
        pipeline.has_mamba_prefix_state(hash).unwrap_or(false)
    }

    fn daemon_restore_mamba_prefix(engine: &Arc<RwLock<Self>>, seq_id: usize, hash: u64) -> bool {
        let guard = engine.read();
        let (pipeline, _) = match guard.get_pipeline(0) {
            Some(p) => p,
            None => return false,
        };
        if pipeline
            .ensure_mamba_slots_for_sequences(&[seq_id])
            .is_err()
        {
            return false;
        }
        pipeline
            .restore_mamba_prefix_state(seq_id, hash)
            .unwrap_or(false)
    }

    fn execute_scheduled_batch_multiprocess(
        engine: &Arc<RwLock<Self>>,
        scheduled: &VecDeque<Arc<SequenceGroup>>,
    ) -> Result<super::BatchExecution> {
        let (pipeline_entry, tokens, positions, metadata, images, is_embedding, model_name) = {
            let mut guard = engine.write();
            let is_embedding = scheduled[0].is_embedding;
            let is_prompt_request = Self::primary_sequence(&scheduled[0]).deref().is_prompt();

            if is_prompt_request {
                guard.restore_mamba_prefix_states_for_prompt(scheduled, 0)?;
            }

            let (pipeline, _) = guard.get_pipeline(0).unwrap();
            let device = pipeline.device();
            let model_name = pipeline.name().to_string();
            #[cfg_attr(not(feature = "flashinfer"), allow(unused_mut))]
            let mut prepared = if is_prompt_request {
                guard.prepare_prompt(scheduled, device, 0)
            } else {
                guard.prepare_decode(scheduled, device, 0)
            }?;

            #[cfg(feature = "flashinfer")]
            if !prepared.metadata.is_prefill {
                let use_cuda_graph = prepared
                    .metadata
                    .flashinfer_metadata
                    .as_ref()
                    .map(|fm| fm.use_cuda_graph)
                    .unwrap_or(false);
                if !use_cuda_graph {
                    guard.ensure_flashinfer_decode_plan(
                        0,
                        device,
                        prepared.tokens.dim(0)?,
                        &mut prepared.metadata,
                    )?;
                }
            }

            let payload = Self::build_forward_payload_from_scheduler(&guard, scheduled, &prepared)?;
            guard.broadcast_forward(&payload);

            let PreparedInputs {
                tokens,
                positions,
                metadata,
            } = prepared;

            let images: Option<ImageData> = if is_prompt_request {
                let seq = Self::primary_sequence(&scheduled[0]);
                let seq_guard = seq.deref();
                let seq_images = seq_guard.get_images();
                let seq_token_ids = seq_guard.get_token_ids();
                let seq_num_cached_tokens = seq_guard.get_num_cached_tokens();
                drop(seq_guard);
                if let Some(images) = seq_images {
                    if scheduled.len() > 1 {
                        candle_core::bail!(
                            "multimodal prefill does not support batching multiple sequence groups"
                        );
                    }
                    if images.image_idx == -1 {
                        None
                    } else {
                        crate::openai::multimodal::compute_image_slice(
                            &seq_token_ids,
                            seq_num_cached_tokens,
                            &images,
                        )
                        .map(|(image_idx, token_offset)| {
                            let mut images = images.clone();
                            images.image_idx = image_idx;
                            images.image_token_offset = token_offset;
                            images
                        })
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let pipeline_entry = guard
                .pipelines
                .remove(&0)
                .ok_or_else(|| candle_core::Error::msg("missing pipeline for rank 0"))?;
            (
                pipeline_entry,
                tokens,
                positions,
                metadata,
                images,
                is_embedding,
                model_name,
            )
        };

        let (pipeline, cache_engine) = (&pipeline_entry.0, &pipeline_entry.1);
        let logits = if is_embedding {
            pipeline.forward_embedding(
                tokens,
                &positions,
                Some(&cache_engine.get_kv_cache()),
                &metadata,
            )
        } else {
            pipeline.forward(
                tokens,
                &positions,
                Some(&cache_engine.get_kv_cache()),
                &metadata,
                images.as_ref(),
            )
        };
        let mut guard = engine.write();
        if guard.pipelines.insert(0, pipeline_entry).is_some() {
            candle_core::bail!("pipeline for rank 0 was replaced while detached");
        }
        let logits = logits?;

        Ok(super::BatchExecution {
            logits,
            is_prompt: metadata.is_prefill,
            is_embedding,
            model_name,
        })
    }

    pub fn run_daemon_loop(engine: Arc<RwLock<Self>>) {
        {
            let e = engine.read();
            e.bind_rank_to_thread(0);
        }
        tracing::warn!("Daemon process entering stateless forward-only loop");

        loop {
            let msg = {
                let e = engine.read();
                let mut dm = e.daemon_manager.write();
                dm.as_mut().unwrap().receive_message()
            };

            match msg {
                Ok(MessageType::RunForward(payload)) => {
                    if let Err(e) = Self::daemon_run_forward(&engine, &payload) {
                        tracing::error!("Daemon forward failed: {:?}", e);
                    }
                }
                Ok(MessageType::FinishSequences(seq_ids)) => {
                    Self::daemon_finish_sequences(&engine, &seq_ids);
                }
                Ok(MessageType::MambaPrefixCapture {
                    seq_id,
                    hash,
                    preserve,
                }) => {
                    let result = Self::daemon_capture_mamba_prefix(&engine, seq_id, hash, preserve);
                    let e = engine.read();
                    let mut dm = e.daemon_manager.write();
                    let _ = dm
                        .as_mut()
                        .unwrap()
                        .send_to_main(&MessageType::MambaPrefixCaptureResponse(result));
                }
                Ok(MessageType::MambaPrefixHas(hash)) => {
                    let result = Self::daemon_has_mamba_prefix(&engine, hash);
                    let e = engine.read();
                    let mut dm = e.daemon_manager.write();
                    let _ = dm
                        .as_mut()
                        .unwrap()
                        .send_to_main(&MessageType::MambaPrefixHasResponse(result));
                }
                Ok(MessageType::MambaPrefixRestore { seq_id, hash }) => {
                    let result = Self::daemon_restore_mamba_prefix(&engine, seq_id, hash);
                    let e = engine.read();
                    let mut dm = e.daemon_manager.write();
                    let _ = dm
                        .as_mut()
                        .unwrap()
                        .send_to_main(&MessageType::MambaPrefixRestoreResponse(result));
                }
                Ok(MessageType::Shutdown) => {
                    tracing::warn!("Daemon: shutdown received, exiting");
                    break;
                }
                Ok(MessageType::Close) => {
                    tracing::warn!("Daemon: close received, exiting");
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::error!("Daemon: IPC error: {:?}, exiting", e);
                    break;
                }
            }
        }
    }

    pub fn generate_once_multiprocess(
        engine: Arc<RwLock<Self>>,
        _rank: usize,
    ) -> Result<HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>> {
        {
            let e = engine.read();
            e.bind_rank_to_thread(0);
        }

        let mut responses: HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)> =
            HashMap::new();
        let mut prompt_finish_times: HashMap<usize, SystemTime> = HashMap::new();

        loop {
            {
                let mut e = engine.write();
                e.move_waiting_tasks_to_scheduler();
            }

            if !engine.read().has_unfinished_sequences() {
                break;
            }

            engine.write().schedule_current_batch(0)?;

            let mut scheduled = engine.read().current_scheduled_groups();
            if scheduled.is_empty() {
                std::thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }

            let is_prompt = Self::primary_sequence(scheduled.front().unwrap())
                .deref()
                .is_prompt();

            if is_prompt {
                let prefix_decisions = {
                    let mut e = engine.write();
                    e.planned_prompt_cache_statuses(&scheduled, 0)?
                };
                let decisions: Vec<(usize, usize)> = prefix_decisions
                    .iter()
                    .map(|(seq_id, cached, available)| {
                        (*seq_id, if *available { *cached } else { 0 })
                    })
                    .collect();
                if decisions.iter().any(|(_, c)| *c == 0) {
                    let targets: Vec<(usize, usize)> = decisions
                        .iter()
                        .filter(|(_, cached)| *cached == 0)
                        .copied()
                        .collect();
                    if !targets.is_empty() {
                        let mut e = engine.write();
                        e.apply_prompt_mamba_targets(&scheduled, &targets)?;
                    }
                }
            }

            let mut batch = Self::execute_scheduled_batch_multiprocess(&engine, &scheduled)?;

            if batch.is_prompt && !batch.is_embedding {
                let aborted_sequences = Self::disconnected_stream_sequence_ids(&scheduled);
                if !aborted_sequences.is_empty() {
                    let kept_indices = {
                        let mut e = engine.write();
                        e.abort_sequences_and_prune_scheduled(&mut scheduled, &aborted_sequences)
                    };
                    if scheduled.is_empty() {
                        engine.read().clear_current_scheduled_groups();
                        continue;
                    }
                    batch.logits = Self::select_logits_rows(&batch.logits, &kept_indices)?;
                }
            }

            {
                let (embedding_done, prefill_continues) = {
                    let mut e = engine.write();
                    let embedding_done = e.process_embedding_batch(&scheduled, &batch, 0)?;
                    let prefill_continues = if embedding_done {
                        false
                    } else {
                        e.process_prefill_progress(&mut scheduled, &mut batch, 0)?
                    };
                    (embedding_done, prefill_continues)
                };
                if embedding_done || prefill_continues {
                    continue;
                }
            }

            let results = {
                let mut e = engine.write();
                let default_pipeline = e.get_mut_pipeline(0usize).unwrap().0.as_mut();
                default_pipeline.sample(&batch.logits, &scheduled)?
            };

            engine.read().clear_current_scheduled_groups();

            {
                let mut e = engine.write();
                e.apply_sample_results(0, &scheduled, results, &mut prompt_finish_times)?;
                e.finalize_post_sampling(&scheduled, 0, &prompt_finish_times, batch.is_prompt)?;
                let _aborted = e.collect_finished_responses(
                    &scheduled,
                    &mut responses,
                    &prompt_finish_times,
                    true,
                );

                let finished_seq_ids: Vec<usize> = scheduled
                    .iter()
                    .filter(|g| g.is_finished())
                    .map(|g| Self::primary_sequence(g).deref().get_id())
                    .collect();
                if !finished_seq_ids.is_empty() {
                    e.broadcast_finish_sequences(&finished_seq_ids);
                }
            }
        }

        engine.write().reset_decoder_for_rank(0);
        tracing::debug!("generate_once_multiprocess: master generation finished");
        Ok(responses)
    }
}
