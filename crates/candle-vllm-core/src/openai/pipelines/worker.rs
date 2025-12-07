//! Legacy inference worker using crossbeam channels.
//!
//! **DEPRECATED**: This module is being replaced by the parking-lot scheduler.
//! Use `crate::parking_lot::LlmExecutor` instead for new implementations.
//!
//! This module is retained for backward compatibility during the migration period.
//! It will be removed in a future version once the migration is complete.

use crossbeam::channel::{select, Receiver};
use tracing::{debug, error, info, warn};

use super::work_item::{StreamingToken, StreamingWorkItem, WorkItem};
use super::DefaultPipeline;
use crate::openai::logits_processor::LogitsProcessor;
use crate::openai::responses::{ChatChoice, ChatChoiceData, ChatCompletionUsageResponse};
use crate::openai::utils::get_created_time_secs;
use crate::scheduler::cache_engine::CacheEngine;
use candle_core::{DType, Result, Tensor};

/// A dedicated inference worker that owns its pipeline and processes
/// work from lock-free channels.
///
/// **DEPRECATED**: Use `crate::parking_lot::LlmExecutor` instead.
///
/// Key properties:
/// - Owns its pipeline (no Arc<RwLock<>>)
/// - Blocks on channel receive (no busy-waiting)
/// - Zero lock contention during inference
/// - Supports both completion and streaming requests
pub struct InferenceWorker {
    /// Rank (GPU index) for this worker
    rank: usize,

    /// Pipeline owned by this worker (moved from main thread)
    pipeline: Box<DefaultPipeline>,

    /// Cache engine owned by this worker
    #[allow(dead_code)]
    cache_engine: CacheEngine,

    /// Receiver for completion work items (lock-free channel)
    work_rx: Receiver<WorkItem>,

    /// Receiver for streaming work items (lock-free channel)
    streaming_work_rx: Receiver<StreamingWorkItem>,

    /// Receiver for shutdown signal
    shutdown_rx: Receiver<()>,
}

impl InferenceWorker {
    pub fn new(
        rank: usize,
        pipeline: Box<DefaultPipeline>,
        cache_engine: CacheEngine,
        work_rx: Receiver<WorkItem>,
        streaming_work_rx: Receiver<StreamingWorkItem>,
        shutdown_rx: Receiver<()>,
    ) -> Self {
        Self {
            rank,
            pipeline,
            cache_engine,
            work_rx,
            streaming_work_rx,
            shutdown_rx,
        }
    }

    /// Main worker loop. Runs until shutdown signal received.
    /// Handles both completion and streaming requests.
    pub fn run(mut self) {
        info!(
            rank = self.rank,
            "Inference worker started, owns pipeline and cache"
        );

        let mut processed_count = 0u64;
        let mut streaming_count = 0u64;

        loop {
            select! {
                recv(self.work_rx) -> msg => {
                    match msg {
                        Ok(work) => {
                            // Check GPU memory before processing
                            if let Err(e) = self.check_gpu_memory() {
                                error!(
                                    rank = self.rank,
                                    request_id = %work.request_id,
                                    error = %e,
                                    "GPU memory check failed"
                                );
                                let _ = work.response_tx.send(Err(format!("GPU memory exhaustion: {}", e)));
                                continue;
                            }

                            self.process_completion_work(work);
                            processed_count += 1;
                        }
                        Err(_) => {
                            // Channel closed - check if we should continue with streaming
                            debug!(rank = self.rank, "Completion work channel closed");
                        }
                    }
                }
                recv(self.streaming_work_rx) -> msg => {
                    match msg {
                        Ok(work) => {
                            // Check GPU memory before processing
                            if let Err(e) = self.check_gpu_memory() {
                                error!(
                                    rank = self.rank,
                                    request_id = %work.request_id,
                                    error = %e,
                                    "GPU memory check failed"
                                );
                                let _ = work.stream_tx.send(Err(format!("GPU memory exhaustion: {}", e)));
                                continue;
                            }

                            self.process_streaming_work(work);
                            streaming_count += 1;
                        }
                        Err(_) => {
                            debug!(rank = self.rank, "Streaming work channel closed");
                        }
                    }
                }
                recv(self.shutdown_rx) -> _ => {
                    info!(
                        rank = self.rank,
                        processed_count,
                        streaming_count,
                        "Received shutdown signal"
                    );
                    break;
                }
            }
        }

        info!(
            rank = self.rank,
            processed_count, streaming_count, "Inference worker terminated gracefully"
        );
    }

    /// Check GPU memory usage to prevent exhaustion
    fn check_gpu_memory(&self) -> Result<()> {
        // Simple memory check - in a real implementation, we would query GPU memory
        // For now, we'll implement a basic check based on pipeline device
        if self.pipeline.device().is_cuda() {
            // TODO: Implement actual CUDA memory checking here
            // For now, just return Ok but log periodically about memory monitoring
            debug!(
                rank = self.rank,
                "GPU memory check enabled for CUDA device (basic implementation)"
            );
        }
        Ok(())
    }

    /// Process a single completion work item. This is the lock-free hot path.
    ///
    /// NOTE: Currently returns a placeholder response. Full integration with
    /// the pipeline's forward/sample methods requires scheduler SequenceGroups
    /// which we don't have in the worker context. This will be implemented
    /// in a future iteration.
    fn process_completion_work(&mut self, work: WorkItem) {
        let start = std::time::Instant::now();

        // ✅ NO LOCKS during this entire function!

        // Convert vectors to tensors on the correct device
        let device = self.pipeline.device();
        let tokens_tensor =
            match Tensor::from_vec(work.tokens.clone(), (work.tokens.len(),), device) {
                Ok(tensor) => tensor,
                Err(e) => {
                    error!(
                        rank = self.rank,
                        request_id = %work.request_id,
                        error = %e,
                        "Failed to create tokens tensor"
                    );
                    let _ = work.response_tx.send(Err(e.to_string()));
                    return;
                }
            };

        let positions_i64: Vec<i64> = work.positions.iter().map(|&pos| pos as i64).collect();
        let positions_tensor =
            match Tensor::from_vec(positions_i64, (work.positions.len(),), device) {
                Ok(tensor) => tensor,
                Err(e) => {
                    error!(
                        rank = self.rank,
                        request_id = %work.request_id,
                        error = %e,
                        "Failed to create positions tensor"
                    );
                    let _ = work.response_tx.send(Err(e.to_string()));
                    return;
                }
            };

        // Step 1: GPU inference forward pass
        // Note: We use the cache_engine's kv_cache but the full autoregressive
        // generation loop needs to be integrated with the scheduler
        let forward_result = self.pipeline.forward(
            tokens_tensor,
            &positions_tensor,
            Some(&self.cache_engine.get_kv_cache()),
            &work.input_metadata,
        );

        match forward_result {
            Ok(logits) => {
                let prefill_elapsed = start.elapsed();
                let decode_start = std::time::Instant::now();

                // Create logits processor for sampling
                let logits_processor = LogitsProcessor::new(
                    299792458, // SAMPLING_SEED
                    work.sampling_params.temperature,
                    work.sampling_params.top_k,
                    work.sampling_params.top_p,
                    work.sampling_params.min_p,
                );

                let stop_token_ids = self.pipeline.get_stop_token_ids();
                let sampling_params = Some(work.sampling_params.clone());
                let max_tokens = work.sampling_params.max_tokens;
                let max_context = work.input_metadata.max_context_len;

                let mut generated_tokens: Vec<u32> = Vec::new();
                let mut generated_text = String::new();
                let mut all_tokens = work.tokens.clone();
                let mut current_logits = logits;
                let mut finish_reason = "length";

                // Autoregressive generation loop
                for step in 0..max_tokens {
                    // Sample next token from logits (take last position)
                    let last_logits = Self::extract_last_logits(&current_logits);

                    let next_tokens = match logits_processor.sample(&last_logits, &sampling_params)
                    {
                        Ok(tokens) => tokens,
                        Err(e) => {
                            error!(
                                rank = self.rank,
                                request_id = %work.request_id,
                                step = step,
                                error = %e,
                                "Token sampling failed"
                            );
                            let _ = work.response_tx.send(Err(e.to_string()));
                            return;
                        }
                    };

                    let next_token = next_tokens[0];
                    generated_tokens.push(next_token);
                    all_tokens.push(next_token);

                    // Check for EOS
                    if stop_token_ids.contains(&next_token) {
                        finish_reason = "stop";
                        break;
                    }

                    // Decode token and accumulate text
                    if let Ok(text) = self.pipeline.decode(&[next_token]) {
                        generated_text.push_str(&text);
                    }

                    // Check for custom stop strings
                    if let Some(ref stop_strs) = work.sampling_params.stop {
                        let mut should_stop = false;
                        for stop in stop_strs.to_vec() {
                            if generated_text.ends_with(&stop) {
                                should_stop = true;
                                break;
                            }
                        }
                        if should_stop {
                            finish_reason = "stop";
                            break;
                        }
                    }

                    // Check context length limit
                    if all_tokens.len() >= max_context {
                        finish_reason = "length";
                        break;
                    }

                    // Next step: run forward with ALL accumulated tokens (no KV cache reuse)
                    // This is less efficient but avoids needing block tables from scheduler
                    let all_tokens_tensor =
                        match Tensor::from_vec(all_tokens.clone(), (all_tokens.len(),), device) {
                            Ok(t) => t,
                            Err(e) => {
                                error!(
                                    rank = self.rank,
                                    request_id = %work.request_id,
                                    step = step,
                                    error = %e,
                                    "Failed to create tokens tensor"
                                );
                                break;
                            }
                        };

                    let positions: Vec<i64> = (0..all_tokens.len() as i64).collect();
                    let positions_tensor =
                        match Tensor::from_vec(positions, (all_tokens.len(),), device) {
                            Ok(t) => t,
                            Err(e) => {
                                error!(
                                    rank = self.rank,
                                    request_id = %work.request_id,
                                    step = step,
                                    error = %e,
                                    "Failed to create positions tensor"
                                );
                                break;
                            }
                        };

                    // Create metadata for full sequence (is_prefill=true to use chunked attention)
                    let seq_len = all_tokens.len();
                    let cu_seqlens = match Tensor::new(&[0u32, seq_len as u32], device) {
                        Ok(t) => t,
                        Err(e) => {
                            error!(
                                rank = self.rank,
                                request_id = %work.request_id,
                                step = step,
                                error = %e,
                                "Failed to create cu_seqlens tensor"
                            );
                            break;
                        }
                    };

                    let step_metadata = crate::InputMetadata {
                        is_prefill: true, // Always use prefill/chunked attention
                        slot_mapping: Tensor::zeros(seq_len, DType::I64, device)
                            .expect("slot_mapping tensor creation failed"),
                        block_tables: None,
                        context_lens: None,
                        cu_seqlens_q: Some(cu_seqlens.clone()),
                        cu_seqlens_k: Some(cu_seqlens),
                        max_seqlen_q: seq_len,
                        max_seqlen_k: seq_len,
                        max_context_len: max_context,
                    };

                    // Forward pass for next step (no KV cache - recompute everything)
                    current_logits = match self.pipeline.forward(
                        all_tokens_tensor,
                        &positions_tensor,
                        None, // Don't use KV cache - recompute each time
                        &step_metadata,
                    ) {
                        Ok(logits) => logits,
                        Err(e) => {
                            error!(
                                rank = self.rank,
                                request_id = %work.request_id,
                                step = step,
                                error = %e,
                                "Forward pass failed at step"
                            );
                            break;
                        }
                    };
                }

                let decode_elapsed = decode_start.elapsed();

                // Create response
                let choices = vec![ChatChoice {
                    index: 0,
                    message: ChatChoiceData {
                        role: "assistant".to_string(),
                        content: Some(generated_text.clone()),
                        tool_calls: None,
                    },
                    finish_reason: Some(finish_reason.to_string()),
                    logprobs: None,
                }];

                let usage = ChatCompletionUsageResponse {
                    request_id: work.request_id.clone(),
                    created: get_created_time_secs(),
                    completion_tokens: generated_tokens.len(),
                    prompt_tokens: work.tokens.len(),
                    total_tokens: work.tokens.len() + generated_tokens.len(),
                    prompt_time_costs: prefill_elapsed.as_millis() as usize,
                    completion_time_costs: decode_elapsed.as_millis() as usize,
                };

                if work.response_tx.send(Ok((choices, usage))).is_err() {
                    warn!(
                        rank = self.rank,
                        request_id = %work.request_id,
                        "Client disconnected before response sent"
                    );
                }

                info!(
                    rank = self.rank,
                    request_id = %work.request_id,
                    generated_tokens = generated_tokens.len(),
                    prefill_ms = prefill_elapsed.as_millis(),
                    decode_ms = decode_elapsed.as_millis(),
                    tokens_per_sec = if decode_elapsed.as_secs_f64() > 0.0 {
                        generated_tokens.len() as f64 / decode_elapsed.as_secs_f64()
                    } else { 0.0 },
                    finish_reason = finish_reason,
                    "Completion request processed"
                );
            }
            Err(e) => {
                error!(
                    rank = self.rank,
                    request_id = %work.request_id,
                    error = %e,
                    "Forward pass failed"
                );
                let _ = work.response_tx.send(Err(e.to_string()));
            }
        }
    }

    /// Extract the last position's logits from a tensor
    fn extract_last_logits(logits: &Tensor) -> Tensor {
        let last_logits = if logits.dims().len() == 3 {
            // Shape: [batch, seq_len, vocab_size] -> take last token
            let seq_len = logits.dim(1).unwrap_or(1);
            logits
                .narrow(1, seq_len - 1, 1)
                .unwrap()
                .squeeze(1)
                .unwrap()
        } else if logits.dims().len() == 2 {
            // Shape: [seq_len, vocab_size]
            let seq_len = logits.dim(0).unwrap_or(1);
            if seq_len > 1 {
                logits.narrow(0, seq_len - 1, 1).unwrap()
            } else {
                logits.clone()
            }
        } else {
            logits.clone()
        };

        // Ensure batch dimension for sample
        if last_logits.dims().len() == 1 {
            last_logits.unsqueeze(0).unwrap()
        } else {
            last_logits
        }
    }

    /// Process a streaming work item with full autoregressive generation.
    /// Streams tokens back to the client as they are generated.
    fn process_streaming_work(&mut self, work: StreamingWorkItem) {
        let start = std::time::Instant::now();
        let device = self.pipeline.device();

        debug!(
            rank = self.rank,
            request_id = %work.request_id,
            prompt_tokens = work.tokens.len(),
            max_tokens = work.sampling_params.max_tokens,
            "Starting streaming generation"
        );

        // Create tensors for initial prefill
        let tokens_tensor =
            match Tensor::from_vec(work.tokens.clone(), (work.tokens.len(),), device) {
                Ok(tensor) => tensor,
                Err(e) => {
                    error!(
                        rank = self.rank,
                        request_id = %work.request_id,
                        error = %e,
                        "Failed to create tokens tensor"
                    );
                    let _ = work.stream_tx.send(Err(e.to_string()));
                    return;
                }
            };

        let positions: Vec<i64> = (0..work.tokens.len() as i64).collect();
        let positions_tensor = match Tensor::from_vec(positions, (work.tokens.len(),), device) {
            Ok(tensor) => tensor,
            Err(e) => {
                error!(
                    rank = self.rank,
                    request_id = %work.request_id,
                    error = %e,
                    "Failed to create positions tensor"
                );
                let _ = work.stream_tx.send(Err(e.to_string()));
                return;
            }
        };

        // Perform initial prefill forward pass
        let forward_result = self.pipeline.forward(
            tokens_tensor,
            &positions_tensor,
            None, // Don't use KV cache - recompute each time for simplicity
            &work.input_metadata,
        );

        match forward_result {
            Ok(logits) => {
                // Create logits processor for sampling
                let logits_processor = LogitsProcessor::new(
                    299792458, // SAMPLING_SEED
                    work.sampling_params.temperature,
                    work.sampling_params.top_k,
                    work.sampling_params.top_p,
                    work.sampling_params.min_p,
                );

                let stop_token_ids = self.pipeline.get_stop_token_ids();
                let sampling_params = Some(work.sampling_params.clone());
                let max_tokens = work.sampling_params.max_tokens;
                let max_context = work.input_metadata.max_context_len;

                let mut generated_count = 0usize;
                let mut all_tokens = work.tokens.clone();
                let mut current_logits = logits;
                let mut generated_text = String::new();

                // Autoregressive streaming generation loop
                for step in 0..max_tokens {
                    // Sample next token
                    let last_logits = Self::extract_last_logits(&current_logits);

                    let next_tokens = match logits_processor.sample(&last_logits, &sampling_params)
                    {
                        Ok(tokens) => tokens,
                        Err(e) => {
                            error!(
                                rank = self.rank,
                                request_id = %work.request_id,
                                step = step,
                                error = %e,
                                "Token sampling failed"
                            );
                            let _ = work.stream_tx.send(Err(e.to_string()));
                            return;
                        }
                    };

                    let next_token = next_tokens[0];
                    generated_count += 1;
                    all_tokens.push(next_token);

                    // Check for EOS
                    let is_eos = stop_token_ids.contains(&next_token);

                    // Decode token to text
                    let token_text = self
                        .pipeline
                        .decode(&[next_token])
                        .unwrap_or_else(|_| String::new());
                    generated_text.push_str(&token_text);

                    // Check for custom stop strings
                    let mut hit_stop_string = false;
                    if let Some(ref stop_strs) = work.sampling_params.stop {
                        for stop in stop_strs.to_vec() {
                            if generated_text.ends_with(&stop) {
                                hit_stop_string = true;
                                break;
                            }
                        }
                    }

                    let is_finished = is_eos || hit_stop_string || all_tokens.len() >= max_context;
                    let finish_reason = if is_finished {
                        Some(
                            if is_eos || hit_stop_string {
                                "stop"
                            } else {
                                "length"
                            }
                            .to_string(),
                        )
                    } else {
                        None
                    };

                    // Send streaming token to client
                    let streaming_token = StreamingToken {
                        text: token_text,
                        token_id: next_token,
                        is_finished,
                        finish_reason: finish_reason.clone(),
                        is_reasoning: false, // TODO: Detect reasoning tokens based on model type
                    };

                    if work.stream_tx.send(Ok(streaming_token)).is_err() {
                        warn!(
                            rank = self.rank,
                            request_id = %work.request_id,
                            step = step,
                            "Client disconnected during streaming"
                        );
                        return;
                    }

                    if is_finished {
                        break;
                    }

                    // Next step: run forward with ALL accumulated tokens (no KV cache reuse)
                    let all_tokens_tensor =
                        match Tensor::from_vec(all_tokens.clone(), (all_tokens.len(),), device) {
                            Ok(t) => t,
                            Err(e) => {
                                error!(
                                    rank = self.rank,
                                    request_id = %work.request_id,
                                    step = step,
                                    error = %e,
                                    "Failed to create tokens tensor"
                                );
                                let _ = work.stream_tx.send(Err(e.to_string()));
                                return;
                            }
                        };

                    let positions: Vec<i64> = (0..all_tokens.len() as i64).collect();
                    let positions_tensor =
                        match Tensor::from_vec(positions, (all_tokens.len(),), device) {
                            Ok(t) => t,
                            Err(e) => {
                                error!(
                                    rank = self.rank,
                                    request_id = %work.request_id,
                                    step = step,
                                    error = %e,
                                    "Failed to create positions tensor"
                                );
                                let _ = work.stream_tx.send(Err(e.to_string()));
                                return;
                            }
                        };

                    // Create metadata for full sequence
                    let seq_len = all_tokens.len();
                    let cu_seqlens = match Tensor::new(&[0u32, seq_len as u32], device) {
                        Ok(t) => t,
                        Err(e) => {
                            error!(
                                rank = self.rank,
                                request_id = %work.request_id,
                                step = step,
                                error = %e,
                                "Failed to create cu_seqlens tensor"
                            );
                            let _ = work.stream_tx.send(Err(e.to_string()));
                            return;
                        }
                    };

                    let step_metadata = crate::InputMetadata {
                        is_prefill: true,
                        slot_mapping: Tensor::zeros(seq_len, DType::I64, device)
                            .expect("slot_mapping tensor creation failed"),
                        block_tables: None,
                        context_lens: None,
                        cu_seqlens_q: Some(cu_seqlens.clone()),
                        cu_seqlens_k: Some(cu_seqlens),
                        max_seqlen_q: seq_len,
                        max_seqlen_k: seq_len,
                        max_context_len: max_context,
                    };

                    // Forward pass for next step
                    current_logits = match self.pipeline.forward(
                        all_tokens_tensor,
                        &positions_tensor,
                        None,
                        &step_metadata,
                    ) {
                        Ok(logits) => logits,
                        Err(e) => {
                            error!(
                                rank = self.rank,
                                request_id = %work.request_id,
                                step = step,
                                error = %e,
                                "Forward pass failed at step"
                            );
                            let _ = work.stream_tx.send(Err(e.to_string()));
                            return;
                        }
                    };
                }

                let elapsed = start.elapsed();
                info!(
                    rank = self.rank,
                    request_id = %work.request_id,
                    generated_tokens = generated_count,
                    elapsed_ms = elapsed.as_millis(),
                    tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
                        generated_count as f64 / elapsed.as_secs_f64()
                    } else { 0.0 },
                    "Streaming generation completed"
                );
            }
            Err(e) => {
                error!(
                    rank = self.rank,
                    request_id = %work.request_id,
                    error = %e,
                    "Forward pass failed during streaming"
                );
                let _ = work.stream_tx.send(Err(e.to_string()));
            }
        }
    }
}
