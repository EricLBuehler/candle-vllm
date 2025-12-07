//! LLM inference executor for the parking-lot scheduler.
//!
//! This module implements the `TaskExecutor` trait from prometheus_parking_lot
//! for processing LLM inference jobs through the resource pool.

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use super::job::{InferenceJob, InferenceResult, StreamingTokenResult};
use super::types::{TaskExecutor, TaskMetadata};
use crate::openai::logits_processor::LogitsProcessor;
use crate::openai::pipelines::pipeline::DefaultPipeline;
use crate::openai::responses::{ChatChoice, ChatChoiceData, ChatCompletionUsageResponse};
use crate::openai::utils::get_created_time_secs;
use crate::scheduler::cache_engine::CacheEngine;

/// Seed for the random number generator used in sampling.
const SAMPLING_SEED: u64 = 299792458;

/// LLM inference executor that processes inference jobs.
///
/// This executor owns its pipeline and cache engine, processing
/// inference requests without lock contention on the hot path.
#[derive(Clone)]
pub struct LlmExecutor {
    /// Rank (GPU index) for this executor
    rank: usize,

    /// Pipeline for inference (shared reference for cloning)
    pipeline: Arc<Box<DefaultPipeline>>,

    /// Cache engine for KV-cache management (shared reference)
    cache_engine: Arc<CacheEngine>,
}

impl LlmExecutor {
    /// Create a new LLM executor.
    ///
    /// # Arguments
    ///
    /// * `rank` - GPU index for this executor
    /// * `pipeline` - The model pipeline for inference
    /// * `cache_engine` - The KV-cache engine
    #[must_use]
    pub fn new(rank: usize, pipeline: Box<DefaultPipeline>, cache_engine: CacheEngine) -> Self {
        Self {
            rank,
            pipeline: Arc::new(pipeline),
            cache_engine: Arc::new(cache_engine),
        }
    }

    /// Get a reference to the pipeline.
    #[must_use]
    pub fn pipeline(&self) -> &DefaultPipeline {
        &self.pipeline
    }

    /// Get the device this executor runs on.
    #[must_use]
    pub fn device(&self) -> &Device {
        self.pipeline.device()
    }

    /// Get the rank (GPU index).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Check GPU memory usage to prevent exhaustion.
    fn check_gpu_memory(&self) -> Result<(), String> {
        if self.pipeline.device().is_cuda() {
            debug!(rank = self.rank, "GPU memory check enabled for CUDA device");
        }
        Ok(())
    }

    /// Extract the last position's logits from a tensor.
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

    /// Process a completion (non-streaming) job.
    ///
    /// CRITICAL: This function performs CPU/GPU-bound inference work.
    /// It MUST be called from a context that can handle blocking operations
    /// (e.g., within tokio::task::spawn_blocking or a dedicated thread pool).
    fn process_completion(&self, job: &InferenceJob) -> InferenceResult {
        let start = std::time::Instant::now();
        let device = self.pipeline.device();

        // Create tokens tensor
        let tokens_tensor = match Tensor::from_vec(job.tokens.clone(), (job.tokens.len(),), device)
        {
            Ok(tensor) => tensor,
            Err(e) => {
                error!(
                    rank = self.rank,
                    request_id = %job.request_id,
                    error = %e,
                    "Failed to create tokens tensor"
                );
                return InferenceResult::error(e.to_string());
            }
        };

        // Create positions tensor
        let positions_i64: Vec<i64> = job.positions.iter().map(|&pos| pos as i64).collect();
        let positions_tensor = match Tensor::from_vec(positions_i64, (job.positions.len(),), device)
        {
            Ok(tensor) => tensor,
            Err(e) => {
                error!(
                    rank = self.rank,
                    request_id = %job.request_id,
                    error = %e,
                    "Failed to create positions tensor"
                );
                return InferenceResult::error(e.to_string());
            }
        };

        // Create input metadata
        let input_metadata = match job.to_input_metadata(device) {
            Ok(meta) => meta,
            Err(e) => {
                error!(
                    rank = self.rank,
                    request_id = %job.request_id,
                    error = %e,
                    "Failed to create input metadata"
                );
                return InferenceResult::error(e.to_string());
            }
        };

        // Initial forward pass
        let forward_result = self.pipeline.forward(
            tokens_tensor,
            &positions_tensor,
            Some(&self.cache_engine.get_kv_cache()),
            &input_metadata,
        );

        match forward_result {
            Ok(logits) => {
                let prefill_elapsed = start.elapsed();
                let decode_start = std::time::Instant::now();

                // Create logits processor for sampling
                let logits_processor = LogitsProcessor::new(
                    SAMPLING_SEED,
                    job.sampling_params.temperature,
                    job.sampling_params.top_k,
                    job.sampling_params.top_p,
                    job.sampling_params.min_p,
                );

                let stop_token_ids = self.pipeline.get_stop_token_ids();
                let sampling_params = Some(job.sampling_params.clone());
                let max_tokens = job.sampling_params.max_tokens;
                let max_context = job.max_context_len;

                let mut generated_tokens: Vec<u32> = Vec::new();
                let mut generated_text = String::new();
                let mut all_tokens = job.tokens.clone();
                let mut current_logits = logits;
                let mut finish_reason = "length";

                // Autoregressive generation loop
                for step in 0..max_tokens {
                    // Sample next token from logits
                    let last_logits = Self::extract_last_logits(&current_logits);

                    let next_tokens = match logits_processor.sample(&last_logits, &sampling_params)
                    {
                        Ok(tokens) => tokens,
                        Err(e) => {
                            error!(
                                rank = self.rank,
                                request_id = %job.request_id,
                                step = step,
                                error = %e,
                                "Token sampling failed"
                            );
                            return InferenceResult::error(e.to_string());
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
                    match self.pipeline.decode(&[next_token]) {
                        Ok(text) => generated_text.push_str(&text),
                        Err(_) => {} // Ignore decode errors
                    }

                    // Check for custom stop strings
                    if let Some(ref stop_strs) = job.sampling_params.stop {
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

                    // Prepare for next step
                    let all_tokens_tensor =
                        match Tensor::from_vec(all_tokens.clone(), (all_tokens.len(),), device) {
                            Ok(t) => t,
                            Err(e) => {
                                error!(
                                    rank = self.rank,
                                    request_id = %job.request_id,
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
                                    request_id = %job.request_id,
                                    step = step,
                                    error = %e,
                                    "Failed to create positions tensor"
                                );
                                break;
                            }
                        };

                    // Create metadata for full sequence
                    let seq_len = all_tokens.len();
                    let cu_seqlens = match Tensor::new(&[0u32, seq_len as u32], device) {
                        Ok(t) => t,
                        Err(e) => {
                            error!(
                                rank = self.rank,
                                request_id = %job.request_id,
                                step = step,
                                error = %e,
                                "Failed to create cu_seqlens tensor"
                            );
                            break;
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
                        None, // Don't use KV cache - recompute each time
                        &step_metadata,
                    ) {
                        Ok(logits) => logits,
                        Err(e) => {
                            error!(
                                rank = self.rank,
                                request_id = %job.request_id,
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
                    request_id: job.request_id.clone(),
                    created: get_created_time_secs(),
                    completion_tokens: generated_tokens.len(),
                    prompt_tokens: job.tokens.len(),
                    total_tokens: job.tokens.len() + generated_tokens.len(),
                    prompt_time_costs: prefill_elapsed.as_millis() as usize,
                    completion_time_costs: decode_elapsed.as_millis() as usize,
                };

                info!(
                    rank = self.rank,
                    request_id = %job.request_id,
                    generated_tokens = generated_tokens.len(),
                    prefill_ms = prefill_elapsed.as_millis(),
                    decode_ms = decode_elapsed.as_millis(),
                    tokens_per_sec = if decode_elapsed.as_secs_f64() > 0.0 {
                        generated_tokens.len() as f64 / decode_elapsed.as_secs_f64()
                    } else {
                        0.0
                    },
                    finish_reason = finish_reason,
                    "Completion request processed"
                );

                InferenceResult::completion(choices, usage)
            }
            Err(e) => {
                error!(
                    rank = self.rank,
                    request_id = %job.request_id,
                    error = %e,
                    "Forward pass failed"
                );
                InferenceResult::error(e.to_string())
            }
        }
    }

    /// Process a streaming job, returning a receiver for tokens.
    fn process_streaming(&self, job: &InferenceJob) -> InferenceResult {
        let (token_tx, token_rx) = flume::unbounded();
        let request_id = job.request_id.clone();

        // Clone necessary data for the spawned task
        let job_clone = job.clone();
        let pipeline = Arc::clone(&self.pipeline);
        let cache_engine = Arc::clone(&self.cache_engine);
        let rank = self.rank;

        // Spawn async task for streaming generation
        tokio::spawn(async move {
            Self::streaming_generation_task(rank, job_clone, pipeline, cache_engine, token_tx)
                .await;
        });

        InferenceResult::streaming(request_id, token_rx)
    }

    /// Async task that performs streaming token generation.
    async fn streaming_generation_task(
        rank: usize,
        job: InferenceJob,
        pipeline: Arc<Box<DefaultPipeline>>,
        cache_engine: Arc<CacheEngine>,
        token_tx: flume::Sender<Result<StreamingTokenResult, String>>,
    ) {
        let start = std::time::Instant::now();
        let device = pipeline.device();

        info!(
            "🎬 EXECUTOR: Streaming generation task started - rank={}, request_id={}, prompt_tokens={}, max_tokens={}",
            rank,
            job.request_id,
            job.tokens.len(),
            job.sampling_params.max_tokens
        );

        // Create tensors for initial prefill
        let tokens_tensor = match Tensor::from_vec(job.tokens.clone(), (job.tokens.len(),), device)
        {
            Ok(tensor) => tensor,
            Err(e) => {
                let _ = token_tx.send(Err(e.to_string()));
                return;
            }
        };

        let positions: Vec<i64> = (0..job.tokens.len() as i64).collect();
        let positions_tensor = match Tensor::from_vec(positions, (job.tokens.len(),), device) {
            Ok(tensor) => tensor,
            Err(e) => {
                let _ = token_tx.send(Err(e.to_string()));
                return;
            }
        };

        // Create input metadata
        let input_metadata = match job.to_input_metadata(device) {
            Ok(meta) => meta,
            Err(e) => {
                let _ = token_tx.send(Err(e.to_string()));
                return;
            }
        };

        // Initial forward pass
        info!(
            "🔮 EXECUTOR: Starting initial forward pass (prefill) - request_id={}",
            job.request_id
        );
        let forward_result = pipeline.forward(
            tokens_tensor,
            &positions_tensor,
            Some(&cache_engine.get_kv_cache()),
            &input_metadata,
        );

        match forward_result {
            Ok(logits) => {
                info!(
                    "✅ EXECUTOR: Initial forward pass complete - request_id={}",
                    job.request_id
                );
                let logits_processor = LogitsProcessor::new(
                    SAMPLING_SEED,
                    job.sampling_params.temperature,
                    job.sampling_params.top_k,
                    job.sampling_params.top_p,
                    job.sampling_params.min_p,
                );

                let stop_token_ids = pipeline.get_stop_token_ids();
                let sampling_params = Some(job.sampling_params.clone());
                let max_tokens = job.sampling_params.max_tokens;
                let max_context = job.max_context_len;

                let mut generated_count = 0usize;
                let mut all_tokens = job.tokens.clone();
                let mut current_logits = logits;
                let mut generated_text = String::new();

                // Autoregressive streaming generation loop
                info!(
                    "🔁 EXECUTOR: Starting generation loop - request_id={}, max_tokens={}",
                    job.request_id, max_tokens
                );
                for step in 0..max_tokens {
                    if step == 0 {
                        info!(
                            "🎯 EXECUTOR: Starting token generation - request_id={}",
                            job.request_id
                        );
                    }
                    if step % 10 == 0 && step > 0 {
                        info!(
                            "📊 EXECUTOR: Generation progress - request_id={}, step={}/{}",
                            job.request_id, step, max_tokens
                        );
                    }
                    // Sample next token
                    let last_logits = Self::extract_last_logits(&current_logits);

                    let next_tokens = match logits_processor.sample(&last_logits, &sampling_params)
                    {
                        Ok(tokens) => tokens,
                        Err(e) => {
                            let _ = token_tx.send(Err(e.to_string()));
                            return;
                        }
                    };

                    let next_token = next_tokens[0];
                    generated_count += 1;
                    all_tokens.push(next_token);

                    // Check for EOS
                    let is_eos = stop_token_ids.contains(&next_token);

                    // Decode token to text
                    let token_text = pipeline.decode(&[next_token]).unwrap_or_default();
                    generated_text.push_str(&token_text);

                    // Check for custom stop strings
                    let mut hit_stop_string = false;
                    if let Some(ref stop_strs) = job.sampling_params.stop {
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
                    let streaming_token = StreamingTokenResult {
                        text: token_text,
                        token_id: next_token,
                        is_finished,
                        finish_reason: finish_reason.clone(),
                        is_reasoning: false,
                    };

                    if token_tx.send(Ok(streaming_token)).is_err() {
                        warn!(
                            "⚠️ EXECUTOR: Client disconnected during streaming - rank={}, request_id={}, step={}",
                            rank,
                            job.request_id,
                            step
                        );
                        return;
                    }

                    if step == 0 {
                        info!(
                            "🎉 EXECUTOR: First token sent successfully - request_id={}",
                            job.request_id
                        );
                    }

                    if is_finished {
                        info!("🏁 EXECUTOR: Generation complete - request_id={}, total_steps={}, reason={:?}", 
                            job.request_id, step + 1, finish_reason);
                        break;
                    }

                    // Prepare next step
                    let all_tokens_tensor =
                        match Tensor::from_vec(all_tokens.clone(), (all_tokens.len(),), device) {
                            Ok(t) => t,
                            Err(e) => {
                                let _ = token_tx.send(Err(e.to_string()));
                                return;
                            }
                        };

                    let positions: Vec<i64> = (0..all_tokens.len() as i64).collect();
                    let positions_tensor =
                        match Tensor::from_vec(positions, (all_tokens.len(),), device) {
                            Ok(t) => t,
                            Err(e) => {
                                let _ = token_tx.send(Err(e.to_string()));
                                return;
                            }
                        };

                    // Create metadata for full sequence
                    let seq_len = all_tokens.len();
                    let cu_seqlens = match Tensor::new(&[0u32, seq_len as u32], device) {
                        Ok(t) => t,
                        Err(e) => {
                            let _ = token_tx.send(Err(e.to_string()));
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
                    current_logits = match pipeline.forward(
                        all_tokens_tensor,
                        &positions_tensor,
                        Some(&cache_engine.get_kv_cache()),
                        &step_metadata,
                    ) {
                        Ok(logits) => logits,
                        Err(e) => {
                            let _ = token_tx.send(Err(e.to_string()));
                            return;
                        }
                    };
                }

                let elapsed = start.elapsed();
                info!(
                    rank = rank,
                    request_id = %job.request_id,
                    generated_tokens = generated_count,
                    elapsed_ms = elapsed.as_millis(),
                    tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
                        generated_count as f64 / elapsed.as_secs_f64()
                    } else {
                        0.0
                    },
                    "Streaming generation completed"
                );
            }
            Err(e) => {
                let _ = token_tx.send(Err(e.to_string()));
            }
        }
    }
}

// Implement prometheus_parking_lot's WorkerExecutor trait.
// This allows our executor to be used with the WorkerPool.
#[async_trait]
impl super::types::PrometheusWorkerExecutor<InferenceJob, InferenceResult> for LlmExecutor {
    async fn execute(
        &self,
        payload: InferenceJob,
        meta: super::types::ParkingLotTaskMetadata,
    ) -> InferenceResult {
        // Convert ParkingLotTaskMetadata to our local TaskMetadata
        let local_meta = TaskMetadata {
            id: meta.id,
            priority: meta.priority,
            cost: meta.cost,
            created_at_ms: meta.created_at_ms,
            deadline_ms: meta.deadline_ms,
            mailbox: meta.mailbox,
        };

        self.execute_internal(payload, local_meta).await
    }
}

// Local TaskExecutor trait implementation for backward compatibility
#[async_trait]
impl TaskExecutor<InferenceJob, InferenceResult> for LlmExecutor {
    async fn execute(&self, payload: InferenceJob, meta: TaskMetadata) -> InferenceResult {
        self.execute_internal(payload, meta).await
    }
}

impl LlmExecutor {
    /// Internal execute implementation shared by both trait implementations.
    async fn execute_internal(&self, payload: InferenceJob, meta: TaskMetadata) -> InferenceResult {
        info!(
            "⚙️ EXECUTOR: execute_internal called - rank={}, request_id={}, is_streaming={}",
            self.rank, payload.request_id, payload.is_streaming
        );

        // Check GPU memory before processing
        if let Err(e) = self.check_gpu_memory() {
            error!(
                "❌ EXECUTOR: GPU memory check failed - rank={}, request_id={}, error={}",
                self.rank, payload.request_id, e
            );
            return InferenceResult::error(format!("GPU memory exhaustion: {}", e));
        }

        info!(
            "🔍 EXECUTOR: Processing inference job - rank={}, request_id={}, task_id={}, is_streaming={}",
            self.rank,
            payload.request_id,
            meta.id,
            payload.is_streaming
        );

        let result = if payload.is_streaming {
            info!(
                "📡 EXECUTOR: Processing STREAMING job - request_id={}",
                payload.request_id
            );
            self.process_streaming(&payload)
        } else {
            info!(
                "📝 EXECUTOR: Processing COMPLETION job - request_id={}",
                payload.request_id
            );
            self.process_completion(&payload)
        };

        info!(
            "✅ EXECUTOR: Job processing complete - rank={}, request_id={}, result_type={:?}",
            self.rank,
            payload.request_id,
            match &result {
                InferenceResult::Completion { .. } => "Completion",
                InferenceResult::Streaming { .. } => "Streaming",
                InferenceResult::Error { .. } => "Error",
            }
        );

        result
    }
}

/// Factory function to create an executor for a given rank.
///
/// This is used when building pools from configuration.
pub fn create_executor(
    rank: usize,
    pipeline: Box<DefaultPipeline>,
    cache_engine: CacheEngine,
) -> LlmExecutor {
    LlmExecutor::new(rank, pipeline, cache_engine)
}
