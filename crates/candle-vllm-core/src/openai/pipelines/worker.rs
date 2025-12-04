use crossbeam::channel::{select, Receiver};
use tracing::{debug, error, info, warn};

use super::work_item::{StreamingToken, StreamingWorkItem, WorkItem};
use super::DefaultPipeline;
use crate::openai::responses::{ChatChoice, ChatChoiceData, ChatCompletionUsageResponse};
use crate::openai::utils::get_created_time_secs;
use crate::scheduler::cache_engine::CacheEngine;
use candle_core::{Result, Tensor};

/// A dedicated inference worker that owns its pipeline and processes
/// work from lock-free channels.
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
            processed_count,
            streaming_count,
            "Inference worker terminated gracefully"
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
            Ok(_logits) => {
                // TODO: Implement proper token sampling without SequenceGroups
                // For now, return a placeholder response to indicate the forward pass succeeded
                let elapsed = start.elapsed();

                // Create placeholder response
                let choices = vec![ChatChoice {
                    index: 0,
                    message: ChatChoiceData {
                        role: "assistant".to_string(),
                        content: Some(
                            "[Worker forward pass completed - sampling integration pending]"
                                .to_string(),
                        ),
                        tool_calls: None,
                    },
                    finish_reason: Some("stop".to_string()),
                    logprobs: None,
                }];

                let usage = ChatCompletionUsageResponse {
                    request_id: work.request_id.clone(),
                    created: get_created_time_secs(),
                    completion_tokens: 1,
                    prompt_tokens: work.tokens.len(),
                    total_tokens: work.tokens.len() + 1,
                    prompt_time_costs: elapsed.as_millis() as usize,
                    completion_time_costs: 0,
                };

                if work.response_tx.send(Ok((choices, usage))).is_err() {
                    warn!(
                        rank = self.rank,
                        request_id = %work.request_id,
                        "Client disconnected before response sent"
                    );
                }

                debug!(
                    rank = self.rank,
                    request_id = %work.request_id,
                    elapsed_ms = elapsed.as_millis(),
                    "Completion request processed (placeholder response)"
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

    /// Process a streaming work item.
    ///
    /// NOTE: Currently returns a placeholder streaming response.
    /// Full autoregressive generation requires scheduler integration.
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

        // Create tensors for the input
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

        // Perform forward pass
        let forward_result = self.pipeline.forward(
            tokens_tensor,
            &positions_tensor,
            Some(&self.cache_engine.get_kv_cache()),
            &work.input_metadata,
        );

        match forward_result {
            Ok(_logits) => {
                // TODO: Implement proper autoregressive generation
                // For now, send a placeholder response
                let placeholder_text =
                    "[Streaming forward pass completed - autoregressive generation pending]";

                // Send streaming token
                let streaming_token = StreamingToken {
                    text: placeholder_text.to_string(),
                    token_id: 0,
                    is_finished: true,
                    finish_reason: Some("stop".to_string()),
                    is_reasoning: false, // TODO: Detect reasoning tokens based on model type
                };

                if work.stream_tx.send(Ok(streaming_token)).is_err() {
                    warn!(
                        rank = self.rank,
                        request_id = %work.request_id,
                        "Client disconnected during streaming"
                    );
                }
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

        let elapsed = start.elapsed();
        debug!(
            rank = self.rank,
            request_id = %work.request_id,
            elapsed_ms = elapsed.as_millis(),
            "Streaming request completed (placeholder response)"
        );
    }
}
