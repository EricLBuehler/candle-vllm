//! LLM inference engine using parking-lot scheduler.
//!
//! This module provides a refactored LLMEngine that uses the parking-lot
//! scheduler for resource-constrained request handling and queueing.

use super::DefaultPipeline;

#[cfg(feature = "nccl")]
use crate::openai::communicator::DaemonManager;
use crate::openai::conversation::default_conversation::{
    DefaultConversation, DefaultConversationSeparators, SeparatorStyle,
};
use crate::openai::conversation::Conversation;
use crate::openai::requests::{FunctionCallDelta, MessageContent, Messages, Tool, ToolCallDelta};
use crate::openai::tool_parser::get_tool_parser;
use crate::parking_lot::{
    InferenceJob, InferenceResult, InferenceWorkerPool, InferenceWorkerPoolConfig, LlmExecutor,
    ResourceAdapter, ResourceCost, ResourceCostExt, SerializableInferenceResult, StreamingRegistry,
    StreamingTokenResult, TaskExecutor, TaskMetadata,
};
use crate::prompt_cache::PromptCacheManager;
use crate::scheduler::Scheduler;
use crate::{
    openai::{
        models::Config,
        responses::{
            ChatChoice, ChatCompletionChunk, ChatCompletionUsageResponse, Choice, ChoiceData,
            PromptTokensDetails,
        },
    },
    scheduler::cache_engine::{CacheConfig, CacheEngine},
};
use candle_core::Result;
use parking_lot::RwLock;
use std::{collections::HashMap, sync::Arc};
use tokenizers::Tokenizer;
use tokio::sync::Notify;
use tracing::{error, info, warn};

/// Configuration for the parking-lot scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerPoolConfig {
    /// Number of dedicated worker threads for inference
    pub worker_threads: usize,
    /// Maximum resource units (GPU blocks) the pool can use
    pub max_units: usize,
    /// Maximum queue depth before rejecting requests
    pub max_queue_depth: usize,
    /// Default timeout in seconds for queued requests
    pub default_timeout_secs: u64,
}

impl Default for SchedulerPoolConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            max_units: 16384,          // ~256K tokens with 16-token blocks
            max_queue_depth: 1000,     // Allow 1000 queued requests
            default_timeout_secs: 120, // 2 minute timeout
        }
    }
}

impl SchedulerPoolConfig {
    /// Create config from cache configuration.
    pub fn from_cache_config(cache_config: &CacheConfig) -> Self {
        let mut config = Self::default();
        config.max_units = cache_config.num_gpu_blocks.unwrap_or(config.max_units);
        config
    }
}

/// Wrapper around a flume receiver that signals cleanup when dropped.
/// This ensures resources are released when the streaming response completes or is abandoned.
pub struct CleanupReceiver {
    inner: flume::Receiver<std::result::Result<StreamingTokenResult, String>>,
    cleanup_signal: Option<flume::Sender<()>>,
    request_id: String,
}

impl CleanupReceiver {
    fn new(
        inner: flume::Receiver<std::result::Result<StreamingTokenResult, String>>,
        cleanup_signal: flume::Sender<()>,
        request_id: String,
    ) -> Self {
        Self {
            inner,
            cleanup_signal: Some(cleanup_signal),
            request_id,
        }
    }
}

// Implement Deref so it acts like the inner receiver
impl std::ops::Deref for CleanupReceiver {
    type Target = flume::Receiver<std::result::Result<StreamingTokenResult, String>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// Signal cleanup when dropped (either naturally or on client disconnect)
impl Drop for CleanupReceiver {
    fn drop(&mut self) {
        if let Some(signal) = self.cleanup_signal.take() {
            info!(
                "🧹 CLEANUP: CleanupReceiver dropped, signaling cleanup - request_id={}",
                self.request_id
            );
            let _ = signal.try_send(());
        }
    }
}

/// LLM inference engine using prometheus_parking_lot for scheduling.
///
/// This engine manages inference requests through a resource pool that:
/// - Tracks GPU resource usage (KV-cache blocks)
/// - Queues requests when resources are exhausted
/// - Automatically dispatches queued work when resources free up
pub struct LLMEngine {
    /// Worker pool with dedicated OS threads for CPU/GPU-bound inference work
    /// Uses prometheus-parking-lot's WorkerPool for proper thread isolation
    worker_pool: Option<Arc<InferenceWorkerPool>>,

    /// LLM executor for processing inference jobs
    executor: Arc<LlmExecutor>,

    /// Resource adapter for cost calculation
    resource_adapter: ResourceAdapter,

    /// Pool configuration
    pool_config: SchedulerPoolConfig,

    /// Scheduler (kept for compatibility but simplified)
    pub scheduler: Arc<parking_lot::Mutex<Scheduler>>,

    /// Cache configuration (public for KV token calculation)
    pub cache_config: CacheConfig,

    /// Model configuration (public for server access)
    pub config: Config,

    /// Notification for engine events
    pub notify: Arc<Notify>,

    /// Tokenizer cloned from pipeline (shared, read-only)
    tokenizer: Tokenizer,

    /// Model name for identification
    model_name: String,

    /// Chat template for prompt generation
    chat_template: Option<String>,

    /// Conversation roles (user, assistant)
    roles: (String, String),

    /// Separator style for conversation formatting
    sep_style: SeparatorStyle,

    /// BOS token for conversation
    bos_token: Option<String>,

    /// EOS token for conversation
    eos_token: Option<String>,

    /// Completed responses keyed by request_id
    pub completion_records: RwLock<HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>>,

    /// Store conversation_id and resource_id per request
    pub request_metadata: RwLock<HashMap<String, (Option<String>, Option<String>)>>,

    /// Track accumulated text per request for incremental tool call parsing
    accumulated_text: RwLock<HashMap<String, String>>,

    /// Track parsed tool calls per request
    parsed_tool_calls: RwLock<HashMap<String, Vec<crate::openai::tool_parser::ParsedToolCall>>>,

    /// Optional daemon manager for multi-process setups (nccl feature)
    #[cfg(feature = "nccl")]
    pub daemon_manager: RwLock<Option<DaemonManager>>,

    /// Current in-flight request count for capacity tracking
    in_flight_requests: Arc<std::sync::atomic::AtomicUsize>,

    /// Current resource units in use
    used_units: Arc<std::sync::atomic::AtomicUsize>,

    /// Optional prompt cache manager for prefix caching
    prompt_cache: Option<Arc<PromptCacheManager>>,

    /// Track cached tokens per request for usage reporting
    cached_tokens: RwLock<HashMap<String, usize>>,
}

impl LLMEngine {
    /// Initialize a new `LLMEngine` with the parking-lot scheduler.
    ///
    /// Unlike the original LLMEngine, this version uses a single executor
    /// and the parking-lot scheduler for resource management.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pipelines: HashMap<usize, (Box<DefaultPipeline>, CacheEngine)>,
        scheduler_config: crate::scheduler::SchedulerConfig,
        cache_config: &CacheConfig,
        config: &Config,
        notify: Arc<Notify>,
        pool_config: Option<SchedulerPoolConfig>,
        #[cfg(feature = "nccl")] daemon_manager: Option<DaemonManager>,
    ) -> Result<Self> {
        Self::new_with_cache(
            pipelines,
            scheduler_config,
            cache_config,
            config,
            notify,
            pool_config,
            None,
            #[cfg(feature = "nccl")]
            daemon_manager,
        )
    }

    /// Create a new `LLMEngine` with optional prompt cache.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_cache(
        pipelines: HashMap<usize, (Box<DefaultPipeline>, CacheEngine)>,
        scheduler_config: crate::scheduler::SchedulerConfig,
        cache_config: &CacheConfig,
        config: &Config,
        notify: Arc<Notify>,
        pool_config: Option<SchedulerPoolConfig>,
        prompt_cache: Option<Arc<PromptCacheManager>>,
        #[cfg(feature = "nccl")] daemon_manager: Option<DaemonManager>,
    ) -> Result<Self> {
        info!(
            "Initializing LLMEngine with parking-lot scheduler ({} pipelines)",
            pipelines.len()
        );

        // We use the first pipeline for the executor
        // Multi-GPU support can be added by creating multiple executors
        let (rank, (pipeline, cache_engine)) = pipelines
            .into_iter()
            .next()
            .ok_or_else(|| candle_core::Error::Msg("no pipelines provided".to_string()))?;

        // Extract shared resources before moving pipeline to executor
        let tokenizer = pipeline.tokenizer().clone();
        let model_name = pipeline.name().to_string();
        let conversation = pipeline.get_past_conversation();
        let roles = conversation.get_roles().clone();

        // Create the executor
        let executor = LlmExecutor::new(rank, pipeline, cache_engine);

        // Create resource adapter from cache config
        let resource_adapter = ResourceAdapter::from_cache_config(cache_config);

        // Use provided pool config or derive from cache config
        let pool_config =
            pool_config.unwrap_or_else(|| SchedulerPoolConfig::from_cache_config(cache_config));

        info!(
            event = "engine_pool_config",
            max_units = pool_config.max_units,
            max_queue_depth = pool_config.max_queue_depth,
            timeout_secs = pool_config.default_timeout_secs,
            worker_threads = pool_config.worker_threads
        );

        // Create thread pool for CPU-bound inference work
        let streaming_registry = StreamingRegistry::with_default_retention();
        let worker_pool_config = InferenceWorkerPoolConfig {
            worker_count: pool_config.worker_threads,
            max_units: pool_config.max_units as u32,
            max_queue_depth: pool_config.max_queue_depth,
            timeout_secs: pool_config.default_timeout_secs,
        };

        let worker_pool = InferenceWorkerPool::new(
            executor.clone(),
            streaming_registry.clone(),
            worker_pool_config,
        )
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create worker pool: {:?}", e)))?;

        info!(
            "🧵 ENGINE: Worker pool created with {} worker threads",
            worker_pool.available_permits()
        );

        let executor = Arc::new(executor);

        // Create scheduler for compatibility
        let scheduler = Arc::new(parking_lot::Mutex::new(Scheduler::new(
            scheduler_config,
            cache_config,
        )));

        let engine = Self {
            worker_pool: Some(Arc::new(worker_pool)),
            executor,
            resource_adapter,
            pool_config,
            scheduler,
            cache_config: cache_config.clone(),
            config: config.clone(),
            notify: notify.clone(),
            tokenizer,
            model_name,
            chat_template: None,
            roles,
            sep_style: SeparatorStyle::ChatML,
            bos_token: None,
            eos_token: None,
            completion_records: RwLock::new(HashMap::new()),
            request_metadata: RwLock::new(HashMap::new()),
            accumulated_text: RwLock::new(HashMap::new()),
            parsed_tool_calls: RwLock::new(HashMap::new()),
            #[cfg(feature = "nccl")]
            daemon_manager: RwLock::new(daemon_manager),
            in_flight_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            used_units: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            prompt_cache: prompt_cache.clone(),
            cached_tokens: RwLock::new(HashMap::new()),
        };

        if prompt_cache.is_some() {
            info!("LLMEngine initialized with prompt caching enabled");
        } else {
            info!("LLMEngine initialized with parking-lot scheduler");
        }

        Ok(engine)
    }

    /// Start the processing loop (async version).
    pub async fn start_processing_loop(&self) {
        info!("Parking-lot scheduler active; processing is request-driven");
    }

    // =========================================================================
    // Accessor methods
    // =========================================================================

    /// Get a reference to the shared tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get the conversation roles.
    pub fn roles(&self) -> &(String, String) {
        &self.roles
    }

    /// Get the model configuration.
    pub fn get_config(&self) -> &Config {
        &self.config
    }

    /// Get the cache configuration.
    pub fn get_cache_config(&self) -> &CacheConfig {
        &self.cache_config
    }

    /// Calculate available KV cache tokens.
    pub fn get_available_kv_tokens(&self) -> usize {
        let total = self.resource_adapter.max_units();
        let used = self.used_units.load(std::sync::atomic::Ordering::Relaxed);
        self.resource_adapter
            .blocks_to_tokens(total.saturating_sub(used))
    }

    /// Get current queue depth.
    pub fn queue_depth(&self) -> usize {
        if let Some(ref pool) = self.worker_pool {
            pool.queue_depth()
        } else {
            // Fallback: track in-flight requests
            self.in_flight_requests
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    /// Check if the engine can accept a new request.
    pub fn can_accept_request(&self, prompt_len: usize, max_tokens: usize) -> bool {
        let cost = self.resource_adapter.calculate_cost(prompt_len, max_tokens);
        let used = self.used_units.load(std::sync::atomic::Ordering::Relaxed);
        let available = self.pool_config.max_units.saturating_sub(used);

        // Check resource capacity
        if (cost.units as usize) > available {
            return false;
        }

        // Check queue depth
        let in_flight = self
            .in_flight_requests
            .load(std::sync::atomic::Ordering::Relaxed);
        if in_flight >= self.pool_config.max_queue_depth {
            return false;
        }

        true
    }

    // =========================================================================
    // Request submission
    // =========================================================================

    /// Add a completion request using the parking-lot scheduler.
    ///
    /// Returns a receiver for the completion result.
    pub async fn add_request(
        &self,
        request_id: String,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        sampling_params: crate::openai::sampling_params::SamplingParams,
        max_context_len: usize,
    ) -> Result<tokio::sync::oneshot::Receiver<InferenceResult>> {
        let prompt_len = tokens.len();
        let max_tokens = sampling_params.max_tokens;

        info!(
            "🏗️ ENGINE: add_request (completion) called - request_id={}, prompt_len={}, max_tokens={}",
            request_id, prompt_len, max_tokens
        );

        // Calculate resource cost
        let cost = self.resource_adapter.calculate_cost(prompt_len, max_tokens);
        info!(
            "💰 ENGINE: Resource cost calculated - request_id={}, cost_units={}, used_units={}/{}, in_flight={}/{}",
            request_id,
            cost.units,
            self.used_units.load(std::sync::atomic::Ordering::Relaxed),
            self.pool_config.max_units,
            self.in_flight_requests.load(std::sync::atomic::Ordering::Relaxed),
            self.pool_config.max_queue_depth
        );

        // Check capacity
        if !self.can_accept_request(prompt_len, max_tokens) {
            error!(
                "🚫 ENGINE: CAPACITY CHECK FAILED - request_id={}, need {} units, have {}/{} used",
                request_id,
                cost.units,
                self.used_units.load(std::sync::atomic::Ordering::Relaxed),
                self.pool_config.max_units
            );
            return Err(candle_core::Error::Msg(format!(
                "Request {} rejected: insufficient capacity (need {} units)",
                request_id, cost.units
            )));
        }
        info!(
            "✅ ENGINE: Capacity check passed - request_id={}",
            request_id
        );

        // Check prompt cache for prefix match
        let _cached_tokens = if let Some(ref cache_manager) = self.prompt_cache {
            if let Ok(Some(cached_match)) = cache_manager.find_cached_prefix(&tokens).await {
                let cached_count = cached_match.cached_tokens;
                info!(
                    "💾 CACHE: Found cached prefix - request_id={}, cached_tokens={}",
                    request_id, cached_count
                );
                // Store cached token count for usage reporting
                self.cached_tokens.write().insert(request_id.clone(), cached_count);
                cached_count
            } else {
                0
            }
        } else {
            0
        };

        // Reserve resources
        info!(
            "📊 ENGINE: Reserving resources - request_id={}, adding {} units",
            request_id, cost.units
        );
        self.used_units
            .fetch_add(cost.units as usize, std::sync::atomic::Ordering::Relaxed);
        self.in_flight_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        info!(
            "📊 ENGINE: Resources reserved - request_id={}, used_units={}/{}, in_flight={}",
            request_id,
            self.used_units.load(std::sync::atomic::Ordering::Relaxed),
            self.pool_config.max_units,
            self.in_flight_requests
                .load(std::sync::atomic::Ordering::Relaxed)
        );

        // Create the inference job
        info!(
            "🎨 ENGINE: Creating completion inference job - request_id={}",
            request_id
        );
        let job = InferenceJob::new_completion(
            request_id.clone(),
            tokens,
            positions,
            sampling_params,
            max_context_len,
        );

        // Create response channel
        let (tx, rx) = tokio::sync::oneshot::channel();

        // Use worker pool if available, otherwise fall back to direct execution
        if let Some(ref pool) = self.worker_pool {
            info!(
                "🏊 ENGINE: Using worker pool for completion request - request_id={}",
                request_id
            );

            let pool = Arc::clone(pool);
            let used_units = Arc::clone(&self.used_units);
            let in_flight = Arc::clone(&self.in_flight_requests);
            let cost_units_usize = cost.units as usize;
            let request_id_task = request_id.clone();

            tokio::spawn(async move {
                let meta =
                    TaskMetadata::new(rand::random::<u64>(), ResourceCost::gpu_vram(cost.units));

                // Submit to worker pool - this will run in a dedicated worker thread
                match pool.submit(job, meta).await {
                    Ok(serializable_result) => {
                        info!(
                            "✅ ENGINE: Worker pool returned result - request_id={}",
                            request_id_task
                        );

                        // Convert back to InferenceResult
                        let result = match serializable_result {
                            SerializableInferenceResult::Completion { choices, usage } => {
                                InferenceResult::Completion { choices, usage }
                            }
                            SerializableInferenceResult::StreamingChannel {
                                request_id,
                                channel_key: _,
                            } => {
                                // This shouldn't happen for completion requests
                                InferenceResult::Error {
                                    message: format!(
                                        "Unexpected streaming result for completion request: {}",
                                        request_id
                                    ),
                                }
                            }
                            SerializableInferenceResult::Error { message } => {
                                InferenceResult::Error { message }
                            }
                        };

                        // Release resources
                        used_units
                            .fetch_sub(cost_units_usize, std::sync::atomic::Ordering::Relaxed);
                        in_flight.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

                        let _ = tx.send(result);
                    }
                    Err(e) => {
                        error!(
                            "❌ ENGINE: Worker pool submission failed - request_id={}, error={:?}",
                            request_id_task, e
                        );
                        used_units
                            .fetch_sub(cost_units_usize, std::sync::atomic::Ordering::Relaxed);
                        in_flight.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        let _ = tx.send(InferenceResult::Error {
                            message: format!("{:?}", e),
                        });
                    }
                }
            });

            info!(
                "🎊 ENGINE: Completion request submitted to worker pool - request_id={}",
                request_id
            );
            Ok(rx)
        } else {
            // Fallback: direct execution (old path)
            warn!(
                "⚠️ ENGINE: No worker pool, using direct execution (may block!) - request_id={}",
                request_id
            );

            let executor = Arc::clone(&self.executor);
            let used_units = Arc::clone(&self.used_units);
            let in_flight = Arc::clone(&self.in_flight_requests);
            let cost_units = cost.units;
            let cost_units_usize = cost.units as usize;

            let _request_id_task = request_id.clone();
            tokio::spawn(async move {
                let meta =
                    TaskMetadata::new(rand::random::<u64>(), ResourceCost::gpu_vram(cost_units));

                let result = executor.execute(job, meta).await;

                used_units.fetch_sub(cost_units_usize, std::sync::atomic::Ordering::Relaxed);
                in_flight.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

                let _ = tx.send(result);
            });

            Ok(rx)
        }
    }

    /// Add a streaming request using the parking-lot scheduler.
    ///
    /// Returns a receiver for streaming tokens.
    pub async fn add_streaming_request(
        &self,
        request_id: String,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        sampling_params: crate::openai::sampling_params::SamplingParams,
        created: u64,
        max_context_len: usize,
    ) -> Result<CleanupReceiver> {
        let prompt_len = tokens.len();
        let max_tokens = sampling_params.max_tokens;

        info!(
            "🏗️ ENGINE: add_streaming_request called - request_id={}, prompt_len={}, max_tokens={}",
            request_id, prompt_len, max_tokens
        );

        // Calculate resource cost
        let cost = self.resource_adapter.calculate_cost(prompt_len, max_tokens);
        info!(
            "💰 ENGINE: Resource cost calculated - request_id={}, cost_units={}, used_units={}/{}, in_flight={}/{}",
            request_id,
            cost.units,
            self.used_units.load(std::sync::atomic::Ordering::Relaxed),
            self.pool_config.max_units,
            self.in_flight_requests.load(std::sync::atomic::Ordering::Relaxed),
            self.pool_config.max_queue_depth
        );

        // Check capacity
        if !self.can_accept_request(prompt_len, max_tokens) {
            error!(
                "🚫 ENGINE: CAPACITY CHECK FAILED - request_id={}, need {} units, have {}/{} used",
                request_id,
                cost.units,
                self.used_units.load(std::sync::atomic::Ordering::Relaxed),
                self.pool_config.max_units
            );
            return Err(candle_core::Error::Msg(format!(
                "Streaming request {} rejected: insufficient capacity (need {} units)",
                request_id, cost.units
            )));
        }
        info!(
            "✅ ENGINE: Capacity check passed - request_id={}",
            request_id
        );

        // Check prompt cache for prefix match
        let _cached_tokens = if let Some(ref cache_manager) = self.prompt_cache {
            if let Ok(Some(cached_match)) = cache_manager.find_cached_prefix(&tokens).await {
                let cached_count = cached_match.cached_tokens;
                info!(
                    "💾 CACHE: Found cached prefix - request_id={}, cached_tokens={}",
                    request_id, cached_count
                );
                // Store cached token count for usage reporting
                self.cached_tokens.write().insert(request_id.clone(), cached_count);
                cached_count
            } else {
                0
            }
        } else {
            0
        };

        // Reserve resources
        info!(
            "📊 ENGINE: Reserving resources - request_id={}, adding {} units",
            request_id, cost.units
        );
        self.used_units
            .fetch_add(cost.units as usize, std::sync::atomic::Ordering::Relaxed);
        self.in_flight_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        info!(
            "📊 ENGINE: Resources reserved - request_id={}, used_units={}/{}, in_flight={}",
            request_id,
            self.used_units.load(std::sync::atomic::Ordering::Relaxed),
            self.pool_config.max_units,
            self.in_flight_requests
                .load(std::sync::atomic::Ordering::Relaxed)
        );

        // Create the inference job
        info!(
            "🎨 ENGINE: Creating streaming inference job - request_id={}",
            request_id
        );
        let job = InferenceJob::new_streaming(
            request_id.clone(),
            tokens,
            positions,
            sampling_params,
            created,
            max_context_len,
        );

        // Use worker pool if available
        if let Some(ref pool) = self.worker_pool {
            info!(
                "🏊 ENGINE: Using worker pool for streaming request - request_id={}",
                request_id
            );

            let pool = Arc::clone(pool);
            let used_units = Arc::clone(&self.used_units);
            let in_flight = Arc::clone(&self.in_flight_requests);
            let cost_units_usize = cost.units as usize;

            let meta = TaskMetadata::new(rand::random::<u64>(), ResourceCost::gpu_vram(cost.units));

            // Submit to worker pool
            match pool.submit(job, meta).await {
                Ok(SerializableInferenceResult::StreamingChannel { channel_key, .. }) => {
                    info!(
                        "✅ ENGINE: Got streaming channel key from worker pool - request_id={}, key={}",
                        request_id, channel_key
                    );

                    // Retrieve the channel from the registry
                    let token_rx = pool
                        .streaming_registry()
                        .retrieve(&channel_key)
                        .ok_or_else(|| {
                            error!(
                                "❌ ENGINE: Streaming channel not found in registry - request_id={}, key={}",
                                request_id, channel_key
                            );
                            candle_core::Error::Msg(format!(
                                "Streaming channel not found: {}",
                                channel_key
                            ))
                        })?;

                    info!(
                        "📻 ENGINE: Retrieved streaming channel from registry - request_id={}",
                        request_id
                    );

                    // Create cleanup signal channel (NOT cloning the token receiver!)
                    let (cleanup_tx, cleanup_rx) = flume::bounded::<()>(1);
                    let used_units_clone = Arc::clone(&used_units);
                    let in_flight_clone = Arc::clone(&in_flight);
                    let streaming_registry_clone = Arc::clone(pool.streaming_registry());
                    let channel_key_cleanup = channel_key.clone();
                    let request_id_cleanup = request_id.clone();

                    // Spawn cleanup task that waits for signal from CleanupReceiver drop
                    tokio::spawn(async move {
                        info!(
                            "🧹 CLEANUP_TASK: Waiting for cleanup signal - request_id={}",
                            request_id_cleanup
                        );

                        // Wait for signal (sent when CleanupReceiver is dropped)
                        let _ = cleanup_rx.recv_async().await;

                        info!(
                            "🏁 CLEANUP_TASK: Received cleanup signal, releasing resources - request_id={}",
                            request_id_cleanup
                        );

                        // Release resources
                        used_units_clone
                            .fetch_sub(cost_units_usize, std::sync::atomic::Ordering::Relaxed);
                        in_flight_clone.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

                        // Remove channel from registry
                        streaming_registry_clone.remove(&channel_key_cleanup);

                        info!(
                            "♻️ CLEANUP_TASK: Resources released and channel cleaned - request_id={}, units_freed={}",
                            request_id_cleanup, cost_units_usize
                        );
                    });

                    // Wrap the receiver with cleanup signal
                    let wrapped_rx = CleanupReceiver::new(token_rx, cleanup_tx, request_id.clone());

                    info!(
                        "🎊 ENGINE: Streaming request setup complete - request_id={}",
                        request_id
                    );

                    // Return the wrapped receiver (acts like normal receiver via Deref)
                    Ok(wrapped_rx)
                }
                Ok(other) => {
                    error!(
                        "❌ ENGINE: Unexpected result type for streaming - request_id={}, got={:?}",
                        request_id, other
                    );
                    self.used_units
                        .fetch_sub(cost.units as usize, std::sync::atomic::Ordering::Relaxed);
                    self.in_flight_requests
                        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    Err(candle_core::Error::Msg(
                        "Expected streaming channel, got different result type".to_string(),
                    ))
                }
                Err(e) => {
                    error!(
                        "❌ ENGINE: Worker pool submission failed - request_id={}, error={:?}",
                        request_id, e
                    );
                    self.used_units
                        .fetch_sub(cost.units as usize, std::sync::atomic::Ordering::Relaxed);
                    self.in_flight_requests
                        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    Err(candle_core::Error::Msg(format!("{:?}", e)))
                }
            }
        } else {
            // Fallback: direct executor call (old path - may block!)
            warn!(
                "⚠️ ENGINE: No thread pool, using direct execution (may block!) - request_id={}",
                request_id
            );

            let executor = Arc::clone(&self.executor);
            let used_units = Arc::clone(&self.used_units);
            let in_flight = Arc::clone(&self.in_flight_requests);
            let cost_units = cost.units;
            let cost_units_usize = cost.units as usize;

            let meta = TaskMetadata::new(rand::random::<u64>(), ResourceCost::gpu_vram(cost_units));

            let result = executor.execute(job, meta).await;

            match result {
                InferenceResult::Streaming { token_rx, .. } => {
                    info!(
                        "📻 ENGINE: Got streaming channel from direct executor - request_id={}",
                        request_id
                    );

                    // Create cleanup signal channel
                    let (cleanup_tx, cleanup_rx) = flume::bounded::<()>(1);
                    let used_units_clone = Arc::clone(&used_units);
                    let in_flight_clone = Arc::clone(&in_flight);
                    let request_id_cleanup = request_id.clone();

                    tokio::spawn(async move {
                        info!(
                            "🧹 CLEANUP_TASK: Waiting for cleanup signal (direct) - request_id={}",
                            request_id_cleanup
                        );

                        let _ = cleanup_rx.recv_async().await;

                        info!(
                            "🏁 CLEANUP_TASK: Received cleanup signal, releasing resources (direct) - request_id={}",
                            request_id_cleanup
                        );

                        used_units_clone
                            .fetch_sub(cost_units_usize, std::sync::atomic::Ordering::Relaxed);
                        in_flight_clone.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

                        info!(
                            "♻️ CLEANUP_TASK: Resources released (direct) - request_id={}",
                            request_id_cleanup
                        );
                    });

                    // Wrap the receiver
                    let wrapped_rx = CleanupReceiver::new(token_rx, cleanup_tx, request_id.clone());

                    Ok(wrapped_rx)
                }
                InferenceResult::Error { message } => {
                    self.used_units
                        .fetch_sub(cost.units as usize, std::sync::atomic::Ordering::Relaxed);
                    self.in_flight_requests
                        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    Err(candle_core::Error::Msg(message))
                }
                _ => {
                    self.used_units
                        .fetch_sub(cost.units as usize, std::sync::atomic::Ordering::Relaxed);
                    self.in_flight_requests
                        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    Err(candle_core::Error::Msg(
                        "Unexpected result type for streaming request".to_string(),
                    ))
                }
            }
        }
    }

    // =========================================================================
    // Prompt building (unchanged from original)
    // =========================================================================

    /// Build a prompt from chat messages.
    pub fn build_prompt(
        &self,
        messages: &Messages,
        tools: Option<&Vec<Tool>>,
        thinking: bool,
    ) -> std::result::Result<String, String> {
        let mut conversation = DefaultConversation::new(
            self.model_name.clone(),
            self.chat_template.clone(),
            Vec::new(),
            self.sep_style.clone(),
            self.bos_token.clone(),
            self.eos_token.clone(),
            self.roles.clone(),
            DefaultConversationSeparators {
                sep: "\n".to_string(),
                sep2: None,
            },
        );

        if let Some(tools) = tools {
            conversation.set_tools(Some(tools.clone()));
        }

        match messages {
            Messages::Literal(msg) => {
                return Ok(msg.clone());
            }
            Messages::Chat(chat_messages) => {
                for message in chat_messages {
                    if message.role == "system" {
                        if let Some(ref content) = message.content {
                            let content_str = Self::extract_text_content(content);
                            conversation.set_system_message(Some(content_str));
                        }
                    }

                    let processed_content =
                        message.content.as_ref().map(Self::extract_text_content);

                    conversation.append_message_ext(
                        message.role.clone(),
                        processed_content,
                        message.tool_calls.clone(),
                        message.tool_call_id.clone(),
                        message.name.clone(),
                    );
                }
            }
            Messages::Map(map_messages) => {
                for message in map_messages {
                    let role = message.get("role").cloned().unwrap_or_default();
                    let content = message.get("content").cloned().unwrap_or_default();

                    if role == "system" {
                        conversation.set_system_message(Some(content.clone()));
                    }
                    conversation.append_message(role, content);
                }
            }
        }

        Ok(conversation.get_prompt(thinking))
    }

    fn extract_text_content(content: &MessageContent) -> String {
        match content {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Parts(parts) => {
                let mut result = String::new();
                for part in parts {
                    match part {
                        crate::openai::requests::ContentPart::Text { text } => {
                            result.push_str(text);
                            result.push('\n');
                        }
                        crate::openai::requests::ContentPart::ImageUrl { .. } => {
                            result.push_str("[Image]");
                            result.push('\n');
                        }
                    }
                }
                result
            }
        }
    }

    // =========================================================================
    // Streaming response building (unchanged from original)
    // =========================================================================

    /// Build a streaming chunk response.
    pub fn get_stream_response(
        &self,
        request_id: String,
        created: u64,
        content: Option<String>,
        finish_reason: Option<String>,
    ) -> ChatCompletionChunk {
        let mut choices = Vec::new();
        let mut tool_calls_delta: Option<Vec<ToolCallDelta>> = None;

        if let Some(ref text_chunk) = content {
            let mut accumulated = self.accumulated_text.write();
            let accumulated_text = accumulated
                .entry(request_id.clone())
                .or_insert_with(String::new);
            accumulated_text.push_str(text_chunk);

            let parser = get_tool_parser(&self.model_name);

            if parser.might_contain_tool_call(accumulated_text) {
                let parsed = parser.parse(accumulated_text);

                if let Some(tool_calls) = parsed.tool_calls() {
                    let mut parsed_tool_calls_map = self.parsed_tool_calls.write();
                    let existing_calls = parsed_tool_calls_map
                        .entry(request_id.clone())
                        .or_insert_with(Vec::new);

                    let mut deltas = Vec::new();
                    for (idx, tool_call) in tool_calls.iter().enumerate() {
                        let is_new = existing_calls.len() <= idx;

                        if is_new {
                            existing_calls.push(tool_call.clone());
                            deltas.push(ToolCallDelta {
                                index: idx,
                                id: Some(tool_call.id.clone()),
                                call_type: Some("function".to_string()),
                                function: Some(FunctionCallDelta {
                                    name: Some(tool_call.name.clone()),
                                    arguments: Some(tool_call.arguments.clone()),
                                }),
                            });
                        } else {
                            let existing = &existing_calls[idx];
                            if existing.arguments != tool_call.arguments {
                                existing_calls[idx] = tool_call.clone();
                                deltas.push(ToolCallDelta {
                                    index: idx,
                                    id: None,
                                    call_type: None,
                                    function: Some(FunctionCallDelta {
                                        name: None,
                                        arguments: Some(tool_call.arguments.clone()),
                                    }),
                                });
                            }
                        }
                    }

                    if !deltas.is_empty() {
                        tool_calls_delta = Some(deltas);
                    }
                }
            }
        }

        if finish_reason.is_some() {
            let mut accumulated = self.accumulated_text.write();
            accumulated.remove(&request_id);
            let mut parsed = self.parsed_tool_calls.write();
            parsed.remove(&request_id);
        }

        let (conversation_id, resource_id) = if finish_reason.is_none() && content.is_some() {
            self.request_metadata
                .read()
                .get(&request_id)
                .cloned()
                .unwrap_or((None, None))
        } else {
            (None, None)
        };

        let choice = Choice {
            delta: ChoiceData {
                role: Some(self.roles.0.clone()),
                content,
                tool_calls: tool_calls_delta,
                reasoning: None,
            },
            finish_reason,
            index: 0,
        };
        choices.push(choice);

        ChatCompletionChunk {
            id: request_id,
            choices,
            created,
            model: self.model_name.clone(),
            object: "chat.completion.chunk",
            system_fingerprint: Some(self.config.system_fingerprint()),
            conversation_id,
            resource_id,
        }
    }

    // =========================================================================
    // Legacy compatibility stubs
    // =========================================================================

    /// Legacy method stub for scheduler synchronization.
    pub fn sync_waiting_task_to_group(&mut self) -> bool {
        false
    }

    /// Legacy method stub for scheduler step processing.
    pub fn process_scheduler_step(&mut self) -> Result<usize> {
        Ok(0)
    }

    /// Get cached token count for a request.
    pub fn get_cached_tokens(&self, request_id: &str) -> Option<usize> {
        self.cached_tokens.read().get(request_id).copied()
    }

    /// Update usage response with cached token information.
    pub fn update_usage_with_cache(
        &self,
        usage: &mut ChatCompletionUsageResponse,
        request_id: &str,
    ) {
        if let Some(cached_count) = self.get_cached_tokens(request_id) {
            if cached_count > 0 {
                usage.prompt_tokens_details = Some(PromptTokensDetails {
                    cached_tokens: Some(cached_count),
                });
            }
        }
    }

    /// Clean up cached token tracking for a completed request.
    pub fn cleanup_cached_tokens(&self, request_id: &str) {
        self.cached_tokens.write().remove(request_id);
    }
}
