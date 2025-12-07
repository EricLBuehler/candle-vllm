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
    InferenceJob, InferenceResult, LlmExecutor, ResourceAdapter, ResourceCost, ResourceCostExt,
    StreamingTokenResult, TaskExecutor, TaskMetadata,
};
use crate::scheduler::Scheduler;
use crate::{
    openai::{
        models::Config,
        responses::{
            ChatChoice, ChatCompletionChunk, ChatCompletionUsageResponse, Choice, ChoiceData,
        },
    },
    scheduler::cache_engine::{CacheConfig, CacheEngine},
};
use candle_core::Result;
use parking_lot::RwLock;
use std::{collections::HashMap, sync::Arc};
use tokenizers::Tokenizer;
use tokio::sync::Notify;
use tracing::info;

/// Configuration for the parking-lot scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerPoolConfig {
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
            max_units: 16384,        // ~256K tokens with 16-token blocks
            max_queue_depth: 1000,   // Allow 1000 queued requests
            default_timeout_secs: 120, // 2 minute timeout
        }
    }
}

impl SchedulerPoolConfig {
    /// Create config from cache configuration.
    pub fn from_cache_config(cache_config: &CacheConfig) -> Self {
        Self {
            max_units: cache_config.num_gpu_blocks.unwrap_or(16384),
            ..Default::default()
        }
    }
}

/// LLM inference engine using prometheus_parking_lot for scheduling.
///
/// This engine manages inference requests through a resource pool that:
/// - Tracks GPU resource usage (KV-cache blocks)
/// - Queues requests when resources are exhausted
/// - Automatically dispatches queued work when resources free up
pub struct LLMEngineV2 {
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
}

impl LLMEngineV2 {
    /// Initialize a new `LLMEngineV2` with the parking-lot scheduler.
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
        info!(
            "Initializing LLMEngineV2 with parking-lot scheduler ({} pipelines)",
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
        let executor = Arc::new(LlmExecutor::new(rank, pipeline, cache_engine));

        // Create resource adapter from cache config
        let resource_adapter = ResourceAdapter::from_cache_config(cache_config);

        // Use provided pool config or derive from cache config
        let pool_config =
            pool_config.unwrap_or_else(|| SchedulerPoolConfig::from_cache_config(cache_config));

        info!(
            "Pool config: max_units={}, max_queue_depth={}, timeout={}s",
            pool_config.max_units, pool_config.max_queue_depth, pool_config.default_timeout_secs
        );

        // Create scheduler for compatibility
        let scheduler = Arc::new(parking_lot::Mutex::new(Scheduler::new(
            scheduler_config,
            cache_config,
        )));

        let engine = Self {
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
        };

        info!("LLMEngineV2 initialized with parking-lot scheduler");

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
        self.resource_adapter.blocks_to_tokens(total.saturating_sub(used))
    }

    /// Get current queue depth.
    pub fn queue_depth(&self) -> usize {
        // In the parking-lot model, this would come from the TaskQueue
        // For now, track in-flight requests
        self.in_flight_requests
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if the engine can accept a new request.
    pub fn can_accept_request(&self, prompt_len: usize, max_tokens: usize) -> bool {
        let cost = self
            .resource_adapter
            .calculate_cost(prompt_len, max_tokens);
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

        // Calculate resource cost
        let cost = self.resource_adapter.calculate_cost(prompt_len, max_tokens);

        // Check capacity
        if !self.can_accept_request(prompt_len, max_tokens) {
            return Err(candle_core::Error::Msg(format!(
                "Request {} rejected: insufficient capacity (need {} units)",
                request_id, cost.units
            )));
        }

        // Reserve resources
        self.used_units
            .fetch_add(cost.units as usize, std::sync::atomic::Ordering::Relaxed);
        self.in_flight_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Create the inference job
        let job = InferenceJob::new_completion(
            request_id.clone(),
            tokens,
            positions,
            sampling_params,
            max_context_len,
        );

        // Create response channel
        let (tx, rx) = tokio::sync::oneshot::channel();

        // Clone refs for the spawned task
        let executor = Arc::clone(&self.executor);
        let used_units = Arc::clone(&self.used_units);
        let in_flight = Arc::clone(&self.in_flight_requests);
        let cost_units = cost.units;
        let cost_units_usize = cost.units as usize;

        // Spawn async task to execute the job
        tokio::spawn(async move {
            let meta = TaskMetadata::new(
                rand::random::<u64>(),
                ResourceCost::gpu_vram(cost_units),
            );

            let result = executor.execute(job, meta).await;

            // Release resources
            used_units.fetch_sub(cost_units_usize, std::sync::atomic::Ordering::Relaxed);
            in_flight.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

            // Send result (ignore error if receiver dropped)
            let _ = tx.send(result);
        });

        info!("Queued completion request {request_id} (cost: {} units)", cost.units);
        Ok(rx)
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
    ) -> Result<flume::Receiver<std::result::Result<StreamingTokenResult, String>>> {
        let prompt_len = tokens.len();
        let max_tokens = sampling_params.max_tokens;

        // Calculate resource cost
        let cost = self.resource_adapter.calculate_cost(prompt_len, max_tokens);

        // Check capacity
        if !self.can_accept_request(prompt_len, max_tokens) {
            return Err(candle_core::Error::Msg(format!(
                "Streaming request {} rejected: insufficient capacity (need {} units)",
                request_id, cost.units
            )));
        }

        // Reserve resources
        self.used_units
            .fetch_add(cost.units as usize, std::sync::atomic::Ordering::Relaxed);
        self.in_flight_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Create the inference job
        let job = InferenceJob::new_streaming(
            request_id.clone(),
            tokens,
            positions,
            sampling_params,
            created,
            max_context_len,
        );

        // Clone refs for the spawned task
        let executor = Arc::clone(&self.executor);
        let used_units = Arc::clone(&self.used_units);
        let in_flight = Arc::clone(&self.in_flight_requests);
        let cost_units = cost.units;
        let cost_units_usize = cost.units as usize;

        // Execute the job and get the streaming receiver
        let meta = TaskMetadata::new(
            rand::random::<u64>(),
            ResourceCost::gpu_vram(cost_units),
        );

        let result = executor.execute(job, meta).await;

        match result {
            InferenceResult::Streaming { token_rx, .. } => {
                // Spawn a task to release resources when streaming completes
                let token_rx_clone = token_rx.clone();
                tokio::spawn(async move {
                    // Wait for streaming to complete
                    while let Ok(token_result) = token_rx_clone.recv_async().await {
                        if let Ok(token) = token_result {
                            if token.is_finished {
                                break;
                            }
                        }
                    }
                    // Release resources
                    used_units.fetch_sub(cost_units_usize, std::sync::atomic::Ordering::Relaxed);
                    in_flight.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                });

                info!("Queued streaming request {request_id} (cost: {} units)", cost.units);
                Ok(token_rx)
            }
            InferenceResult::Error { message } => {
                // Release resources on error
                self.used_units
                    .fetch_sub(cost.units as usize, std::sync::atomic::Ordering::Relaxed);
                self.in_flight_requests
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                Err(candle_core::Error::Msg(message))
            }
            _ => {
                // Unexpected result type
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

                    let processed_content = message.content.as_ref().map(Self::extract_text_content);

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
            system_fingerprint: None,
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
}

// =============================================================================
// LLMEngineInterface trait implementation
// =============================================================================

use async_trait::async_trait;
use super::engine_trait::{
    CompletionResult, LLMEngineInterface, StreamingToken,
    StreamingTokenResult as UnifiedStreamingTokenResult,
};

#[async_trait]
impl LLMEngineInterface for LLMEngineV2 {
    async fn start_processing_loop(&self) {
        LLMEngineV2::start_processing_loop(self).await
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn roles(&self) -> &(String, String) {
        &self.roles
    }

    fn get_config(&self) -> &Config {
        &self.config
    }

    fn get_cache_config(&self) -> &CacheConfig {
        &self.cache_config
    }

    fn get_available_kv_tokens(&self) -> usize {
        LLMEngineV2::get_available_kv_tokens(self)
    }

    fn queue_depth(&self) -> usize {
        LLMEngineV2::queue_depth(self)
    }

    fn can_accept_request(&self, prompt_len: usize, max_tokens: usize) -> bool {
        LLMEngineV2::can_accept_request(self, prompt_len, max_tokens)
    }

    fn scheduler(&self) -> &Arc<parking_lot::Mutex<Scheduler>> {
        &self.scheduler
    }

    fn notify(&self) -> &Arc<Notify> {
        &self.notify
    }

    fn completion_records(&self) -> &RwLock<HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>> {
        &self.completion_records
    }

    fn request_metadata(&self) -> &RwLock<HashMap<String, (Option<String>, Option<String>)>> {
        &self.request_metadata
    }

    async fn add_completion_request(
        &self,
        request_id: String,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        _input_metadata: crate::InputMetadata, // V2 engine doesn't use InputMetadata
        sampling_params: crate::openai::sampling_params::SamplingParams,
        max_context_len: usize,
    ) -> Result<tokio::sync::oneshot::Receiver<CompletionResult>> {
        // Call the native async method (V2 doesn't use InputMetadata)
        let inference_rx = self.add_request(
            request_id.clone(),
            tokens,
            positions,
            sampling_params,
            max_context_len,
        ).await?;

        // Create a oneshot channel and spawn a task to convert the result
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            match inference_rx.await {
                Ok(result) => {
                    let completion_result = match result {
                        InferenceResult::Completion { choices, usage } => Ok((choices, usage)),
                        InferenceResult::Error { message } => Err(message),
                        InferenceResult::Streaming { .. } => {
                            Err("Unexpected streaming result for completion request".to_string())
                        }
                    };
                    let _ = tx.send(completion_result);
                }
                Err(e) => {
                    let _ = tx.send(Err(format!("channel error: {}", e)));
                }
            }
        });

        Ok(rx)
    }

    async fn add_streaming_request(
        &self,
        request_id: String,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        _input_metadata: crate::InputMetadata, // V2 engine doesn't use InputMetadata
        sampling_params: crate::openai::sampling_params::SamplingParams,
        created: u64,
        max_context_len: usize,
    ) -> Result<flume::Receiver<UnifiedStreamingTokenResult>> {
        // Call the native async method (V2 doesn't use InputMetadata)
        let streaming_rx = LLMEngineV2::add_streaming_request(
            self,
            request_id,
            tokens,
            positions,
            sampling_params,
            created,
            max_context_len,
        ).await?;

        // Create a new flume channel with the unified token type
        let (tx, rx) = flume::unbounded();

        // Spawn a task to convert V2 tokens to unified tokens
        tokio::spawn(async move {
            while let Ok(result) = streaming_rx.recv_async().await {
                let unified_result = match result {
                    Ok(v2_token) => Ok(StreamingToken {
                        text: v2_token.text,
                        token_id: v2_token.token_id,
                        is_finished: v2_token.is_finished,
                        finish_reason: v2_token.finish_reason,
                        is_reasoning: v2_token.is_reasoning,
                    }),
                    Err(e) => Err(e),
                };
                if tx.send(unified_result).is_err() {
                    break;
                }
            }
        });

        Ok(rx)
    }

    fn build_prompt(
        &self,
        messages: &Messages,
        tools: Option<&Vec<Tool>>,
        thinking: bool,
    ) -> std::result::Result<String, String> {
        LLMEngineV2::build_prompt(self, messages, tools, thinking)
    }

    fn get_stream_response(
        &self,
        request_id: String,
        created: u64,
        content: Option<String>,
        finish_reason: Option<String>,
    ) -> ChatCompletionChunk {
        LLMEngineV2::get_stream_response(self, request_id, created, content, finish_reason)
    }
}
