use super::work_item::{StreamingWorkItem, WorkItem};
use super::worker::InferenceWorker;
use super::DefaultPipeline;

#[cfg(feature = "nccl")]
use crate::openai::communicator::DaemonManager;
use crate::openai::conversation::default_conversation::{
    DefaultConversation, DefaultConversationSeparators, SeparatorStyle,
};
use crate::openai::conversation::Conversation;
use crate::openai::requests::{FunctionCallDelta, MessageContent, Messages, Tool, ToolCallDelta};
use crate::openai::tool_parser::get_tool_parser;
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
use candle_core::{Result, Tensor};
use crossbeam::channel::Sender as CrossbeamSender;
use parking_lot::RwLock;
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    thread::JoinHandle,
};
use tokenizers::Tokenizer;
use tokio::sync::Notify;
use tracing::{error, info, warn};

/// Prepared model inputs used in some helper paths.
/// In the lock-free worker design, actual input preparation happens inside workers,
/// but we keep this for potential future reuse and compatibility.
#[allow(dead_code)]
struct PreparedInputs {
    tokens: Tensor,
    positions: Tensor,
    metadata: crate::InputMetadata,
}

#[allow(dead_code)]
const PREFILL_CHUNK_SIZE: usize = 8192;

/// Lock-free inference engine fronting a pool of stateless workers.
///
/// Key design points:
/// - Each `InferenceWorker` owns its `DefaultPipeline` and `CacheEngine`.
/// - Requests are submitted via a lock-free `crossbeam-channel` queue.
/// - No `Arc<RwLock<LLMEngine>>` is used in the inference hot path.
/// - Scheduler is retained only for higher-level coordination, not per-token locking.
pub struct LLMEngine {
    /// Lock-free channel for work distribution.
    work_tx: CrossbeamSender<WorkItem>,

    /// Lock-free channel for streaming work distribution.
    streaming_work_tx: CrossbeamSender<StreamingWorkItem>,

    /// Worker thread handles (for graceful shutdown).
    worker_handles: Vec<JoinHandle<()>>,

    /// Shutdown signal senders.
    shutdown_txs: Vec<CrossbeamSender<()>>,

    /// Scheduler (kept for compatibility but not used in the inference hot path).
    pub scheduler: Arc<parking_lot::Mutex<Scheduler>>,

    /// Cache configuration (public for KV token calculation).
    pub cache_config: CacheConfig,

    /// Model configuration (public for server access).
    pub config: Config,

    pub notify: Arc<Notify>,

    /// Tokenizer cloned from first pipeline (shared, read-only).
    tokenizer: Tokenizer,

    /// Model name for identification.
    model_name: String,

    /// Chat template for prompt generation.
    chat_template: Option<String>,

    /// Conversation roles (user, assistant).
    roles: (String, String),

    /// Separator style for conversation formatting.
    sep_style: SeparatorStyle,

    /// BOS token for conversation.
    bos_token: Option<String>,

    /// EOS token for conversation.
    eos_token: Option<String>,

    /// Completed responses keyed by request_id.
    pub completion_records: RwLock<HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>>,

    /// Store conversation_id and resource_id per request.
    pub request_metadata: RwLock<HashMap<String, (Option<String>, Option<String>)>>,

    /// Track accumulated text per request for incremental tool call parsing.
    accumulated_text: RwLock<HashMap<String, String>>,

    /// Track parsed tool calls per request.
    parsed_tool_calls: RwLock<HashMap<String, Vec<crate::openai::tool_parser::ParsedToolCall>>>,

    /// Optional daemon manager for multi-process setups (nccl feature).
    #[cfg(feature = "nccl")]
    pub daemon_manager: RwLock<Option<DaemonManager>>,

    /// Optional prefill chunk size for prompt processing.
    #[allow(dead_code)]
    prefill_chunk_size: Option<usize>,
}

impl LLMEngine {
    /// Initialize a new lock-free `LLMEngine` with a worker pool.
    ///
    /// `pipelines` is a map from rank (GPU index) to `(pipeline, cache_engine)` pair.
    /// Each entry is moved into a dedicated worker thread.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pipelines: HashMap<usize, (Box<DefaultPipeline>, CacheEngine)>,
        scheduler_config: crate::scheduler::SchedulerConfig,
        cache_config: &CacheConfig,
        config: &Config,
        notify: Arc<Notify>,
        _holding_time: usize,
        _num_shards: usize,
        _multi_process: bool,
        #[cfg(feature = "nccl")] daemon_manager: Option<DaemonManager>,
        prefill_chunk_size: Option<usize>,
    ) -> Result<Self> {
        info!("Initializing LLMEngine with {} workers", pipelines.len());

        // Extract shared resources from first pipeline before moving to workers.
        // These are read-only and can be cloned/shared.
        let first_pipeline = pipelines
            .values()
            .next()
            .ok_or_else(|| candle_core::Error::Msg("no pipelines provided".to_string()))?;

        let tokenizer = first_pipeline.0.tokenizer().clone();
        let model_name = first_pipeline.0.name().to_string();
        let conversation = first_pipeline.0.get_past_conversation();
        let roles = conversation.get_roles().clone();

        // Extract chat template and conversation settings from pipeline's conversation
        // We'll use a default separator style since we're doing stateless prompt building
        let chat_template = None; // Will be set from tokenizer_config if available
        let sep_style = SeparatorStyle::ChatML; // Default, works for most models
        let bos_token = None;
        let eos_token = None;

        // Create unbounded work channel for completion requests.
        let (work_tx, work_rx) = crossbeam::channel::unbounded::<WorkItem>();

        // Create unbounded work channel for streaming requests.
        let (streaming_work_tx, streaming_work_rx) =
            crossbeam::channel::unbounded::<StreamingWorkItem>();

        let mut worker_handles = Vec::new();
        let mut shutdown_txs = Vec::new();

        // Spawn one worker per pipeline (typically per GPU rank).
        for (rank, (pipeline, cache_engine)) in pipelines {
            let (shutdown_tx, shutdown_rx) = crossbeam::channel::bounded::<()>(1);
            let worker_rx = work_rx.clone();
            let streaming_rx = streaming_work_rx.clone();

            let worker = InferenceWorker::new(
                rank,
                pipeline,
                cache_engine,
                worker_rx,
                streaming_rx,
                shutdown_rx,
            );

            let handle = std::thread::Builder::new()
                .name(format!("inference-worker-{rank}"))
                .stack_size(8 * 1024 * 1024) // 8MB stack for large models
                .spawn(move || worker.run())
                .map_err(|e| candle_core::Error::Msg(format!("failed to spawn worker: {e}")))?;

            worker_handles.push(handle);
            shutdown_txs.push(shutdown_tx);

            info!("Spawned inference worker for rank {rank}");
        }

        let scheduler = Arc::new(parking_lot::Mutex::new(Scheduler::new(
            scheduler_config,
            cache_config,
        )));

        let engine = Self {
            work_tx,
            streaming_work_tx,
            worker_handles,
            shutdown_txs,
            scheduler,
            cache_config: cache_config.clone(),
            config: config.clone(),
            notify: notify.clone(),
            tokenizer,
            model_name,
            chat_template,
            roles,
            sep_style,
            bos_token,
            eos_token,
            completion_records: RwLock::new(HashMap::new()),
            request_metadata: RwLock::new(HashMap::new()),
            accumulated_text: RwLock::new(HashMap::new()),
            parsed_tool_calls: RwLock::new(HashMap::new()),
            #[cfg(feature = "nccl")]
            daemon_manager: RwLock::new(daemon_manager),
            prefill_chunk_size,
        };

        info!("LLMEngine initialized with lock-free worker pool");

        Ok(engine)
    }

    /// Start the main processing loop (no-op in lock-free design).
    ///
    /// Workers continuously pull work from the channel and process requests.
    pub async fn start_processing_loop(&self) {
        info!("Lock-free worker pool active; processing loop is worker-driven");
    }

    /// Legacy scheduler synchronization is not used in the lock-free worker design.
    /// Kept as a no-op for compatibility with existing call sites.
    pub fn sync_waiting_task_to_group(&mut self) -> bool {
        false
    }

    // =========================================================================
    // Accessor methods for shared resources (lock-free, read-only)
    // =========================================================================

    /// Get a reference to the shared tokenizer.
    ///
    /// The tokenizer is cloned from the first pipeline during initialization
    /// and is safe to use without locks.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get the conversation roles (user_role, assistant_role).
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

    /// Calculate available KV cache tokens based on cache configuration.
    ///
    /// This is computed from the cache config and doesn't require pipeline access.
    pub fn get_available_kv_tokens(&self) -> usize {
        self.cache_config
            .num_gpu_blocks
            .unwrap_or(0)
            .saturating_mul(self.cache_config.block_size)
    }

    /// Build a prompt from chat messages using stateless conversation handling.
    ///
    /// This method creates a temporary conversation, populates it with messages,
    /// and generates the prompt string without requiring pipeline access.
    pub fn build_prompt(
        &self,
        messages: &Messages,
        tools: Option<&Vec<Tool>>,
        thinking: bool,
    ) -> std::result::Result<String, String> {
        // Create a temporary conversation for this request
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

        // Set tools if provided
        if let Some(tools) = tools {
            conversation.set_tools(Some(tools.clone()));
        }

        // Process messages based on type
        match messages {
            Messages::Literal(msg) => {
                return Ok(msg.clone());
            }
            Messages::Chat(chat_messages) => {
                for message in chat_messages {
                    // Handle system message
                    if message.role == "system" {
                        if let Some(ref content) = message.content {
                            let content_str = Self::extract_text_content(content);
                            conversation.set_system_message(Some(content_str));
                        }
                    }

                    // Extract content as string
                    let processed_content = message.content.as_ref().map(Self::extract_text_content);

                    // Append message with full tool support
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
                // Legacy format - convert to simple messages
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

    /// Extract text content from MessageContent enum.
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
                            // Images need to be processed separately by vision tool
                            result.push_str("[Image]");
                            result.push('\n');
                        }
                    }
                }
                result
            }
        }
    }

    /// Add a request to the lock-free work queue.
    ///
    /// Returns a channel that yields a single completion result:
    /// `(Vec<ChatChoice>, ChatCompletionUsageResponse)` or an error string.
    /// Add a completion request to the lock-free work queue.
    ///
    /// Returns a channel that yields a single completion result:
    /// `(Vec<ChatChoice>, ChatCompletionUsageResponse)` or an error string.
    pub fn add_request(
        &self,
        request_id: String,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        input_metadata: crate::InputMetadata,
        sampling_params: crate::openai::sampling_params::SamplingParams,
    ) -> Result<
        crossbeam::channel::Receiver<
            std::result::Result<
                (
                    Vec<crate::openai::responses::ChatChoice>,
                    crate::openai::responses::ChatCompletionUsageResponse,
                ),
                String,
            >,
        >,
    > {
        let (response_tx, response_rx) = crossbeam::channel::bounded(1);

        let work = WorkItem::new(
            request_id.clone(),
            tokens,
            positions,
            input_metadata,
            sampling_params,
            response_tx,
        );

        if let Err(e) = self.work_tx.send(work) {
            return Err(candle_core::Error::Msg(format!(
                "failed to enqueue work item for request {request_id}: {e}"
            )));
        }

        info!("Queued completion request {request_id} for processing by workers");
        Ok(response_rx)
    }

    /// Add a streaming request to the lock-free work queue.
    ///
    /// Returns a flume receiver that yields individual tokens as they are generated.
    pub fn add_streaming_request(
        &self,
        request_id: String,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        input_metadata: crate::InputMetadata,
        sampling_params: crate::openai::sampling_params::SamplingParams,
        created: u64,
    ) -> Result<flume::Receiver<std::result::Result<super::work_item::StreamingToken, String>>> {
        let (stream_tx, stream_rx) = flume::unbounded();

        let work = StreamingWorkItem::new(
            request_id.clone(),
            tokens,
            positions,
            input_metadata,
            sampling_params,
            stream_tx,
            created,
        );

        if let Err(e) = self.streaming_work_tx.send(work) {
            return Err(candle_core::Error::Msg(format!(
                "failed to enqueue streaming work item for request {request_id}: {e}"
            )));
        }

        info!("Queued streaming request {request_id} for processing by workers");
        Ok(stream_rx)
    }

    /// Gracefully shutdown all workers and wait for completion.
    pub fn shutdown(self) -> Result<()> {
        info!(
            "Shutting down LLMEngine with {} workers...",
            self.worker_handles.len()
        );

        for (i, tx) in self.shutdown_txs.into_iter().enumerate() {
            if let Err(_) = tx.send(()) {
                warn!("Worker {i} already terminated");
            }
        }

        for (i, handle) in self.worker_handles.into_iter().enumerate() {
            match handle.join() {
                Ok(_) => info!("Worker {i} terminated gracefully"),
                Err(e) => error!("Worker {i} panicked: {e:?}"),
            }
        }

        info!("LLMEngine shutdown complete");
        Ok(())
    }

    /// Build a streaming chunk response, including incremental tool-call deltas.
    ///
    /// This logic is independent of the internal worker/scheduler architecture and
    /// only depends on accumulated text and the model's tool-calling format.
    /// No pipeline access required - uses stored model_name and roles.
    pub fn get_stream_response(
        &self,
        request_id: String,
        created: u64,
        content: Option<String>,
        finish_reason: Option<String>,
    ) -> ChatCompletionChunk {
        let mut choices = Vec::new();
        let mut tool_calls_delta: Option<Vec<ToolCallDelta>> = None;

        // If we have content, accumulate it and check for tool calls.
        if let Some(ref text_chunk) = content {
            let mut accumulated = self.accumulated_text.write();
            let accumulated_text = accumulated
                .entry(request_id.clone())
                .or_insert_with(String::new);
            accumulated_text.push_str(text_chunk);

            // Parse tool calls incrementally using stored model name.
            let parser = get_tool_parser(&self.model_name);

            if parser.might_contain_tool_call(accumulated_text) {
                let parsed = parser.parse(accumulated_text);

                if let Some(tool_calls) = parsed.tool_calls() {
                    let mut parsed_tool_calls_map = self.parsed_tool_calls.write();
                    let existing_calls = parsed_tool_calls_map
                        .entry(request_id.clone())
                        .or_insert_with(Vec::new);

                    // Generate deltas for new or updated tool calls.
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

        // If finished, clean up accumulated text and parsed tool state.
        if finish_reason.is_some() {
            let mut accumulated = self.accumulated_text.write();
            accumulated.remove(&request_id);
            let mut parsed = self.parsed_tool_calls.write();
            parsed.remove(&request_id);
        }

        // Retrieve conversation_id and resource_id (only include when streaming content).
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

    /// In the lock-free worker design, scheduler cache operations and block tables
    /// are delegated to workers and lower-level components. This method is kept as
    /// a stub to avoid compile breakage for any remaining call sites.
    pub fn process_scheduler_step(&mut self) -> Result<usize> {
        Ok(0)
    }

    /// Placeholder for legacy API that prepared block tables from sequence groups.
    ///
    /// The lock-free architecture handles KV cache and block mapping inside workers.
    pub fn prepare_block_tables(
        &self,
        _groups: &VecDeque<Arc<crate::scheduler::sequence::SequenceGroup>>,
        _device: &candle_core::Device,
    ) -> Result<Tensor> {
        candle_core::bail!("prepare_block_tables is not used in the lock-free worker design")
    }

    /// Placeholder for legacy API that prepared prompt inputs.
    ///
    /// The lock-free architecture handles prompt preparation inside workers via
    /// `InferenceWorker::process_work`.
    #[allow(dead_code)]
    fn prepare_prompt(
        &self,
        _groups: &VecDeque<Arc<crate::scheduler::sequence::SequenceGroup>>,
        _device: &candle_core::Device,
    ) -> Result<PreparedInputs> {
        candle_core::bail!("prepare_prompt is not used in the lock-free worker design")
    }
}
