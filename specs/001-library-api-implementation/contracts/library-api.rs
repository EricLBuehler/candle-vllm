//! Library API Contract Definitions
//!
//! This file defines the public API surface for candle-vllm as a library.
//! These are contracts - the actual implementations live in the crates.

// =============================================================================
// candle-vllm-core Public API
// =============================================================================

/// Core inference engine - wraps model, scheduler, and cache management.
pub trait InferenceEngineApi {
    /// Create a new builder for configuring the engine.
    fn builder() -> InferenceEngineBuilder;

    /// Load a model and create the engine.
    async fn new(config: EngineConfig) -> Result<Self>;

    /// Tokenize text input to token IDs.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;

    /// Detokenize token IDs back to text.
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;

    /// Generate tokens from a prompt (non-streaming).
    async fn generate(
        &mut self,
        prompt: Vec<u32>,
        params: GenerationParams,
    ) -> Result<GenerationOutput>;

    /// Generate tokens with streaming output.
    async fn generate_stream(
        &mut self,
        prompt: Vec<u32>,
        params: GenerationParams,
    ) -> Result<impl Stream<Item = Result<TokenOutput>>>;

    /// Add a generation request to the queue (for batching).
    fn add_request(&mut self, request: InferenceRequest) -> RequestHandle;

    /// Cancel an in-flight request.
    fn cancel_request(&mut self, handle: RequestHandle);

    /// Execute one step of generation (for custom scheduling).
    fn step(&mut self) -> Result<Vec<RequestUpdate>>;

    /// Get model information.
    fn model_info(&self) -> &ModelInfo;

    /// Get current statistics.
    fn stats(&self) -> EngineStats;
}

/// Builder for InferenceEngine configuration.
pub trait InferenceEngineBuilderApi {
    fn model_path(self, path: impl Into<String>) -> Self;
    fn device(self, device: Device) -> Self;
    fn dtype(self, dtype: DType) -> Self;
    fn max_batch_size(self, size: usize) -> Self;
    fn max_sequence_length(self, length: usize) -> Self;
    fn kv_cache_memory(self, bytes: usize) -> Self;
    fn enable_cuda_graph(self, enable: bool) -> Self;
    fn enable_chunked_prefill(self, enable: bool) -> Self;
    fn prefill_chunk_size(self, size: usize) -> Self;
    async fn build(self) -> Result<InferenceEngine>;
}

/// Configuration for the inference engine.
pub struct EngineConfig {
    pub model_path: String,
    pub device: Device,
    pub dtype: DType,
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub kv_cache_memory: usize,
    pub enable_cuda_graph: bool,
    pub enable_chunked_prefill: bool,
    pub prefill_chunk_size: usize,
}

/// Parameters for text generation.
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub stop_sequences: Vec<String>,
    pub logprobs: bool,
    pub seed: Option<u64>,
}

/// Result of a generation operation.
pub struct GenerationOutput {
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    pub logprobs: Option<Vec<f32>>,
    pub stats: GenerationStats,
}

/// Why generation stopped.
pub enum FinishReason {
    Stop,
    Length,
    StopSequence(String),
    Cancelled,
    Error(String),
}

/// Generation performance statistics.
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_time_ms: u64,
    pub tokens_per_second: f32,
}

// =============================================================================
// candle-vllm-openai Public API
// =============================================================================

/// OpenAI-compatible adapter wrapping the inference engine.
pub trait OpenAIAdapterApi {
    /// Create a new adapter from an inference engine.
    fn new(engine: InferenceEngine) -> Self;

    /// Create with custom configuration.
    fn with_config(engine: InferenceEngine, config: AdapterConfig) -> Self;

    /// Handle a chat completion request.
    async fn chat_completion(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse>;

    /// Handle a streaming chat completion request.
    async fn chat_completion_stream(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Result<impl Stream<Item = Result<ChatCompletionChunk>>>;

    /// Get the underlying engine (for custom operations).
    fn engine(&self) -> &InferenceEngine;

    /// Get mutable access to the engine.
    fn engine_mut(&mut self) -> &mut InferenceEngine;
}

// Request/Response types are already defined in candle-vllm-core/src/openai/requests.rs
// and candle-vllm-core/src/openai/responses.rs

// =============================================================================
// candle-vllm-responses Public API
// =============================================================================

/// MCP client for server communication.
pub trait McpClientApi {
    /// Connect to an MCP server.
    async fn connect(config: McpServerConfig) -> Result<Self>;

    /// List available tools from the server.
    async fn list_tools(&self) -> Result<Vec<Value>>;

    /// Execute a tool call.
    async fn call_tool(&self, name: &str, payload: Value) -> Result<Value>;
}

/// Configuration for MCP server connection.
pub struct McpServerConfig {
    pub url: String,
    pub auth: Option<String>,
    pub timeout_secs: u64,
}

/// High-level session for multi-turn conversations.
pub trait ResponsesSessionApi {
    /// Create a new session builder.
    fn builder() -> ResponsesSessionBuilder;

    /// Create a session from configuration.
    async fn new(config: SessionConfig) -> Result<Self>;

    /// Add an MCP server to the session.
    async fn add_mcp_server(
        &mut self,
        name: &str,
        url: &str,
        auth: Option<String>,
    ) -> Result<()>;

    /// List all available tools from connected MCP servers.
    async fn list_openai_tools(
        &self,
        allowed_tools: Option<Vec<String>>,
    ) -> Result<Vec<Tool>>;

    /// Run a multi-turn conversation with automatic tool execution.
    async fn run_conversation(
        &mut self,
        initial_messages: Vec<Message>,
        options: ConversationOptions,
    ) -> Result<ConversationResult>;

    /// Execute a specific tool by server and tool name.
    async fn call_tool(
        &self,
        server: &str,
        tool_name: &str,
        payload: Value,
    ) -> Result<Value>;
}

/// Builder for ResponsesSession.
pub trait ResponsesSessionBuilderApi {
    fn model_path(self, path: impl Into<String>) -> Self;
    fn device(self, device: Device) -> Self;
    fn add_mcp_server(
        self,
        name: impl Into<String>,
        url: impl Into<String>,
        auth: Option<String>,
    ) -> Self;
    async fn build(self) -> Result<ResponsesSession>;
}

/// Configuration for a responses session.
pub struct SessionConfig {
    pub model_path: Option<String>,
    pub device: Option<usize>,
}

/// Options for running a conversation.
pub struct ConversationOptions {
    pub max_turns: usize,
    pub allowed_tools: Option<Vec<String>>,
}

/// Result of a completed conversation.
pub struct ConversationResult {
    pub final_message: String,
    pub tool_calls_executed: Vec<ToolCall>,
    pub turns_taken: usize,
}

// =============================================================================
// candle-vllm-server Public API (for embedding)
// =============================================================================

/// Model manager for handling model lifecycle.
pub trait ModelManagerApi {
    /// Create a new model manager.
    fn new(models_state: ModelsState, max_queue: usize) -> Self;

    /// Enqueue a request to switch to a model.
    fn enqueue_switch(&self, model: &str) -> Result<SwitchResult>;

    /// Start the next switch if pending.
    fn begin_next_switch(&self) -> Option<ModelSwitchRequest>;

    /// Mark switch as complete.
    async fn complete_switch(&self, model: String);

    /// Mark switch as failed.
    fn fail_switch(&self, err: String);

    /// Get current status.
    fn status(&self) -> ModelStatus;
}

/// Result of a switch enqueue operation.
pub enum SwitchResult {
    Enqueued,
    AlreadyActive,
}

/// Model lifecycle status.
pub enum ModelLifecycleStatus {
    Idle,
    Ready,
    Switching,
    Loading,
    Error,
}

/// Status information for model manager.
pub struct ModelStatus {
    pub active_model: Option<String>,
    pub status: ModelLifecycleStatus,
    pub last_error: Option<String>,
    pub in_flight_requests: usize,
    pub switch_requested_at: Option<u64>,
}

// =============================================================================
// Error Types
// =============================================================================

/// Error type for library operations.
pub enum Error {
    /// Model loading error
    ModelLoad(String),

    /// Tokenization error
    Tokenization(String),

    /// Generation error
    Generation(String),

    /// Device error (CUDA/Metal)
    Device(String),

    /// Configuration error
    Config(String),

    /// MCP connection error
    McpConnection(String),

    /// MCP tool execution error
    McpToolExecution(String),

    /// Request cancelled
    Cancelled,

    /// IO error
    Io(std::io::Error),

    /// Candle error
    Candle(candle_core::Error),

    /// Other errors
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

