use crate::openai::responses::{ChatChoice, ChatCompletionUsageResponse};
use crate::openai::sampling_params::SamplingParams;
use crate::InputMetadata;

/// A work item sent to inference workers via lock-free channels.
/// Contains all data needed to process one inference request (non-streaming).
pub struct WorkItem {
    /// Unique request ID for tracing
    pub request_id: String,

    /// Input token IDs
    pub tokens: Vec<u32>,

    /// Token positions for positional encoding
    pub positions: Vec<usize>,

    /// Fully prepared input metadata for this request
    pub input_metadata: InputMetadata,

    /// Sampling parameters for generation
    pub sampling_params: SamplingParams,

    /// Channel to send response back to client
    /// Uses oneshot for single response
    pub response_tx: ResponseSender,
}

/// Type alias for completion response sender (single response).
pub type ResponseSender =
    crossbeam::channel::Sender<Result<(Vec<ChatChoice>, ChatCompletionUsageResponse), String>>;

impl WorkItem {
    pub fn new(
        request_id: String,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        input_metadata: InputMetadata,
        sampling_params: SamplingParams,
        response_tx: ResponseSender,
    ) -> Self {
        Self {
            request_id,
            tokens,
            positions,
            input_metadata,
            sampling_params,
            response_tx,
        }
    }
}

/// A streaming work item for token-by-token generation.
/// Workers will send individual tokens or chunks back via the streaming channel.
pub struct StreamingWorkItem {
    /// Unique request ID for tracing
    pub request_id: String,

    /// Input token IDs
    pub tokens: Vec<u32>,

    /// Token positions for positional encoding
    pub positions: Vec<usize>,

    /// Fully prepared input metadata for this request
    pub input_metadata: InputMetadata,

    /// Sampling parameters for generation
    pub sampling_params: SamplingParams,

    /// Channel to send streaming tokens back to client
    pub stream_tx: StreamingResponseSender,

    /// Created timestamp for response metadata
    pub created: u64,
}

/// Token chunk sent during streaming generation.
#[derive(Debug, Clone)]
pub struct StreamingToken {
    /// The generated token text
    pub text: String,
    /// Token ID
    pub token_id: u32,
    /// Whether this is the final token
    pub is_finished: bool,
    /// Finish reason if finished
    pub finish_reason: Option<String>,
    /// Whether this token is part of reasoning/thinking output
    /// Reasoning tokens are emitted by models that support chain-of-thought
    /// or thinking capabilities when thinking mode is enabled
    pub is_reasoning: bool,
}

/// Type alias for streaming response sender (multiple tokens).
pub type StreamingResponseSender = flume::Sender<Result<StreamingToken, String>>;

impl StreamingWorkItem {
    pub fn new(
        request_id: String,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        input_metadata: InputMetadata,
        sampling_params: SamplingParams,
        stream_tx: StreamingResponseSender,
        created: u64,
    ) -> Self {
        Self {
            request_id,
            tokens,
            positions,
            input_metadata,
            sampling_params,
            stream_tx,
            created,
        }
    }
}

/// GPU resource monitoring structure to track memory usage
#[derive(Debug, Clone)]
pub struct GpuResourceInfo {
    pub device_id: usize,
    pub memory_used_mb: usize,
    pub memory_total_mb: usize,
    pub utilization_percent: f32,
    pub temperature_c: Option<f32>,
}

impl GpuResourceInfo {
    pub fn memory_usage_percent(&self) -> f32 {
        if self.memory_total_mb == 0 {
            0.0
        } else {
            (self.memory_used_mb as f32 / self.memory_total_mb as f32) * 100.0
        }
    }

    pub fn is_memory_critical(&self, threshold_percent: f32) -> bool {
        self.memory_usage_percent() > threshold_percent
    }
}

/// Resource monitor for tracking GPU memory usage across workers
pub struct GpuResourceMonitor {
    warning_threshold_percent: f32,
    critical_threshold_percent: f32,
}

impl GpuResourceMonitor {
    pub fn new(warning_threshold: f32, critical_threshold: f32) -> Self {
        Self {
            warning_threshold_percent: warning_threshold,
            critical_threshold_percent: critical_threshold,
        }
    }

    /// Check GPU memory usage and return resource info
    pub fn check_gpu_resources(&self, device_id: usize) -> Result<GpuResourceInfo, String> {
        // TODO: Implement actual GPU memory monitoring
        // For now, return mock data but structure is ready for real implementation

        #[cfg(feature = "cuda")]
        {
            // In a real implementation, we would use CUDA APIs to get actual memory info
            // This is a placeholder structure
            Ok(GpuResourceInfo {
                device_id,
                memory_used_mb: 4000,      // Mock: 4GB used
                memory_total_mb: 12000,    // Mock: 12GB total (e.g., RTX 3080Ti)
                utilization_percent: 75.0, // Mock: 75% utilization
                temperature_c: Some(68.0), // Mock: 68°C
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            // For non-CUDA builds, return minimal info
            Ok(GpuResourceInfo {
                device_id,
                memory_used_mb: 0,
                memory_total_mb: 0,
                utilization_percent: 0.0,
                temperature_c: None,
            })
        }
    }

    pub fn is_safe_to_process(&self, resource_info: &GpuResourceInfo) -> Result<(), String> {
        if resource_info.is_memory_critical(self.critical_threshold_percent) {
            return Err(format!(
                "GPU {} memory usage critical: {:.1}% (>{:.1}% threshold)",
                resource_info.device_id,
                resource_info.memory_usage_percent(),
                self.critical_threshold_percent
            ));
        }

        if resource_info.is_memory_critical(self.warning_threshold_percent) {
            tracing::warn!(
                device_id = resource_info.device_id,
                memory_percent = resource_info.memory_usage_percent(),
                threshold = self.warning_threshold_percent,
                "GPU memory usage approaching warning threshold"
            );
        }

        Ok(())
    }
}
