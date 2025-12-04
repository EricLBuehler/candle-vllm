use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelLifecycleStatus {
    Loading,
    Ready,
    Switching,
    Error,
    Idle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub active_model: Option<String>,
    pub status: ModelLifecycleStatus,
    pub last_error: Option<String>,
    pub in_flight_requests: usize,
    pub switch_requested_at: Option<u64>,
    /// Queue length for each model
    #[serde(default)]
    pub queue_lengths: std::collections::HashMap<String, usize>,
}

/// Status information about request queues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStatus {
    /// Model name
    pub model: String,
    /// Current queue length
    pub length: usize,
    /// Maximum queue size
    pub max_size: usize,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerStatus {
    pub name: String,
    pub healthy: bool,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub model: ModelStatus,
    pub mcp_servers: Vec<McpServerStatus>,
}
