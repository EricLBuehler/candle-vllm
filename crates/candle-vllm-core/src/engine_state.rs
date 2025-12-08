use crate::api::{InferenceEngine, ModelInfo};
use crate::openai::image_tool::ImageDescriptionTool;
use crate::openai::local_vision_tool::LocalVisionModelTool;
use crate::openai::vision_proxy::{PreprocessingStats, VisionProxyConfig};
use crate::vision::{VisionError, VisionResult};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use tracing::{debug, info};

/// Overall health status for the engine
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineHealthStatus {
    /// Engine is fully operational
    Healthy,
    /// Engine has minor issues but is functional
    Degraded,
    /// Engine has significant issues
    Unhealthy,
    /// Engine is not initialized or starting up
    Initializing,
    /// Engine is shutting down
    ShuttingDown,
}

/// Health information for a vision tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionToolHealth {
    pub name: String,
    pub is_available: bool,
    pub last_check: SystemTime,
    pub response_time_ms: Option<u64>,
    pub error_count: usize,
    pub success_count: usize,
    pub model_info: Option<HashMap<String, serde_json::Value>>,
}

/// Statistics for engine operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    pub requests_processed: usize,
    pub total_processing_time_ms: u64,
    pub average_processing_time_ms: Option<u64>,
    pub errors_encountered: usize,
    pub vision_preprocessing_stats: Option<PreprocessingStats>,
    pub uptime_ms: u64,
    pub started_at: SystemTime,
}

impl Default for EngineStats {
    fn default() -> Self {
        Self {
            requests_processed: 0,
            total_processing_time_ms: 0,
            average_processing_time_ms: None,
            errors_encountered: 0,
            vision_preprocessing_stats: None,
            uptime_ms: 0,
            started_at: SystemTime::now(),
        }
    }
}

impl EngineStats {
    /// Update statistics with a new request processing result
    pub fn update_request_stats(&mut self, processing_time_ms: u64, success: bool) {
        self.requests_processed += 1;
        self.total_processing_time_ms += processing_time_ms;

        if !success {
            self.errors_encountered += 1;
        }

        self.average_processing_time_ms = if self.requests_processed > 0 {
            Some(self.total_processing_time_ms / self.requests_processed as u64)
        } else {
            None
        };

        self.uptime_ms = SystemTime::now()
            .duration_since(self.started_at)
            .unwrap_or_default()
            .as_millis() as u64;
    }

    /// Update vision preprocessing statistics
    pub fn update_vision_stats(&mut self, preprocessing_stats: PreprocessingStats) {
        self.vision_preprocessing_stats = Some(preprocessing_stats);
    }
}

/// Configuration for engine state management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStateConfig {
    /// How often to perform health checks (in seconds)
    pub health_check_interval_secs: u64,
    /// How many consecutive failures before marking a tool as unhealthy
    pub max_consecutive_failures: usize,
    /// Whether to enable automatic recovery attempts
    pub enable_auto_recovery: bool,
    /// Timeout for health check operations (in milliseconds)
    pub health_check_timeout_ms: u64,
    /// Whether to include detailed statistics in health reports
    pub include_detailed_stats: bool,
}

impl Default for EngineStateConfig {
    fn default() -> Self {
        Self {
            health_check_interval_secs: 60,
            max_consecutive_failures: 3,
            enable_auto_recovery: true,
            health_check_timeout_ms: 5000,
            include_detailed_stats: true,
        }
    }
}

/// Comprehensive engine state manager
pub struct EngineStateManager {
    /// Primary text inference engine
    primary_engine: Arc<InferenceEngine>,

    /// Optional vision tool for multimodal processing
    vision_tool: Option<Arc<LocalVisionModelTool>>,

    /// Configuration for vision proxy preprocessing
    vision_proxy_config: Option<VisionProxyConfig>,

    /// Current health status
    health_status: Arc<RwLock<EngineHealthStatus>>,

    /// Vision tool health information
    vision_tool_health: Arc<RwLock<Option<VisionToolHealth>>>,

    /// Engine statistics
    stats: Arc<RwLock<EngineStats>>,

    /// State management configuration
    config: EngineStateConfig,

    /// Timestamp of last health check
    last_health_check: Arc<RwLock<Option<Instant>>>,

    /// Flag to indicate if shutdown is in progress
    shutting_down: Arc<RwLock<bool>>,
}

impl EngineStateManager {
    /// Create a new engine state manager
    pub fn new(
        primary_engine: Arc<InferenceEngine>,
        vision_tool: Option<Arc<LocalVisionModelTool>>,
        vision_proxy_config: Option<VisionProxyConfig>,
        config: EngineStateConfig,
    ) -> Self {
        let initial_health = if vision_tool.is_some() {
            EngineHealthStatus::Initializing
        } else {
            EngineHealthStatus::Healthy
        };

        Self {
            primary_engine,
            vision_tool,
            vision_proxy_config,
            health_status: Arc::new(RwLock::new(initial_health)),
            vision_tool_health: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(EngineStats::default())),
            config,
            last_health_check: Arc::new(RwLock::new(None)),
            shutting_down: Arc::new(RwLock::new(false)),
        }
    }

    /// Get the current health status
    pub fn get_health_status(&self) -> EngineHealthStatus {
        self.health_status.read().clone()
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> EngineStats {
        self.stats.read().clone()
    }

    /// Get information about the primary model
    pub fn get_primary_model_info(&self) -> &ModelInfo {
        self.primary_engine.model_info()
    }

    /// Check if vision support is available
    pub async fn is_vision_available(&self) -> bool {
        if let Some(ref tool) = self.vision_tool {
            tool.is_available().await
        } else {
            false
        }
    }

    /// Get vision tool health information
    pub fn get_vision_tool_health(&self) -> Option<VisionToolHealth> {
        self.vision_tool_health.read().clone()
    }

    /// Update engine statistics with request processing results
    pub fn update_request_stats(&self, processing_time_ms: u64, success: bool) {
        self.stats
            .write()
            .update_request_stats(processing_time_ms, success);
    }

    /// Update vision preprocessing statistics
    pub fn update_vision_stats(&self, preprocessing_stats: PreprocessingStats) {
        self.stats.write().update_vision_stats(preprocessing_stats);
    }

    /// Perform a comprehensive health check
    pub async fn perform_health_check(&self) -> VisionResult<()> {
        if *self.shutting_down.read() {
            return Ok(());
        }

        debug!("Performing engine health check");
        let start_time = Instant::now();

        // Check primary engine health (basic check)
        let primary_healthy = self.check_primary_engine_health().await;

        // Check vision tool health if available
        let vision_health = if let Some(ref tool) = self.vision_tool {
            Some(self.check_vision_tool_health(tool).await)
        } else {
            None
        };

        // Update overall health status
        let overall_status = self
            .determine_overall_health_status(primary_healthy, &vision_health)
            .await;
        *self.health_status.write() = overall_status;

        // Update vision tool health info
        if let Some(health) = vision_health {
            *self.vision_tool_health.write() = Some(health);
        }

        // Update last health check timestamp
        *self.last_health_check.write() = Some(start_time);

        let check_duration = start_time.elapsed().as_millis() as u64;
        debug!("Health check completed in {}ms", check_duration);

        Ok(())
    }

    /// Check if a health check is needed based on the configured interval
    pub fn should_perform_health_check(&self) -> bool {
        if *self.shutting_down.read() {
            return false;
        }

        let last_check = self.last_health_check.read();
        match *last_check {
            Some(last) => {
                let elapsed = last.elapsed().as_secs();
                elapsed >= self.config.health_check_interval_secs
            }
            None => true, // Never checked before
        }
    }

    /// Get a comprehensive health report
    pub async fn get_health_report(&self) -> VisionResult<serde_json::Value> {
        let status = self.get_health_status();
        let stats = if self.config.include_detailed_stats {
            Some(self.get_stats())
        } else {
            None
        };

        let vision_health = self.get_vision_tool_health();
        let primary_info = self.get_primary_model_info();

        Ok(serde_json::json!({
            "status": status,
            "primary_model": {
                "path": primary_info.model_path,
                "max_sequence_length": primary_info.max_sequence_length,
                "max_batch_size": primary_info.max_batch_size,
                "dtype": primary_info.dtype
            },
            "vision_support": {
                "enabled": self.vision_tool.is_some(),
                "available": self.is_vision_available().await,
                "health": vision_health,
                "config": self.vision_proxy_config.as_ref()
            },
            "statistics": stats,
            "last_health_check": self.last_health_check.read()
                .map(|t| t.elapsed().as_secs()),
            "shutting_down": *self.shutting_down.read()
        }))
    }

    /// Initiate graceful shutdown
    pub async fn shutdown(&self) -> VisionResult<()> {
        info!("Initiating engine state manager shutdown");

        *self.shutting_down.write() = true;
        *self.health_status.write() = EngineHealthStatus::ShuttingDown;

        // Perform any cleanup operations here
        // Note: The actual engine cleanup would be handled by the engines themselves

        info!("Engine state manager shutdown completed");
        Ok(())
    }

    /// Check if the engine is shutting down
    pub fn is_shutting_down(&self) -> bool {
        *self.shutting_down.read()
    }

    /// Force a health status update (for external monitoring)
    pub fn set_health_status(&self, status: EngineHealthStatus) {
        *self.health_status.write() = status;
    }

    // Private helper methods

    async fn check_primary_engine_health(&self) -> bool {
        // For now, assume the primary engine is healthy if it exists
        // In a real implementation, this could check for memory usage,
        // response times, error rates, etc.
        true
    }

    async fn check_vision_tool_health(&self, tool: &Arc<LocalVisionModelTool>) -> VisionToolHealth {
        let start_time = Instant::now();

        let is_available = tool.is_available().await;
        let response_time = start_time.elapsed().as_millis() as u64;

        let health_check_result = tool.health_check().await;
        let model_info = health_check_result.ok();

        // Get current health to update counters
        let mut current_health =
            self.vision_tool_health
                .read()
                .clone()
                .unwrap_or_else(|| VisionToolHealth {
                    name: tool.name().to_string(),
                    is_available: false,
                    last_check: SystemTime::now(),
                    response_time_ms: None,
                    error_count: 0,
                    success_count: 0,
                    model_info: None,
                });

        // Update counters
        if is_available {
            current_health.success_count += 1;
        } else {
            current_health.error_count += 1;
        }

        VisionToolHealth {
            name: tool.name().to_string(),
            is_available,
            last_check: SystemTime::now(),
            response_time_ms: Some(response_time),
            error_count: current_health.error_count,
            success_count: current_health.success_count,
            model_info,
        }
    }

    async fn determine_overall_health_status(
        &self,
        primary_healthy: bool,
        vision_health: &Option<VisionToolHealth>,
    ) -> EngineHealthStatus {
        if *self.shutting_down.read() {
            return EngineHealthStatus::ShuttingDown;
        }

        if !primary_healthy {
            return EngineHealthStatus::Unhealthy;
        }

        // If vision is configured but not available, consider degraded
        if self.vision_tool.is_some() {
            match vision_health {
                Some(health) if !health.is_available => {
                    if health.error_count >= self.config.max_consecutive_failures {
                        EngineHealthStatus::Degraded
                    } else {
                        EngineHealthStatus::Healthy
                    }
                }
                Some(_) => EngineHealthStatus::Healthy,
                None => EngineHealthStatus::Initializing,
            }
        } else {
            EngineHealthStatus::Healthy
        }
    }
}

/// Builder for creating EngineStateManager instances
pub struct EngineStateManagerBuilder {
    primary_engine: Option<Arc<InferenceEngine>>,
    vision_tool: Option<Arc<LocalVisionModelTool>>,
    vision_proxy_config: Option<VisionProxyConfig>,
    config: EngineStateConfig,
}

impl EngineStateManagerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            primary_engine: None,
            vision_tool: None,
            vision_proxy_config: None,
            config: EngineStateConfig::default(),
        }
    }

    /// Set the primary inference engine
    pub fn with_primary_engine(mut self, engine: Arc<InferenceEngine>) -> Self {
        self.primary_engine = Some(engine);
        self
    }

    /// Set the vision tool
    pub fn with_vision_tool(mut self, tool: Arc<LocalVisionModelTool>) -> Self {
        self.vision_tool = Some(tool);
        self
    }

    /// Set the vision proxy configuration
    pub fn with_vision_proxy_config(mut self, config: VisionProxyConfig) -> Self {
        self.vision_proxy_config = Some(config);
        self
    }

    /// Set the state management configuration
    pub fn with_config(mut self, config: EngineStateConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the EngineStateManager
    pub fn build(self) -> VisionResult<EngineStateManager> {
        let primary_engine = self
            .primary_engine
            .ok_or_else(|| VisionError::InternalError {
                message: "Primary engine is required".to_string(),
            })?;

        Ok(EngineStateManager::new(
            primary_engine,
            self.vision_tool,
            self.vision_proxy_config,
            self.config,
        ))
    }
}

impl Default for EngineStateManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::EngineConfig;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_engine_state_manager_creation() {
        // Create a mock inference engine for testing
        let engine_config = EngineConfig::from_model_path(PathBuf::from("/fake/path"));
        // Note: This would fail in a real test due to model loading
        // but demonstrates the API structure

        let builder = EngineStateManagerBuilder::new().with_config(EngineStateConfig::default());

        // Would need a real engine for this to work
        // let state_manager = builder.build().unwrap();
        // assert_eq!(state_manager.get_health_status(), EngineHealthStatus::Healthy);
    }

    #[test]
    fn test_engine_stats_update() {
        let mut stats = EngineStats::default();
        assert_eq!(stats.requests_processed, 0);
        assert_eq!(stats.average_processing_time_ms, None);

        stats.update_request_stats(100, true);
        assert_eq!(stats.requests_processed, 1);
        assert_eq!(stats.average_processing_time_ms, Some(100));
        assert_eq!(stats.errors_encountered, 0);

        stats.update_request_stats(200, false);
        assert_eq!(stats.requests_processed, 2);
        assert_eq!(stats.average_processing_time_ms, Some(150));
        assert_eq!(stats.errors_encountered, 1);
    }

    #[test]
    fn test_engine_state_config_defaults() {
        let config = EngineStateConfig::default();
        assert_eq!(config.health_check_interval_secs, 60);
        assert_eq!(config.max_consecutive_failures, 3);
        assert_eq!(config.enable_auto_recovery, true);
        assert_eq!(config.health_check_timeout_ms, 5000);
        assert_eq!(config.include_detailed_stats, true);
    }
}
