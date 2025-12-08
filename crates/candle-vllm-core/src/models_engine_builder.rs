use crate::api::{Error, InferenceEngine, Result};
use crate::engine_builder_ext::ExtendedEngineBuilder;
use crate::engine_params::EngineParams;
use crate::engine_state::{EngineStateConfig, EngineStateManager, EngineStateManagerBuilder};
use crate::models_config::{ModelEntry, ModelsFile};
use crate::openai::image_tool::ImageDescriptionConfig;
use crate::openai::local_vision_tool::LocalVisionModelTool;
use crate::openai::vision_proxy::VisionProxyConfig;
use crate::vision::VisionError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Result of building engines from a models file
pub struct ModelsEngineBuilderResult {
    /// Primary text model engine
    pub primary_engine: Arc<InferenceEngine>,
    /// Optional vision model tool for multimodal processing
    pub vision_tool: Option<Arc<LocalVisionModelTool>>,
    /// Engine state manager for monitoring and lifecycle management
    pub state_manager: EngineStateManager,
    /// Configuration that was used to build the engines
    pub config_used: ModelEntry,
}

/// Configuration for the models engine builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsEngineBuilderConfig {
    /// Whether to enable automatic vision model loading for compatible models
    pub enable_vision_auto_load: bool,
    /// Default vision configuration to use when not specified in model entry
    pub default_vision_config: Option<ImageDescriptionConfig>,
    /// Default vision proxy configuration
    pub default_vision_proxy_config: Option<VisionProxyConfig>,
    /// Engine state management configuration
    pub state_management_config: EngineStateConfig,
    /// Fallback engine parameters for models without explicit configuration
    pub fallback_params: EngineParams,
}

impl Default for ModelsEngineBuilderConfig {
    fn default() -> Self {
        Self {
            enable_vision_auto_load: true,
            default_vision_config: Some(ImageDescriptionConfig::default()),
            default_vision_proxy_config: Some(VisionProxyConfig::default()),
            state_management_config: EngineStateConfig::default(),
            fallback_params: EngineParams::default(),
        }
    }
}

/// Builder for creating engines from models configuration files
pub struct ModelsEngineBuilder {
    config: ModelsEngineBuilderConfig,
}

impl ModelsEngineBuilder {
    /// Create a new models engine builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ModelsEngineBuilderConfig::default(),
        }
    }

    /// Create a new models engine builder with custom configuration
    pub fn with_config(config: ModelsEngineBuilderConfig) -> Self {
        Self { config }
    }

    /// Build engines from a models configuration file
    pub async fn build_from_file<P: AsRef<Path>>(
        &self,
        models_file_path: P,
        model_name: &str,
    ) -> Result<ModelsEngineBuilderResult> {
        info!(
            "Loading models configuration from: {}",
            models_file_path.as_ref().display()
        );

        let models_file = ModelsFile::load(models_file_path.as_ref())
            .map_err(|e| Error::Config(format!("Failed to load models file: {}", e)))?;

        self.build_from_models_file(models_file, model_name).await
    }

    /// Build engines from a parsed models configuration
    pub async fn build_from_models_file(
        &self,
        models_file: ModelsFile,
        model_name: &str,
    ) -> Result<ModelsEngineBuilderResult> {
        info!("Building engine for model: {}", model_name);

        // Find the requested model in the configuration
        let model_entry = models_file
            .models
            .iter()
            .find(|m| m.name == model_name)
            .ok_or_else(|| {
                Error::Config(format!("Model '{}' not found in configuration", model_name))
            })?;

        self.build_from_model_entry(model_entry.clone()).await
    }

    /// Build engines from a single model entry
    pub async fn build_from_model_entry(
        &self,
        model_entry: ModelEntry,
    ) -> Result<ModelsEngineBuilderResult> {
        info!("Building engine for model entry: {}", model_entry.name);

        // Validate and resolve model path
        let primary_model_path = self.resolve_model_path(&model_entry)?;
        debug!(
            "Resolved primary model path: {}",
            primary_model_path.display()
        );

        // Determine if we should load a vision model
        let (vision_model_path, vision_params, vision_config) =
            if self.should_load_vision_model(&model_entry) {
                self.prepare_vision_model_config(&model_entry).await?
            } else {
                debug!(
                    "Vision model not configured for model: {}",
                    model_entry.name
                );
                (None, None, None)
            };

        // Use the extended engine builder for concurrent loading
        let primary_result = ExtendedEngineBuilder::build_inference_engine_from_params_async(
            primary_model_path,
            model_entry.params.clone(),
            vision_model_path,
            vision_params,
            vision_config.clone(),
        )
        .await?;

        info!(
            "Successfully built engines for '{}'. Primary: ✓, Vision: {}",
            model_entry.name,
            if primary_result.vision_tool.is_some() {
                "✓"
            } else {
                "✗"
            }
        );

        // Create state manager
        let primary_engine_arc = Arc::new(primary_result.primary_engine);
        let vision_tool_arc = primary_result.vision_tool.clone();
        let state_manager = self.create_state_manager(
            primary_engine_arc.clone(),
            vision_tool_arc.clone(),
            vision_config,
        )?;

        Ok(ModelsEngineBuilderResult {
            primary_engine: primary_engine_arc,
            vision_tool: vision_tool_arc,
            state_manager,
            config_used: model_entry,
        })
    }

    /// List all available models in a configuration file
    pub fn list_models<P: AsRef<Path>>(models_file_path: P) -> Result<Vec<String>> {
        let models_file = ModelsFile::load(models_file_path.as_ref())
            .map_err(|e| Error::Config(format!("Failed to load models file: {}", e)))?;

        Ok(models_file.models.iter().map(|m| m.name.clone()).collect())
    }

    /// Get detailed information about a model from the configuration file
    pub fn get_model_info<P: AsRef<Path>>(
        models_file_path: P,
        model_name: &str,
    ) -> Result<ModelEntry> {
        let models_file = ModelsFile::load(models_file_path.as_ref())
            .map_err(|e| Error::Config(format!("Failed to load models file: {}", e)))?;

        models_file
            .models
            .into_iter()
            .find(|m| m.name == model_name)
            .ok_or_else(|| Error::Config(format!("Model '{}' not found", model_name)))
    }

    /// Validate a models configuration file
    pub fn validate_models_file<P: AsRef<Path>>(models_file_path: P) -> Result<Vec<String>> {
        let models_file = ModelsFile::load(models_file_path.as_ref())
            .map_err(|e| Error::Config(format!("Failed to load models file: {}", e)))?;

        let mut issues = Vec::new();

        for model in &models_file.models {
            // Check model path resolution
            match Self::resolve_model_path_static(model) {
                Ok(_) => {}
                Err(e) => issues.push(format!("Model '{}': {}", model.name, e)),
            }

            // Check parameter validation
            if let Err(e) = model.params.validate() {
                issues.push(format!("Model '{}' params: {}", model.name, e));
            }

            // Check for duplicate names
            let duplicate_count = models_file
                .models
                .iter()
                .filter(|m| m.name == model.name)
                .count();
            if duplicate_count > 1 {
                issues.push(format!("Duplicate model name: '{}'", model.name));
            }
        }

        Ok(issues)
    }

    // Private helper methods

    fn resolve_model_path(&self, model_entry: &ModelEntry) -> Result<PathBuf> {
        Self::resolve_model_path_static(model_entry)
    }

    fn resolve_model_path_static(model_entry: &ModelEntry) -> Result<PathBuf> {
        if let Some(ref local_path) = model_entry.local_path {
            let path = PathBuf::from(local_path);
            if path.exists() {
                Ok(path)
            } else {
                Err(Error::Config(format!(
                    "Local model path does not exist: {}",
                    path.display()
                )))
            }
        } else if let Some(ref hf_id) = model_entry.hf_id {
            // For HuggingFace models, we'd need to implement download logic
            // For now, return an error indicating this is not yet supported
            Err(Error::Config(format!(
                "HuggingFace model downloading not yet implemented: {}",
                hf_id
            )))
        } else {
            Err(Error::Config(
                "Model entry must have either local_path or hf_id".to_string(),
            ))
        }
    }

    fn should_load_vision_model(&self, model_entry: &ModelEntry) -> bool {
        if !self.config.enable_vision_auto_load {
            return false;
        }

        // Check if the model has vision capabilities enabled
        model_entry.capabilities.has_vision()
            && model_entry.capabilities.get_vision_model_id().is_some()
    }

    async fn prepare_vision_model_config(
        &self,
        model_entry: &ModelEntry,
    ) -> Result<(
        Option<PathBuf>,
        Option<EngineParams>,
        Option<ImageDescriptionConfig>,
    )> {
        let vision_model_path = model_entry
            .capabilities
            .get_vision_model_id()
            .map(|p| PathBuf::from(p));

        let vision_model_path = match vision_model_path {
            Some(path) => {
                if path.exists() {
                    Some(path)
                } else {
                    warn!(
                        "Vision model path does not exist: {}, skipping vision support",
                        path.display()
                    );
                    return Ok((None, None, None));
                }
            }
            None => return Ok((None, None, None)),
        };

        // Use vision-optimized engine parameters
        let vision_params = EngineParams::vision_model_defaults();

        // Use configured or default vision configuration
        let vision_config = self
            .config
            .default_vision_config
            .clone()
            .unwrap_or_default();

        Ok((vision_model_path, Some(vision_params), Some(vision_config)))
    }

    fn create_state_manager(
        &self,
        primary_engine: Arc<InferenceEngine>,
        vision_tool: Option<Arc<LocalVisionModelTool>>,
        vision_config: Option<ImageDescriptionConfig>,
    ) -> Result<EngineStateManager> {
        let mut builder = EngineStateManagerBuilder::new()
            .with_primary_engine(primary_engine)
            .with_config(self.config.state_management_config.clone());

        if let Some(vision_tool) = vision_tool {
            builder = builder.with_vision_tool(vision_tool);
        }

        if let Some(_config) = vision_config {
            if let Some(ref proxy_config) = self.config.default_vision_proxy_config {
                builder = builder.with_vision_proxy_config(proxy_config.clone());
            }
        }

        builder.build().map_err(|e| match e {
            VisionError::InternalError { message } => Error::Other(message),
            VisionError::ModelNotFound { model_path } => {
                Error::ModelLoad(format!("Vision model not found: {}", model_path))
            }
            VisionError::InvalidImageData { message } => Error::Config(message),
            VisionError::ModelError { message } => Error::Generation(message),
            _ => Error::Other(format!("Vision error: {}", e)),
        })
    }
}

impl Default for ModelsEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory for creating pre-configured models engine builders
pub struct ModelsEngineBuilderFactory;

impl ModelsEngineBuilderFactory {
    /// Create a builder optimized for production use
    pub fn production() -> ModelsEngineBuilder {
        let config = ModelsEngineBuilderConfig {
            enable_vision_auto_load: true,
            state_management_config: EngineStateConfig {
                health_check_interval_secs: 30,
                max_consecutive_failures: 5,
                enable_auto_recovery: true,
                health_check_timeout_ms: 3000,
                include_detailed_stats: false,
            },
            ..Default::default()
        };
        ModelsEngineBuilder::with_config(config)
    }

    /// Create a builder optimized for development use
    pub fn development() -> ModelsEngineBuilder {
        let config = ModelsEngineBuilderConfig {
            enable_vision_auto_load: true,
            state_management_config: EngineStateConfig {
                health_check_interval_secs: 10,
                max_consecutive_failures: 2,
                enable_auto_recovery: true,
                health_check_timeout_ms: 1000,
                include_detailed_stats: true,
            },
            ..Default::default()
        };
        ModelsEngineBuilder::with_config(config)
    }

    /// Create a builder with vision support disabled
    pub fn text_only() -> ModelsEngineBuilder {
        let config = ModelsEngineBuilderConfig {
            enable_vision_auto_load: false,
            default_vision_config: None,
            default_vision_proxy_config: None,
            ..Default::default()
        };
        ModelsEngineBuilder::with_config(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelCapabilities;
    use std::collections::HashMap;

    fn create_test_model_entry() -> ModelEntry {
        ModelEntry {
            name: "test-model".to_string(),
            hf_id: None,
            local_path: Some("/fake/path/to/model".to_string()),
            weight_file: None,
            params: EngineParams::default(),
            capabilities: ModelCapabilities::default(),
            notes: Some("Test model for unit tests".to_string()),
        }
    }

    #[test]
    fn test_models_engine_builder_creation() {
        let builder = ModelsEngineBuilder::new();
        assert!(builder.config.enable_vision_auto_load);

        let custom_config = ModelsEngineBuilderConfig {
            enable_vision_auto_load: false,
            ..Default::default()
        };
        let custom_builder = ModelsEngineBuilder::with_config(custom_config);
        assert!(!custom_builder.config.enable_vision_auto_load);
    }

    #[test]
    fn test_should_load_vision_model() {
        use crate::config::{VisionMode, VisionProxyConfig};

        let builder = ModelsEngineBuilder::new();

        // Model without vision support
        let model_no_vision = create_test_model_entry();
        assert!(!builder.should_load_vision_model(&model_no_vision));

        // Model with vision support (via proxy)
        let mut model_with_vision = create_test_model_entry();
        model_with_vision.capabilities.vision_mode = VisionMode::Proxy;
        model_with_vision.capabilities.vision_proxy = Some(VisionProxyConfig {
            hf_id: "llava-hf/llava-1.5-7b-hf".to_string(),
            prompt_template: None,
        });
        assert!(builder.should_load_vision_model(&model_with_vision));

        // Builder with vision auto-load disabled
        let no_vision_builder = ModelsEngineBuilderFactory::text_only();
        assert!(!no_vision_builder.should_load_vision_model(&model_with_vision));
    }

    #[test]
    fn test_factory_configurations() {
        let prod_builder = ModelsEngineBuilderFactory::production();
        assert_eq!(
            prod_builder
                .config
                .state_management_config
                .health_check_interval_secs,
            30
        );
        assert!(
            !prod_builder
                .config
                .state_management_config
                .include_detailed_stats
        );

        let dev_builder = ModelsEngineBuilderFactory::development();
        assert_eq!(
            dev_builder
                .config
                .state_management_config
                .health_check_interval_secs,
            10
        );
        assert!(
            dev_builder
                .config
                .state_management_config
                .include_detailed_stats
        );

        let text_only_builder = ModelsEngineBuilderFactory::text_only();
        assert!(!text_only_builder.config.enable_vision_auto_load);
    }

    #[test]
    fn test_resolve_model_path_static() {
        // Test with missing local_path and hf_id
        let mut invalid_model = create_test_model_entry();
        invalid_model.local_path = None;
        invalid_model.hf_id = None;
        assert!(ModelsEngineBuilder::resolve_model_path_static(&invalid_model).is_err());

        // Test with HuggingFace ID (should fail as not implemented)
        let mut hf_model = create_test_model_entry();
        hf_model.local_path = None;
        hf_model.hf_id = Some("microsoft/DialoGPT-medium".to_string());
        assert!(ModelsEngineBuilder::resolve_model_path_static(&hf_model).is_err());

        // Test with non-existent local path
        let non_existent_model = create_test_model_entry();
        assert!(ModelsEngineBuilder::resolve_model_path_static(&non_existent_model).is_err());
    }
}
