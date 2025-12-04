use crate::config::ModelCapabilities;
use crate::engine_params::EngineParams;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Entry for a single model in the models configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Unique identifier for the model
    pub name: String,

    /// HuggingFace model identifier
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_id: Option<String>,

    /// Local filesystem path to model files
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_path: Option<String>,

    /// Specific weight file name (for GGUF or specific checkpoints)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_file: Option<String>,

    /// Engine parameters for this model
    #[serde(default)]
    pub params: EngineParams,

    /// Model capabilities including vision support
    #[serde(default)]
    pub capabilities: ModelCapabilities,

    /// Optional notes or description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

/// Complete models configuration file structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelsFile {
    /// List of model configurations
    #[serde(default)]
    pub models: Vec<ModelEntry>,

    /// Global idle timeout for model unloading (seconds)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_unload_secs: Option<u64>,

    /// Global configuration overrides
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_params: Option<EngineParams>,
}

impl ModelEntry {
    /// Create a new text-only model entry
    pub fn new(name: impl Into<String>, hf_id: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            hf_id: Some(hf_id.into()),
            local_path: None,
            weight_file: None,
            params: EngineParams::text_model_defaults(),
            capabilities: ModelCapabilities::text_only(),
            notes: None,
        }
    }

    /// Create a new vision model entry
    pub fn new_vision_model(
        name: impl Into<String>,
        hf_id: impl Into<String>,
        vision_hf_id: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            hf_id: Some(hf_id.into()),
            local_path: None,
            weight_file: None,
            params: EngineParams::text_model_defaults(),
            capabilities: ModelCapabilities::with_proxy_vision(vision_hf_id.into()),
            notes: None,
        }
    }

    /// Create a local model entry
    pub fn from_local_path(name: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            hf_id: None,
            local_path: Some(path.into()),
            weight_file: None,
            params: EngineParams::text_model_defaults(),
            capabilities: ModelCapabilities::text_only(),
            notes: None,
        }
    }

    /// Check if the model has a valid source (either HF ID or local path)
    pub fn has_source(&self) -> bool {
        self.hf_id.is_some() || self.local_path.is_some()
    }

    /// Get the model source identifier (HF ID or local path)
    pub fn get_source(&self) -> Option<&str> {
        self.hf_id.as_deref().or(self.local_path.as_deref())
    }

    /// Check if this model supports vision processing
    pub fn has_vision(&self) -> bool {
        self.capabilities.has_vision()
    }

    /// Get the vision model HF ID if configured
    pub fn get_vision_model_id(&self) -> Option<&str> {
        self.capabilities.get_vision_model_id()
    }

    /// Validate the model entry for consistency
    pub fn validate(&self) -> Result<(), String> {
        // Check that we have a source
        if !self.has_source() {
            return Err(format!("Model '{}' must have either hf_id or local_path", self.name));
        }

        // Validate both can't be set
        if self.hf_id.is_some() && self.local_path.is_some() {
            return Err(format!("Model '{}' cannot have both hf_id and local_path", self.name));
        }

        // Validate engine parameters
        if let Err(e) = self.params.validate() {
            return Err(format!("Model '{}' has invalid params: {}", self.name, e));
        }

        // Validate capabilities
        if let Err(e) = self.capabilities.validate() {
            return Err(format!("Model '{}' has invalid capabilities: {}", self.name, e));
        }

        Ok(())
    }

    /// Apply global parameter overrides
    pub fn apply_global_params(&mut self, global_params: &EngineParams) {
        self.params = self.params.clone().merge_with(global_params);
    }
}

impl ModelsFile {
    /// Load models configuration from YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        let mut models_file: ModelsFile = serde_yaml::from_str(&content)?;

        // Apply global parameters to all models
        if let Some(global_params) = &models_file.global_params {
            for model in &mut models_file.models {
                model.apply_global_params(global_params);
            }
        }

        // Validate all models
        models_file.validate().map_err(|e| anyhow::anyhow!(e))?;

        Ok(models_file)
    }

    /// Save models configuration to YAML file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let content = serde_yaml::to_string(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Validate the entire models file
    pub fn validate(&self) -> Result<(), String> {
        // Check for duplicate model names
        let mut seen_names = std::collections::HashSet::new();
        for model in &self.models {
            if !seen_names.insert(&model.name) {
                return Err(format!("Duplicate model name: '{}'", model.name));
            }
        }

        // Validate each model
        for model in &self.models {
            model.validate()?;
        }

        // Validate global params if present
        if let Some(global_params) = &self.global_params {
            global_params.validate()
                .map_err(|e| format!("Invalid global_params: {}", e))?;
        }

        Ok(())
    }

    /// Get a model by name
    pub fn get_model(&self, name: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.name == name)
    }

    /// Get all models that support vision
    pub fn get_vision_models(&self) -> Vec<&ModelEntry> {
        self.models.iter().filter(|m| m.has_vision()).collect()
    }

    /// Get all text-only models
    pub fn get_text_models(&self) -> Vec<&ModelEntry> {
        self.models.iter().filter(|m| !m.has_vision()).collect()
    }

    /// Create a model name to entry mapping
    pub fn to_hashmap(&self) -> HashMap<String, ModelEntry> {
        self.models.iter()
            .map(|model| (model.name.clone(), model.clone()))
            .collect()
    }

    /// Add a model to the configuration
    pub fn add_model(&mut self, model: ModelEntry) -> Result<(), String> {
        // Check for duplicate names
        if self.models.iter().any(|m| m.name == model.name) {
            return Err(format!("Model name '{}' already exists", model.name));
        }

        model.validate()?;
        self.models.push(model);
        Ok(())
    }

    /// Remove a model by name
    pub fn remove_model(&mut self, name: &str) -> bool {
        if let Some(pos) = self.models.iter().position(|m| m.name == name) {
            self.models.remove(pos);
            true
        } else {
            false
        }
    }

    /// Get idle timeout duration
    pub fn get_idle_timeout_secs(&self) -> u64 {
        self.idle_unload_secs.unwrap_or(300) // 5 minutes default
    }
}

impl Default for ModelEntry {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            hf_id: None,
            local_path: None,
            weight_file: None,
            params: EngineParams::default(),
            capabilities: ModelCapabilities::default(),
            notes: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_model_entry_creation() {
        let model = ModelEntry::new("test-model", "test/model-id");
        assert_eq!(model.name, "test-model");
        assert_eq!(model.hf_id, Some("test/model-id".to_string()));
        assert!(!model.has_vision());
        assert!(model.has_source());
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_vision_model_creation() {
        let model = ModelEntry::new_vision_model("vision-model", "text/model", "vision/model");
        assert_eq!(model.name, "vision-model");
        assert!(model.has_vision());
        assert_eq!(model.get_vision_model_id(), Some("vision/model"));
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_local_model_creation() {
        let model = ModelEntry::from_local_path("local-model", "/path/to/model");
        assert_eq!(model.name, "local-model");
        assert_eq!(model.local_path, Some("/path/to/model".to_string()));
        assert!(model.has_source());
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_model_validation() {
        // Invalid: no source
        let invalid_model = ModelEntry {
            name: "invalid".to_string(),
            hf_id: None,
            local_path: None,
            ..Default::default()
        };
        assert!(invalid_model.validate().is_err());

        // Invalid: both sources
        let invalid_model = ModelEntry {
            name: "invalid".to_string(),
            hf_id: Some("test".to_string()),
            local_path: Some("/path".to_string()),
            ..Default::default()
        };
        assert!(invalid_model.validate().is_err());
    }

    #[test]
    fn test_models_file_validation() {
        let mut models_file = ModelsFile::default();

        // Add duplicate names - should fail
        models_file.models.push(ModelEntry::new("test", "model1"));
        models_file.models.push(ModelEntry::new("test", "model2"));

        assert!(models_file.validate().is_err());
    }

    #[test]
    fn test_models_file_operations() {
        let mut models_file = ModelsFile::default();

        let model1 = ModelEntry::new("text-model", "text/model");
        let model2 = ModelEntry::new_vision_model("vision-model", "text/model", "vision/model");

        assert!(models_file.add_model(model1).is_ok());
        assert!(models_file.add_model(model2).is_ok());

        assert_eq!(models_file.get_text_models().len(), 1);
        assert_eq!(models_file.get_vision_models().len(), 1);

        assert!(models_file.remove_model("text-model"));
        assert_eq!(models_file.models.len(), 1);
    }

    #[test]
    fn test_yaml_serialization() {
        let mut models_file = ModelsFile::default();
        models_file.add_model(ModelEntry::new("test-model", "test/model")).unwrap();
        models_file.idle_unload_secs = Some(600);

        let yaml = serde_yaml::to_string(&models_file).unwrap();
        let deserialized: ModelsFile = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(models_file.models.len(), deserialized.models.len());
        assert_eq!(models_file.idle_unload_secs, deserialized.idle_unload_secs);
    }

    #[test]
    fn test_file_io() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let yaml_content = r#"
models:
  - name: "test-model"
    hf_id: "test/model-id"
    params:
      mem: 2048
      max_num_seqs: 8
    capabilities:
      vision_mode: "disabled"
idle_unload_secs: 600
"#;
        temp_file.write_all(yaml_content.as_bytes()).unwrap();

        let models_file = ModelsFile::load(temp_file.path()).unwrap();
        assert_eq!(models_file.models.len(), 1);
        assert_eq!(models_file.models[0].name, "test-model");
        assert_eq!(models_file.get_idle_timeout_secs(), 600);
    }
}