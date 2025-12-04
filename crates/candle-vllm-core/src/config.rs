use serde::{Deserialize, Serialize};

/// Vision processing mode for models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VisionMode {
    /// Vision processing is disabled
    Disabled,
    /// Use a separate vision model to generate captions (proxy approach)
    Proxy,
    /// Native multimodal model (future extension)
    Native,
}

impl Default for VisionMode {
    fn default() -> Self {
        VisionMode::Disabled
    }
}

/// Configuration for proxy vision model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionProxyConfig {
    /// HuggingFace model ID for the vision model
    pub hf_id: String,
    /// Optional prompt template for vision model captioning
    /// If not provided, uses a default template
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,
}

/// Model capabilities configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    /// Vision processing mode
    #[serde(default)]
    pub vision_mode: VisionMode,
    /// Proxy vision configuration (required when vision_mode is Proxy)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_proxy: Option<VisionProxyConfig>,
    /// Image token for native VLMs (reserved for future use)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_token: Option<String>,
}

impl ModelCapabilities {
    /// Create capabilities with vision disabled
    pub fn text_only() -> Self {
        Self::default()
    }

    /// Create capabilities with proxy vision
    pub fn with_proxy_vision(hf_id: impl Into<String>) -> Self {
        Self {
            vision_mode: VisionMode::Proxy,
            vision_proxy: Some(VisionProxyConfig {
                hf_id: hf_id.into(),
                prompt_template: None,
            }),
            image_token: None,
        }
    }

    /// Check if vision processing is enabled
    pub fn has_vision(&self) -> bool {
        matches!(self.vision_mode, VisionMode::Proxy | VisionMode::Native)
    }

    /// Get the vision model HF ID if proxy mode is enabled
    pub fn get_vision_model_id(&self) -> Option<&str> {
        match &self.vision_proxy {
            Some(config) => Some(&config.hf_id),
            None => None,
        }
    }

    /// Get the prompt template for vision model
    pub fn get_vision_prompt_template(&self) -> &str {
        match &self.vision_proxy {
            Some(config) => config.prompt_template.as_deref()
                .unwrap_or("Describe this image in detail:"),
            None => "Describe this image in detail:",
        }
    }
}

/// Validate that the vision configuration is consistent
impl ModelCapabilities {
    pub fn validate(&self) -> Result<(), String> {
        match self.vision_mode {
            VisionMode::Proxy => {
                if self.vision_proxy.is_none() {
                    return Err("Proxy vision mode requires vision_proxy configuration".to_string());
                }
            }
            VisionMode::Disabled => {
                if self.vision_proxy.is_some() {
                    return Err("Disabled vision mode should not have vision_proxy configuration".to_string());
                }
            }
            VisionMode::Native => {
                // Future validation for native VLMs
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_capabilities() {
        let caps = ModelCapabilities::default();
        assert_eq!(caps.vision_mode, VisionMode::Disabled);
        assert!(!caps.has_vision());
        assert!(caps.get_vision_model_id().is_none());
    }

    #[test]
    fn test_proxy_vision_capabilities() {
        let caps = ModelCapabilities::with_proxy_vision("Qwen/Qwen2-VL-7B-Instruct");
        assert_eq!(caps.vision_mode, VisionMode::Proxy);
        assert!(caps.has_vision());
        assert_eq!(caps.get_vision_model_id(), Some("Qwen/Qwen2-VL-7B-Instruct"));
        assert_eq!(caps.get_vision_prompt_template(), "Describe this image in detail:");
    }

    #[test]
    fn test_vision_config_validation() {
        // Valid proxy config
        let caps = ModelCapabilities::with_proxy_vision("test-model");
        assert!(caps.validate().is_ok());

        // Invalid proxy config (missing vision_proxy)
        let invalid_caps = ModelCapabilities {
            vision_mode: VisionMode::Proxy,
            vision_proxy: None,
            image_token: None,
        };
        assert!(invalid_caps.validate().is_err());

        // Invalid disabled config (has vision_proxy)
        let invalid_caps = ModelCapabilities {
            vision_mode: VisionMode::Disabled,
            vision_proxy: Some(VisionProxyConfig {
                hf_id: "test".to_string(),
                prompt_template: None,
            }),
            image_token: None,
        };
        assert!(invalid_caps.validate().is_err());
    }

    #[test]
    fn test_serde_roundtrip() {
        let caps = ModelCapabilities::with_proxy_vision("Qwen/Qwen2-VL-7B-Instruct");
        let serialized = serde_yaml::to_string(&caps).unwrap();
        let deserialized: ModelCapabilities = serde_yaml::from_str(&serialized).unwrap();

        assert_eq!(caps.vision_mode, deserialized.vision_mode);
        assert_eq!(caps.get_vision_model_id(), deserialized.get_vision_model_id());
    }
}