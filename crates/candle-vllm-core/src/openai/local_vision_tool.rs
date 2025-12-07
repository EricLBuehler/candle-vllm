use crate::engine_params::EngineParams;
use crate::openai::image_tool::{
    ImageDescription, ImageDescriptionConfig, ImageDescriptionTool, ImageMetadata,
};
use crate::openai::requests::ImageUrl;
use crate::vision::{VisionError, VisionResult};
use async_trait::async_trait;
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::Mutex;
use tokio::time::timeout;
use tracing::{debug, info};

/// Configuration specific to local vision models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalVisionConfig {
    /// Path to the vision model weights
    pub model_path: String,
    /// Device to run the vision model on
    pub device: String,
    /// Model precision (bf16, fp16, fp32)
    pub dtype: String,
    /// Maximum sequence length for the vision model
    pub max_seq_len: Option<usize>,
    /// Vision model specific parameters
    #[serde(default)]
    pub model_params: HashMap<String, serde_json::Value>,
}

impl Default for LocalVisionConfig {
    fn default() -> Self {
        Self {
            model_path: "/path/to/qwen2-vl-7b".to_string(),
            device: "cuda:0".to_string(),
            dtype: "bf16".to_string(),
            max_seq_len: Some(8192),
            model_params: HashMap::new(),
        }
    }
}

/// State of the local vision model
#[derive(Debug, Clone)]
pub enum ModelState {
    /// Model not initialized yet
    Uninitialized,
    /// Model is loading
    Loading,
    /// Model is ready for inference
    Ready,
    /// Model encountered an error
    Error(String),
}

/// Internal vision model wrapper (placeholder for actual vision model implementation)
pub struct VisionModelWrapper {
    // This would contain the actual vision model components
    // For now, we'll use placeholder fields
    pub model_path: String,
    pub device: Device,
    pub state: ModelState,
}

impl VisionModelWrapper {
    pub fn new(config: &LocalVisionConfig) -> VisionResult<Self> {
        // Parse device specification
        let device = Self::parse_device(&config.device)?;

        Ok(Self {
            model_path: config.model_path.clone(),
            device,
            state: ModelState::Uninitialized,
        })
    }

    fn parse_device(device_str: &str) -> VisionResult<Device> {
        match device_str {
            "cpu" => Ok(Device::Cpu),
            s if s.starts_with("cuda:") => {
                let device_id: usize = s
                    .strip_prefix("cuda:")
                    .ok_or_else(|| VisionError::InternalError {
                        message: "Invalid CUDA device format".to_string(),
                    })?
                    .parse()
                    .map_err(|_| VisionError::InternalError {
                        message: "Invalid CUDA device ID".to_string(),
                    })?;
                Device::new_cuda(device_id).map_err(|e| VisionError::InternalError {
                    message: format!("Failed to create CUDA device: {}", e),
                })
            }
            s if s.starts_with("metal") => {
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0).map_err(|e| VisionError::InternalError {
                        message: format!("Failed to create Metal device: {}", e),
                    })
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(VisionError::InternalError {
                        message: "Metal support not compiled in".to_string(),
                    })
                }
            }
            _ => Err(VisionError::InternalError {
                message: format!("Unsupported device: {}", device_str),
            }),
        }
    }

    pub async fn load_model(&mut self) -> VisionResult<()> {
        self.state = ModelState::Loading;
        info!("Loading vision model from: {}", self.model_path);

        // TODO: Implement actual model loading
        // This would involve:
        // 1. Loading tokenizer from model path
        // 2. Loading model weights
        // 3. Initializing the vision encoder and text decoder
        // 4. Setting up the model for inference

        // For now, simulate loading time
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check if model path exists
        if !std::path::Path::new(&self.model_path).exists() {
            let error_msg = format!("Vision model path does not exist: {}", self.model_path);
            self.state = ModelState::Error(error_msg.clone());
            return Err(VisionError::ModelNotFound {
                model_path: self.model_path.clone(),
            });
        }

        self.state = ModelState::Ready;
        info!("Vision model loaded successfully");
        Ok(())
    }

    pub async fn describe_image(&self, image_data: &[u8], prompt: &str) -> VisionResult<String> {
        match self.state {
            ModelState::Ready => {
                debug!("Processing image with vision model, prompt: {}", prompt);

                // TODO: Implement actual image description
                // This would involve:
                // 1. Preprocessing the image (resize, normalize, etc.)
                // 2. Encoding the image with the vision encoder
                // 3. Generating text description using the language model
                // 4. Postprocessing the output

                // For now, return a placeholder description
                tokio::time::sleep(Duration::from_millis(50)).await;

                // Simulate different responses based on image size for testing
                let description = if image_data.len() > 1000000 {
                    "A high-resolution image showing detailed visual content with multiple objects and complex composition."
                } else if image_data.len() > 100000 {
                    "A medium-resolution image containing several objects with clear visual details."
                } else {
                    "A small image with basic visual elements and simple composition."
                };

                Ok(format!("{} {}", prompt, description))
            }
            ModelState::Uninitialized => Err(VisionError::ModelNotReady {
                message: "Vision model not initialized".to_string(),
            }),
            ModelState::Loading => Err(VisionError::ModelNotReady {
                message: "Vision model still loading".to_string(),
            }),
            ModelState::Error(ref e) => Err(VisionError::ModelError { message: e.clone() }),
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.state, ModelState::Ready)
    }

    pub fn get_health_info(&self) -> HashMap<String, serde_json::Value> {
        let mut info = HashMap::new();

        info.insert(
            "model_path".to_string(),
            serde_json::Value::String(self.model_path.clone()),
        );
        info.insert(
            "device".to_string(),
            serde_json::Value::String(match &self.device {
                Device::Cpu => "cpu".to_string(),
                Device::Cuda(_) => "cuda".to_string(),
                #[cfg(feature = "metal")]
                Device::Metal(_) => "metal".to_string(),
                #[cfg(not(feature = "metal"))]
                _ => "unknown".to_string(),
            }),
        );
        info.insert(
            "state".to_string(),
            serde_json::Value::String(match &self.state {
                ModelState::Uninitialized => "uninitialized".to_string(),
                ModelState::Loading => "loading".to_string(),
                ModelState::Ready => "ready".to_string(),
                ModelState::Error(e) => format!("error: {}", e),
            }),
        );

        info
    }
}

/// Local vision model tool implementation
pub struct LocalVisionModelTool {
    config: ImageDescriptionConfig,
    local_config: LocalVisionConfig,
    model: Arc<Mutex<VisionModelWrapper>>,
    name: String,
}

impl LocalVisionModelTool {
    /// Create a new local vision model tool
    pub async fn new(
        config: ImageDescriptionConfig,
        local_config: LocalVisionConfig,
    ) -> VisionResult<Self> {
        let model_wrapper = VisionModelWrapper::new(&local_config)?;
        let model = Arc::new(Mutex::new(model_wrapper));

        let tool = Self {
            config,
            local_config,
            model,
            name: "local_vision_model".to_string(),
        };

        // Initialize the model
        tool.initialize().await?;

        Ok(tool)
    }

    /// Create from engine parameters and model path
    pub async fn from_engine_params(
        engine_params: &EngineParams,
        model_path: String,
        base_config: ImageDescriptionConfig,
    ) -> VisionResult<Self> {
        let local_config = LocalVisionConfig {
            model_path,
            device: engine_params
                .device_ids
                .as_ref()
                .and_then(|ids| ids.first())
                .map(|&id| format!("cuda:{}", id))
                .unwrap_or_else(|| "cpu".to_string()),
            dtype: engine_params
                .dtype
                .clone()
                .unwrap_or_else(|| "bf16".to_string()),
            max_seq_len: Some(8192), // Use a reasonable default
            model_params: HashMap::new(),
        };

        Self::new(base_config, local_config).await
    }

    async fn initialize(&self) -> VisionResult<()> {
        let mut model = self.model.lock().await;
        model.load_model().await
    }

    async fn download_image(&self, url: &str) -> VisionResult<Vec<u8>> {
        if url.starts_with("data:") {
            // Handle data URLs
            self.decode_data_url(url)
        } else {
            // Handle web URLs
            self.download_web_image(url).await
        }
    }

    fn decode_data_url(&self, data_url: &str) -> VisionResult<Vec<u8>> {
        // Parse data URL format: data:image/jpeg;base64,<data>
        let parts: Vec<&str> = data_url.splitn(2, ',').collect();
        if parts.len() != 2 {
            return Err(VisionError::InvalidImageData {
                message: "Invalid data URL format".to_string(),
            });
        }

        let header = parts[0];
        let data = parts[1];

        // Check if it's base64 encoded
        if !header.contains("base64") {
            return Err(VisionError::InvalidImageData {
                message: "Only base64 data URLs are supported".to_string(),
            });
        }

        // Decode base64
        use base64::{engine::general_purpose, Engine as _};
        general_purpose::STANDARD
            .decode(data)
            .map_err(|e| VisionError::InvalidImageData {
                message: format!("Failed to decode base64 data: {}", e),
            })
    }

    async fn download_web_image(&self, url: &str) -> VisionResult<Vec<u8>> {
        let timeout_duration = Duration::from_secs(self.config.timeout_secs);

        let response = timeout(timeout_duration, reqwest::get(url))
            .await
            .map_err(|_| VisionError::NetworkTimeout {
                url: url.to_string(),
                timeout_secs: self.config.timeout_secs,
            })?
            .map_err(|e| VisionError::NetworkError {
                message: format!("Failed to fetch {}: {}", url, e),
            })?;

        if !response.status().is_success() {
            return Err(VisionError::NetworkError {
                message: format!(
                    "HTTP {} from {}: {}",
                    response.status(),
                    url,
                    response.status().canonical_reason().unwrap_or("Unknown")
                ),
            });
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| VisionError::NetworkError {
                message: format!("Failed to read response body from {}: {}", url, e),
            })?;

        Ok(bytes.to_vec())
    }

    fn extract_image_metadata(&self, image_data: &[u8]) -> ImageMetadata {
        // TODO: Implement actual image metadata extraction
        // This would use an image processing library to get dimensions, format, etc.

        ImageMetadata {
            width: Some(1024),                // Placeholder
            height: Some(1024),               // Placeholder
            format: Some("jpeg".to_string()), // Placeholder
            size_bytes: Some(image_data.len() as u64),
            was_resized: false,
            original_dimensions: None,
        }
    }
}

#[async_trait]
impl ImageDescriptionTool for LocalVisionModelTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &ImageDescriptionConfig {
        &self.config
    }

    async fn is_available(&self) -> bool {
        let model = self.model.lock().await;
        model.is_ready()
    }

    async fn health_check(&self) -> VisionResult<HashMap<String, serde_json::Value>> {
        let model = self.model.lock().await;

        let mut health_info = model.get_health_info();
        health_info.insert(
            "tool_name".to_string(),
            serde_json::Value::String(self.name.clone()),
        );
        health_info.insert(
            "config".to_string(),
            serde_json::to_value(&self.config).unwrap_or(serde_json::Value::Null),
        );
        health_info.insert(
            "local_config".to_string(),
            serde_json::to_value(&self.local_config).unwrap_or(serde_json::Value::Null),
        );

        Ok(health_info)
    }

    async fn describe_image(
        &self,
        image_url: &ImageUrl,
        custom_prompt: Option<&str>,
    ) -> VisionResult<ImageDescription> {
        let start_time = Instant::now();

        // Validate the image first
        self.validate_image(image_url)?;

        // Download the image
        let image_data = self.download_image(&image_url.url).await?;

        // Check image size constraints if configured
        if let Some((max_width, max_height)) = self.config.max_image_size {
            // For now, just check file size as a proxy
            let max_size_bytes = (max_width * max_height * 3) as u64; // Rough estimate
            if image_data.len() as u64 > max_size_bytes {
                return Err(VisionError::InvalidImageData {
                    message: format!(
                        "Image too large: {} bytes, max allowed: {} bytes",
                        image_data.len(),
                        max_size_bytes
                    ),
                });
            }
        }

        // Use custom prompt or default
        let prompt = custom_prompt
            .or(self.config.prompt_template.as_deref())
            .unwrap_or("Describe this image in detail.");

        // Generate description using the vision model
        let model = self.model.lock().await;
        let description_text = model.describe_image(&image_data, prompt).await?;
        drop(model); // Release the lock
        let processing_time = start_time.elapsed().as_millis() as u64;

        // Extract metadata if requested
        let metadata = if self.config.include_metadata {
            self.extract_image_metadata(&image_data)
        } else {
            ImageMetadata::default()
        };

        // Create the description result
        let mut description = ImageDescription::new(description_text)
            .with_processing_time(processing_time)
            .with_metadata(metadata);

        // Add model-specific data
        description = description.with_model_data(
            "model_path",
            serde_json::Value::String(self.local_config.model_path.clone()),
        );
        description = description.with_model_data(
            "device",
            serde_json::Value::String(self.local_config.device.clone()),
        );
        description = description.with_model_data(
            "dtype",
            serde_json::Value::String(self.local_config.dtype.clone()),
        );

        Ok(description)
    }

    async fn describe_images(
        &self,
        image_urls: &[ImageUrl],
        custom_prompt: Option<&str>,
    ) -> VisionResult<Vec<ImageDescription>> {
        // Process images sequentially for now
        // TODO: Consider parallel processing with appropriate concurrency limits
        let mut results = Vec::new();

        for image_url in image_urls {
            let description = self.describe_image(image_url, custom_prompt).await?;
            results.push(description);
        }

        Ok(results)
    }

    fn supports_format(&self, format: &str) -> bool {
        // Local vision models typically support common formats
        matches!(
            format.to_lowercase().as_str(),
            "jpeg" | "jpg" | "png" | "webp" | "bmp" | "tiff" | "gif"
        )
    }

    async fn estimate_processing_time(&self, image_url: &ImageUrl) -> u64 {
        // Estimate based on image type and size
        if image_url.is_data_url() {
            // Data URLs are typically smaller, faster to process
            self.config.timeout_secs / 2
        } else {
            // Web URLs need download time plus processing
            self.config.timeout_secs
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_vision_config_default() {
        let config = LocalVisionConfig::default();
        assert_eq!(config.model_path, "/path/to/qwen2-vl-7b");
        assert_eq!(config.device, "cuda:0");
        assert_eq!(config.dtype, "bf16");
        assert_eq!(config.max_seq_len, Some(8192));
    }

    #[test]
    fn test_vision_model_wrapper_parse_device() {
        // Test CPU device
        let device = VisionModelWrapper::parse_device("cpu").unwrap();
        assert!(matches!(device, Device::Cpu));

        // Test CUDA device parsing - only if CUDA is available
        #[cfg(feature = "cuda")]
        {
            let device_result = VisionModelWrapper::parse_device("cuda:0");
            assert!(device_result.is_ok());
        }

        // Without CUDA feature, parsing cuda device should fail
        #[cfg(not(feature = "cuda"))]
        {
            let device_result = VisionModelWrapper::parse_device("cuda:0");
            assert!(device_result.is_err());
        }

        // Test invalid device
        let device_result = VisionModelWrapper::parse_device("invalid");
        assert!(device_result.is_err());
    }

    #[tokio::test]
    async fn test_vision_model_wrapper_initialization() {
        let config = LocalVisionConfig {
            model_path: "/tmp/test_model".to_string(),
            device: "cpu".to_string(),
            dtype: "fp32".to_string(),
            max_seq_len: Some(4096),
            model_params: HashMap::new(),
        };

        let wrapper_result = VisionModelWrapper::new(&config);
        assert!(wrapper_result.is_ok());

        let wrapper = wrapper_result.unwrap();
        assert!(!wrapper.is_ready());
        assert_eq!(wrapper.model_path, "/tmp/test_model");
    }

    #[test]
    fn test_model_state_transitions() {
        let state = ModelState::Uninitialized;
        assert!(matches!(state, ModelState::Uninitialized));

        let error_state = ModelState::Error("Test error".to_string());
        if let ModelState::Error(msg) = error_state {
            assert_eq!(msg, "Test error");
        } else {
            panic!("Expected error state");
        }
    }

    #[tokio::test]
    async fn test_local_vision_config_serialization() {
        let config = LocalVisionConfig {
            model_path: "/test/model".to_string(),
            device: "cuda:1".to_string(),
            dtype: "fp16".to_string(),
            max_seq_len: Some(2048),
            model_params: {
                let mut params = HashMap::new();
                params.insert("temperature".to_string(), serde_json::json!(0.7));
                params
            },
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: LocalVisionConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.model_path, deserialized.model_path);
        assert_eq!(config.device, deserialized.device);
        assert_eq!(config.dtype, deserialized.dtype);
        assert_eq!(config.max_seq_len, deserialized.max_seq_len);
    }
}
