use crate::openai::requests::ImageUrl;
use crate::vision::{VisionError, VisionResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for image description tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageDescriptionConfig {
    /// Maximum image size to process (width × height)
    #[serde(default)]
    pub max_image_size: Option<(u32, u32)>,
    /// Timeout for image processing in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Whether to include image metadata in descriptions
    #[serde(default)]
    pub include_metadata: bool,
    /// Custom prompt template for image descriptions
    #[serde(default)]
    pub prompt_template: Option<String>,
    /// Additional model-specific parameters
    #[serde(default)]
    pub model_params: HashMap<String, serde_json::Value>,
}

fn default_timeout() -> u64 {
    30 // 30 seconds default timeout
}

impl Default for ImageDescriptionConfig {
    fn default() -> Self {
        Self {
            max_image_size: Some((2048, 2048)),
            timeout_secs: default_timeout(),
            include_metadata: false,
            prompt_template: Some("Describe this image in detail.".to_string()),
            model_params: HashMap::new(),
        }
    }
}

/// Metadata about an image being processed
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageMetadata {
    /// Image width in pixels
    pub width: Option<u32>,
    /// Image height in pixels
    pub height: Option<u32>,
    /// Image format (e.g., "jpeg", "png", "webp")
    pub format: Option<String>,
    /// Image file size in bytes (if known)
    pub size_bytes: Option<u64>,
    /// Whether the image was resized during processing
    pub was_resized: bool,
    /// Original dimensions if the image was resized
    pub original_dimensions: Option<(u32, u32)>,
}

impl Default for ImageMetadata {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            format: None,
            size_bytes: None,
            was_resized: false,
            original_dimensions: None,
        }
    }
}

/// Result of image description processing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageDescription {
    /// The generated description text
    pub description: String,
    /// Confidence score (0.0 to 1.0) if available
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Processing time in milliseconds
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub processing_time_ms: Option<u64>,
    /// Image metadata
    #[serde(default)]
    pub metadata: ImageMetadata,
    /// Model-specific additional data
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub model_data: HashMap<String, serde_json::Value>,
}

impl ImageDescription {
    /// Create a new image description with just the description text
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            confidence: None,
            processing_time_ms: None,
            metadata: ImageMetadata::default(),
            model_data: HashMap::new(),
        }
    }

    /// Set the confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Set the processing time
    pub fn with_processing_time(mut self, ms: u64) -> Self {
        self.processing_time_ms = Some(ms);
        self
    }

    /// Set the image metadata
    pub fn with_metadata(mut self, metadata: ImageMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add model-specific data
    pub fn with_model_data(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.model_data.insert(key.into(), value);
        self
    }

    /// Check if the description has high confidence (>= 0.8)
    pub fn has_high_confidence(&self) -> bool {
        self.confidence.map_or(true, |c| c >= 0.8)
    }

    /// Check if processing was fast (< 5000ms)
    pub fn was_fast(&self) -> bool {
        self.processing_time_ms.map_or(true, |ms| ms < 5000)
    }
}

/// Trait for image description tools that can process images and generate captions
///
/// This trait provides a pluggable interface for different vision model implementations,
/// allowing the system to swap between local models, remote APIs, or other vision services.
#[async_trait]
pub trait ImageDescriptionTool: Send + Sync {
    /// Get a human-readable name for this tool
    fn name(&self) -> &str;

    /// Get the configuration used by this tool
    fn config(&self) -> &ImageDescriptionConfig;

    /// Check if the tool is available and ready for use
    async fn is_available(&self) -> bool;

    /// Get health status and diagnostic information
    async fn health_check(&self) -> VisionResult<HashMap<String, serde_json::Value>>;

    /// Process a single image and generate a description
    ///
    /// # Arguments
    /// * `image_url` - The image to process (supports both web URLs and data URLs)
    /// * `custom_prompt` - Optional custom prompt to override the default
    ///
    /// # Returns
    /// * `Ok(ImageDescription)` - Success with description and metadata
    /// * `Err(VisionError)` - Processing error (format unsupported, network error, etc.)
    async fn describe_image(
        &self,
        image_url: &ImageUrl,
        custom_prompt: Option<&str>,
    ) -> VisionResult<ImageDescription>;

    /// Process multiple images concurrently
    ///
    /// Default implementation processes images sequentially. Implementations may override
    /// for better performance with batch processing or parallel execution.
    async fn describe_images(
        &self,
        image_urls: &[ImageUrl],
        custom_prompt: Option<&str>,
    ) -> VisionResult<Vec<ImageDescription>> {
        let mut results = Vec::new();
        for image_url in image_urls {
            let description = self.describe_image(image_url, custom_prompt).await?;
            results.push(description);
        }
        Ok(results)
    }

    /// Validate if the image can be processed by this tool
    ///
    /// This method should check image format, size, and other constraints without
    /// actually processing the image. Used for early validation.
    fn validate_image(&self, image_url: &ImageUrl) -> VisionResult<()> {
        // Default validation checks basic format support
        if image_url.is_data_url() {
            // Extract format from data URL
            if let Some(format) = extract_data_url_format(&image_url.url) {
                if !self.supports_format(&format) {
                    return Err(VisionError::UnsupportedFormat { format });
                }
            }
        } else if image_url.is_web_url() {
            // Basic URL validation - implementations may override for more thorough checks
            if !image_url.url.starts_with("https://") && !image_url.url.starts_with("http://") {
                return Err(VisionError::InvalidImageData {
                    message: "Invalid web URL format".to_string(),
                });
            }
        } else {
            return Err(VisionError::InvalidImageData {
                message: "URL must be either a data URL or web URL".to_string(),
            });
        }
        Ok(())
    }

    /// Check if the tool supports a specific image format
    ///
    /// Default implementation supports common formats. Tools should override
    /// to specify their actual capabilities.
    fn supports_format(&self, format: &str) -> bool {
        matches!(
            format.to_lowercase().as_str(),
            "jpeg" | "jpg" | "png" | "webp" | "gif" | "bmp"
        )
    }

    /// Get the maximum image size supported by this tool
    fn max_image_size(&self) -> Option<(u32, u32)> {
        self.config().max_image_size
    }

    /// Estimate processing time for an image (used for timeout planning)
    ///
    /// Default implementation returns the configured timeout. Tools may override
    /// to provide more accurate estimates based on image size, model complexity, etc.
    async fn estimate_processing_time(&self, _image_url: &ImageUrl) -> u64 {
        self.config().timeout_secs
    }
}

/// Extract image format from data URL (e.g., "data:image/jpeg;base64,..." -> "jpeg")
fn extract_data_url_format(url: &str) -> Option<String> {
    if !url.starts_with("data:image/") {
        return None;
    }

    let format_part = url.strip_prefix("data:image/")?.split(';').next()?;

    Some(format_part.to_lowercase())
}

/// Factory trait for creating image description tools
pub trait ImageDescriptionToolFactory: Send + Sync {
    /// The type of tool this factory creates
    type Tool: ImageDescriptionTool;

    /// Create a new tool instance with the given configuration
    fn create_tool(&self, config: ImageDescriptionConfig) -> VisionResult<Self::Tool>;

    /// Get the name of the tool type this factory creates
    fn tool_type_name(&self) -> &str;

    /// Get default configuration for this tool type
    fn default_config(&self) -> ImageDescriptionConfig {
        ImageDescriptionConfig::default()
    }

    /// Validate configuration before creating a tool
    fn validate_config(&self, config: &ImageDescriptionConfig) -> VisionResult<()> {
        // Basic validation - ensure timeout is reasonable
        if config.timeout_secs == 0 {
            return Err(VisionError::InternalError {
                message: "Timeout must be greater than 0".to_string(),
            });
        }

        // Validate max image size if specified
        if let Some((width, height)) = config.max_image_size {
            if width == 0 || height == 0 {
                return Err(VisionError::InternalError {
                    message: "Image dimensions must be greater than 0".to_string(),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_description_config_default() {
        let config = ImageDescriptionConfig::default();
        assert_eq!(config.max_image_size, Some((2048, 2048)));
        assert_eq!(config.timeout_secs, 30);
        assert!(!config.include_metadata);
        assert_eq!(
            config.prompt_template,
            Some("Describe this image in detail.".to_string())
        );
        assert!(config.model_params.is_empty());
    }

    #[test]
    fn test_image_metadata_default() {
        let metadata = ImageMetadata::default();
        assert_eq!(metadata.width, None);
        assert_eq!(metadata.height, None);
        assert_eq!(metadata.format, None);
        assert_eq!(metadata.size_bytes, None);
        assert!(!metadata.was_resized);
        assert_eq!(metadata.original_dimensions, None);
    }

    #[test]
    fn test_image_description_builder() {
        let description = ImageDescription::new("A beautiful sunset over the ocean")
            .with_confidence(0.95)
            .with_processing_time(2500)
            .with_metadata(ImageMetadata {
                width: Some(1920),
                height: Some(1080),
                format: Some("jpeg".to_string()),
                ..Default::default()
            })
            .with_model_data("model_version", serde_json::json!("v2.1"));

        assert_eq!(description.description, "A beautiful sunset over the ocean");
        assert_eq!(description.confidence, Some(0.95));
        assert_eq!(description.processing_time_ms, Some(2500));
        assert_eq!(description.metadata.width, Some(1920));
        assert_eq!(description.metadata.height, Some(1080));
        assert_eq!(description.metadata.format, Some("jpeg".to_string()));
        assert!(description.has_high_confidence());
        assert!(description.was_fast());
    }

    #[test]
    fn test_confidence_clamping() {
        let description1 = ImageDescription::new("test").with_confidence(-0.5);
        assert_eq!(description1.confidence, Some(0.0));

        let description2 = ImageDescription::new("test").with_confidence(1.5);
        assert_eq!(description2.confidence, Some(1.0));

        let description3 = ImageDescription::new("test").with_confidence(0.5);
        assert_eq!(description3.confidence, Some(0.5));
    }

    #[test]
    fn test_extract_data_url_format() {
        assert_eq!(
            extract_data_url_format("data:image/jpeg;base64,/9j/4AAQ..."),
            Some("jpeg".to_string())
        );
        assert_eq!(
            extract_data_url_format("data:image/png;base64,iVBORw0KGgo..."),
            Some("png".to_string())
        );
        assert_eq!(
            extract_data_url_format("data:image/webp;base64,UklGRi..."),
            Some("webp".to_string())
        );
        assert_eq!(extract_data_url_format("not a data url"), None);
        assert_eq!(
            extract_data_url_format("data:text/plain;base64,SGVs..."),
            None
        );
    }

    #[test]
    fn test_high_confidence_detection() {
        let high_conf = ImageDescription::new("test").with_confidence(0.9);
        assert!(high_conf.has_high_confidence());

        let med_conf = ImageDescription::new("test").with_confidence(0.7);
        assert!(!med_conf.has_high_confidence());

        let no_conf = ImageDescription::new("test");
        assert!(no_conf.has_high_confidence()); // Default to true when no confidence provided
    }

    #[test]
    fn test_fast_processing_detection() {
        let fast = ImageDescription::new("test").with_processing_time(3000);
        assert!(fast.was_fast());

        let slow = ImageDescription::new("test").with_processing_time(8000);
        assert!(!slow.was_fast());

        let unknown = ImageDescription::new("test");
        assert!(unknown.was_fast()); // Default to true when no timing provided
    }

    #[test]
    fn test_image_url_types() {
        let data_url = ImageUrl::new("data:image/jpeg;base64,/9j/4AAQ...");
        assert!(data_url.is_data_url());
        assert!(!data_url.is_web_url());

        let web_url = ImageUrl::new("https://example.com/image.jpg");
        assert!(!web_url.is_data_url());
        assert!(web_url.is_web_url());

        let http_url = ImageUrl::new("http://example.com/image.png");
        assert!(!http_url.is_data_url());
        assert!(http_url.is_web_url());
    }

    #[test]
    fn test_serialization() {
        let description = ImageDescription::new("Test description")
            .with_confidence(0.8)
            .with_processing_time(1500);

        let json = serde_json::to_string(&description).unwrap();
        let deserialized: ImageDescription = serde_json::from_str(&json).unwrap();

        assert_eq!(description, deserialized);
    }

    #[test]
    fn test_config_serialization() {
        let config = ImageDescriptionConfig {
            max_image_size: Some((1024, 1024)),
            timeout_secs: 60,
            include_metadata: true,
            prompt_template: Some("Custom prompt".to_string()),
            model_params: {
                let mut params = HashMap::new();
                params.insert("temperature".to_string(), serde_json::json!(0.7));
                params
            },
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ImageDescriptionConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.max_image_size, deserialized.max_image_size);
        assert_eq!(config.timeout_secs, deserialized.timeout_secs);
        assert_eq!(config.include_metadata, deserialized.include_metadata);
        assert_eq!(config.prompt_template, deserialized.prompt_template);
    }
}
