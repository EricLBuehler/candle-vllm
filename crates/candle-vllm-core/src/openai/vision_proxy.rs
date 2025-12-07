use crate::openai::image_tool::{ImageDescription, ImageDescriptionTool};
use crate::openai::requests::{
    ChatCompletionRequest, ChatMessage, ContentPart, ImageUrl, MessageContent, Messages,
};
use crate::vision::{VisionError, VisionResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Configuration for vision proxy preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionProxyConfig {
    /// Whether to enable vision proxy preprocessing
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Custom prompt template for integrating image descriptions
    #[serde(default)]
    pub image_description_template: Option<String>,
    /// Whether to include confidence scores in descriptions
    #[serde(default)]
    pub include_confidence: bool,
    /// Whether to include processing time in descriptions
    #[serde(default)]
    pub include_timing: bool,
    /// Maximum number of images to process per message
    #[serde(default = "default_max_images")]
    pub max_images_per_message: usize,
    /// Maximum number of images to process per request
    #[serde(default = "default_max_images_per_request")]
    pub max_images_per_request: usize,
}

fn default_enabled() -> bool {
    true
}

fn default_max_images() -> usize {
    5
}

fn default_max_images_per_request() -> usize {
    20
}

impl Default for VisionProxyConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            image_description_template: Some("[Image: {description}]".to_string()),
            include_confidence: false,
            include_timing: false,
            max_images_per_message: default_max_images(),
            max_images_per_request: default_max_images_per_request(),
        }
    }
}

/// Statistics about preprocessing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingStats {
    /// Total number of images processed
    pub images_processed: usize,
    /// Total preprocessing time in milliseconds
    pub total_time_ms: u64,
    /// Number of successful image descriptions
    pub successful_descriptions: usize,
    /// Number of failed image descriptions
    pub failed_descriptions: usize,
    /// Average processing time per image in milliseconds
    pub avg_time_per_image_ms: Option<u64>,
}

impl Default for PreprocessingStats {
    fn default() -> Self {
        Self {
            images_processed: 0,
            total_time_ms: 0,
            successful_descriptions: 0,
            failed_descriptions: 0,
            avg_time_per_image_ms: None,
        }
    }
}

impl PreprocessingStats {
    /// Update statistics with a new image processing result
    pub fn update(&mut self, processing_time_ms: u64, success: bool) {
        self.images_processed += 1;
        self.total_time_ms += processing_time_ms;

        if success {
            self.successful_descriptions += 1;
        } else {
            self.failed_descriptions += 1;
        }

        self.avg_time_per_image_ms = if self.images_processed > 0 {
            Some(self.total_time_ms / self.images_processed as u64)
        } else {
            None
        };
    }
}

/// Result of preprocessing a chat completion request
#[derive(Debug, Clone)]
pub struct PreprocessingResult {
    /// The converted request with images replaced by text descriptions
    pub request: ChatCompletionRequest,
    /// Statistics about the preprocessing operation
    pub stats: PreprocessingStats,
    /// Whether any images were processed
    pub has_images: bool,
}

/// Vision proxy preprocessor that converts multimodal requests to text-only requests
pub struct VisionProxyPreprocessor<T: ImageDescriptionTool> {
    /// Vision tool for generating image descriptions
    vision_tool: Arc<T>,
    /// Configuration for preprocessing behavior
    config: VisionProxyConfig,
    /// Name identifier for this preprocessor
    name: String,
}

impl<T: ImageDescriptionTool> VisionProxyPreprocessor<T> {
    /// Create a new vision proxy preprocessor
    pub fn new(vision_tool: Arc<T>, config: VisionProxyConfig) -> Self {
        Self {
            vision_tool,
            config,
            name: "vision_proxy_preprocessor".to_string(),
        }
    }

    /// Create a preprocessor with default configuration
    pub fn with_defaults(vision_tool: Arc<T>) -> Self {
        Self::new(vision_tool, VisionProxyConfig::default())
    }

    /// Get the preprocessor configuration
    pub fn config(&self) -> &VisionProxyConfig {
        &self.config
    }

    /// Check if the vision tool is available
    pub async fn is_available(&self) -> bool {
        self.config.enabled && self.vision_tool.is_available().await
    }

    /// Get health information about the preprocessor
    pub async fn health_check(&self) -> VisionResult<serde_json::Value> {
        let tool_health = self.vision_tool.health_check().await?;

        Ok(serde_json::json!({
            "preprocessor_name": self.name,
            "enabled": self.config.enabled,
            "is_available": self.is_available().await,
            "config": self.config,
            "vision_tool": tool_health
        }))
    }

    /// Process a chat completion request, converting multimodal content to text
    pub async fn preprocess_request(
        &self,
        mut request: ChatCompletionRequest,
    ) -> VisionResult<PreprocessingResult> {
        let start_time = Instant::now();
        let mut stats = PreprocessingStats::default();
        let mut has_images = false;

        // If vision is disabled, return the request unchanged
        if !self.config.enabled {
            debug!("Vision proxy preprocessing is disabled");
            return Ok(PreprocessingResult {
                request,
                stats,
                has_images,
            });
        }

        // Check if vision tool is available
        if !self.vision_tool.is_available().await {
            warn!("Vision tool is not available, skipping image processing");
            return Ok(PreprocessingResult {
                request,
                stats,
                has_images,
            });
        }

        // Convert messages to chat format for processing
        let mut chat_messages = request.messages.to_chat_messages();

        info!(
            "Starting vision proxy preprocessing for {} messages",
            chat_messages.len()
        );

        // Count total images first for validation
        let total_images = self.count_images_in_chat_messages(&chat_messages);
        if total_images > self.config.max_images_per_request {
            return Err(VisionError::InvalidImageData {
                message: format!(
                    "Request contains {} images, maximum allowed is {}",
                    total_images, self.config.max_images_per_request
                ),
            });
        }

        // Process each message
        for message in &mut chat_messages {
            let message_images = self.count_images_in_message(message);
            if message_images > self.config.max_images_per_message {
                return Err(VisionError::InvalidImageData {
                    message: format!(
                        "Message contains {} images, maximum allowed is {}",
                        message_images, self.config.max_images_per_message
                    ),
                });
            }

            self.preprocess_message(message, &mut stats).await?;
            if message_images > 0 {
                has_images = true;
            }
        }

        // Update the request with processed messages
        request.messages = Messages::Chat(chat_messages);

        let total_time = start_time.elapsed().as_millis() as u64;
        info!(
            "Vision proxy preprocessing completed in {}ms. Processed {} images ({} successful, {} failed)",
            total_time, stats.images_processed, stats.successful_descriptions, stats.failed_descriptions
        );

        Ok(PreprocessingResult {
            request,
            stats,
            has_images,
        })
    }

    /// Process a single chat message, converting image content to text
    async fn preprocess_message(
        &self,
        message: &mut ChatMessage,
        stats: &mut PreprocessingStats,
    ) -> VisionResult<()> {
        // Only process messages with multimodal content
        let content_parts = match &mut message.content {
            Some(MessageContent::Text(_)) => return Ok(()),
            Some(MessageContent::Parts(parts)) => parts,
            None => return Ok(()), // Skip messages with no content
        };

        let mut current_text = String::new();

        for part in content_parts.iter() {
            match part {
                ContentPart::Text { text } => {
                    if !current_text.is_empty() {
                        current_text.push(' ');
                    }
                    current_text.push_str(text);
                }
                ContentPart::ImageUrl { image_url } => {
                    // Process the image and add description as text
                    let description = self.process_image(image_url, stats).await;
                    let description_text = self.format_image_description(description);

                    if !current_text.is_empty() {
                        current_text.push(' ');
                    }
                    current_text.push_str(&description_text);
                }
            }
        }

        // Convert to text-only message
        if !current_text.is_empty() {
            message.content = Some(MessageContent::Text(current_text));
        }

        Ok(())
    }

    /// Process a single image and generate a description
    async fn process_image(
        &self,
        image_url: &ImageUrl,
        stats: &mut PreprocessingStats,
    ) -> Result<ImageDescription, VisionError> {
        let start_time = Instant::now();

        debug!("Processing image: {}", self.truncate_url(&image_url.url));

        let result = self.vision_tool.describe_image(image_url, None).await;
        let processing_time = start_time.elapsed().as_millis() as u64;

        match &result {
            Ok(_) => {
                debug!("Successfully processed image in {}ms", processing_time);
                stats.update(processing_time, true);
            }
            Err(e) => {
                warn!("Failed to process image: {}", e);
                stats.update(processing_time, false);
            }
        }

        result
    }

    /// Format an image description according to the configured template
    fn format_image_description(
        &self,
        description_result: Result<ImageDescription, VisionError>,
    ) -> String {
        let template = self
            .config
            .image_description_template
            .as_deref()
            .unwrap_or("[Image: {description}]");

        match description_result {
            Ok(description) => {
                let mut formatted = template.replace("{description}", &description.description);

                if self.config.include_confidence {
                    if let Some(confidence) = description.confidence {
                        formatted = formatted.replace(
                            "{description}",
                            &format!(
                                "{} (confidence: {:.1}%)",
                                description.description,
                                confidence * 100.0
                            ),
                        );
                    }
                }

                if self.config.include_timing {
                    if let Some(timing) = description.processing_time_ms {
                        formatted.push_str(&format!(" [processed in {}ms]", timing));
                    }
                }

                formatted
            }
            Err(e) => {
                warn!("Using fallback description due to processing error: {}", e);
                template.replace("{description}", "image content unavailable")
            }
        }
    }

    /// Count images in a single message
    fn count_images_in_message(&self, message: &ChatMessage) -> usize {
        match &message.content {
            Some(MessageContent::Text(_)) => 0,
            Some(MessageContent::Parts(parts)) => parts
                .iter()
                .filter(|part| matches!(part, ContentPart::ImageUrl { .. }))
                .count(),
            None => 0,
        }
    }

    /// Count total images in a list of chat messages
    fn count_images_in_chat_messages(&self, messages: &[ChatMessage]) -> usize {
        messages
            .iter()
            .map(|msg| self.count_images_in_message(msg))
            .sum()
    }

    /// Truncate URL for logging (to avoid logging sensitive data)
    fn truncate_url(&self, url: &str) -> String {
        if url.len() > 100 {
            format!("{}...", &url[..97])
        } else {
            url.to_string()
        }
    }
}

/// Trait for factory creation of vision proxy preprocessors
#[async_trait]
pub trait VisionProxyPreprocessorFactory: Send + Sync {
    /// The type of vision tool this factory uses
    type Tool: ImageDescriptionTool;

    /// Create a new preprocessor with the given configuration
    async fn create_preprocessor(
        &self,
        vision_tool: Arc<Self::Tool>,
        config: VisionProxyConfig,
    ) -> VisionResult<VisionProxyPreprocessor<Self::Tool>>;

    /// Get default configuration for this preprocessor type
    fn default_config(&self) -> VisionProxyConfig {
        VisionProxyConfig::default()
    }

    /// Validate configuration before creating preprocessor
    fn validate_config(&self, config: &VisionProxyConfig) -> VisionResult<()> {
        if config.max_images_per_message == 0 {
            return Err(VisionError::InternalError {
                message: "max_images_per_message must be greater than 0".to_string(),
            });
        }

        if config.max_images_per_request == 0 {
            return Err(VisionError::InternalError {
                message: "max_images_per_request must be greater than 0".to_string(),
            });
        }

        if config.max_images_per_message > config.max_images_per_request {
            return Err(VisionError::InternalError {
                message: "max_images_per_message cannot exceed max_images_per_request".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::image_tool::ImageDescriptionConfig;
    use crate::openai::requests::{ChatMessage, ContentPart, ImageUrl, MessageContent, Messages};
    use std::collections::HashMap;

    // Mock vision tool for testing
    struct MockVisionTool {
        available: bool,
        description: String,
    }

    impl MockVisionTool {
        fn new(available: bool, description: String) -> Self {
            Self {
                available,
                description,
            }
        }
    }

    #[async_trait]
    impl ImageDescriptionTool for MockVisionTool {
        fn name(&self) -> &str {
            "mock_vision_tool"
        }

        fn config(&self) -> &ImageDescriptionConfig {
            // Return a static config for testing
            use std::sync::OnceLock;
            static CONFIG: OnceLock<ImageDescriptionConfig> = OnceLock::new();
            CONFIG.get_or_init(|| ImageDescriptionConfig {
                max_image_size: Some((1024, 1024)),
                timeout_secs: 30,
                include_metadata: false,
                prompt_template: None,
                model_params: HashMap::new(),
            })
        }

        async fn is_available(&self) -> bool {
            self.available
        }

        async fn health_check(&self) -> VisionResult<HashMap<String, serde_json::Value>> {
            Ok(HashMap::new())
        }

        async fn describe_image(
            &self,
            _image_url: &ImageUrl,
            _custom_prompt: Option<&str>,
        ) -> VisionResult<ImageDescription> {
            Ok(ImageDescription::new(&self.description)
                .with_confidence(0.95)
                .with_processing_time(500))
        }
    }

    #[tokio::test]
    async fn test_preprocessor_with_available_tool() {
        let tool = Arc::new(MockVisionTool::new(
            true,
            "A beautiful landscape".to_string(),
        ));
        let preprocessor = VisionProxyPreprocessor::with_defaults(tool);

        assert!(preprocessor.is_available().await);
    }

    #[tokio::test]
    async fn test_preprocessor_with_unavailable_tool() {
        let tool = Arc::new(MockVisionTool::new(false, "N/A".to_string()));
        let preprocessor = VisionProxyPreprocessor::with_defaults(tool);

        assert!(!preprocessor.is_available().await);
    }

    #[tokio::test]
    async fn test_text_only_message_unchanged() {
        let tool = Arc::new(MockVisionTool::new(
            true,
            "A beautiful landscape".to_string(),
        ));
        let preprocessor = VisionProxyPreprocessor::with_defaults(tool);

        let request = ChatCompletionRequest {
            messages: Messages::Chat(vec![ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text("Hello, how are you?".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }]),
            model: "test-model".to_string(),
            temperature: None,
            top_p: None,
            min_p: None,
            n: None,
            max_tokens: None,
            stop: None,
            stream: None,
            presence_penalty: None,
            repeat_last_n: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            top_k: None,
            best_of: None,
            use_beam_search: None,
            ignore_eos: None,
            skip_special_tokens: None,
            stop_token_ids: None,
            logprobs: None,
            thinking: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            conversation_id: None,
            resource_id: None,
        };

        let result = preprocessor
            .preprocess_request(request.clone())
            .await
            .unwrap();
        assert!(!result.has_images);
        assert_eq!(result.stats.images_processed, 0);

        // Content should be unchanged
        let chat_messages = result.request.messages.to_chat_messages();
        if let Some(MessageContent::Text(text)) = &chat_messages[0].content {
            assert_eq!(text, "Hello, how are you?");
        } else {
            panic!("Expected text content");
        }
    }

    #[tokio::test]
    async fn test_multimodal_message_conversion() {
        let tool = Arc::new(MockVisionTool::new(
            true,
            "A beautiful landscape".to_string(),
        ));
        let preprocessor = VisionProxyPreprocessor::with_defaults(tool);

        let request = ChatCompletionRequest {
            messages: Messages::Chat(vec![ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "What's in this image?".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl::new("data:image/jpeg;base64,/9j/4AAQ..."),
                    },
                ])),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }]),
            model: "test-model".to_string(),
            temperature: None,
            top_p: None,
            min_p: None,
            n: None,
            max_tokens: None,
            stop: None,
            stream: None,
            presence_penalty: None,
            repeat_last_n: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            top_k: None,
            best_of: None,
            use_beam_search: None,
            ignore_eos: None,
            skip_special_tokens: None,
            stop_token_ids: None,
            logprobs: None,
            thinking: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            conversation_id: None,
            resource_id: None,
        };

        let result = preprocessor.preprocess_request(request).await.unwrap();
        assert!(result.has_images);
        assert_eq!(result.stats.images_processed, 1);
        assert_eq!(result.stats.successful_descriptions, 1);

        // Content should be converted to text
        let chat_messages = result.request.messages.to_chat_messages();
        if let Some(MessageContent::Text(text)) = &chat_messages[0].content {
            assert!(text.contains("What's in this image?"));
            assert!(text.contains("[Image: A beautiful landscape]"));
        } else {
            panic!("Expected text content after preprocessing");
        }
    }

    #[test]
    fn test_config_validation() {
        let factory = TestFactory;

        // Valid config
        let valid_config = VisionProxyConfig::default();
        assert!(factory.validate_config(&valid_config).is_ok());

        // Invalid config - zero max images per message
        let invalid_config = VisionProxyConfig {
            max_images_per_message: 0,
            ..Default::default()
        };
        assert!(factory.validate_config(&invalid_config).is_err());

        // Invalid config - max per message exceeds max per request
        let invalid_config = VisionProxyConfig {
            max_images_per_message: 10,
            max_images_per_request: 5,
            ..Default::default()
        };
        assert!(factory.validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_preprocessing_stats() {
        let mut stats = PreprocessingStats::default();
        assert_eq!(stats.images_processed, 0);
        assert_eq!(stats.avg_time_per_image_ms, None);

        stats.update(500, true);
        assert_eq!(stats.images_processed, 1);
        assert_eq!(stats.successful_descriptions, 1);
        assert_eq!(stats.avg_time_per_image_ms, Some(500));

        stats.update(1000, false);
        assert_eq!(stats.images_processed, 2);
        assert_eq!(stats.successful_descriptions, 1);
        assert_eq!(stats.failed_descriptions, 1);
        assert_eq!(stats.avg_time_per_image_ms, Some(750));
    }

    // Test factory implementation
    struct TestFactory;

    #[async_trait]
    impl VisionProxyPreprocessorFactory for TestFactory {
        type Tool = MockVisionTool;

        async fn create_preprocessor(
            &self,
            vision_tool: Arc<Self::Tool>,
            config: VisionProxyConfig,
        ) -> VisionResult<VisionProxyPreprocessor<Self::Tool>> {
            self.validate_config(&config)?;
            Ok(VisionProxyPreprocessor::new(vision_tool, config))
        }
    }

    #[tokio::test]
    async fn test_factory_creation() {
        let factory = TestFactory;
        let tool = Arc::new(MockVisionTool::new(true, "Test description".to_string()));
        let config = VisionProxyConfig::default();

        let preprocessor = factory.create_preprocessor(tool, config).await.unwrap();
        assert!(preprocessor.is_available().await);
    }

    #[test]
    fn test_config_serialization() {
        let config = VisionProxyConfig {
            enabled: false,
            image_description_template: Some("[IMG: {description}]".to_string()),
            include_confidence: true,
            include_timing: true,
            max_images_per_message: 3,
            max_images_per_request: 15,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: VisionProxyConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.enabled, deserialized.enabled);
        assert_eq!(
            config.image_description_template,
            deserialized.image_description_template
        );
        assert_eq!(config.include_confidence, deserialized.include_confidence);
        assert_eq!(config.include_timing, deserialized.include_timing);
        assert_eq!(
            config.max_images_per_message,
            deserialized.max_images_per_message
        );
        assert_eq!(
            config.max_images_per_request,
            deserialized.max_images_per_request
        );
    }
}
