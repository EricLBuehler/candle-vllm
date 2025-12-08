use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents the availability status of a vision backend
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum VisionBackend {
    /// Vision backend is available and ready for use
    Available {
        /// The model ID being used for vision processing
        model_id: String,
        /// Additional metadata about the backend
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<VisionBackendMetadata>,
    },
    /// Vision backend is unavailable
    Unavailable {
        /// Reason why the backend is unavailable
        reason: UnavailableReason,
        /// Optional error message with more details
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
}

/// Additional metadata about a vision backend
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisionBackendMetadata {
    /// Maximum image size supported (width × height)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_image_size: Option<(u32, u32)>,
    /// Supported image formats (e.g., "jpeg", "png", "webp")
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported_formats: Vec<String>,
    /// Whether base64 data URLs are supported
    #[serde(default)]
    pub supports_data_urls: bool,
    /// Whether web URLs are supported
    #[serde(default)]
    pub supports_web_urls: bool,
}

/// Reasons why a vision backend might be unavailable
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnavailableReason {
    /// Vision is disabled in configuration
    Disabled,
    /// Model failed to load
    ModelLoadFailed,
    /// Model is currently loading
    Loading,
    /// Out of memory or resources
    InsufficientResources,
    /// Network error (for remote models)
    NetworkError,
    /// Unsupported model format or architecture
    UnsupportedModel,
    /// Generic initialization error
    InitializationFailed,
    /// Temporary error, might recover
    TemporaryError,
}

impl VisionBackend {
    /// Create an available vision backend
    pub fn available(model_id: impl Into<String>) -> Self {
        Self::Available {
            model_id: model_id.into(),
            metadata: None,
        }
    }

    /// Create an available vision backend with metadata
    pub fn available_with_metadata(
        model_id: impl Into<String>,
        metadata: VisionBackendMetadata,
    ) -> Self {
        Self::Available {
            model_id: model_id.into(),
            metadata: Some(metadata),
        }
    }

    /// Create an unavailable vision backend with reason
    pub fn unavailable(reason: UnavailableReason) -> Self {
        Self::Unavailable {
            reason,
            message: None,
        }
    }

    /// Create an unavailable vision backend with reason and message
    pub fn unavailable_with_message(reason: UnavailableReason, message: impl Into<String>) -> Self {
        Self::Unavailable {
            reason,
            message: Some(message.into()),
        }
    }

    /// Check if the vision backend is available
    pub fn is_available(&self) -> bool {
        matches!(self, VisionBackend::Available { .. })
    }

    /// Check if the vision backend is unavailable
    pub fn is_unavailable(&self) -> bool {
        !self.is_available()
    }

    /// Get the model ID if available
    pub fn model_id(&self) -> Option<&str> {
        match self {
            VisionBackend::Available { model_id, .. } => Some(model_id),
            VisionBackend::Unavailable { .. } => None,
        }
    }

    /// Get the unavailability reason if unavailable
    pub fn unavailable_reason(&self) -> Option<&UnavailableReason> {
        match self {
            VisionBackend::Available { .. } => None,
            VisionBackend::Unavailable { reason, .. } => Some(reason),
        }
    }

    /// Get the error message if unavailable
    pub fn error_message(&self) -> Option<&str> {
        match self {
            VisionBackend::Available { .. } => None,
            VisionBackend::Unavailable { message, .. } => message.as_deref(),
        }
    }

    /// Get metadata if available
    pub fn metadata(&self) -> Option<&VisionBackendMetadata> {
        match self {
            VisionBackend::Available { metadata, .. } => metadata.as_ref(),
            VisionBackend::Unavailable { .. } => None,
        }
    }

    /// Check if the backend can handle the given image format
    pub fn supports_format(&self, format: &str) -> bool {
        match self.metadata() {
            Some(meta) => {
                meta.supported_formats.is_empty()
                    || meta.supported_formats.contains(&format.to_lowercase())
            }
            None => true, // Assume support if no metadata available
        }
    }

    /// Check if the backend can handle data URLs
    pub fn supports_data_urls(&self) -> bool {
        match self.metadata() {
            Some(meta) => meta.supports_data_urls,
            None => true, // Assume support if no metadata available
        }
    }

    /// Check if the backend can handle web URLs
    pub fn supports_web_urls(&self) -> bool {
        match self.metadata() {
            Some(meta) => meta.supports_web_urls,
            None => true, // Assume support if no metadata available
        }
    }
}

impl fmt::Display for VisionBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VisionBackend::Available { model_id, .. } => {
                write!(f, "Available ({})", model_id)
            }
            VisionBackend::Unavailable { reason, message } => {
                if let Some(msg) = message {
                    write!(f, "Unavailable ({:?}): {}", reason, msg)
                } else {
                    write!(f, "Unavailable ({:?})", reason)
                }
            }
        }
    }
}

impl fmt::Display for UnavailableReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnavailableReason::Disabled => write!(f, "disabled"),
            UnavailableReason::ModelLoadFailed => write!(f, "model load failed"),
            UnavailableReason::Loading => write!(f, "loading"),
            UnavailableReason::InsufficientResources => write!(f, "insufficient resources"),
            UnavailableReason::NetworkError => write!(f, "network error"),
            UnavailableReason::UnsupportedModel => write!(f, "unsupported model"),
            UnavailableReason::InitializationFailed => write!(f, "initialization failed"),
            UnavailableReason::TemporaryError => write!(f, "temporary error"),
        }
    }
}

impl Default for VisionBackendMetadata {
    fn default() -> Self {
        Self {
            max_image_size: None,
            supported_formats: vec!["jpeg".to_string(), "png".to_string(), "webp".to_string()],
            supports_data_urls: true,
            supports_web_urls: true,
        }
    }
}

/// Result type for vision operations
pub type VisionResult<T> = Result<T, VisionError>;

/// Errors that can occur during vision processing
#[derive(thiserror::Error, Debug)]
pub enum VisionError {
    #[error("vision backend is unavailable: {reason}")]
    BackendUnavailable { reason: UnavailableReason },

    #[error("image format not supported: {format}")]
    UnsupportedFormat { format: String },

    #[error("image too large: {width}x{height} (max: {max_width}x{max_height})")]
    ImageTooLarge {
        width: u32,
        height: u32,
        max_width: u32,
        max_height: u32,
    },

    #[error("invalid image data: {message}")]
    InvalidImageData { message: String },

    #[error("network error: {message}")]
    NetworkError { message: String },

    #[error("network timeout: {url} (timeout: {timeout_secs}s)")]
    NetworkTimeout { url: String, timeout_secs: u64 },

    #[error("model not found: {model_path}")]
    ModelNotFound { model_path: String },

    #[error("model not ready: {message}")]
    ModelNotReady { message: String },

    #[error("model error: {message}")]
    ModelError { message: String },

    #[error("processing timeout")]
    Timeout,

    #[error("internal error: {message}")]
    InternalError { message: String },
}

impl From<VisionBackend> for VisionResult<()> {
    fn from(backend: VisionBackend) -> Self {
        match backend {
            VisionBackend::Available { .. } => Ok(()),
            VisionBackend::Unavailable { reason, .. } => {
                Err(VisionError::BackendUnavailable { reason })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_backend_available() {
        let backend = VisionBackend::available("qwen2-vl-7b-instruct");
        assert!(backend.is_available());
        assert!(!backend.is_unavailable());
        assert_eq!(backend.model_id(), Some("qwen2-vl-7b-instruct"));
        assert_eq!(backend.unavailable_reason(), None);
    }

    #[test]
    fn test_vision_backend_unavailable() {
        let backend = VisionBackend::unavailable_with_message(
            UnavailableReason::ModelLoadFailed,
            "CUDA out of memory",
        );
        assert!(!backend.is_available());
        assert!(backend.is_unavailable());
        assert_eq!(backend.model_id(), None);
        assert_eq!(
            backend.unavailable_reason(),
            Some(&UnavailableReason::ModelLoadFailed)
        );
        assert_eq!(backend.error_message(), Some("CUDA out of memory"));
    }

    #[test]
    fn test_vision_backend_with_metadata() {
        let metadata = VisionBackendMetadata {
            max_image_size: Some((2048, 2048)),
            supported_formats: vec!["jpeg".to_string(), "png".to_string()],
            supports_data_urls: true,
            supports_web_urls: false,
        };

        let backend = VisionBackend::available_with_metadata("test-model", metadata.clone());
        assert!(backend.is_available());
        assert_eq!(backend.metadata(), Some(&metadata));
        assert!(backend.supports_format("jpeg"));
        assert!(!backend.supports_format("gif"));
        assert!(backend.supports_data_urls());
        assert!(!backend.supports_web_urls());
    }

    #[test]
    fn test_vision_backend_serialization() {
        let backend = VisionBackend::available("test-model");
        let json = serde_json::to_string(&backend).unwrap();
        let deserialized: VisionBackend = serde_json::from_str(&json).unwrap();
        assert_eq!(backend, deserialized);

        let backend = VisionBackend::unavailable(UnavailableReason::Disabled);
        let json = serde_json::to_string(&backend).unwrap();
        let deserialized: VisionBackend = serde_json::from_str(&json).unwrap();
        assert_eq!(backend, deserialized);
    }

    #[test]
    fn test_vision_result_conversion() {
        let available_backend = VisionBackend::available("test-model");
        let result: VisionResult<()> = available_backend.into();
        assert!(result.is_ok());

        let unavailable_backend = VisionBackend::unavailable(UnavailableReason::Disabled);
        let result: VisionResult<()> = unavailable_backend.into();
        assert!(result.is_err());
        match result {
            Err(VisionError::BackendUnavailable { reason }) => {
                assert_eq!(reason, UnavailableReason::Disabled);
            }
            _ => panic!("Expected BackendUnavailable error"),
        }
    }

    #[test]
    fn test_display_implementations() {
        let backend = VisionBackend::available("test-model");
        assert_eq!(backend.to_string(), "Available (test-model)");

        let backend = VisionBackend::unavailable_with_message(
            UnavailableReason::ModelLoadFailed,
            "Error details",
        );
        assert_eq!(
            backend.to_string(),
            "Unavailable (ModelLoadFailed): Error details"
        );

        let reason = UnavailableReason::Disabled;
        assert_eq!(reason.to_string(), "disabled");
    }
}
