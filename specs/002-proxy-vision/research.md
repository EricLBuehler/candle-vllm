# Research: Proxy Vision Support Implementation

**Feature**: Proxy Vision Support for candle-vllm
**Date**: 2025-12-03
**Phase**: 0 - Research & Decision Making

## Research Overview

This document consolidates research findings for implementing proxy vision support in candle-vllm, enabling multimodal chat completions through a separate vision model that generates image captions for integration with the primary text model.

## Key Technical Decisions

### 1. Multi-Model Architecture Pattern

**Decision**: Resource Pool Pattern with Lazy Loading
**Rationale**:
- Enables concurrent execution of text and vision models with controlled resource allocation
- Supports graceful degradation when vision model fails or is unavailable
- Allows independent memory management for each model type
- Maintains existing text-only performance when vision not needed

**Implementation Pattern**:
```rust
pub struct ModelResourceManager {
    text_model_pool: Arc<Semaphore>,
    vision_model_pool: Arc<Semaphore>,
    total_memory_limit: usize,
}
```

**Alternatives considered**:
- Sequential processing: Rejected due to performance impact
- Shared resource pool: Rejected due to resource contention concerns
- Always-on both models: Rejected due to memory waste when vision not needed

### 2. Error Handling and Graceful Degradation

**Decision**: Cascading Fallback Chain with Circuit Breaker
**Rationale**:
- Ensures system remains functional when vision model fails
- Provides clear error boundaries and recovery mechanisms
- Prevents cascade failures from vision model affecting text processing
- Aligns with FR-005 requirement for graceful vision model failure handling

**Implementation Pattern**:
```rust
pub enum VisionBackend {
    Available { model_name: String, adapter: Arc<OpenAIAdapter> },
    Unavailable { reason: String },
}
```

**Alternatives considered**:
- Hard failure on vision errors: Rejected due to requirement for graceful degradation
- Silent failure without notification: Rejected due to need for observability
- Synchronous retry mechanisms: Rejected due to performance impact

### 3. OpenAI API Compatibility

**Decision**: Untagged Enum for Message Content with Backward Compatibility
**Rationale**:
- Maintains 100% compatibility with existing text-only clients
- Supports both legacy string format and new multimodal array format
- Follows OpenAI API specification exactly
- Enables gradual migration path for clients

**Implementation Pattern**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),           // Legacy format
    Parts(Vec<ContentPart>), // Multimodal format
}
```

**Alternatives considered**:
- Breaking change to array-only format: Rejected due to backward compatibility requirement
- Custom message format: Rejected due to OpenAI compatibility requirement
- Separate endpoints for multimodal: Rejected due to API complexity

### 4. Image Processing Strategy

**Decision**: Async Download with Validation Pipeline and Caching
**Rationale**:
- Supports both HTTP URLs and data URLs as per OpenAI specification
- Implements proper validation for security and resource management
- Uses timeout-based downloading to prevent hanging requests
- Provides clear error messages for different failure modes

**Implementation Pattern**:
```rust
pub struct ImageProcessor {
    client: reqwest::Client,
    max_size: usize,
    timeout_duration: Duration,
}
```

**Supported formats**: JPEG, PNG, WebP, GIF (matching OpenAI support)
**Size limits**: 20MB per image (matching OpenAI limits)
**Security measures**: URL validation, format verification, size limits

**Alternatives considered**:
- Synchronous image processing: Rejected due to performance impact
- No format restrictions: Rejected due to security and resource concerns
- Local file system access: Rejected due to security implications in server context

### 5. Configuration Integration

**Decision**: Extend existing models.yaml with Vision Capabilities
**Rationale**:
- Leverages existing configuration infrastructure
- Maintains consistency with current candle-vllm patterns
- Supports multiple vision models and configurations
- Enables runtime switching of vision models

**Implementation Pattern**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub vision_mode: VisionMode,
    pub vision_proxy: Option<VisionProxyConfig>,
}

pub enum VisionMode {
    Disabled,
    Proxy,   // Current implementation
    Native,  // Future extension
}
```

**Alternatives considered**:
- Separate vision configuration file: Rejected due to configuration fragmentation
- Command-line only configuration: Rejected due to limited flexibility
- Hard-coded vision model selection: Rejected due to inflexibility

## Performance and Resource Considerations

### Memory Management
- **Text Model Memory**: Configurable per model.yaml, typically 512MB-2GB
- **Vision Model Memory**: Configurable per model.yaml, typically 2GB-8GB
- **Concurrent Processing**: Separate memory pools prevent resource contention
- **Graceful Degradation**: System continues operation if vision model OOM

### Latency Targets
- **Vision Processing**: <10 seconds per image (SC-004 requirement)
- **Text-Only Requests**: No performance impact (SC-002 requirement)
- **Multimodal Requests**: <30 seconds total including vision processing (SC-001 requirement)

### Throughput Considerations
- **Concurrent Requests**: Semaphore-based limiting to prevent resource exhaustion
- **Batch Processing**: Support for multiple images in single request
- **Async Processing**: Non-blocking I/O for image downloads and model inference

## Integration Points with Existing Codebase

### Core Infrastructure
- **InferenceEngine**: Extend existing pattern for vision model loading
- **OpenAIAdapter**: Wrap vision models with same interface as text models
- **EngineConfig/EngineConfigBuilder**: Extend for vision-specific parameters

### Server Components
- **Chat Completion Handler**: Extend to support multimodal preprocessing
- **Error Handling**: Integrate with existing error types and patterns
- **Logging**: Use existing tracing infrastructure for vision processing

### Configuration System
- **models.yaml**: Extend ModelEntry with capabilities field
- **CLI Parameters**: Maintain existing override patterns
- **Environment Variables**: Support existing configuration precedence

## Testing Strategy

### Unit Tests
- **Message Content Parsing**: Test both legacy and multimodal formats
- **Image Processing**: Test URL validation, format detection, error handling
- **Vision Preprocessing**: Test caption integration and fallback logic

### Integration Tests
- **End-to-End Multimodal**: Test full request/response cycle with images
- **Backward Compatibility**: Ensure text-only requests unchanged
- **Graceful Degradation**: Test behavior when vision model unavailable
- **Error Scenarios**: Test various image-related failure modes

### Performance Tests
- **Memory Usage**: Verify resource limits respected
- **Latency**: Confirm performance targets met
- **Concurrency**: Test behavior under load

## Security Considerations

### Image Processing Security
- **URL Validation**: Restrict to HTTP/HTTPS, prevent file:// access
- **Size Limits**: 20MB per image, total request size limits
- **Format Validation**: Magic byte checking, not just extension
- **Timeout Protection**: Prevent hanging on slow/malicious image sources

### Model Security
- **Resource Limits**: Memory and compute quotas per model
- **Input Validation**: Sanitize image captions before text model input
- **Error Information**: Avoid leaking internal system details

## Dependencies and Prerequisites

### Required Dependencies
- **reqwest**: HTTP client for image downloads
- **image**: Image format detection and basic processing
- **base64**: Data URL decoding
- **tokio**: Async runtime support

### System Requirements
- **GPU Memory**: Additional 2-8GB for vision model
- **Network Access**: For downloading images from URLs (configurable)
- **File System**: Temporary space for image caching (optional)

## Migration and Deployment Considerations

### Backward Compatibility
- **Existing Clients**: Continue to work without changes
- **Configuration**: Existing models.yaml remains valid
- **API Endpoints**: No breaking changes to existing endpoints

### Deployment Strategy
- **Optional Feature**: Vision support can be disabled via configuration
- **Resource Scaling**: Can deploy text-only instances for cost optimization
- **Model Management**: Support for different vision models per deployment

This research provides the foundation for implementing proxy vision support while maintaining the reliability, performance, and compatibility requirements of the candle-vllm system.