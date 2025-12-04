# Data Model: Proxy Vision Support

**Feature**: Proxy Vision Support
**Date**: 2025-12-03
**Phase**: 1 - Design & Data Modeling

## Core Entities

### MessageContent
**Purpose**: Represents the content of a chat message, supporting both legacy text-only and new multimodal formats for backward compatibility.

**Fields**:
- `content_type`: Enum (Text, Parts) - discriminates between legacy and multimodal formats
- `text_content`: Optional String - legacy text-only content
- `parts`: Optional Vec<ContentPart> - multimodal content parts

**Relationships**:
- Contained within ChatMessage
- Contains multiple ContentPart entities when multimodal

**Validation Rules**:
- Must have either text_content OR parts, not both
- If parts provided, must contain at least one element
- Parts array cannot be empty when selected

**State Transitions**:
- Immutable once created
- Can be converted from text-only to multimodal format during preprocessing

---

### ContentPart
**Purpose**: Individual component of multimodal message content, representing either text or image elements.

**Fields**:
- `part_type`: Enum (Text, ImageUrl) - discriminates content type
- `text`: Optional String - text content when part_type is Text
- `image_url`: Optional ImageUrl - image reference when part_type is ImageUrl

**Relationships**:
- Contained within MessageContent parts array
- Contains ImageUrl entity when representing images

**Validation Rules**:
- Must have exactly one of text OR image_url based on part_type
- Text content cannot be empty string when provided
- ImageUrl must be valid when provided

**State Transitions**:
- Immutable once created
- Text parts remain unchanged during processing
- ImageUrl parts may be converted to Text parts during vision preprocessing

---

### ImageUrl
**Purpose**: Reference to an image, including the source URL and processing detail level, following OpenAI API specification.

**Fields**:
- `url`: String - HTTP/HTTPS URL or data URL for the image
- `detail`: String - processing detail level ("auto", "low", "high")

**Relationships**:
- Contained within ContentPart when part_type is ImageUrl
- Referenced by VisionProcessor for image analysis

**Validation Rules**:
- URL must be valid HTTP/HTTPS URL or properly formatted data URL
- Data URLs must use supported image formats (JPEG, PNG, WebP, GIF)
- Detail must be one of: "auto", "low", "high"
- Image size must not exceed 20MB when decoded

**State Transitions**:
- Immutable once created
- Processed by VisionProcessor to generate image captions
- May trigger image download and caching operations

---

### VisionCapabilities
**Purpose**: Configuration defining the vision processing capabilities and settings for a model.

**Fields**:
- `vision_mode`: VisionMode Enum (Disabled, Proxy, Native) - type of vision support
- `vision_proxy`: Optional VisionProxyConfig - proxy vision model configuration
- `image_token`: Optional String - reserved for native VLMs

**Relationships**:
- Contained within ModelCapabilities
- References VisionProxyConfig when proxy mode enabled

**Validation Rules**:
- vision_proxy must be provided when vision_mode is Proxy
- vision_proxy must be None when vision_mode is Disabled
- image_token reserved for future native VLM support

**State Transitions**:
- Loaded at startup from models.yaml configuration
- Remains static during runtime operation
- May be updated through configuration reload

---

### VisionProxyConfig
**Purpose**: Configuration for proxy vision model, specifying the model to use and optional prompt templates.

**Fields**:
- `hf_id`: String - HuggingFace model identifier for vision model
- `prompt_template`: Optional String - system/user prompt template for captioning

**Relationships**:
- Contained within VisionCapabilities
- Used by VisionBackend for model loading
- Referenced by LocalVisionModelTool for prompt construction

**Validation Rules**:
- hf_id must be valid HuggingFace model identifier
- prompt_template must be valid template string if provided
- Model specified by hf_id must be accessible and compatible

**State Transitions**:
- Loaded from configuration at startup
- Used during VisionBackend initialization
- Remains static during operation

---

### ModelEntry
**Purpose**: Complete configuration for a single model, including engine parameters and capabilities.

**Fields**:
- `name`: String - unique identifier for the model
- `hf_id`: Optional String - HuggingFace model identifier
- `local_path`: Optional String - local filesystem path to model
- `weight_file`: Optional String - specific weight file name
- `params`: EngineParams - engine configuration parameters
- `capabilities`: ModelCapabilities - model capabilities including vision

**Relationships**:
- Contains EngineParams for engine configuration
- Contains ModelCapabilities for feature definitions
- Part of ModelsFile configuration collection

**Validation Rules**:
- Must have either hf_id OR local_path, not both
- name must be unique within ModelsFile
- params must be valid EngineParams
- capabilities must be consistent with model type

**State Transitions**:
- Loaded from models.yaml at startup
- Used for model initialization and capability determination
- Static during runtime operation

---

### VisionBackend
**Purpose**: Runtime representation of vision model availability and access, supporting graceful degradation.

**Fields**:
- `backend_type`: Enum (Available, Unavailable) - discriminates availability
- `model_name`: Optional String - name of loaded vision model when available
- `adapter`: Optional Arc<OpenAIAdapter> - model adapter when available
- `reason`: Optional String - unavailability reason when not available

**Relationships**:
- Created from ModelEntry during startup
- Used by VisionProxyPreprocessor for image processing
- Referenced by EngineState for request routing

**Validation Rules**:
- When Available: must have model_name and adapter
- When Unavailable: must have reason, model_name and adapter are None
- adapter must be valid OpenAIAdapter when provided

**State Transitions**:
1. Initialize as Unavailable during startup
2. Transition to Available when model loading succeeds
3. May transition back to Unavailable if health checks fail
4. Remains in state until explicit health check or restart

---

### ImageCaption
**Purpose**: Generated textual description of image content produced by vision model for integration with text prompts.

**Fields**:
- `caption_text`: String - descriptive text generated by vision model
- `image_index`: usize - position of image in original request (1-based)
- `confidence`: Optional f32 - vision model confidence score
- `processing_time_ms`: u64 - time taken to generate caption

**Relationships**:
- Generated by LocalVisionModelTool from ImageUrl
- Integrated into text prompts by VisionProxyPreprocessor
- Temporary entity, not persisted

**Validation Rules**:
- caption_text cannot be empty
- image_index must be positive
- confidence must be between 0.0 and 1.0 if provided
- processing_time_ms must be reasonable (<30000ms)

**State Transitions**:
1. Created by vision model processing
2. Formatted and integrated into text prompt
3. Discarded after request processing complete

## Entity Relationships Diagram

```
ModelsFile
├── ModelEntry[]
    ├── EngineParams
    └── ModelCapabilities
        └── VisionCapabilities
            └── VisionProxyConfig

ChatMessage
└── MessageContent
    └── ContentPart[]
        └── ImageUrl (optional)

VisionBackend ←→ ModelEntry
    └── OpenAIAdapter (when available)

VisionProxyPreprocessor
├── VisionBackend
└── ImageDescriptionTool
    └── ImageCaption[]
```

## Data Flow Summary

1. **Configuration Loading**: ModelsFile → ModelEntry → VisionCapabilities
2. **Startup**: ModelEntry → VisionBackend (Available/Unavailable)
3. **Request Processing**: ChatMessage → MessageContent → ContentPart → ImageUrl
4. **Vision Processing**: ImageUrl → VisionBackend → ImageCaption
5. **Response Generation**: ImageCaption → text prompt → ChatMessage

## Persistence and Caching

### Non-Persistent Entities
- **ImageCaption**: Temporary, discarded after request
- **VisionBackend**: Runtime state, recreated on startup
- **MessageContent/ContentPart**: Request-scoped, not stored

### Configuration Entities
- **ModelEntry**: Loaded from models.yaml, cached in memory
- **VisionCapabilities**: Part of configuration, static after load
- **EngineParams**: Configuration-driven, cached during operation

### Caching Strategy
- **Image Data**: Optional HTTP response caching (not modeled)
- **Model Loading**: Lazy loading with in-memory retention
- **Configuration**: Load-once pattern with reload capability