# Feature Specification: Proxy Vision Support

**Feature Branch**: `002-proxy-vision`
**Created**: 2025-12-03
**Status**: Draft
**Input**: User description: "vision support based on @docs/images/README.md is what we want."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Basic Image Analysis in Chat (Priority: P1)

A user sends a chat message containing both text and an image to the candle-vllm API and receives a meaningful response that demonstrates understanding of both the text question and image content.

**Why this priority**: This is the core value proposition - enabling multimodal conversations where users can discuss images with the AI. This is the fundamental capability that justifies the entire feature.

**Independent Test**: Can be fully tested by sending a single multimodal chat request with an image URL and text prompt, and verifying the response includes relevant image content analysis.

**Acceptance Scenarios**:

1. **Given** a running candle-vllm server with vision support enabled, **When** user sends a chat request with text "What do you see in this image?" and an image URL, **Then** the response contains a description of the image content
2. **Given** the system is configured with Ministral-3-3B as primary model and Qwen2-VL as vision model, **When** a multimodal request is sent, **Then** the vision model processes the image and the text model generates a contextual response
3. **Given** an image analysis request, **When** the vision model generates a caption, **Then** the caption is seamlessly integrated into the prompt sent to the primary model

---

### User Story 2 - Text-Only Conversations Remain Unaffected (Priority: P2)

A user sends text-only chat messages and receives the same quality responses as before, with no performance degradation or changes in behavior.

**Why this priority**: Maintaining backward compatibility ensures existing users aren't negatively impacted by the vision feature addition.

**Independent Test**: Can be tested by sending text-only chat requests and comparing response quality and speed to pre-vision implementation.

**Acceptance Scenarios**:

1. **Given** a candle-vllm server with vision support, **When** user sends text-only chat requests, **Then** responses match the quality and speed of text-only deployments
2. **Given** vision model loading fails during startup, **When** user sends text-only requests, **Then** the service continues operating normally without vision capabilities

---

### User Story 3 - Multiple Images in Single Conversation (Priority: P3)

A user can include multiple images in a single chat message and receive a response that addresses all images coherently.

**Why this priority**: Enhances the multimodal experience by supporting richer interactions, but not essential for basic vision functionality.

**Independent Test**: Can be tested by sending a chat request with multiple image URLs and verifying all images are referenced in the response.

**Acceptance Scenarios**:

1. **Given** a chat request with multiple image URLs, **When** the system processes the request, **Then** each image receives a separate caption numbered sequentially
2. **Given** multiple images and accompanying text, **When** the vision preprocessing occurs, **Then** all image captions are integrated with the user text in a readable format

---

### Edge Cases

- What happens when image URLs are invalid or inaccessible?
- How does system handle extremely large images that might cause memory issues?
- What occurs when vision model fails during a request but primary model is available?
- How does the system behave when image formats are unsupported?
- What happens when the vision model is unavailable but image requests are received?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept OpenAI-compatible multimodal chat requests with both text content and image URLs
- **FR-002**: System MUST process images using a configured vision model to generate textual descriptions
- **FR-003**: System MUST integrate image captions into text prompts sent to the primary reasoning model
- **FR-004**: System MUST maintain backward compatibility with text-only chat requests
- **FR-005**: System MUST gracefully handle vision model failures without affecting text-only functionality
- **FR-006**: System MUST support configurable vision models through the models.yaml configuration file
- **FR-007**: System MUST load both primary and vision models at startup with appropriate error handling
- **FR-008**: System MUST preserve the existing OpenAI API contract for chat completions
- **FR-009**: System MUST support multiple images within a single chat message
- **FR-010**: System MUST provide clear logging of vision processing steps and any errors

### Key Entities *(include if feature involves data)*

- **Chat Message**: Contains role, content (text and/or images), represents user input with multimodal content
- **Image Reference**: URL or data reference to image content, includes detail level specification
- **Vision Capabilities Configuration**: Defines vision mode (disabled/proxy/native), vision model settings, prompt templates
- **Model Configuration**: Extended model definition including vision proxy settings, engine parameters
- **Image Caption**: Generated textual description of image content, produced by vision model for integration with text prompt

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can send multimodal chat requests and receive relevant responses that demonstrate image understanding within 30 seconds
- **SC-002**: Text-only requests maintain the same response times and quality as before vision feature addition
- **SC-003**: System successfully starts up with both primary and vision models loaded, or gracefully degrades to text-only mode when vision model fails
- **SC-004**: Vision processing adds no more than 10 seconds to total response time for single-image requests
- **SC-005**: System handles at least 95% of common image formats (JPEG, PNG, WebP) without errors
- **SC-006**: Multimodal requests process successfully with both models consuming appropriate amounts of configured memory limits

## Assumptions

- Vision model (Qwen2-VL-7B-Instruct) and primary model (Ministral-3-3B-Reasoning-2512) can run concurrently within available system memory
- Image URLs provided by users are publicly accessible or properly authenticated
- Standard web image formats are sufficient for initial implementation
- Proxy-based approach (separate vision and text models) is acceptable performance trade-off compared to native multimodal models
- Current models.yaml configuration pattern can be extended to include vision capabilities without breaking existing setups

## Dependencies

- Existing candle-vllm core inference engine must support loading multiple models simultaneously
- OpenAI API compatibility layer must be extended to handle multimodal message content
- Configuration system must support the new vision-related parameters defined in the design document