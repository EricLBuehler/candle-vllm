# Implementation Plan: Proxy Vision Support

**Branch**: `002-proxy-vision` | **Date**: 2025-12-03 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-proxy-vision/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Enable multimodal chat interactions in candle-vllm by implementing a proxy vision architecture. The system will accept OpenAI-compatible chat requests containing both text and images, use a separate vision model (Qwen2-VL-7B-Instruct) to generate image captions, and integrate these captions into prompts for the primary text model (Ministral-3-3B-Reasoning-2512). This approach maintains backward compatibility while adding vision capabilities through configuration-driven model orchestration.

## Technical Context

**Language/Version**: Rust 1.75+ (existing candle-vllm codebase)
**Primary Dependencies**: candle-core, candle-nn, tokio, axum, serde, anyhow, tracing
**Storage**: N/A (stateless processing, models loaded from HuggingFace/local paths)
**Testing**: cargo test with integration tests for multimodal API endpoints
**Target Platform**: Linux/macOS servers with CUDA/Metal GPU support
**Project Type**: Library extension (extending existing candle-vllm inference service)
**Performance Goals**: Vision processing adds <10s to response time, supports concurrent text/vision inference
**Constraints**: Must gracefully degrade when vision model unavailable, maintain existing text-only performance
**Scale/Scope**: Support for primary + vision model concurrent loading, OpenAI API compatibility, configurable via models.yaml

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Since the constitution file contains only template placeholders, no specific constitutional violations can be evaluated. The implementation will follow standard Rust/candle-vllm patterns:

✅ **Library-First Approach**: Extending existing candle-vllm-core library with vision capabilities
✅ **Backward Compatibility**: Text-only functionality remains unchanged
✅ **Configuration-Driven**: Using models.yaml for vision model configuration
✅ **Error Handling**: Graceful degradation when vision model unavailable

## Project Structure

### Documentation (this feature)

```text
specs/002-proxy-vision/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
# Extending existing candle-vllm modular architecture
crates/
├── candle-vllm-core/
│   ├── src/
│   │   ├── config.rs           # Extended with VisionMode, VisionProxyConfig
│   │   ├── engine_params.rs    # New: EngineParams struct
│   │   ├── engine_builder_ext.rs # New: build_inference_engine_from_params_async
│   │   ├── models_config.rs    # New: ModelsFile, ModelEntry
│   │   └── openai/
│   │       ├── vision_backend.rs     # New: VisionBackend enum
│   │       ├── image_tool.rs         # New: ImageDescriptionTool trait
│   │       ├── image_tool_local_model.rs # New: LocalVisionModelTool
│   │       ├── vision_proxy.rs       # New: VisionProxyPreprocessor
│   │       └── requests.rs           # Extended: MessageContent, ImageUrl
├── candle-vllm-server/
│   ├── src/
│   │   ├── engine_state.rs     # New: EngineState with vision support
│   │   ├── build_engines.rs    # New: build_engines from ModelsFile
│   │   └── handlers/
│   │       └── chat.rs         # Modified: chat_completions_handler
└── src/
    └── main.rs                 # Modified: startup with vision model loading

tests/
├── integration/
│   ├── multimodal_api_tests.rs     # New: end-to-end vision tests
│   ├── text_compatibility_tests.rs # New: backward compatibility tests
│   └── vision_failure_tests.rs     # New: graceful degradation tests
└── unit/
    ├── vision_preprocessing_tests.rs # New: VisionProxyPreprocessor tests
    ├── image_tool_tests.rs          # New: ImageDescriptionTool tests
    └── models_config_tests.rs       # New: ModelEntry/ModelsFile tests
```

**Structure Decision**: Extending the existing modular candle-vllm architecture by adding vision-specific components to the core library and server crates. This maintains the library-first approach while integrating seamlessly with existing inference and API layers.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No constitutional violations identified. The implementation follows established patterns within the existing candle-vllm architecture.