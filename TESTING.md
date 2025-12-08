# Testing Guide for candle-vllm

This document describes how to run tests for candle-vllm, including unit tests and integration tests that require LLM access.

## Test Structure

Tests are organized by crate:
- `crates/candle-vllm-core/tests/` - Core inference engine tests
- `crates/candle-vllm-openai/tests/` - OpenAI adapter tests
- `crates/candle-vllm-responses/tests/` - MCP and multi-turn conversation tests
- `crates/candle-vllm-server/tests/` - Server integration tests

## Prerequisites

### Model Availability

Most integration tests require a model to be available. You can specify the model path using:

```bash
export CANDLE_VLLM_TEST_MODEL=/path/to/model
```

Or place a model at one of these default locations:
- `./test-models/mistral-7b`
- `./models/mistral-7b`
- `../test-models/mistral-7b`

### Device Selection

By default, tests run on CPU. To use GPU:

```bash
export CANDLE_VLLM_TEST_DEVICE=cuda  # or "metal" for Apple Silicon
```

## Running Tests

### All Tests

```bash
# Run all tests (will skip tests that require models if not available)
cargo test --all --all-features

# Run tests for a specific crate
cargo test -p candle-vllm-core
cargo test -p candle-vllm-openai
cargo test -p candle-vllm-responses
cargo test -p candle-vllm-server
```

### With Model Access

```bash
# Set model path and run tests
export CANDLE_VLLM_TEST_MODEL=/path/to/model
cargo test --all --all-features

# Run specific test
cargo test -p candle-vllm-core test_generation_basic
```

### CPU-Only Tests (No Model Required)

Some tests don't require models and can run without setup:

```bash
# These tests will run even without CANDLE_VLLM_TEST_MODEL set
cargo test -p candle-vllm-core test_model_info
cargo test -p candle-vllm-server test_model_status_structure
```

## Test Categories

### Unit Tests

Tests that don't require model access:
- API structure validation
- Serialization/deserialization
- Error handling
- Configuration validation

### Integration Tests

Tests that require model access:
- Tokenization/detokenization
- Text generation
- Chat completions
- Tool calling
- Multi-turn conversations
- Streaming (when implemented)

## Test Coverage

### Core Inference Engine (`candle-vllm-core`)

- ✅ Engine creation and configuration
- ✅ Tokenization and detokenization
- ✅ Text generation with various parameters
- ✅ Stop sequences
- ✅ Temperature effects
- ✅ Max tokens limits
- ✅ Model information retrieval
- ✅ Generation statistics
- ✅ **Prompt caching (37 tests)**
  - Storage backend tests (memory, sled, redis)
  - Cache manager tests (hashing, prefix matching)
  - Integration tests (end-to-end flows)
  - See [Prompt Cache Testing Documentation](docs/PROMPT_CACHE_TESTING.md) for details

### OpenAI Adapter (`candle-vllm-openai`)

- ✅ Adapter creation
- ✅ Basic chat completions
- ✅ Multi-turn conversations
- ✅ Tool calling support
- ✅ Parameter handling (temperature, top_p, etc.)

### Responses API (`candle-vllm-responses`)

- ✅ Session creation
- ✅ Basic conversations
- ✅ Max turns limit
- ✅ Tool listing
- ✅ MCP server integration (when servers available)

### Server (`candle-vllm-server`)

- ✅ Status structure serialization
- ✅ Prompt cache integration tests (3 tests, require models)
- ⏳ Full HTTP endpoint tests (require running server)

## Writing New Tests

### Test Template

```rust
#[tokio::test]
async fn test_feature_name() {
    skip_if_no_model!();
    
    // Setup
    let engine = test_utils::create_test_engine().await.unwrap();
    
    // Test
    let result = engine.some_method().await;
    
    // Assert
    assert!(result.is_ok());
}
```

### Skipping Tests Without Models

Use the `skip_if_no_model!()` macro to skip tests when no model is available:

```rust
#[tokio::test]
async fn test_requires_model() {
    skip_if_no_model!();
    // Test code here
}
```

## Continuous Integration

In CI environments:

1. **Without models**: Tests that don't require models will run
2. **With models**: Set `CANDLE_VLLM_TEST_MODEL` to run full test suite
3. **GPU tests**: Set `CANDLE_VLLM_TEST_DEVICE` for GPU-specific tests

## Troubleshooting

### Tests Skip Unexpectedly

- Check that `CANDLE_VLLM_TEST_MODEL` is set correctly
- Verify the model path exists and is accessible
- Check test output for skip messages

### Model Loading Fails

- Verify model format is supported
- Check device compatibility (CPU/CUDA/Metal)
- Ensure sufficient memory is available

### Slow Tests

- Integration tests with models can be slow
- Use smaller models for faster tests
- Consider running tests in parallel: `cargo test -- --test-threads=4`

## Prompt Cache Testing

The prompt caching feature has comprehensive test coverage:

- **37 unit tests** covering all caching functionality
- **3 integration tests** for real inference scenarios (require models)
- **Multiple backend tests** (memory, sled, redis)

See [Prompt Cache Testing Documentation](docs/PROMPT_CACHE_TESTING.md) for complete test definitions and coverage details.

### Quick Test Commands

```bash
# All prompt cache tests
cargo test --package candle-vllm-core --lib prompt_cache

# With persistent backends
cargo test --package candle-vllm-core --lib --features prompt-cache prompt_cache

# Integration tests with real inference
cargo test --package candle-vllm-server --test prompt_cache_integration_test -- --ignored
```

## Future Test Enhancements

- [ ] Streaming tests
- [ ] Full HTTP server integration tests
- [ ] MCP server mock for testing
- [ ] Performance benchmarks
- [ ] Memory leak tests
- [ ] Concurrent request tests
- [ ] Prompt cache performance benchmarks
- [ ] Prompt cache stress tests

