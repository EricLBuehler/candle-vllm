# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. When touching configuration or SDK docs, cross-reference `.example.env`, `example.models.yaml`, and the guides in `docs/`.

## Project Overview

Candle-vLLM is a Rust-based efficient inference platform for Large Language Models (LLMs) with OpenAI-compatible API server capabilities. The project has undergone a significant architectural transformation from a monolithic design to a modular, library-first approach.

## Architecture

The codebase follows a three-layer modular architecture:

### Crate Structure
```
candle-vllm/
├── crates/
│   ├── candle-vllm-core/         # Core inference engine
│   ├── candle-vllm-openai/       # OpenAI API compatibility layer
│   ├── candle-vllm-server/       # HTTP server implementation
│   └── candle-vllm-responses/    # MCP (Model Context Protocol) integration
└── src/                          # Main binary entry point
```

### Key Components

1. **Core Inference Engine** (`candle-vllm-core`)
   - Model-agnostic inference orchestration
   - Request scheduling and batching
   - KV cache management with PagedAttention
   - Support for multiple model architectures (Mistral, Llama, Qwen, etc.)

2. **OpenAI Compatibility Layer** (`candle-vllm-openai`)
   - OpenAI-compatible request/response handling
   - Tool calling (function calling) support
   - Conversation management
   - Streaming generation

3. **HTTP Server Layer** (`candle-vllm-server`)
   - Axum-based HTTP server
   - OpenAI API endpoints
   - Built-in ChatGPT-like Web UI
   - Middleware for CORS, logging, etc.

4. **MCP Integration** (`candle-vllm-responses`)
   - Model Context Protocol server integration
   - Multi-server tool orchestration
   - High-level conversation API with automatic tool execution

## Development Commands

### Building the Project

Basic build:
```bash
cargo build --release
```

Platform-specific builds:
```bash
# Mac/Metal (single-node only)
cargo build --release --features metal

# CUDA (single GPU or multi-GPU single machine)
cargo build --release --features cuda,nccl

# CUDA with graph optimization
cargo build --release --features cuda,nccl,graph

# CUDA with flash attention (requires CUDA_ARCH >= 800)
cargo build --release --features cuda,nccl,graph,flash-attn

# Multi-node with MPI
cargo build --release --features cuda,nccl,mpi
```

### Running the Server

Run with uncompressed models:
```bash
# Local model path
target/release/candle-vllm --p 2000 --d 0,1 --w /path/to/model/ --isq q4k --ui-server

# HuggingFace model ID
target/release/candle-vllm --m deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --ui-server
```

Run with GGUF models:
```bash
# Local GGUF file
target/release/candle-vllm --f /path/to/model.gguf --ui-server

# HuggingFace GGUF model
target/release/candle-vllm --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server
```

### Key Parameters

- `--p`: Server port
- `--d`: Device IDs (comma-separated for multi-GPU)
- `--w`: Weight path (safetensors folder)
- `--f`: GGUF file path
- `--m`: HuggingFace model ID
- `--isq`: In-situ quantization (q4_0, q4_1, q5_0, q5_1, q8_0, q2k, q3k, q4k, q5k, q6k)
- `--mem`: KV cache memory in MB (default varies, increase for large batches)
- `--dtype`: Data type (bf16, fp16, fp32)
- `--prefill-chunk-size`: Chunked prefill size (default 8K, 0 to disable)
- `--ui-server`: Enable built-in web UI

### Testing and Development

Run tests:
```bash
cargo test
```

Run with debug logging:
```bash
RUST_LOG=debug cargo run --release --features cuda,nccl -- --log --p 2000
```

## Supported Model Architectures

The system supports multiple model families:
- **LLAMA** (including Llama 3.x variants)
- **Mistral/Ministral** (including Mistral 3, supports nested rope_parameters)
- **Phi** (Phi-3, etc.)
- **QWen2/Qwen3**
- **Yi**
- **StableLM**
- **Gemma-2/Gemma-3**
- **DeepSeek** (R1, V2/V3 variants)
- **QwQ-32B**
- **GLM4**
- **QWen MoE** (QWen2/QWen3 MoE variants)

## Key Features

1. **OpenAI-Compatible API**: Full support for chat completions, streaming, and tool calling (streaming now emits incremental tool-call deltas)
2. **Tool Calling**: Multi-format support (Mistral, Llama, Qwen, generic JSON) with automatic MCP tool injection driven by `mcp.json`
3. **Quantization**: In-situ quantization, GPTQ/Marlin format support
4. **Multi-GPU Support**: Both multi-process and multi-threaded modes
5. **Performance Optimizations**: CUDA graphs, chunked prefill, PagedAttention
6. **Hardware Support**: CUDA, Metal (Apple Silicon), CPU
7. **Batch Processing**: Continuous batching for high throughput
8. **MCP Integration**: Model Context Protocol for tool orchestration

## Model Format Support

- **Safetensors**: Primary format for uncompressed models
- **GGUF**: Quantized format with various precision levels
- **GPTQ/Marlin**: 4-bit quantized format with hardware acceleration
- **AWQ**: After conversion to Marlin-compatible format

## Development Notes

### Important Considerations

1. **Mistral 3/Ministral Models**: Use BF16/FP16 variants only, FP8 models not supported
2. **Flash Attention**: Requires CUDA_ARCH >= 800 for optimal performance
3. **Multi-GPU**: Number of GPUs must align to 2^n (2, 4, 8)
4. **Memory Management**: KV cache memory is critical for batch performance
5. **Quantization**: Marlin format only works on CUDA, provides significant speedups
6. **Configuration**: Keep `.example.env` / `docs/CONFIGURATION.md` in sync when adding env vars or config files.

### Tool Calling Patterns

The system auto-detects multiple tool calling formats:
- Mistral: `[TOOL_CALLS] [{"name": "...", "arguments": {...}}]`
- Llama: `<function=func_name>{"arg": "value"}</function>`
- Qwen: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- Generic JSON: `{"name": "...", "arguments": {...}}`

### Performance Tuning

1. **Batch Size**: Increase `--mem` for larger batches
2. **Chunked Prefill**: Use `--prefill-chunk-size` for long contexts
3. **CUDA Graphs**: Enable with `--graph` feature flag
4. **Quantization**: Use `--isq q4k` for memory-constrained environments
5. **Data Types**: BF16 generally provides best performance/memory balance

## Library Usage

The project is designed as a library-first architecture. Key APIs:

- **InferenceEngine**: Core model inference
- **OpenAIAdapter**: OpenAI compatibility layer
- **ResponsesSession**: High-level MCP integration
- **Tool calling**: Automatic tool execution with MCP servers

See `LIBRARY_API.md` for comprehensive API documentation.

## Multi-Node Deployment

For large models like DeepSeek-R1 (671B), the system supports:
1. Multi-node MPI deployment
2. CPU offloading for experts
3. NUMA binding for optimal performance
4. Network interface configuration

## Troubleshooting

Common issues:
1. **Model Loading**: Ensure correct model format (safetensors for uncompressed)
2. **Memory Issues**: Adjust `--mem` parameter or reduce batch size
3. **Multi-GPU**: Disable P2P with `NCCL_P2P_DISABLE=1` if needed
4. **Performance**: Enable CUDA graph and chunked prefill for optimization