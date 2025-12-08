# Quickstart: Proxy Vision Support

**Feature**: Proxy Vision Support for candle-vllm
**Date**: 2025-12-03
**Prerequisites**: Existing candle-vllm installation

## Overview

This quickstart guide shows how to enable and use multimodal (text + image) chat completions in candle-vllm using the proxy vision architecture. The system uses a separate vision model to analyze images and integrates the results with your primary text model.

## Quick Setup

### 1. Configure Models

Create or update your `models.yaml` file:

```yaml
models:
  # Primary text model (required)
  - name: "ministral-3-3b-reasoning"
    hf_id: "mistralai/Ministral-3-3B-Reasoning-2512"
    params:
      mem: 2048  # Memory in MB
      dtype: "bf16"
      max_num_seqs: 16
    capabilities:
      vision_mode: "disabled"  # Text-only model

  # Vision model (optional, enables multimodal support)
  - name: "qwen2-vl-7b"
    hf_id: "Qwen/Qwen2-VL-7B-Instruct"
    params:
      mem: 8192  # Vision models need more memory
      dtype: "bf16"
      max_num_seqs: 4
    capabilities:
      vision_mode: "proxy"
      vision_proxy:
        hf_id: "Qwen/Qwen2-VL-7B-Instruct"
        prompt_template: "Describe this image in detail:"

# Global settings
idle_unload_secs: 300
```

### 2. Start the Server

```bash
# Start with vision support enabled
./candle-vllm --models-config models.yaml --port 2000 --ui-server

# Or with specific models from command line (fallback)
./candle-vllm --m mistralai/Ministral-3-3B-Reasoning-2512 \
              --vision-model Qwen/Qwen2-VL-7B-Instruct \
              --port 2000 --ui-server
```

### 3. Test Text-Only (Baseline)

```bash
curl -X POST http://localhost:2000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ministral-3-3b-reasoning",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

### 4. Test Multimodal (Vision)

```bash
curl -X POST http://localhost:2000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ministral-3-3b-reasoning",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What do you see in this image?"},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/image.jpg",
              "detail": "high"
            }
          }
        ]
      }
    ],
    "max_tokens": 200
  }'
```

## Usage Examples

### Single Image Analysis

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:2000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="ministral-3-3b-reasoning",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/photo.jpg"}
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### Multiple Images Comparison

```python
response = client.chat.completions.create(
    model="ministral-3-3b-reasoning",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image1.jpg", "detail": "high"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image2.jpg", "detail": "high"}
                }
            ]
        }
    ]
)
```

### Base64 Image Data

```python
import base64

# Read and encode image
with open("local_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = client.chat.completions.create(
    model="ministral-3-3b-reasoning",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]
)
```

### Streaming Multimodal Response

```python
stream = client.chat.completions.create(
    model="ministral-3-3b-reasoning",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

## Configuration Options

### Vision Model Settings

```yaml
capabilities:
  vision_mode: "proxy"        # "disabled", "proxy", or "native"
  vision_proxy:
    hf_id: "Qwen/Qwen2-VL-7B-Instruct"
    prompt_template: "Describe this image in detail:"  # Optional
```

### Performance Tuning

```yaml
params:
  mem: 8192                   # Vision models need more GPU memory
  max_num_seqs: 4            # Lower batch size for vision models
  dtype: "bf16"              # Use BF16 for best performance
  prefill_chunk_size: 4096   # Adjust based on GPU memory
```

### Resource Management

```yaml
# Global resource limits
idle_unload_secs: 300        # Unload unused models after 5 minutes
```

## Troubleshooting

### Vision Model Not Loading

**Symptoms**: "Vision model unavailable" errors
**Solution**:
1. Check GPU memory: `nvidia-smi` or system memory
2. Reduce vision model memory: `mem: 4096` instead of `mem: 8192`
3. Use CPU offloading: Add `--device cpu` flag

### Image Download Failures

**Symptoms**: "Image download timeout" or "Image URL not accessible"
**Solution**:
1. Verify image URL is publicly accessible
2. Check network connectivity from server
3. Use base64 data URLs for local images

### Out of Memory Errors

**Symptoms**: CUDA OOM or system memory errors
**Solution**:
1. Reduce total memory allocation in `models.yaml`
2. Lower `max_num_seqs` for both models
3. Use smaller vision model variants

### Performance Issues

**Symptoms**: Slow response times (>30 seconds)
**Solution**:
1. Enable GPU acceleration: `--features cuda`
2. Use smaller image detail: `"detail": "low"`
3. Optimize `prefill_chunk_size` parameter

## Monitoring and Logs

### Enable Detailed Logging

```bash
RUST_LOG=info ./candle-vllm --models-config models.yaml --log
```

### Key Log Messages

- Model loading: `Building InferenceEngine for model`
- Vision availability: `Vision backend: Available` or `Unavailable`
- Request processing: `Processing multimodal request with N images`
- Performance: `Vision processing completed in Xms`

## API Compatibility

The implementation maintains 100% compatibility with OpenAI's Chat Completions API:

- ✅ Legacy text-only requests work unchanged
- ✅ New multimodal format supported
- ✅ Streaming responses work with images
- ✅ Error responses follow OpenAI format
- ✅ All existing SDKs and tools compatible

## Next Steps

1. **Production Deployment**: Configure load balancing and monitoring
2. **Custom Vision Models**: Swap in different vision models via configuration
3. **Performance Optimization**: Tune memory and batch settings for your workload
4. **Integration**: Connect with existing applications using OpenAI SDK

For detailed implementation information, see the [technical specification](./spec.md) and [API contracts](./contracts/openapi.yaml).