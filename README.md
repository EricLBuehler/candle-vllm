<p align="center">
    <img src="./res/candle_vllm_logo.png" alt="candle vLLM" width=55%/>
</p>

<p align="center">
  <b>Efficient, easy-to-use platform for inference and serving local LLMs including an OpenAI compatible API server.</b><br>
  <a href="./README.md">English</a> | <a href="./README-CN.md">简体中文</a>
</p>

---

## ✨ Why Candle-vLLM?

| | Feature | Details |
|---|---|---|
| **⚡** | High Performance | Native Flash Attention, FlashInfer, CUDA Graphs, continuous batching, prefix caching. |
| **🗜️** | Aggressive KV Compression | TurboQuant (`2–4 bit` KV cache) extends context up to **4.7×** with minimal quality loss |
| **🌍** | Cross-platform | CUDA (Linux), Metal (macOS). Same codebase, same API |
| **🏭** | Production-ready | OpenAI-compatible API server, built-in ChatGPT-style Web UI, MCP tool calling, streaming |
| **📦** | Easy to deploy | One-line install script, Docker images, or build from source |
| **🔧** | Extensible | Trait-based architecture for rapid implementation of new model pipelines |
| **🖥️** | Multi-GPU & Multi-Node | Multi-process and multi-threaded tensor parallelism, TCP-based multi-node inference |

---

## 🚀 Quick Start

### 📦 Install

**Option 1 — One-line install (DEB or binary)**
```bash
curl -sSL https://ericlbuehler.github.io/candle-vllm/install.sh | bash
```

**Option 2 — Build from source**
```bash
git clone git@github.com:EricLBuehler/candle-vllm.git
cd candle-vllm

# CUDA (11+, 12+, 13.0) — remove flashinfer,cutlass for sm_70/sm_75
cargo install --features cuda,nccl,flashinfer,cutlass --path .

# macOS/Metal
cargo install --features metal --path .
```

**Option 3 — Docker**
```bash
# Pass custom SM version and CUDA version: ./build_docker.sh "cuda,nccl,flashinfer,cutlass" sm_90 13.0.0
./build_docker.sh "cuda,nccl,flashinfer,cutlass"
```

---

### ▶️ Run

**Using HuggingFace Model ID:**
```bash
candle-vllm --m Qwen/Qwen3.6-27B-FP8 --ui-server

candle-vllm --m unsloth/Qwen3.5-122B-A10B-GGUF --f Q3_K_S --d 0,1 --ui-server

candle-vllm --m zai-org/GLM-5.2-FP8 --d 0,1,2,3,4,5,6,7 --ui-server
```

**Using local model path:**
```bash
# Local safetensors directory
candle-vllm --d 0,1,2,3,4,5,6,7 --m /home/data/GLM-5.2-FP8/ --ui-server

# Local GGUF file (single or split-shard)
candle-vllm --d 0,1 --m /home/data/model-Q4_K_M.gguf --ui-server

# Local directory containing GGUF files (auto-detected)
candle-vllm --d 0,1 --m /home/data/Qwen3.5-35B-A3B-GGUF/ --ui-server
```

> **Tip:** Add `--ui-server` to launch the built-in ChatGPT-style Web UI. The UI server uses the API port minus one (e.g., API on `2000`, UI on `1999`).

---

## 📈 Performance

> Single-request decode speed (input 4k, output 1k, on `Hopper` 80G)

| # | Model | BF16 (Decode Speed / req) | Quantized |
|---|---|---|---|
| 1 | **LLAMA** | 119 tks/s (8B) | 163 tks/s (8B, Q4K), 171 tks/s (8B, **Marlin**) |
| 2 | **Mistral** | 122 tks/s (7B) | 181 tks/s (7B, Q4K), 190 tks/s (7B, **Marlin**) |
| 3 | **Phi3/Phi4** | 153 tks/s (3.8B) | 196 tks/s (3.8B, Q4K) |
| 4 | **QWen2/Qwen3 Dense** | 127 tks/s (8B) | 154 tks/s **(8B, Q4K)** |
| 5 | **QWen3 MoE** | 102 tks/s **(30B)** | 124 tks/s **(30B, Q4K)** |
| 6 | **QWen3-Next MoE** | 80 tks/s **(80B, BF16, tp=2)** | TBD |
| 7 | **QWen3.5/3.6 Dense** | 36 tks/s **(27B, BF16)** | ~49 tks/s **(27B, Q4K / FP8)** |
| 8 | **QWen3.5/3.6 MoE** | 90 tks/s **(35B)** | 105 tks/s **(35B, Q4K)** |
| 9 | **Yi** | 168 tks/s (6B) | 199 tks/s (6B, Q4K) |
| 10 | **StableLM** | 251 tks/s (3B) | - |
| 11 | **Gemma-2/Gemma-3** | 103 tks/s (9B) | 130 tks/s (9B, **Marlin**) |
| 12 | **DeepSeek V2/V3/V3.2/R1** | TBD | ~20 tks **(AWQ 671B, tp=8, offloading)** |
| 13 | **QwQ-32B** | 51 tks/s **(32B, tp=2)** | 70 tks/s **(32B, Q4K)** |
| 14 | **GLM4** | 96 tks/s **(9B)** | 139 tks/s **(9B, Q4K)** |
| 15 | **GLM4.7 Flash** | TBD | 82 tks/s **(31B, Software NVFP4)** |
| 16 | **LLama4** | TBD | 47 tks/s **(107B, Software NVFP4)** |
| 17 | **Gemma4** | (26B) 83 tks/s | 82 tks/s **(26B, Software NVFP4)** |
| 18 | **MiniMax-M2.5/M2.7** | TBD | 72 tks/s **(229B, Software NVFP4, TP=2)** |
| 19 | **GLM-5.2** | TBD | Supported **(FP8, tp=8)** |

<details>
<summary><b>Demo Video — GPU & Apple Silicon</b></summary>

Chat demo on **GPU** (A100, BF16, QWen3-8B Reasoning Model)
<img src="res/Qwen3-8B-Reasoning-A100.gif" width="85%" height="85%" >

Chat demo on **Apple Silicon** (M4, 16GB unified memory, Q2K, QWen3-8B)
<img src="res/Qwen3-8B-Apple-M4.gif" width="85%" height="85%" >

</details>

---

## 🧠 Features

- OpenAI compatible API server for serving LLMs
- Streaming support in generation
- Efficient KV cache management with PagedAttention
- Continuous batching (batched decoding for incoming requests over time)
- `In-situ` quantization (and `In-situ` Marlin format conversion)
- `GPTQ/Marlin` format quantization (4-bit)
- Support `Mac/Metal` devices
- Support `Multi-GPU` inference (both `multi-process` and `multi-threaded` mode)
- Support `Multi-node` inference via TCP-based coordination
- Support Chunked Prefilling (default chunk size 8K)
- Support CUDA Graph
- Support Model Context Protocol (MCP) and OpenAI-compatible tool calling
- Support Prefix Caching
- Support Block-wise FP8 Models (SM90+, Qwen3 Series)
- Support FP8 KV Cache on all CUDA and Metal platforms
- Support TurboQuant KV Cache (turbo8/turbo4/turbo3) with native flash attention kernels
- Support Flashinfer Backend
- Support manual YaRN RoPE scaling override via `--yarn-scaling-factor`
- Support MXFP4/NVFP4 models
- Support DeepSeek V3.2 and GLM-5.2 FP8 models

---

## 📘 Usage

### Running Models

> **Tip:** By default, candle-vllm starts an OpenAI-compatible API server at `http://localhost:2000`. Add `--ui-server` to also launch the built-in ChatGPT-style Web UI.

```bash
# FP8 model + Web UI
candle-vllm --m Qwen/Qwen3.6-27B-FP8 --ui-server

# GLM-5.2 FP8 model
candle-vllm --d 0,1,2,3,4,5,6,7 --m zai-org/GLM-5.2-FP8 --ui-server

# Unquantized Safetensors (multi-GPU)
candle-vllm --d 0,1 --w /home/Qwen3-30B-A3B-Instruct-2507/

# ISQ on-the-fly quantization
candle-vllm --m Qwen/Qwen3.6-27B --isq q4k

# FP4 Model
candle-vllm --m GadflyII/GLM-4.7-Flash-NVFP4 --ui-server

# GGUF model
candle-vllm --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server

# Manual YaRN scaling
candle-vllm --m Qwen/Qwen3.6-35B-A3B --yarn-scaling-factor 4.0 --ui-server
```

<details open>
<summary><b>FP8 / FP4 models</b></summary>

```bash
# FP8 Model (block-wise quant, build with cutlass feature)
candle-vllm --m Qwen/Qwen3.6-27B-FP8 --ui-server

# Faster GDN prefill on Hopper with slight precision loss
SM90_LOWER_PRECISION_GDN_PREFILL=1 candle-vllm --m Qwen/Qwen3.5-35B-A3B-FP8

# GLM-5.2 FP8 Model
candle-vllm --d 0,1,2,3,4,5,6,7 --m zai-org/GLM-5.2-FP8 --ui-server

# FP8 on MacOS/Metal (Dense)
candle-vllm --m Qwen/Qwen3-4B-Instruct-2507-FP8 --ui-server

# FP4 Model (MXFP4/NVFP4, MLX quantized format not supported)
candle-vllm --m GadflyII/GLM-4.7-Flash-NVFP4 --ui-server

# MXFP4
candle-vllm --m nm-testing/Qwen3-30B-A3B-MXFP4A16 --ui-server
```

</details>

<details>
<summary><b>GGUF models</b></summary>

```bash
# Local GGUF file via --m (recommended)
candle-vllm --m /home/data/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server

# Local GGUF file via --f (legacy)
candle-vllm --f /home/data/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server

# Local directory containing GGUF (auto-detected, mmproj loaded on demand)
candle-vllm --m /home/data/Qwen3.5-35B-A3B-GGUF/ --ui-server

# From HuggingFace (exact file)
candle-vllm --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server

# From HuggingFace (subfolder — downloads all GGUF files in the remote path)
candle-vllm --m unsloth/Qwen3.5-122B-A10B-GGUF --f Q3_K_S --d 0,1 --ui-server

# Local multi-shard GGUF (split files auto-discovered from local path)
candle-vllm --m /home/data/model-00001-of-00003.gguf --d 0,1 --ui-server

# GGUF on Apple Silicon
candle-vllm --m /home/qwq-32b-q4_k_m.gguf --ui-server
candle-vllm --m Qwen/QwQ-32B-GGUF --f qwq-32b-q4_k_m.gguf --ui-server
```

**Multi-shard GGUF:** Split GGUF files (e.g., `model-00001-of-00005.gguf`) are automatically discovered — both locally (from the same directory) and remotely (from the HuggingFace repo). When `--f` is a subfolder name (not ending in `.gguf`), all GGUF files in that remote subfolder are downloaded. Vision tower auxiliary files (`mmproj*.gguf`) are loaded on demand for multimodal models.

</details>

<details>
<summary><b>ISQ (In-situ quantization)</b></summary>

Simply add `--isq` parameter when running unquantized models:

```bash
candle-vllm --m Qwen/Qwen3.6-27B --isq q4k
```

Options: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2k`, `q3k`, `q4k`, `q5k`, `q6k`

</details>

<details>
<summary><b>GPTQ / AWQ / Marlin models</b></summary>

```bash
# Marlin-compatible GPTQ (4-bit, 128-group, desc_act=False)
candle-vllm --m thesven/Llama-3-8B-GPTQ-4bit

# Convert uncompressed model to Marlin-compatible format
python3 examples/convert_marlin.py --src /home/DeepSeek-R1-Distill-Qwen-14B/ --dst /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g
candle-vllm --w /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g

# Convert AWQ to Marlin-compatible format
python3 examples/convert_awq_marlin.py --src /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --dst /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ --bits 4 --method awq --group 128 --nk False
candle-vllm --d 0 --w /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/

# Direct Marlin-format model
candle-vllm --w /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ-Marlin/
```

</details>

---

### 🗜️ TurboQuant KV Cache

TurboQuant compresses the KV cache using Walsh-Hadamard transform for higher throughput and longer context:

| Mode | Description | KV Cache Compression | Recommended Use |
|------|-------------|---------------------|-----------------|
| `turbo8` | FP8 K + 4-bit V | ~2.6x | Best quality-compression trade-off |
| `turbo4` | 4-bit K + 4-bit V | ~3.7x | Balanced quality and memory savings |
| `turbo3` | 3-bit K + 4-bit V | ~4.7x | Maximum memory savings |

```bash
# Turbo4 (4-bit KV cache, ~3.7x compression)
candle-vllm --w /data/Qwen3.5-27B-FP8/ --kvcache-dtype turbo4

# Turbo8 (FP8 K + 4-bit V, ~2.6x compression)
candle-vllm --w /data/Qwen3.5-27B-FP8/ --kvcache-dtype turbo8

# Turbo3 (3-bit K + 4-bit V, ~4.7x compression)
candle-vllm --w /data/Qwen3.5-27B-FP8/ --kvcache-dtype turbo3

# FP8 KV Cache
candle-vllm --w /data/Qwen3.5-35B-A3B-FP8/ --kvcache-dtype fp8
```

> **Note**: TurboQuant uses native flash attention kernels (flashinfer is automatically disabled). Supported on both CUDA (SM70+) and Metal (Apple Silicon) platforms. MLA models (DeepSeek, GLM4/GLM-5.2) auto-fallback to standard KV cache as TurboQuant is incompatible with their compressed KV layout.

---

### 🖥️ Multi-GPU Inference

<details>
<summary><b>Multi-process mode (recommended)</b></summary>

```bash
# QwQ-32B BF16 on two GPUs
candle-vllm --d 0,1 --w /home/QwQ-32B/

# QwQ-32B 4-bit AWQ on two GPUs
python3 examples/convert_awq_marlin.py --src /home/QwQ-32B-AWQ/ --dst /home/QwQ-32B-AWQ-Marlin/ --bits 4 --method awq --group 128 --nk False
candle-vllm --d 0,1 --w /home/QwQ-32B-AWQ-Marlin/
```

**Note:** Number of GPUs (`--d`) must be a power of 2 (e.g., 2, 4, or 8).

</details>

<details>
<summary><b>Multi-threaded mode (debug)</b></summary>

```bash
# Add --multithread parameter
candle-vllm --multithread --d 0,1 --w /home/QwQ-32B/

# Troubleshooting
export NCCL_P2P_DISABLE=1  # disable P2P if encountering illegal memory access
```

</details>

---

### 🌐 Multi-Node Inference

Distribute inference across multiple machines using TCP-based NCCL bootstrap. No MPI required.

```bash
# On master node (192.168.1.100):
candle-vllm --d 0,1,2,3,4,5,6,7 --w /data/DeepSeek-R1-AWQ-Marlin/ \
  --num-nodes 2 --node-rank 0 --master-addr 192.168.1.100 --master-port 29500

# On worker node (192.168.1.101):
candle-vllm --d 0,1,2,3,4,5,6,7 --w /data/DeepSeek-R1-AWQ-Marlin/ \
  --num-nodes 2 --node-rank 1 --master-addr 192.168.1.100 --master-port 29500
```

All nodes must have model weights locally and be TCP-reachable on `--master-port` (default 29500).

| Flag | Description |
|------|-------------|
| `--num-nodes N` | Total number of nodes in the cluster |
| `--node-rank R` | This node's rank (0 = master) |
| `--master-addr ADDR` | IP address of the master node |
| `--master-port PORT` | Port for NCCL ID exchange (default: 29500) |

---

### 📐 NUMA Binding

<details>
<summary><b>Show command</b></summary>

```bash
sudo apt-get install numactl

# 8 GPUs, 2 NUMA nodes
MAP_NUMA_NODE=0,0,0,0,1,1,1,1 numactl --cpunodebind=0 --membind=0 candle-vllm --d 0,1,2,3,4,5,6,7 --w /home/data/DeepSeek-V2-Chat-AWQ-Marlin

# 4 GPUs
MAP_NUMA_NODE=0,0,0,0 numactl --cpunodebind=0 --membind=0 candle-vllm --d 0,1,2,3 --w /home/data/DeepSeek-V2-Chat-AWQ-Marlin
```

`numactl --cpunodebind=0 --membind=0` specifies the master rank's NUMA binding and must match `MAP_NUMA_NODE`.

</details>

---

## ⚙️ CLI Reference

| Flag | Description |
|------|-------------|
| `--h` | Bind address (default `0.0.0.0`). Supports `host`, `host:port`, `[ipv6]:port`, `tcp://host[:port]`, `file:///path`, `socket:///path`, `unix:///path` |
| `--p` | TCP server port when `--h` does not include a port (default `2000`) |
| `--d` | Device IDs (e.g. `--d 0,1`) |
| `--m` | Model source: HuggingFace model ID, local directory, or local `.gguf` file. Auto-detects GGUF vs safetensors in directories |
| `--w` | Local weight directory (safetensors or GGUF). Prefer `--m <local_dir>` for new commands |
| `--f` | GGUF file or subfolder: `--m repo --f file.gguf` (exact file), `--m repo --f subfolder` (all GGUFs in path), or local GGUF path |
| `--dtype` | Data type (`bf16`, `f16`) |
| `--isq` | In-situ quantization: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2k`, `q3k`, `q4k`, `q5k`, `q6k` |
| `--kvcache-dtype` | KV cache quantization: `auto`, `fp8`, `turbo8`, `turbo4`, `turbo3` |
| `--kv-fraction` | Auto-size KV cache as fraction of remaining GPU memory (default `0.6`) |
| `--mem` | Fixed KV cache budget in MB |
| `--prefill-chunk-size` | Prefill chunk size (default 8K, `0` to disable) |
| `--max-gen-tokens` | Max output tokens per response (default: 1/5 of max_sequence_len) |
| `--frequency-penalty` | Frequency penalty (−2.0 to 2.0) |
| `--presence-penalty` | Presence penalty (−2.0 to 2.0) |
| `--yarn-scaling-factor` | YaRN RoPE context extension factor |
| `--enforce-parser` | Force tool parser backend: `qwen_coder`, `qwen`, `json`, `mistral` |
| `--ui-server` | Start with built-in ChatGPT-like Web UI |
| `--multithread` | Use multi-threaded mode (debug) |
| `--num-nodes` | Total nodes in cluster (multi-node) |
| `--node-rank` | This node's rank (0 = master) |
| `--master-addr` | Master node IP address |
| `--master-port` | NCCL ID exchange port (default `29500`) |
| `--disable-prefix-cache` | Disable prefix caching (enabled by default) |
| `--prefix-cache-max-tokens` | Cap prefix cache size |
| `--disable-cuda-graph` | Disable CUDA graph capture (enabled by default on CUDA) |

**Binding examples:**
```bash
candle-vllm --h 127.0.0.1 --p 8000 --m Qwen/Qwen3.6-27B-FP8
candle-vllm --h 127.0.0.1:8000 --m Qwen/Qwen3.6-27B-FP8
candle-vllm --h '[::1]:8000' --m Qwen/Qwen3.6-27B-FP8
candle-vllm --h unix:///tmp/candle-vllm.sock --m Qwen/Qwen3.6-27B-FP8
```

---

## 📚 Documentation

| Guide | Description |
|---|---|
| [Rust Crate Usage](docs/rust_crate.md) | Use as a Rust library |
| [Embedding Models](docs/embedding.md) | Text embedding API |
| [MCP & Tool Calling](docs/mcp_tool_calling.md) | Model Context Protocol integration |
| [Tool Call Parsing](docs/tool_parsing.md) | Tool call detection and parsing |
| [Prefix Cache](docs/prefix_cache.md) | Automatic KV cache reuse |
| [Multimodal Models](docs/multimodal.md) | Vision-language models |

**Using Agents under Candle-vLLM backend:** [xbot](docs/xbot.md) · [OpenCode](docs/opencode.md) · [Kilo Code](docs/kilocode.md)

---

## 🛠️ Roadmap

* [x] OpenAI-compatible API server (streaming)
* [x] Continuous batching
* [x] Flash Attention (CUDA)
* [x] FlashInfer backend
* [x] CUDA Graph
* [x] Chunked Prefill
* [x] Prefix Caching (CUDA & Metal)
* [x] Multi-GPU inference (multi-process & multi-threaded)
* [x] Multi-node tensor parallelism (TCP-based NCCL, no MPI)
* [x] In-situ quantization (GGML/GGUF + Marlin)
* [x] FP8 KV Cache (CUDA & Metal, all backends)
* [x] TurboQuant KV Cache (2–4 bit compression)
* [x] FP8 Models (block-wise, SM90+)
* [x] MXFP4/NVFP4 Model Support
* [x] DeepSeek V3.2 and GLM-5.2 FP8 Model Support
* [x] MCP Integration & Tool Calling
* [x] Built-in ChatGPT-style Web UI

---

## 📚 References

- Python implementation: [`vllm-project`](https://github.com/vllm-project/vllm)
- [`vllm` paper](https://arxiv.org/abs/2309.06180)

## Report Issue

If you encounter any problems, please create an [issue](https://github.com/EricLBuehler/candle-vllm/issues).
