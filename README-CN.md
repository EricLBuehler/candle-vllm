<p align="center">
    <img src="./res/candle_vllm_logo.png" alt="candle vLLM" width=55%/>
</p>

<p align="center">
  <b>高效、易用的本地大语言模型（LLM）推理与服务平台，提供 OpenAI 兼容的 API 服务。</b><br>
  <a href="./README.md">English</a> | <a href="./README-CN.md">简体中文</a>
</p>

---

## ✨ 为什么选择 Candle-vLLM？

| | 特性 | 详情 |
|---|---|---|
| **⚡** | 极致性能 | 原生 Flash Attention、FlashInfer、CUDA Graphs、持续批处理、前缀缓存。|
| **🗜️** | 极致 KV 压缩 | TurboQuant（`2–4 位` KV 缓存）以极小的质量损失将上下文扩展至 **4.7 倍** |
| **🌍** | 跨平台 | CUDA（Linux）、Metal（macOS），统一代码库，统一 API |
| **🏭** | 生产就绪 | OpenAI 兼容 API 服务、内置 ChatGPT 风格 Web UI、MCP 工具调用、流式输出 |
| **📦** | 部署简单 | 一键安装脚本、Docker 镜像或源码编译 |
| **🔧** | 高度可扩展 | 基于 trait 的架构，支持快速实现新的模型 |
| **🖥️** | 多 GPU & 多节点 | 多进程及多线程张量并行，TCP 多节点推理 |

---

## 🚀 快速开始

### 📦 安装

**方案 1 — 一键安装（DEB 或二进制）**
```bash
curl -sSL https://ericlbuehler.github.io/candle-vllm/install.sh | bash
```

**方案 2 — 从源代码构建**
```bash
git clone git@github.com:EricLBuehler/candle-vllm.git
cd candle-vllm

# CUDA（11+, 12+, 13.0）— sm_70/sm_75 需去除 flashinfer,cutlass
cargo install --features cuda,nccl,flashinfer,cutlass --path .

# macOS/Metal
cargo install --features metal --path .
```

**方案 3 — Docker**
```bash
# 可指定 SM 版本和 CUDA 版本：./build_docker.sh "cuda,nccl,flashinfer,cutlass" sm_90 13.0.0
./build_docker.sh "cuda,nccl,flashinfer,cutlass"
```

---

### ▶️ 运行

**使用 HuggingFace 模型 ID：**
```bash
candle-vllm --m Qwen/Qwen3.6-27B-FP8 --ui-server

candle-vllm --m unsloth/Qwen3.5-122B-A10B-GGUF --f Q3_K_S --d 0,1 --ui-server

candle-vllm --m zai-org/GLM-5.2-FP8 --d 0,1,2,3,4,5,6,7 --ui-server
```

**使用本地模型路径：**
```bash
# 本地 Safetensors 目录
candle-vllm --d 0,1,2,3,4,5,6,7 --m /home/data/GLM-5.2-FP8/ --ui-server

# 本地 GGUF 文件（单文件或分片）
candle-vllm --d 0,1 --m /home/data/model-Q4_K_M.gguf --ui-server

# 本地目录（自动检测 GGUF 文件）
candle-vllm --d 0,1 --m /home/data/Qwen3.5-35B-A3B-GGUF/ --ui-server
```

> **提示：** 添加 `--ui-server` 可启动内置 ChatGPT 风格 Web UI。UI 服务端口为 API 端口减一（例如 API 为 `2000`，UI 为 `1999`）。

---

## 📈 性能

> 单请求解码速度（输入 4k，输出 1k，`Hopper` 80G）

| # | 模型 | BF16（解码速度 / 单请求） | 量化 |
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
| 19 | **GLM-5.2** | TBD | 支持 **(FP8, tp=8)** |

<details>
<summary><b>演示视频 — GPU 与 Apple Silicon</b></summary>

**GPU**（A100, BF16, QWen3-8B 推理模型）上的聊天演示
<img src="res/Qwen3-8B-Reasoning-A100.gif" width="85%" height="85%" >

**Apple Silicon**（M4, 16GB 统一内存, Q2K, QWen3-8B）上的聊天演示
<img src="res/Qwen3-8B-Apple-M4.gif" width="85%" height="85%" >

</details>

---

## 🧠 功能特性

- 提供 OpenAI 兼容的 API 服务，用于部署 LLM
- 生成过程中支持流式（stream）传输
- 使用 PagedAttention 高效管理 KV 缓存
- 持续批处理（continuous batching，不同时间段的请求 decoding 阶段聚合为批量处理）
- 原位（In-situ）量化（及原位 Marlin 格式转换）
- 支持 `GPTQ/Marlin` 格式量化（4 位）
- 支持 `Mac/Metal` 设备
- 支持 `多 GPU` 推理（包括 `多进程` 和 `多线程` 模式）
- 支持 `多节点` 推理（基于 TCP 协调）
- 支持分块 Prefilling（默认块大小 8K）
- 支持 CUDA Graph
- 支持 Model Context Protocol（MCP）和 OpenAI 兼容工具调用
- 支持 Prefix Caching
- 支持硬件 FP8 模型推理加速（SM90+, Qwen3 系列，Block-wise FP8 量化）
- 支持 FP8 KV Cache（兼容 FlashInfer、FlashAttention 及 Prefix Cache，适用于所有 CUDA 和 Metal 平台）
- 支持 TurboQuant KV Cache（turbo8/turbo4/turbo3），使用原生 Flash 注意力内核实现高压缩比 KV 缓存
- 支持 Flashinfer 后端
- 支持通过命令行参数 `--yarn-scaling-factor` 手动设置 YaRN RoPE 缩放因子
- 支持 MXFP4/NVFP4 模型
- 支持 DeepSeek V3.2 和 GLM-5.2 FP8 模型

---

## 📘 使用方法

### 运行模型

> **提示：** 默认启动 OpenAI 兼容 API 服务（`http://localhost:2000`）。添加 `--ui-server` 可同时启动内置 ChatGPT 风格 Web UI。

```bash
# FP8 模型 + Web UI
candle-vllm --m Qwen/Qwen3.6-27B-FP8 --ui-server

# GLM-5.2 FP8 模型
candle-vllm --d 0,1,2,3,4,5,6,7 --m zai-org/GLM-5.2-FP8 --ui-server

# 未量化 Safetensors（多 GPU）
candle-vllm --d 0,1 --w /home/Qwen3-30B-A3B-Instruct-2507/

# ISQ 即时量化
candle-vllm --m Qwen/Qwen3.6-27B --isq q4k

# FP4 模型
candle-vllm --m GadflyII/GLM-4.7-Flash-NVFP4 --ui-server

# GGUF 模型
candle-vllm --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server

# 手动 YaRN 缩放
candle-vllm --m Qwen/Qwen3.6-35B-A3B --yarn-scaling-factor 4.0 --ui-server
```

<details open>
<summary><b>FP8 / FP4 模型</b></summary>

```bash
# FP8 模型（block-wise 量化，需启用 cutlass 特性）
candle-vllm --m Qwen/Qwen3.6-27B-FP8 --ui-server

# GLM-5.2 FP8 模型
candle-vllm --d 0,1,2,3,4,5,6,7 --m zai-org/GLM-5.2-FP8 --ui-server

# MacOS/Metal 上的 FP8（Dense）
candle-vllm --m Qwen/Qwen3-4B-Instruct-2507-FP8 --ui-server

# FP4 模型（MXFP4/NVFP4，暂不支持 MLX 量化格式）
candle-vllm --m GadflyII/GLM-4.7-Flash-NVFP4 --ui-server

# MXFP4
candle-vllm --m nm-testing/Qwen3-30B-A3B-MXFP4A16 --ui-server
```

</details>

<details>
<summary><b>GGUF 模型</b></summary>

```bash
# 本地 GGUF 文件（推荐使用 --m）
candle-vllm --m /home/data/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server

# 本地 GGUF 文件（--f 传统方式）
candle-vllm --f /home/data/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server

# 本地目录（自动检测 GGUF，按需加载 mmproj 视觉文件）
candle-vllm --m /home/data/Qwen3.5-35B-A3B-GGUF/ --ui-server

# 从 HuggingFace 下载（精确文件）
candle-vllm --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server

# 从 HuggingFace 下载（子文件夹 — 下载远程路径下所有 GGUF 文件）
candle-vllm --m unsloth/Qwen3.5-122B-A10B-GGUF --f Q3_K_S --d 0,1 --ui-server

# 本地多分片 GGUF（从本地路径自动发现分片文件）
candle-vllm --m /home/data/model-00001-of-00003.gguf --d 0,1 --ui-server

# Apple Silicon 上的 GGUF
candle-vllm --m /home/qwq-32b-q4_k_m.gguf --ui-server
candle-vllm --m Qwen/QwQ-32B-GGUF --f qwq-32b-q4_k_m.gguf --ui-server
```

**多分片 GGUF：** 分片 GGUF 文件（如 `model-00001-of-00005.gguf`）支持自动发现 — 本地（从同一目录）和远程（从 HuggingFace 仓库）均可。当 `--f` 为子文件夹名（不以 `.gguf` 结尾）时，会下载该远程子文件夹中的所有 GGUF 文件。视觉塔辅助文件（`mmproj*.gguf`）在多模态模型中按需加载。

</details>

<details>
<summary><b>ISQ 原位量化</b></summary>

运行未量化模型时只需添加 `--isq` 参数：

```bash
candle-vllm --m Qwen/Qwen3.6-27B --isq q4k
```

可选值：`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2k`, `q3k`, `q4k`, `q5k`, `q6k`

</details>

<details>
<summary><b>GPTQ / AWQ / Marlin 模型</b></summary>

```bash
# Marlin 兼容的 GPTQ（4 位，128 分组，desc_act=False）
candle-vllm --m thesven/Llama-3-8B-GPTQ-4bit

# 将未压缩模型转换为 Marlin 兼容格式
python3 examples/convert_marlin.py --src /home/DeepSeek-R1-Distill-Qwen-14B/ --dst /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g
candle-vllm --w /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g

# 将 AWQ 转换为 Marlin 兼容格式
python3 examples/convert_awq_marlin.py --src /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --dst /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ --bits 4 --method awq --group 128 --nk False
candle-vllm --d 0 --w /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/

# 直接使用 Marlin 格式模型
candle-vllm --w /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ-Marlin/
```

</details>

---

### 🗜️ TurboQuant KV 缓存

TurboQuant 通过 Walsh-Hadamard 变换压缩 KV 缓存，实现更高吞吐量和更长上下文：

| 模式 | 描述 | KV 缓存压缩比 | 推荐用途 |
|------|------|--------------|---------|
| `turbo8` | FP8 K + 4-bit V | ~2.6x | 最佳质量-压缩比平衡 |
| `turbo4` | 4-bit K + 4-bit V | ~3.7x | 质量与显存节省兼顾 |
| `turbo3` | 3-bit K + 4-bit V | ~4.7x | 最大限度节省显存 |

```bash
# Turbo4（4-bit KV 缓存，约 3.7 倍压缩）
candle-vllm --w /data/Qwen3.5-27B-FP8/ --kvcache-dtype turbo4

# Turbo8（FP8 K + 4-bit V，约 2.6 倍压缩）
candle-vllm --w /data/Qwen3.5-27B-FP8/ --kvcache-dtype turbo8

# Turbo3（3-bit K + 4-bit V，约 4.7 倍压缩）
candle-vllm --w /data/Qwen3.5-27B-FP8/ --kvcache-dtype turbo3

# FP8 KV Cache
candle-vllm --w /data/Qwen3.5-35B-A3B-FP8/ --kvcache-dtype fp8
```

> **注意**：TurboQuant 使用原生 Flash 注意力内核（flashinfer 自动禁用）。支持 CUDA（SM70+）和 Metal（Apple Silicon）两种平台。MLA 模型（DeepSeek、GLM4/GLM-5.2）因 KV 压缩布局不兼容会自动回退到标准 KV 缓存。

---

### 🖥️ 多 GPU 推理

<details>
<summary><b>多进程模式（推荐）</b></summary>

```bash
# 两块 GPU 运行 QwQ-32B BF16
candle-vllm --d 0,1 --w /home/QwQ-32B/

# 两块 GPU 运行 QwQ-32B 4 位 AWQ
python3 examples/convert_awq_marlin.py --src /home/QwQ-32B-AWQ/ --dst /home/QwQ-32B-AWQ-Marlin/ --bits 4 --method awq --group 128 --nk False
candle-vllm --d 0,1 --w /home/QwQ-32B-AWQ-Marlin/
```

**注意：** GPU 数量（`--d`）必须为 2 的幂次方（例如 2、4 或 8）。

</details>

<details>
<summary><b>多线程模式（调试用途）</b></summary>

```bash
# 添加 --multithread 参数
candle-vllm --multithread --d 0,1 --w /home/QwQ-32B/

# 问题排查
export NCCL_P2P_DISABLE=1  # 禁用 P2P 以避免非法内存访问
```

</details>

---

### 🌐 多节点推理

跨多台机器分布式推理，基于 TCP 的 NCCL 引导，无需 MPI。

```bash
# 在主节点 (192.168.1.100) 上运行：
candle-vllm --d 0,1,2,3,4,5,6,7 --w /data/DeepSeek-R1-AWQ-Marlin/ \
  --num-nodes 2 --node-rank 0 --master-addr 192.168.1.100 --master-port 29500

# 在工作节点 (192.168.1.101) 上运行：
candle-vllm --d 0,1,2,3,4,5,6,7 --w /data/DeepSeek-R1-AWQ-Marlin/ \
  --num-nodes 2 --node-rank 1 --master-addr 192.168.1.100 --master-port 29500
```

所有节点需本地存放模型权重，并通过 TCP 连接 `--master-port`（默认 29500）。

| 参数 | 说明 |
|------|------|
| `--num-nodes N` | 集群中的节点总数 |
| `--node-rank R` | 本节点的排名（0 = 主节点） |
| `--master-addr ADDR` | 主节点的 IP 地址 |
| `--master-port PORT` | NCCL ID 交换端口（默认：29500） |

---

### 📐 NUMA 绑定

<details>
<summary><b>显示命令</b></summary>

```bash
sudo apt-get install numactl

# 8 张 GPU，2 个 NUMA 节点
MAP_NUMA_NODE=0,0,0,0,1,1,1,1 numactl --cpunodebind=0 --membind=0 candle-vllm --d 0,1,2,3,4,5,6,7 --w /home/data/DeepSeek-V2-Chat-AWQ-Marlin

# 4 张 GPU
MAP_NUMA_NODE=0,0,0,0 numactl --cpunodebind=0 --membind=0 candle-vllm --d 0,1,2,3 --w /home/data/DeepSeek-V2-Chat-AWQ-Marlin
```

`numactl --cpunodebind=0 --membind=0` 指定 master 进程的 NUMA 绑定，必须与 `MAP_NUMA_NODE` 相匹配。

</details>

---

## ⚙️ 命令行参数

| 参数 | 说明 |
|------|------|
| `--h` | 绑定地址（默认 `0.0.0.0`），支持 `host`、`host:port`、`[ipv6]:port`、`tcp://host[:port]`、`file:///path`、`socket:///path`、`unix:///path` |
| `--p` | 当 `--h` 未包含端口时使用的 TCP 服务端口（默认 `2000`） |
| `--d` | 设备 ID（如 `--d 0,1`） |
| `--m` | 模型来源：HuggingFace 模型 ID、本地目录或本地 `.gguf` 文件。目录中自动检测 GGUF 或 Safetensors |
| `--w` | 本地权重目录（Safetensors 或 GGUF）。新命令建议使用 `--m <本地目录>` |
| `--f` | GGUF 文件或子文件夹：`--m 仓库 --f 文件.gguf`（精确文件），`--m 仓库 --f 子文件夹`（下载路径下所有 GGUF），或本地 GGUF 路径 |
| `--dtype` | 数据类型（`bf16`, `f16`） |
| `--isq` | 原位量化：`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2k`, `q3k`, `q4k`, `q5k`, `q6k` |
| `--kvcache-dtype` | KV 缓存量化：`auto`, `fp8`, `turbo8`, `turbo4`, `turbo3` |
| `--kv-fraction` | 模型加载后按剩余显存比例自动计算 KV 缓存大小（默认 `0.6`） |
| `--mem` | 固定 KV 缓存预算（MB） |
| `--prefill-chunk-size` | 预填充分块大小（默认 8K，`0` 为禁用） |
| `--max-gen-tokens` | 每次响应最大输出 token 数（默认：max_sequence_len 的 1/5） |
| `--frequency-penalty` | 频率惩罚（−2.0 到 2.0） |
| `--presence-penalty` | 存在惩罚（−2.0 到 2.0） |
| `--yarn-scaling-factor` | YaRN RoPE 上下文扩展因子 |
| `--enforce-parser` | 强制指定工具解析器后端：`qwen_coder`, `qwen`, `json`, `mistral` |
| `--ui-server` | 启动内置 ChatGPT 风格 Web UI |
| `--multithread` | 使用多线程模式（调试用途） |
| `--num-nodes` | 集群中节点总数（多节点推理） |
| `--node-rank` | 本节点排名（0 = 主节点） |
| `--master-addr` | 主节点 IP 地址 |
| `--master-port` | NCCL ID 交换端口（默认 `29500`） |
| `--disable-prefix-cache` | 禁用前缀缓存（默认开启） |
| `--prefix-cache-max-tokens` | 前缀缓存大小上限 |
| `--disable-cuda-graph` | 禁用 CUDA Graph 捕获（CUDA 构建默认开启） |

**绑定示例：**
```bash
candle-vllm --h 127.0.0.1 --p 8000 --m Qwen/Qwen3.6-27B-FP8
candle-vllm --h 127.0.0.1:8000 --m Qwen/Qwen3.6-27B-FP8
candle-vllm --h '[::1]:8000' --m Qwen/Qwen3.6-27B-FP8
candle-vllm --h unix:///tmp/candle-vllm.sock --m Qwen/Qwen3.6-27B-FP8
```

---

## 📚 文档

| 指南 | 说明 |
|---|---|
| [Rust Crate 用法](docs/rust_crate.md) | 作为 Rust 库使用 |
| [Embedding 模型](docs/embedding.md) | 文本嵌入 API |
| [MCP & 工具调用](docs/mcp_tool_calling.md) | Model Context Protocol 集成 |
| [工具调用解析](docs/tool_parsing.md) | 工具调用检测与解析 |
| [Prefix Cache](docs/prefix_cache.md) | 自动 KV 缓存复用 |
| [多模态模型](docs/multimodal.md) | 视觉语言模型 |

**在 Candle-vLLM 后端下使用 Agent：** [xbot](docs/xbot.md) · [OpenCode](docs/opencode.md) · [Kilo Code](docs/kilocode.md)

---

## 🛠️ 开发计划

* [x] OpenAI 兼容 API 服务器（流式输出）
* [x] 持续批处理
* [x] Flash Attention（CUDA）
* [x] FlashInfer 后端
* [x] CUDA Graph
* [x] 分块预填充
* [x] 前缀缓存（CUDA 和 Metal）
* [x] 多 GPU 推理（多进程及多线程）
* [x] 多节点张量并行推理（基于 TCP 的 NCCL，无需 MPI）
* [x] 原位量化（GGML/GGUF + Marlin）
* [x] FP8 KV Cache（CUDA 和 Metal，所有后端）
* [x] TurboQuant KV Cache（2–4 位压缩）
* [x] FP8 模型（Block-wise，SM90+）
* [x] MXFP4/NVFP4 模型支持
* [x] DeepSeek V3.2 和 GLM-5.2 FP8 模型支持
* [x] MCP 集成与工具调用
* [x] 内置 ChatGPT 风格 Web UI

---

## 📚 参考

- Python 实现：[`vllm-project`](https://github.com/vllm-project/vllm)
- [`vllm` 论文](https://arxiv.org/abs/2309.06180)

## 报告问题

如果遇到任何问题，请创建 [issue](https://github.com/EricLBuehler/candle-vllm/issues)。
