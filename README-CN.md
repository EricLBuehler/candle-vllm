<p align="center">
    <img src="./res/candle_vllm_logo.png" alt="candle vLLM" width=55%/>
</p>

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README-CN.md">简体中文</a> |
</p>

[![Continuous integration](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml)

高效、易用的本地大语言模型（LLM）推理与服务平台，提供OpenAI兼容的API服务。

## 功能特性
- 提供OpenAI兼容的API服务，用于部署LLM。
- 高度可扩展的基于trait的系统，支持快速实现新的模型服务。
- 生成过程中支持流式（stream）传输。
- 使用Paged Attention高效管理KV缓存。
- 持续批处理（continuous batching，不同时间段的请求decoding阶段聚合为批量处理）。
- 原位（In-situ）量化（及原位Marlin格式转换）。
- 支持`GPTQ/Marlin`格式量化（4位）。
- 支持`Mac/Metal`设备。
- 支持`多GPU`推理（包括`多进程`和`多线程`模式）。
- 支持`多节点`推理（使用MPI运行）。
- 支持分块Prefilling (默认块大小8K)
- 支持CUDA Graph
- 支持Prefix Caching
- 支持硬件FP8模型推理加速（SM90+, Qwen3系列，Block-wise FP8量化）
- 支持 Flashinfer 后端
- 支持通过命令行参数 `--yarn-scaling-factor` 手动设置 YaRN RoPE 缩放因子

## 支持的模型
- 目前，candle-vllm支持以下模型结构的推理服务。
  <details open>
    <summary>显示支持的模型架构</summary>

    | 模型ID | 模型类型 | 解码速度 / 单请求（`BF16`, Hopper） | 量化（`Q4K`或`Marlin`） |
    |--|--|--|--|
    | #1 | **LLAMA** |105 tks/s (8B) | 154 tks/s (8B, Q4k), 163 tks/s (8B, **Marlin**) |
    | #2 | **Mistral** |112 tks/s (7B)| 171 tks/s (7B, Q4k), 175 tks/s (7B, **Marlin**) |
    | #3 | **Phi3/Phi4** |139 tks/s (3.8B)|180 tks/s (3.8B, Q4k)|
    | #4 | **QWen2/Qwen3 Dense** |96 tks/s (8B)|135 tks/s **(8B, Q4k)**|
    | #5 | **QWen3 MoE** |92 tks/s **(30B)**|114 tks/s **(30B, Q4K)** |
    | #6 | **QWen3-Next MoE** |71 tks/s **(80B, BF16, tp=2)**|TBD|
    | #7 | **QWen3.5 Dense** |30 tks/s **(27B, BF16)**|~42 tks/s **(27B, Q4K / FP8)** |
    | #8 | **QWen3.5 MoE** |82 tks/s **(35B)**|93 tks/s **(35B, Q4K)** |
    | #9 | **Yi** |148 tks/s (6B)| 180 tks/s (6B, Q4k)|
    | #10 | **StableLM** |223 tks/s (3B)|-|
    | #11 | **Gemma-2/Gemma-3** |92 tks/s (9B)|115 tks/s (9B, **Marlin**)|
    | #12 | **DeepSeek V2/V3/R1** |TBD|~20 tks **(AWQ 671B, tp=8, offloading)**|
    | #13 | **QwQ-32B** |45 tks/s **(32B, tp=2)**|63 tks/s **(32B, Q4K)**|
    | #14 | **GLM4** |89 tks/s **(9B)**|124 tks/s **(9B, Q4K)**|
  </details>

### 演示视频
- GPU与Apple Silicon
  <details>
    <summary>显示演示视频</summary>

    **GPU**（A100, BF16, QWen3-8B推理模型）上的聊天演示
    <img src="res/Qwen3-8B-Reasoning-A100.gif" width="85%" height="85%" >

    **Apple Silicon**（M4，16GB统一内存，Q2K, QWen3-8B）上的聊天演示
    <img src="res/Qwen3-8B-Apple-M4.gif" width="85%" height="85%" >
  </details>

## 基本用法
### 安装Candle-vLLM
**下载源代码**
```shell
git clone git@github.com:EricLBuehler/candle-vllm.git
cd candle-vllm
```
**CUDA平台（11+, 12+, 13.0）**

 > 方案 1 (安装进docker)
```bash
# 主机驱动版本需要 >= 选定的CUDA版本；启用`flashattn`或`flashinfer`特性需要更长的编译时间
# 将 `sm_80` 更改为当前硬件特性, 例如, sm_75 (V100), sm_80 (Ampere, A100), sm_86/89 (RTX30xx, RTX40xx), sm_90 (Hopper, H100/H200), sm_100/sm_120 (Blackwell, RTX50xx)
./build_docker.sh "cuda,nccl,graph,flashinfer,cutlass" sm_90 13.0.0

# 或切换为 Flah attention 后端, 或 传 1 使用Rust 中国区镜像 (适用于中国大陆)
./build_docker.sh "cuda,nccl,graph,flashattn,cutlass" sm_80 12.9.0 1
```

 > 方案 2 (手动安装)

安装依赖项
```shell
sudo apt update
sudo apt install libssl-dev pkg-config curl -y
# 安装 CUDA toolkit (可选)
sudo apt-get install -y cuda-toolkit-12-9 #需要匹配主机驱动版本
# 安装Rust，需要1.83.0及以上版本
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 确保CUDA Toolkit在系统PATH中
export PATH=$PATH:/usr/local/cuda/bin/
```

适用于单节点推理
```shell
# sm+70/sm_75硬件平台需要去除“flashattn,flashinfer,cutlass”特性
# 将 `flashinfer` 替换为 `flashattn` 则启用Flash attention后端
cargo install --features cuda,nccl,graph,flashinfer,cutlass --path .
```

适用于多节点推理
```shell
sudo apt install git libopenmpi-dev openmpi-bin -y #安装MPI
sudo apt install clang libclang-dev
cargo install --features cuda,nccl,flashattn,cutlass,mpi --path . #同时包含flash attention与MPI功能

# FlashInfer 后端
cargo install --features cuda,nccl,graph,flashinfer,cutlass,mpi --path .
```

**Mac/Metal平台**

安装 [Xcode command line tools](https://mac.install.guide/commandlinetools/)

安装带有 `metal` 特性
```shell
cargo install --features metal --path .
```

### 直接运行（非安装方式）

- [`ENV_PARAM`] cargo run [`BUILD_PARAM`] -- [`PROGRAM_PARAM`] [`MODEL_ID/MODEL_WEIGHT_PATH`] [`CACHE CONFIG`]
  <details open>
    <summary>显示详情</summary>

    **示例:**

    ```shell
    [RUST_LOG=warn] cargo run [--release --features cuda,nccl,flashinfer,cutlass,graph] -- [--log --dtype bf16 --p 2000 --d 0,1 --gpu-memory-fraction 0.7 --isq q4k --prefill-chunk-size 8192 --frequency-penalty 1.1 --presence-penalty 1.1 --enforce-parser qwen_coder --yarn-scaling-factor 4.0] [--m Qwen/Qwen3.5-27B-FP8] [--fp8-kvcache] [--ui-server]
    ```

    `ENV_PARAM`: RUST_LOG=warn

    `BUILD_PARAM`: --release --features cuda,nccl,flashinfer,cutlass,graph

    `PROGRAM_PARAM`：--log --dtype bf16 --p 2000 --d 0,1 --gpu-memory-fraction 0.7 --isq q4k --prefill-chunk-size 8192 --frequency-penalty 1.1 --presence-penalty 1.1 --enforce-parser qwen_coder --yarn-scaling-factor 4.0

    `MODEL_ID/MODEL_WEIGHT_PATH`: --m Qwen/Qwen3.5-27B-FP8（或使用 `--w` 指定本地模型路径）

    `CACHE CONFIG`: --fp8-kvcache

    `WEB UI`: --ui-server

    其中，`--p`: 服务端口; `--d`: 设备序列号; `--w`: 权重路径 (safetensors路径); `--f`: 权重文件 (GGUF模型使用); `--m`: Huggingface model-id; `--isq`将权重在加载过程中量化为`q4k`格式；`--prefill-chunk-size`指定分块prefill时的块大小（默认8K，`0`为禁用），`--frequency-penalty`和`--presence-penalty`为重复输出惩罚项 (取值-2.0到2.0)；`--mem` (`kvcache-mem-gpu`) 用于以 MB 为单位设置固定 KV Cache 预算；`--gpu-memory-fraction` 会在模型加载完成后按 `fraction * 总显存 - 当前占用显存` 自动计算 KV Cache 大小；`--enforce-parser` 用于强制指定 tool calling 解析器后端，例如 `qwen_coder`、`qwen`、`json` 或 `mistral`；`--yarn-scaling-factor` 用于手动注入 YaRN RoPE 缩放因子，例如 `4.0`，以在支持的模型上扩展有效上下文长度；`--fp8-kvcache` 参数用于启用 FP8 KV Cache；`--prefix-cache` 启用前缀缓存复用；`--prefix-cache-max-tokens` 限制前缀缓存大小；`--ui-server` 启动内置 Web UI。若要使用 Flash attention 后端，可将示例中的 `flashinfer` 替换为 `flashattn`。
  </details>

## 如何运行模型？

- **注意:** 通过Docker安装后需执行以下命令进入candle-vllm Docker:
```shell
docker run --rm -it --gpus all --network host -v /home:/home -v /data:/data candle-vllm:latest bash
```

- 运行**未压缩**模型 
  <details open>
    <summary>显示命令</summary>

    **本地路径（指定端口、设备）**
    ```shell
    candle-vllm --p 8000 --d 0,1 --w /home/Qwen3-30B-A3B-Instruct-2507/
    ```

    **本地路径 (ISQ量化, +UI Server)**
    ```shell
    candle-vllm --p 8000 --d 0,1 --w /home/Qwen3.5-27B/ --isq q4k --ui-server --prefix-cache
    ```

    **模型ID（从Huggingface下载）**

    ```shell
    candle-vllm --m Qwen/Qwen3.5-35B-A3B --ui-server --prefix-cache
    ```

    **手动设置 YaRN 缩放**
    ```shell
    candle-vllm --m Qwen/Qwen3.5-35B-A3B --yarn-scaling-factor 4.0 --ui-server --prefix-cache
    ```

    **FP8 模型** (block-wise量化, 通过增加`cutlass`特性构建)
    ```shell
    candle-vllm --m Qwen/Qwen3.5-27B-FP8 --ui-server --prefix-cache
    ```

  </details>

- 运行**GGUF**模型 
  <details open>
    <summary>显示命令</summary>

    **本地路径**

    ```shell
    candle-vllm --f /home/data/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
    ```

    **模型ID（从Huggingface下载）**

    ```shell
    candle-vllm --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
    ```

  </details>

- 在**Apple Silicon**上运行**GGUF**模型
  <details>
    <summary>显示命令</summary>

    **本地路径（假设模型已下载到/home）**

    ```shell
    candle-vllm -- --f /home/qwq-32b-q4_k_m.gguf
    ```

    **模型ID（从Huggingface下载）**

    ```shell
    candle-vllm -- --m Qwen/QwQ-32B-GGUF --f qwq-32b-q4_k_m.gguf
    ```

  </details>

- 将未压缩模型使用**原位（in-situ）量化**加载并运行为量化模型
  <details>
    <summary>显示命令</summary>

    **只需在运行未量化模型时添加`isq`参数**

    ```shell
    candle-vllm --m Qwen/Qwen3.5-27B --isq q4k
    ```

    注：原位量化加载可能需要更长的加载时间，原位`isq`参数选项：["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k","q4k","q5k","q6k"]
  </details>

- 运行**Marlin兼容的GPTQ模型**（4位GPTQ，128分组，desc_act=False）
  <details>
    <summary>显示命令</summary>

    **本地路径**

    ```shell
    candle-vllm --w /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g
    ```

    **模型ID（从Huggingface下载）**

    ```shell
    candle-vllm --m thesven/Llama-3-8B-GPTQ-4bit
    ```

    **将未压缩模型转换为Marlin兼容格式**
    ```shell
    python3 examples/convert_marlin.py --src /home/DeepSeek-R1-Distill-Qwen-14B/ --dst /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g
    candle-vllm --w /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g
    ```

  </details>

- 运行**Marlin兼容的AWQ模型**
  <details>
    <summary>显示命令</summary>

    **将AWQ模型转换为Marlin兼容格式**
    ```shell
    python3 examples/convert_awq_marlin.py --src /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --dst /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ --bits 4 --method awq --group 128 --nk False
    ```

    **运行转换后的AWQ模型**
    ```shell
    candle-vllm --d 0 --w /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/
    ```

  </details>

- 运行**Marlin格式模型**
  <details>
    <summary>显示命令</summary>

    ```shell
    candle-vllm --w /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ-Marlin/
    ```

  </details>

- 使用**多进程模式（多GPU）**运行**大型模型**
  <details>
    <summary>显示命令</summary>

    **在两块GPU上运行QwQ-32B BF16模型**
    ```shell
    candle-vllm --d 0,1 --w /home/QwQ-32B/
    ```

    **在两块GPU上运行QwQ-32B 4位AWQ模型**

    1) 将AWQ模型转换为Marlin兼容格式
    ```shell
    python3 examples/convert_awq_marlin.py --src /home/QwQ-32B-AWQ/ --dst /home/QwQ-32B-AWQ-Marlin/ --bits 4 --method awq --group 128 --nk False
    ```

    2) 运行转换后的AWQ模型
    ```shell
    candle-vllm --d 0,1 --w /home/QwQ-32B-AWQ-Marlin/
    ```

    **注意**：使用的GPU数量（`--d`）必须为2的幂次方（例如2、4或8）。
  </details>

- 使用**多线程模式（多GPU，调试用途）**运行**大型模型**
  <details>
    <summary>显示命令</summary>

    只需添加`--multithread`参数。

    **在两块GPU上运行QwQ-32B BF16模型**
    ```shell
    candle-vllm --multithread --d 0,1 --w /home/QwQ-32B/
    ```

    如果在多线程多GPU模式下遇到问题，可以尝试：
    ```shell
    export NCCL_P2P_DISABLE=1 #在某些环境中禁用P2P功能以避免非法内存访问
    ```

  </details>

- 在**低显存GPU（CPU卸载）**上运行**DeepSeek-R1（671B/685B）**
  <details>
    <summary>显示命令</summary>

    **1. 将DeepSeek-R1-AWQ模型转换为Marlin兼容格式**
    ```shell
    python3 examples/convert_awq_marlin.py --src /data/DeepSeek-R1-AWQ/ --dst /data/DeepSeek-R1-AWQ-Marlin/ 
    ```

    **2. 在8块A100（40GB）上运行DeepSeek-R1模型**
    ```shell
    candle-vllm --log --d 0,1,2,3,4,5,6,7 --w /data/DeepSeek-R1-AWQ-Marlin --num-experts-offload-per-rank 15
    ```

    **注意**：此设置将每个rank的15个专家（总共256个专家中的120个）卸载到CPU（需要约150GB的额外主机内存）。在推理过程中，这些卸载的专家会根据需要交换回GPU内存。如果GPU内存更少，可以增加`--num-experts-offload-per-rank`参数（最大支持每个rank卸载32个专家）。

  </details>

- 在**多节点**上运行**DeepSeek-R1（671B/685B）**
  <details>
    <summary>显示命令</summary>

    **1. 安装MPI并构建MPI功能**
    ```shell
    sudo apt update
    sudo apt install libopenmpi-dev openmpi-bin -y #安装MPI
    sudo apt install clang libclang-dev
    #在两个节点的相同目录下克隆仓库并构建
    cargo install --features cuda,nccl,mpi #构建MPI功能
    ```

    **2. 将AWQ DeepSeek模型转换为Marlin兼容格式**
    ```shell
    python3 examples/convert_awq_marlin.py --src /data/DeepSeek-R1-AWQ/ --dst /data/DeepSeek-R1-AWQ-Marlin/ 
    ```

    **3. 配置多节点环境**

    MPI运行器要求所有节点具有`相同的`硬件和软件配置，请确保权重和candle-vllm二进制文件位于不同节点的相同文件夹中。节点之间需要通过SSH（端口22）无密码互相访问（如果是`--allow-run-as-root`则需要root用户）。`%NET_INTERFACE%`是通过命令`ifconfig -a`获取的活动网络接口。如果节点中没有InfiniBand，可以通过插入环境变量`-x NCCL_IB_DISABLE=1`来禁用它。`hostfile`可以定义如下：

    示例（两个节点，每个节点8块GPU）：
    ```
    192.168.1.100 slots=8
    192.168.1.101 slots=8
    ```

    **4. 使用MPI运行器在两个节点上运行模型**
    ```shell
    sudo mpirun -np 16 -x RUST_LOG=info -hostfile ./hostfile --allow-run-as-root -bind-to none -map-by slot --mca plm_rsh_args "-p 22" --mca btl_tcp_if_include %NET_INTERFACE% candle-vllm --log --d 0,1,2,3,4,5,6,7 --w /data/DeepSeek-R1-AWQ-Marlin/
    ```
  </details>

- 在多核CPU机器上使用 **NUMA绑定**运行模型
  <details>
    <summary>显示命令</summary>

    **前置条件**
    请确保你的机器有多个 NUMA 节点（即多个物理 CPU），并安装 numactl：

    ```shell
    sudo apt-get install numactl
    ```

    假设你的机器有 8 张 GPU 和 2 个 NUMA 节点，每 4 张 GPU 绑定到一个不同的 NUMA 节点。
    如果你想使用全部 GPU 进行推理，下面的 NUMA 绑定配置可以获得最佳性能：

    ```shell
    MAP_NUMA_NODE=0,0,0,0,1,1,1,1 numactl --cpunodebind=0 --membind=0 candle-vllm --d 0,1,2,3,4,5,6,7 --w /home/data/DeepSeek-V2-Chat-AWQ-Marlin
    ```

    如果你只使用 4 张 GPU，可以使用如下的 NUMA 绑定方式：
    
    ```shell
    MAP_NUMA_NODE=0,0,0,0 numactl --cpunodebind=0 --membind=0 candle-vllm --d 0,1,2,3 --w /home/data/DeepSeek-V2-Chat-AWQ-Marlin
    ```

    以上命令中 `numactl --cpunodebind=0 --membind=0`指定了master进程（master rank）绑定的NUMA node，其必须与 `MAP_NUMA_NODE`相匹配。
    
    注意： 绑定顺序可能会根据你的硬件配置有所不同。
  </details>

## 📚 其它文档
- [Crate Usage](docs/rust_crate.md)
- [Embedding模型使用](docs/embedding.md)
- [MCP & Tool Calling](docs/mcp_tool_calling.md)
- [Prefix Cache](docs/prefix_cache.md)
- [OpenCode + Candle-vLLM后端](docs/opencode.md)

## 如何向后端发送请求？

**启动后端服务后运行聊天前端**

聊天前端（任何与OpenAI API兼容的前端，以下是简单选项）：

- **选项1：使用Chat.py聊天（用于简单测试）**
  <details>
    <summary>显示选项1</summary>
    
    安装API和聊天机器人依赖（openai包仅用于与candle-vllm本地聊天）

    ```shell
    python3 -m pip install openai rich click
    ```

    使用迷你聊天机器人（纯文本）
    ```shell
    python3 examples/chat.py
    ```

    传递生成参数（使用`--thinking True`与推理模型交互）
    ```shell
    python3 examples/chat.py --temperature 0.7 --top_k 64 --top_p 0.9 --thinking True --system_prompt "Thinking big!"
    ```

    使用迷你聊天机器人（实时更新Markdown，可能会闪烁）
    ```shell
    python3 examples/chat.py --live
    ```

    注：**开启VPN状态下，VPN服务可能会影响本地聊天请求**
  <details>

- **选项2：使用简单的ChatUI（或流行的dify前端）**
  <details>
    <summary>显示选项2</summary>

    安装简单的ChatUI及其依赖：

    ```
    git clone git@github.com:guoqingbao/candle-vllm-demo.git
    cd candle-vllm-demo
    apt install npm #如果需要，安装npm
    npm install n -g #如果需要，更新Node.js
    n stable #如果需要，更新Node.js
    npm i -g pnpm #安装pnpm管理器
    pnpm install #安装ChatUI依赖
    ```

    启动ChatUI：
    ```
    pnpm run dev #运行ChatUI
    ```

    **Nodejs错误处理**
    `ENOSPC: System limit for number of file watchers reached`
    ```
    echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
    ```
  </details>

- **选项3：通过HTTP post发送聊天完成请求**
  <details>
    <summary>显示选项3</summary>

    ``` shell
    curl -X POST "http://127.0.0.1:2000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer YOUR_API_KEY" \
        -d '{
            "model": "llama7b",
            "messages": [
                {"role": "user", "content": "解释如何最好地学习Rust。"}
            ],
            "temperature": 0.7,
            "max_tokens": 128,
            "stop": {"Single":"</s>"}
        }'
    ```
    示例响应：

    ```
    {"id":"cmpl-53092967-c9cf-40e0-ae26-d7ac786d59e8","choices":[{"message":{"content":" 学习任何编程语言都需要理论、实践和专注的结合。以下是帮助你高效学习Rust的一些步骤和资源：\n\n1. 从基础开始：\n\t* 理解Rust程序的语法和基本结构。\n\t* 学习变量、数据类型、循环和控制结构。\n\t* 熟悉Rust的所有权系统和借用机制。\n2. 阅读Rust官方书籍：\n\t* Rust官方书籍是全面介绍该语言的资源。\n\t* 它涵盖了诸如","role":"[INST]"},"finish_reason":"length","index":0,"logprobs":null}],"created":1718784498,"model":"llama7b","object":"chat.completion","usage":{"completion_tokens":129,"prompt_tokens":29,"total_tokens":158}}
    ```
  </details>

- **选项4：使用openai包发送聊天完成请求**
  <details>
    <summary>显示选项4</summary>

    在终端中运行`pip install openai`安装`openai` Python包。这里使用的是`1.3.5`版本。

    然后，创建一个新的Python文件并写入以下代码：
    ```python
    import openai

    openai.api_key = "EMPTY"

    openai.base_url = "http://localhost:2000/v1/"

    completion = openai.chat.completions.create(
        model="llama",
        messages=[
            {
                "role": "user",
                "content": "解释如何最好地学习Rust。",
            },
        ],
        max_tokens = 64,
    )
    print(completion.choices[0].message.content)
    ```
    在`candle-vllm`服务运行后，运行Python脚本即可享受高效的推理服务！


    **批量请求**

    首先安装openai API：
    ```
    python3 -m pip install openai
    ```

    运行基准测试：
    ``` shell
    python3 examples/benchmark.py --batch 16 --max_tokens 1024
    ```
    参考`examples/benchmark.py`：

    ``` python
    async def benchmark():
        model = "mistral7b"
        max_tokens = 1024
        # 16个请求
        prompts = ["解释如何最好地学习Rust。", 
                "请用100字谈谈深度学习。", 
                "你知道中国的首都是哪个城市吗？谈谈你所知道的细节。", 
                "谁是世界上最好的女演员？解释原因。",
                "如何应对抑郁症？",
                "如何在短时间内赚钱？",
                "大语言模型的未来趋势是什么？",
                "世界上著名的科技公司。",
                "解释如何最好地学习Rust。", 
                "请用100字谈谈深度学习。", 
                "你知道中国的首都是哪个城市吗？谈谈你所知道的细节。", 
                "谁是世界上最好的女演员？解释原因。",
                "如何应对抑郁症？",
                "如何在短时间内赚钱？",
                "大语言模型的未来趋势是什么？",
                "世界上著名的科技公司。"]
        
        # 同时发送16个聊天请求
        tasks: List[asyncio.Task] = []
        for i in range(len(prompts)):
            tasks.append(
                asyncio.create_task(
                    chat_completion(model, max_tokens, prompts[i]))
            )

        # 获取每个请求的流对象
        outputs: List[Stream[ChatCompletionChunk]] = await asyncio.gather(*tasks)

        # 流式聊天响应的任务
        tasks_stream: List[asyncio.Task] = []
        for i in range(len(outputs)):
            tasks_stream.append(
                asyncio.create_task(
                    stream_response(i, outputs[i]))
            )

        # 收集响应文本
        outputs: List[(int, str)] = await asyncio.gather(*tasks_stream)

        # 打印结果，可以在后端服务器（即candle-vllm）中查看聊天完成统计信息
        for idx, output in outputs:
            print("\n\n 响应 {}: \n\n {}".format(idx, output))


    asyncio.run(benchmark())
    ```
  </details>

## 原位（in-situ）量化
- **原位量化和原位Marlin格式转换**
  <details>
    <summary>显示量化配置</summary>

    Candle-vllm支持在模型加载时将默认权重（F32/F16/BF16）转换为任何GGML/GGUF格式，或将`4位GPTQ/AWQ`权重转换为`Marlin`格式进行加速。此功能有助于节省GPU内存（或通过Marlin内核加速推理性能），使其更适合消费级GPU（例如RTX 4090）。要使用此功能，只需在运行candle-vllm时传递相应`isq`参数。

    **对于未量化模型：**

    ```
    candle-vllm --w /home/Meta-Llama-3.1-8B-Instruct/ --isq q4k
    ```

    `quant`参数选项：["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k","q4k","q5k","q6k"]

    **对于4位GPTQ量化模型：**

    ```
    candle-vllm --w /home/mistral_7b-int4/ --isq marlin
    ```

    **关于Marlin的注意事项**：

    1) 将F32/F16/BF16模型加载为量化格式可能需要几分钟时间；

    2) 原位Marlin格式转换仅支持4位GPTQ（`sym=True`，`groupsize=128`或-1，`desc_act=False`）和4位AWQ（使用给定脚本转换后）；

    3) Marlin格式仅在CUDA平台上支持。
  </details>

## 其他用法
- KV缓存配置、采样参数等

  <details>
    <summary>显示详情</summary>
    `--mem` (`kvcache-mem-gpu`) 用于以 MB 为单位设置固定 KV Cache 预算，默认值为 `4096` MB。

    `--gpu-memory-fraction` 提供一个更轻量的自动模式。不显式指定时，默认值为 `0.7`。模型加载完成后，candle-vllm 会探测每张已加载的 CUDA 或 Metal 设备，并按以下公式计算 KV Cache 预算：

    ```
    gpu_memory_fraction * total_gpu_memory - current_memory_usage
    ```

    多卡场景下，会取所有 rank 中最小的结果作为每个 rank 的 KV Cache 预算。例如：

    ```
    candle-vllm --w /home/Qwen3-Coder-30B-A3B-Instruct-FP8 --d 0,1 --gpu-memory-fraction 0.7
    ```

    当你需要显式固定缓存预算时，用 `--mem`。当你希望服务根据模型加载后的可用显存自动调整时，用 `--gpu-memory-fraction`。

    `--enforce-parser` 用于强制指定 tool calling 解析器后端，而不是使用基于模型名称的自动检测。这在模型兼容某个 parser、但自动识别不准确时很有用。常见值包括 `qwen_coder`、`qwen`、`json` 和 `mistral`。例如：

    ```
    candle-vllm --w /home/Qwen3-Coder-30B-A3B-Instruct-FP8 --enforce-parser qwen_coder
    ```

    无效的 parser 名称会在启动时直接报错。

    对于聊天历史设置，将`record_conversation`设置为`true`以让candle-vllm记住聊天历史。默认情况下，candle-vllm`不会`记录聊天历史；相反，客户端将消息和上下文历史一起发送给candle-vllm。如果`record_conversation`设置为`true`，客户端仅发送新的聊天消息给candle-vllm，而candle-vllm负责记录之前的聊天消息。然而，这种方法需要每会话聊天记录，目前尚未实现，因此推荐使用默认方法`record_conversation=false`。

    对于聊天流式传输，需要在聊天请求中将`stream`标志设置为`True`。

    你可以传递`frequency-penalty`、`presence-penalty`和`temperature`参数给模型以**防止潜在的重复**，例如：

    ```
    candle-vllm --w /home/mistral_7b/
    ```

    `--max-gen-tokens`参数用于控制每次聊天响应的最大输出令牌数。默认值将设置为`max_sequence_len`的1/5。

    对于`消费级GPU`，建议以GGML格式（或Marlin格式）运行模型，例如：

    ```
    candle-vllm --w /home/Meta-Llama-3.1-8B-Instruct/ --isq q4k
    ```

    其中`isq`可选值为：["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k","q4k","q5k","q6k", "awq", "gptq", "marlin", "gguf", "ggml"]。
  </details>

- **GPTQ/AWQ模型通过Marlin Kernel加速**
  <details>
    <summary>显示详情</summary>

    Candle-vllm现在支持GPTQ/AWQ（Marlin内核），如果你有`Marlin`格式的量化权重，可以传递`quant`（marlin）参数，例如：

    ```shell
    candle-vllm --w /home/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-Marlin/
    ```

    或者，将现有的AWQ 4位模型转换为Marlin兼容格式：
    ```shell
    python3 examples/convert_awq_marlin.py --src /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --dst /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ --bits 4 --method awq --group 128 --nk False
    candle-vllm --d 0 --w /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/
    ```

    你也可以使用`GPTQModel`通过脚本`examples/convert_marlin.py`将模型转换为Marlin兼容格式。

    **注意**：目前仅支持4位GPTQ量化的Marlin快速内核。
  </details>

## 报告问题
安装`candle-vllm`非常简单，只需按照以下步骤操作。如果遇到任何问题，请创建[issue](https://github.com/EricLBuehler/candle-vllm/issues)。


## 参考
- Python实现：[`vllm-project`](https://github.com/vllm-project/vllm)
- [`vllm`论文](https://arxiv.org/abs/2309.06180)
