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

## 支持的模型
- 目前，candle-vllm支持以下模型结构的推理服务。
  <details>
    <summary>显示支持的模型架构</summary>

    | 模型ID | 模型类型 | 是否支持 | 速度（A100, `BF16`） | 吞吐量（`BF16`, `bs=16`） | 量化（A100, `Q4K`或`Marlin`） | 量化吞吐量（`GTPQ/Marlin`, `bs=16`） |
    |--|--|--|--|--|--|--|
    | #1 | **LLAMA** |✅|65 tks/s (8B) | 553 tks/s (8B) | 75 tks/s (8B), 115 tks/s (8B, **Marlin**) |968 tks/s (8B)|
    | #2 | **Mistral** |✅|70 tks/s (7B)| 585 tks/s (7B) | 96 tks/s (7B), 115 tks/s (7B, **Marlin**) |981 tks/s (7B)|
    | #3 | **Phi** |✅|107 tks/s (3.8B)| 744 tks/s (3.8B)|135 tks/s (3.8B)|TBD|
    | #4 | **QWen2/Qwen3** |✅|81 tks/s (8B)|831 tks/s (8B) |-|TBD|S
    | #4 | **Yi** |✅|75 tks/s (6B)| 566 tks/s (6B) | 105 tks/s (6B)|TBD|
    | #5 | **StableLM** |✅|99 tks/s (3B)|TBD|-|TBD|
    | #6 | **Gemma-2/Gemma-3** |✅|60 tks/s (9B)|TBD |73 tks/s (9B, **Marlin**) |587 tks/s (9B)|
    | #7 | **DeepSeek-R1-Distill-QWen** |✅|48 tks (14B)|TBD|62 tks (14B)|TBD|
    | #8 | **DeepSeek-R1-Distill-LLaMa** |✅|65 tks (8B)|TBD|108 tks (8B)|TBD|
    | #9 | **DeepSeek V2/V3/R1** |✅|TBD|TBD|~20 tks **(AWQ 671B, tp=8, offloading)**|TBD|
    | #10 | **QwQ-32B** |✅|30 tks/s **(32B, tp=2)**|TBD |36 tks/s **(32B, Q4K, GGUF)**|TBD|
    | #11 | **GLM4** |✅|55 tks/s **(9B)**|TBD |92 tks/s **(9B, Q4K, GGUF)**|TBD|
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
### 构建Candle-vLLM

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh #安装Rust，需要1.83.0及以上版本
sudo apt install libssl-dev pkg-config -y
git clone git@github.com:EricLBuehler/candle-vllm.git
cd candle-vllm

#Mac/Metal平台编译命令
cargo build --release --features metal

#CUDA平台：确保CUDA Toolkit在系统PATH中
export PATH=$PATH:/usr/local/cuda/bin/

#CUDA平台：单节点（单机单卡或单机多卡）编译命令
cargo build --release --features cuda,nccl

#CUDA平台：单节点（使用flash attention kernel，适用于长上下文推理）编译命令
cargo build --release --features cuda,nccl,flash-attn

#CUDA平台：多节点（多机推理）编译命令
sudo apt update
sudo apt install libopenmpi-dev openmpi-bin -y #安装MPI
sudo apt install clang libclang-dev
cargo build --release --features cuda,nccl,mpi #包含MPI功能
#或
cargo build --release --features cuda,nccl,flash-attn,mpi #同时包含flash attention与MPI功能
```

### 构建/运行参数

- [`ENV_PARAM`] cargo run [`BUILD_PARAM`] -- [`PROGRAM_PARAM`] [`MODEL_ID/MODEL_WEIGHT_PATH`] [`MODEL_TYPE`] [`MODEL_PARAM`]  
  <details>
    <summary>显示详情</summary>

    **示例:**

    ```shell
    [RUST_LOG=warn] cargo run [--release --features cuda,nccl] -- [--multi-process --log --dtype bf16 --port 2000 --device-ids "0,1" --kvcache-mem-gpu 8192] [--weight-path /home/weights/Qwen3-27B-GPTQ-4Bit] [qwen3] [--quant gptq --temperature 0.7 --penalty 1.0 --top-k 32 --top-p 0.95 --thinking]
    ```

    `ENV_PARAM`: RUST_LOG=warn

    `BUILD_PARAM`: --release --features cuda,nccl

    `PROGRAM_PARAM`：--multi-process --log --dtype bf16 --port 2000 --device-ids "0,1" --kvcache-mem-gpu 8192

    `MODEL_WEIGHT_PATH`: --weight-path /home/weights/Qwen3-27B-GPTQ-4Bit

    `MODEL_TYPE`: qwen3

    `MODEL_PARAM`: --quant gptq --temperature 0.7 --penalty 1.0 --top-k 32 --top-p 0.95 --thinking

    其中，`kvcache-mem-gpu`参数控制KV Cache缓存，长文本或批量推理量请增大缓存；`MODEL_TYPE`可选值为：["llama", "llama3", "mistral", "phi2", "phi3", "qwen2", "qwen3", "glm4", "gemma", "gemma3", "yi", "stable-lm", "deep-seek"]
  </details>

## 如何运行？

- 运行**未压缩**模型 
  <details>
    <summary>显示命令</summary>

    **本地路径**

    ```shell
    target/release/candle-vllm --port 2000 --weight-path /home/DeepSeek-R1-Distill-Llama-8B/ llama3 --temperature 0. --penalty 1.0
    ```

    **模型ID（从Huggingface下载）**

    ```shell
    target/release/candle-vllm --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B qwen3
    ```

  </details>

- 运行**GGUF**模型 
  <details>
    <summary>显示命令</summary>

    **本地路径（指定端口、数据类型、采样参数）**

    ```shell
    target/release/candle-vllm --port 2000 --dtype bf16 --weight-file /home/data/DeepSeek-R1-0528-Qwen3-8B-Q2_K.gguf qwen3 --quant gguf --temperature 0.7 --penalty 1.1
    ```

    **模型ID（从Huggingface下载）**

    ```shell
    target/release/candle-vllm --model-id unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF --weight-file DeepSeek-R1-0528-Qwen3-8B-Q2_K.gguf qwen3 --quant gguf
    ```

  </details>

- 在**Apple Silicon**上运行**GGUF**模型
  <details>
    <summary>显示命令</summary>

    **本地路径（假设模型已下载到/home）**

    ```shell
    cargo run --release --features metal -- --port 2000 --dtype bf16 --weight-file /home/qwq-32b-q4_k_m.gguf qwen2 --quant gguf --temperature 0. --penalty 1.0
    ```

    **模型ID（从Huggingface下载）**

    ```shell
    cargo run --release --features metal -- --port 2000 --dtype bf16 --model-id Qwen/QwQ-32B-GGUF --weight-file qwq-32b-q4_k_m.gguf qwen2 --quant gguf --temperature 0. --penalty 1.0
    ```

  </details>

- 将未压缩模型使用**原位（in-situ）量化**加载并运行为量化模型
  <details>
    <summary>显示命令</summary>

    **只需在运行未量化模型时添加`quant`参数**

    ```shell
    target/release/candle-vllm --port 2000 --weight-path /home/DeepSeek-R1-Distill-Llama-8B/ llama3 --quant q4k --temperature 0. --penalty 1.0
    ```

    注：原位量化加载可能需要更长的加载时间，原位`quant`参数选项：["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k","q4k","q5k","q6k"]
  </details>

- 运行**Marlin兼容的GPTQ模型**（4位GPTQ，128分组，desc_act=False）
  <details>
    <summary>显示命令</summary>

    **本地路径**

    ```shell
    target/release/candle-vllm --dtype bf16 --port 2000 --weight-path /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g qwen2 --quant gptq --temperature 0. --penalty 1.0
    ```

    **模型ID（从Huggingface下载）**

    ```shell
    target/release/candle-vllm --model-id thesven/Llama-3-8B-GPTQ-4bit llama3 --quant gptq
    ```

    **将未压缩模型转换为Marlin兼容格式**
    ```shell
    python3 examples/convert_marlin.py --src /home/DeepSeek-R1-Distill-Qwen-14B/ --dst /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g
    target/release/candle-vllm --dtype bf16 --port 2000 --weight-path /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g qwen2 --quant gptq --temperature 0. --penalty 1.0
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
    target/release/candle-vllm --multi-process --dtype f16 --port 2000 --device-ids "0" --weight-path /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ llama3 --quant awq --temperature 0. --penalty 1.0
    ```

  </details>

- 运行**Marlin格式模型**
  <details>
    <summary>显示命令</summary>

    ```shell
    target/release/candle-vllm --dtype bf16 --port 2000 --weight-path /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ-Marlin/ qwen2 --quant marlin --penalty 1.0 --temperature 0.
    ```

  </details>

- 使用**多进程模式（多GPU）**运行**大型模型**
  <details>
    <summary>显示命令</summary>

    **在两块GPU上运行QwQ-32B BF16模型**
    ```shell
    cargo run --release --features cuda,nccl -- --multi-process --dtype bf16 --port 2000 --device-ids "0,1" --weight-path /home/QwQ-32B/ qwen2 --penalty 1.0 --temperature 0.
    ```

    **在两块GPU上运行QwQ-32B 4位AWQ模型**

    1) 将AWQ模型转换为Marlin兼容格式
    ```shell
    python3 examples/convert_awq_marlin.py --src /home/QwQ-32B-AWQ/ --dst /home/QwQ-32B-AWQ-Marlin/ --bits 4 --method awq --group 128 --nk False
    ```

    2) 运行转换后的AWQ模型
    ```shell
    cargo run --release --features cuda,nccl -- --multi-process --dtype bf16 --port 2000 --device-ids "0,1" --weight-path /home/QwQ-32B-AWQ-Marlin/ qwen2 --quant awq --penalty 1.0 --temperature 0.
    ```

    **注意**：使用的GPU数量（`--device-ids`）必须为2的幂次方（例如2、4或8）。
  </details>

- 使用**多线程模式（多GPU，调试用途）**运行**大型模型**
  <details>
    <summary>显示命令</summary>

    只需移除`--multi-process`参数。

    **在两块GPU上运行QwQ-32B BF16模型**
    ```shell
    cargo run --release --features cuda,nccl -- --dtype bf16 --port 2000 --device-ids "0,1" --weight-path /home/QwQ-32B/ qwen2 --penalty 1.0 --temperature 0.
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
    cargo run --release --features cuda,nccl -- --log --multi-process --dtype bf16 --port 2000 --device-ids "0,1,2,3,4,5,6,7" --weight-path /data/DeepSeek-R1-AWQ-Marlin/ deep-seek --quant awq --temperature 0. --penalty 1.0 --num-experts-offload-per-rank 15
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
    cargo build --release --features cuda,nccl,mpi #构建MPI功能
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
    sudo mpirun -np 16 -x RUST_LOG=info -hostfile ./hostfile --allow-run-as-root -bind-to none -map-by slot --mca plm_rsh_args "-p 22" --mca btl_tcp_if_include %NET_INTERFACE% target/release/candle-vllm --log --multi-process --dtype bf16 --port 2000 --device-ids "0,1,2,3,4,5,6,7" --weight-path /data/DeepSeek-R1-AWQ-Marlin/ deep-seek --quant awq --temperature 0. --penalty 1.0
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
    MAP_NUMA_NODE=0,0,0,0,1,1,1,1 numactl --cpunodebind=0 --membind=0 cargo run --release --features cuda,nccl -- --multi-process --dtype bf16 --port 2000 --device-ids "0,1,2,3,4,5,6,7" --weight-path /home/data/DeepSeek-V2-Chat-AWQ-Marlin deep-seek --quant awq --temperature 0. --penalty 1.0
    ```

    如果你只使用 4 张 GPU，可以使用如下的 NUMA 绑定方式：
    
    ```shell
    MAP_NUMA_NODE=0,0,0,0 numactl --cpunodebind=0 --membind=0 cargo run --release --features cuda,nccl -- --multi-process --dtype bf16 --port 2000 --device-ids "0,1,2,3" --weight-path /home/data/DeepSeek-V2-Chat-AWQ-Marlin deep-seek --quant awq --temperature 0. --penalty 1.0
    ```

    以上命令中 `numactl --cpunodebind=0 --membind=0`指定了master进程（master rank）绑定的NUMA node，其必须与 `MAP_NUMA_NODE`相匹配。
    
    注意： 绑定顺序可能会根据你的硬件配置有所不同。
  </details>

- 使用**Qwen3-Reranker**进行知识检索
  <details>
    <summary>显示命令</summary>

    1) 启动`Qwen3-Reranker`模型服务
    ```shell
    target/release/candle-vllm --port 2000 --multi-process --weight-file /home/data/Qwen3-Reranker-4B-q4_k_m.gguf qwen3 --quant gguf
    ```

    2) 启动迷你聊天机器人并传入`system prompt`
    ```shell
    python3 examples/chat.py --thinking True --system_prompt "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    ```

    3) 使用query/doc对进行知识检查，例如：
    ```shell
    <Query>: What is the capital of China?\n\n<Document>: The capital of China is Beijing.
    ```

    观察输出结果：
    
    ```shell
    🙋 Please Input (Ctrl+C to start a new chat or exit): <Query>: What is the capital of China?\n\n<Document>: The capital of China is Beijing.
    Candle-vLLM: ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    <think>
    Okay, the user is asking for the capital of China. The document provided is a direct answer: "The capital of China is Beijing." I need to check if this is correct. From my knowledge, Beijing is indeed the capital of China. The answer is correct and straightforward. The document meets the requirement as it provides the accurate information. So the answer is yes.
    </think>

    yes
    ```
  </details>

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

    Candle-vllm支持在模型加载时将默认权重（F32/F16/BF16）转换为任何GGML/GGUF格式，或将`4位GPTQ/AWQ`权重转换为`Marlin`格式进行加速。此功能有助于节省GPU内存（或通过Marlin内核加速推理性能），使其更适合消费级GPU（例如RTX 4090）。要使用此功能，只需在运行candle-vllm时传递相应`quant`参数。

    **对于未量化模型：**

    ```
    cargo run --release --features cuda -- --port 2000 --weight-path /home/Meta-Llama-3.1-8B-Instruct/ llama3 --quant q4k
    ```

    `quant`参数选项：["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k","q4k","q5k","q6k"]

    **对于4位GPTQ量化模型：**

    ```
    cargo run --release --features cuda -- --port 2000 --weight-path /home/mistral_7b-int4/ mistral --quant marlin
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
    对于KV缓存配置，设置`kvcache-mem-gpu`，默认为4GB GPU内存用于KV缓存，长文本或批量推理时请增大KV缓存。

    对于聊天历史设置，将`record_conversation`设置为`true`以让candle-vllm记住聊天历史。默认情况下，candle-vllm`不会`记录聊天历史；相反，客户端将消息和上下文历史一起发送给candle-vllm。如果`record_conversation`设置为`true`，客户端仅发送新的聊天消息给candle-vllm，而candle-vllm负责记录之前的聊天消息。然而，这种方法需要每会话聊天记录，目前尚未实现，因此推荐使用默认方法`record_conversation=false`。

    对于聊天流式传输，需要在聊天请求中将`stream`标志设置为`True`。

    你可以传递`penalty`和`temperature`参数给模型以**防止潜在的重复**，例如：

    ```
    cargo run --release --features cuda -- --port 2000 --weight-path /home/mistral_7b/ mistral --repeat-last-n 64 --penalty 1.1 --temperature 0.7
    ```

    `--max-gen-tokens`参数用于控制每次聊天响应的最大输出令牌数。默认值将设置为`max_sequence_len`的1/5。

    对于`消费级GPU`，建议以GGML格式（或Marlin格式）运行模型，例如：

    ```
    cargo run --release --features cuda -- --port 2000 --weight-path /home/Meta-Llama-3.1-8B-Instruct/ llama3 --quant q4k
    ```

    其中`quant`可选值为：["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k","q4k","q5k","q6k", "awq", "gptq", "marlin", "gguf", "ggml"]。
  </details>

- **GPTQ/AWQ模型通过Marlin Kernel加速**
  <details>
    <summary>显示详情</summary>

    Candle-vllm现在支持GPTQ/AWQ（Marlin内核），如果你有`Marlin`格式的量化权重，可以传递`quant`（marlin）参数，例如：

    ```shell
    cargo run --release --features cuda -- --port 2000 --dtype f16 --weight-path /home/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-Marlin/ llama3 --quant marlin --temperature 0. --penalty 1.
    ```

    或者，将现有的AWQ 4位模型转换为Marlin兼容格式：
    ```shell
    python3 examples/convert_awq_marlin.py --src /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --dst /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ --bits 4 --method awq --group 128 --nk False
    cargo run --release --features cuda,nccl -- --multi-process --dtype f16 --port 2000 --device-ids "0" --weight-path /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ llama3 --quant awq --temperature 0. --penalty 1.0
    ```

    你也可以使用`GPTQModel`通过脚本`examples/convert_marlin.py`将模型转换为Marlin兼容格式。

    **注意**：目前仅支持4位GPTQ量化的Marlin快速内核。
  </details>

## 报告问题
安装`candle-vllm`非常简单，只需按照以下步骤操作。如果遇到任何问题，请创建[issue](https://github.com/EricLBuehler/candle-vllm/issues)。


## 参考
- Python实现：[`vllm-project`](https://github.com/vllm-project/vllm)
- [`vllm`论文](https://arxiv.org/abs/2309.06180)