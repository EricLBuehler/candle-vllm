<p align="center">
    <img src="./res/candle_vllm_logo.png" alt="candle vLLM" width=55%/>
</p>

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README-CN.md">简体中文</a> |
</p>

[![Continuous integration](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml)

Efficient, easy-to-use platform for inference and serving local LLMs including an OpenAI compatible API server.

## Features
- OpenAI compatible API server provided for serving LLMs.
- Highly extensible trait-based system to allow rapid implementation of new module pipelines,
- Streaming support in generation.
- Efficient management of key-value cache with PagedAttention.
- Continuous batching (batched decoding for incoming requests over time).
- `In-situ` quantization (and `In-situ` marlin format conversion)
- `GPTQ/Marlin` format quantization (4-bit)
- Support `Mac/Metal` devices
- Support `Multi-GPU` inference (both `multi-process` and  `multi-threaded` mode)
- Support `Multi-node` inference with MPI runner
- Support Chunked Prefilling (default chunk size 8K)
- Support CUDA Graph

## Supported Models
- Currently, candle-vllm supports chat serving for the following model structures.
  <details>
    <summary>Show supported model architectures</summary>

    | Model ID | Model Type | Supported | Speed (A100, `BF16`) | Throughput (`BF16`, `bs=16`) | Quantized (A100, `Q4K` or `Marlin`) | Throughput (`GTPQ/Marlin`, `bs=16`) |
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
    | #12 | **QWen2 MoE** |✅|TBD|TBD |65 tks/s (14B, Q4K)|TBD|
    | #13 | **QWen3 MoE** |✅|TBD|TBD |76 tks/s **(32B, Q4K)** |TBD|
  </details>

### Demo Video
- Nvidia GPU and Apple Silicon

  <details>
    <summary>Show Demo Video</summary>
    Chat demo on **GPU** (A100, BF16, QWen3-8B Reasoning Model)
    <img src="res/Qwen3-8B-Reasoning-A100.gif" width="85%" height="85%" >

    Chat demo on **Apple Silicon** (M4 with 16GB unified memory, Q2K, QWen3-8B)
    <img src="res/Qwen3-8B-Apple-M4.gif" width="85%" height="85%" >
  </details>

## General Usage
### Build Candle-vLLM

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh #install rust, 1.83.0+ required
sudo apt install libssl-dev pkg-config -y
git clone git@github.com:EricLBuehler/candle-vllm.git
cd candle-vllm

## Mac/Metal (single-node only)
cargo build --release --features metal

#Make sure the CUDA Toolkit can be found in the system PATH
export PATH=$PATH:/usr/local/cuda/bin/

#CUDA: single-node compilation (single gpu, or multi-gpus on single machine)
cargo build --release --features cuda,nccl

#CUDA: single-node compilation (+CUDA Graph)
cargo build --release --features cuda,nccl,graph

#CUDA: single-node compilation with flash attention for prefill only (requires CUDA_ARCH >= 800)
cargo build --release --features cuda,nccl,graph,flash-attn

#CUDA: single-node compilation with flash attention for both prefill and decoding 
#(takes few minutes for the first build, faster inference for long-context, requires CUDA_ARCH >= 800)
cargo build --release --features cuda,nccl,graph,flash-attn,flash-decoding

#CUDA: multinode compilation with MPI (multi-gpus, multiple machines)
sudo apt update
sudo apt install libopenmpi-dev openmpi-bin -y #install mpi
sudo apt install clang libclang-dev
cargo build --release --features cuda,nccl,mpi #build with mpi feature
# or
cargo build --release --features cuda,nccl,flash-attn,mpi #build with flash-attn and mpi features
```

### Build/Run Parameters

- [`ENV_PARAM`] cargo run [`BUILD_PARAM`] -- [`PROGRAM_PARAM`] [`MODEL_ID/MODEL_WEIGHT_PATH`] [`CACHE CONFIG`] [`WEB UI`]
  <details open>
    <summary>Show details</summary>

    **Example:**

    ```shell
    [RUST_LOG=warn] cargo run [--release --features cuda,nccl] -- [--log --dtype bf16 --p 2000 --d 0,1 --mem 4096 --isq q4k --prefill-chunk-size 8192 --frequency-penalty 1.1 --presence-penalty 1.1] [--w /home/weights/Qwen3-30B-A3B-Instruct-2507] [--fp8-kvcache] [--ui-server]
    ```

    `ENV_PARAM`: RUST_LOG=warn

    `BUILD_PARAM`: --release --features cuda,nccl

    `PROGRAM_PARAM`：--log --dtype bf16 --p 2000 --d 0,1 --mem 4096 --isq q4k --prefill-chunk-size 8192 --frequency-penalty 1.1 --presence-penalty 1.1

    `MODEL_ID/MODEL_WEIGHT_PATH`: --w /home/weights/Qwen3-30B-A3B-Instruct-2507 (or `--m` specify model-id)

    `CACHE CONFIG`: --fp8-kvcache

    `WEB UI`: --ui-server

    where, `--p`: server port; `--d`: device ids; `--w`: weight path (safetensors folder); `--f`: weight file (for gguf); `--m`: huggingface model-id; `--isq q4k`: convert weights into `q4k` format during model loading; `--prefill-chunk-size` chunk the prefill into size defined in this flag (default 8K, `0` for disable); `--frequency-penalty` and `presence-penalty` repetition penalty (value from -2.0 to 2.0); `--mem` (`kvcache-mem-gpu`) is the key parameter to control KV cache usage (increase this for large batch); `--fp8-kvcache` used to enable fp8 kvcache; `--ui-server` start with a built-in ChatGPT-like Web UI sever.
  </details>



## How to run?

- Run **Uncompressed** models 
  <details open>
    <summary>Show command</summary>

    **Local Path (with port, device and isq specified)**

    ```shell
    target/release/candle-vllm --p 2000 --d 0,1 --w /home/Qwen3-30B-A3B-Instruct-2507/ --isq q4k --ui-server
    ```

    **Model-ID (download from Huggingface)**

    ```shell
    target/release/candle-vllm --m deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --ui-server
    ```

  </details>

- Run **GGUF** models 
  <details open>
    <summary>Show command</summary>

    **Local Path**

    ```shell
    target/release/candle-vllm --f /home/data/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server
    ```

    **Model-ID (download from Huggingface)**

    ```shell
    target/release/candle-vllm --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server
    ```

  </details>

- Run **GGUF** models on **Apple Silicon**
  <details>
    <summary>Show command</summary>

    **Local Path (assume model downloaded in /home)**

    ```shell
    cargo run --release --features metal -- --f /home/qwq-32b-q4_k_m.gguf --ui-server
    ```

    **Model-ID (download from Huggingface)**

    ```shell
    cargo run --release --features metal -- --m Qwen/QwQ-32B-GGUF --f qwq-32b-q4_k_m.gguf --ui-server
    ```

  </details>

- Run **Any uncompressed models as quantized with in-situ quantization**
  <details>
    <summary>Show command</summary>

    **Simply add `isq` parameter when running unquantized models**

    ```shell
    target/release/candle-vllm --p 2000 --w /home/DeepSeek-R1-Distill-Llama-8B/ --isq q4k
    ```

    Options for in-site `isq` parameters: ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k","q4k","q5k","q6k"]

  </details>

- Run **Marlin-compatible GPTQ models** models (4-bit GPTQ, 128-group, desc_act=False)
  <details>
    <summary>Show command</summary>

    **Local Path**

    ```shell
    target/release/candle-vllm --w /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g
    ```

    **Model-ID (download from Huggingface)**

    ```shell
    target/release/candle-vllm --m thesven/Llama-3-8B-GPTQ-4bit
    ```

    **Convert Any uncompressed model to marlin-compatible format**
    ```shell
    python3 examples/convert_marlin.py --src /home/DeepSeek-R1-Distill-Qwen-14B/ --dst /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g
    target/release/candle-vllm --w /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ_4bit-128g
    ```

  </details>

- Run **Marlin-compatible AWQ models** models
  <details>
    <summary>Show command</summary>

    **Convert AWQ model to Marlin-compatible format**
    ```shell
    python3 examples/convert_awq_marlin.py --src /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --dst /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ --bits 4 --method awq --group 128 --nk False
    ```

    **Run the converted AWQ model**
    ```shell
    target/release/candle-vllm --d 0 --w /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/
    ```

  </details>

- Run **Marlin-format** models
  <details>
    <summary>Show command</summary>

    ```shell
    target/release/candle-vllm --w /home/DeepSeek-R1-Distill-Qwen-14B-GPTQ-Marlin/
    ```

  </details>


- Run **Large models using multi-process mode (Multi-GPU)**
  <details>
    <summary>Show command</summary>

    **QwQ-32B BF16 model on two GPUs**
    ```shell
    cargo run --release --features cuda,nccl -- --d 0,1 --w /home/QwQ-32B/
    ```

    **QwQ-32B 4-bit AWQ model on two GPUs**

    1) Convert AWQ model to Marlin-compatible format
    ```shell
    python3 examples/convert_awq_marlin.py --src /home/QwQ-32B-AWQ/ --dst /home/QwQ-32B-AWQ-Marlin/ --bits 4 --method awq --group 128 --nk False
    ```

    2) Run the converted AWQ model
    ```shell
    cargo run --release --features cuda,nccl -- --d 0,1 --w /home/QwQ-32B-AWQ-Marlin/
    ```

    **Note:** number of GPUs (`--d`) used must be aligned to 2^n (e.g., 2, 4, or 8).
  </details>

- Run **Large models using multi-threaded mode (Multi-GPU, for debug purpose)**
  <details>
    <summary>Show command</summary>

    Simply add the `--multithread` parameter

    **QwQ-32B BF16 model on two GPUs**
    ```shell
    cargo run --release --features cuda,nccl -- --multithread --d 0,1 --w /home/QwQ-32B/
    ```

    If you encountered problems under Multi-threaded Multi-GPU mode, you may:
    ```shell
    export NCCL_P2P_DISABLE=1 # disable p2p cause this feature can cause illegal memory access in certain environments
    ```

  </details>

- Run **DeepSeek-R1 (671B/685B) on Lower GPU Memories (CPU offloading)**
  <details>
    <summary>Show command</summary>

    **1. Convert DeepSeek-R1-AWQ model to Marlin-compatible format**
    ```shell
    python3 examples/convert_awq_marlin.py --src /data/DeepSeek-R1-AWQ/ --dst /data/DeepSeek-R1-AWQ-Marlin/ 
    ```

    **2. Run DeepSeek-R1 model on 8 x A100(40GB)**
    ```shell
    cargo run --release --features cuda,nccl -- --log --d 0,1,2,3,4,5,6,7 --w /data/DeepSeek-R1-AWQ-Marlin/--num-experts-offload-per-rank 15
    ```

    **Note:** This setup offloads 15 experts per rank (a total of 120 out of 256 experts) to the CPU (around 150GB additional host memory required). During inference, these offloaded experts are swapped back into GPU memory as needed. If you have even less GPU memory, consider increasing the `--num-experts-offload-per-rank` parameter (up to a maximum of 32 experts per rank in this case).

  </details>

- Run **DeepSeek-R1 (671B/685B) on Multi-node**
  <details>
    <summary>Show command</summary>

    **1. Install MPI and build with MPI feature**
    ```shell
    sudo apt update
    sudo apt install libopenmpi-dev openmpi-bin -y #install mpi
    sudo apt install clang libclang-dev
    #clone the repo on the same directory of the two node and build
    cargo build --release --features cuda,nccl,mpi #build with mpi feature
    ```

    **2. Convert AWQ deepseek to Marlin-compatible format**
    ```shell
    python3 examples/convert_awq_marlin.py --src /data/DeepSeek-R1-AWQ/ --dst /data/DeepSeek-R1-AWQ-Marlin/ 
    ```

    **3. Config Multi-node Environment**

    MPI Runner requires `identical` hardware and software configurations for all nodes, please ensure weights and candle-vllm binaries located in the identical folders in difference nodes. The the nodes need to be ssh (port 22 in this case) passwordless for each other (root user if `--allow-run-as-root`). `%NET_INTERFACE%` is the active network interface obtained through command 'ifconfig -a'. You may disable InfiniBand if it's not available in the nodes by insert env "-x NCCL_IB_DISABLE=1". Where, `hostfile` can be defined as:

    Example (two nodes, each with 8 GPUs)
    ```
    192.168.1.100 slots=8
    192.168.1.101 slots=8
    ```

    **4. Run the model on two nodes with MPI runner**
    ```shell
    sudo mpirun -np 16 -x RUST_LOG=info -hostfile ./hostfile --allow-run-as-root -bind-to none -map-by slot --mca plm_rsh_args "-p 22" --mca btl_tcp_if_include %NET_INTERFACE% target/release/candle-vllm --log --d 0,1,2,3,4,5,6,7 --w /data/DeepSeek-R1-AWQ-Marlin/
    ```
  </details>

- Run with **NUMA binding**
  <details>
    <summary>Show command</summary>

    **Prerequisite**
    Ensure your machine has more than one NUMA node (i.e., more than one physical CPU), and install numactl:
    ```shell
    sudo apt-get install numactl
    ```

    Suppose your machine has 8 GPUs and 2 NUMA nodes, with each set of 4 GPUs bound to a different NUMA node.
    To achieve optimal performance during inference using all GPUs, use the following NUMA binding:

    ```shell
    MAP_NUMA_NODE=0,0,0,0,1,1,1,1 numactl --cpunodebind=0 --membind=0 target/release/candle-vllm --d 0,1,2,3,4,5,6,7 --w /home/data/DeepSeek-V2-Chat-AWQ-Marlin
    ```

    To use only 4 GPUs, you can apply this NUMA binding:
    
    ```shell
    MAP_NUMA_NODE=0,0,0,0 numactl --cpunodebind=0 --membind=0 target/release/candle-vllm --d 0,1,2,3 --w /home/data/DeepSeek-V2-Chat-AWQ-Marlin
    ```
    *where* `numactl --cpunodebind=0 --membind=0` above indicates NUMA binding for the master rank (master process) which should be matched to `MAP_NUMA_NODE`.

    Note: The exact NUMA binding sequence may vary depending on your hardware configuration.
  </details>

## 📚 Work as a Rust crate
- [Rust Crate Usage](docs/rust_crate.md)
- [Embedding Model Usage](docs/embedding.md)

## How to send request(s) to the backend?

**Run chat frontend after starting the backend service**

Chat frontend (any frontend compatible with openai API, simple options available below):

- **Option 1: Chat with Chat.py (for simple tests)**
  <details>
    <summary>Show Option 1</summary>
    
    Install API and chatbot dependencies (openai package is only used for local chat with candle-vllm)

    ```shell
    python3 -m pip install openai rich click
    ```

    Chat with the mini chatbot (plain text)
    ```shell
    python3 examples/chat.py
    ```

    Pass generation parameters (to reasoning models with `--thinking True`)
    ```shell
    python3 examples/chat.py --temperature 0.7 --top_k 64 --top_p 0.9 --thinking True --system_prompt "Thinking big!"
    ```

    Chat with the mini chatbot (live update with Markdown, may cause flick)
    ```shell
    python3 examples/chat.py --live
    ```
  <details>

- **Option 2: Chat with naive ChatUI (or popular dify frontend)**
  <details>
    <summary>Show Option 2</summary>

    Install naive ChatUI and its dependencies:

    ```
    git clone git@github.com:guoqingbao/candle-vllm-demo.git
    cd candle-vllm-demo
    apt install npm #install npm if needed
    npm install n -g #update node js if needed
    n stable #update node js if needed
    npm i -g pnpm #install pnpm manager
    pnpm install #install ChatUI dependencies
    ```

    Launching the ChatUI:
    ```
    pnpm run dev # run the ChatUI
    ```

    **Trouble shooting for Nodejs error**
    `ENOSPC: System limit for number of file watchers reached`
    ```
    echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
    ```
  </details>

- **Option 3: Chat completion request with HTTP post**
  <details>
    <summary>Show Option 3</summary>

    ``` shell
    curl -X POST "http://127.0.0.1:2000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer YOUR_API_KEY" \
        -d '{
            "model": "llama7b",
            "messages": [
                {"role": "user", "content": "Explain how to best learn Rust."}
            ],
            "temperature": 0.7,
            "max_tokens": 128,
            "stop": {"Single":"</s>"}
        }'
    ```
    Sample response:

    ```
    {"id":"cmpl-53092967-c9cf-40e0-ae26-d7ac786d59e8","choices":[{"message":{"content":" Learning any programming language requires a combination of theory, practice, and dedication. Here are some steps and resources to help you learn Rust effectively:\n\n1. Start with the basics:\n\t* Understand the syntax and basic structure of Rust programs.\n\t* Learn about variables, data types, loops, and control structures.\n\t* Familiarize yourself with Rust's ownership system and borrowing mechanism.\n2. Read the Rust book:\n\t* The Rust book is an official resource that provides a comprehensive introduction to the language.\n\t* It covers topics such","role":"[INST]"},"finish_reason":"length","index":0,"logprobs":null}],"created":1718784498,"model":"llama7b","object":"chat.completion","usage":{"completion_tokens":129,"prompt_tokens":29,"total_tokens":158}}
    ```
  </details>

- **Option 4: Chat completion with with openai package**
  <details>
    <summary>Show Option 4</summary>

    In your terminal, install the `openai` Python package by running `pip install openai`. I use version `1.3.5`.

    Then, create a new Python file and write the following code:
    ```python
    import openai

    openai.api_key = "EMPTY"

    openai.base_url = "http://localhost:2000/v1/"

    completion = openai.chat.completions.create(
        model="llama",
        messages=[
            {
                "role": "user",
                "content": "Explain how to best learn Rust.",
            },
        ],
        max_tokens = 64,
    )
    print(completion.choices[0].message.content)
    ```
    After the `candle-vllm` service is running, run the Python script and enjoy efficient inference with an OpenAI compatible API server!


    **Batched requests**

    Install openai API first
    ```
    python3 -m pip install openai
    ```

    Run the benchmark test
    ``` shell
    python3 examples/benchmark.py --batch 16 --max_tokens 1024
    ```
    Refer to `examples/benchmark.py`

    ``` python
    async def benchmark():
        model = "mistral7b"
        max_tokens = 1024
        # 16 requests
        prompts = ["Explain how to best learn Rust.", 
                "Please talk about deep learning in 100 words.", 
                "Do you know the capital city of China? Talk the details of you known.", 
                "Who is the best female actor in the world? Explain why.",
                "How to dealing with depression?",
                "How to make money in short time?",
                "What is the future trend of large language model?",
                "The famous tech companies in the world.",
                "Explain how to best learn Rust.", 
                "Please talk about deep learning in 100 words.", 
                "Do you know the capital city of China? Talk the details of you known.", 
                "Who is the best female actor in the world? Explain why.",
                "How to dealing with depression?",
                "How to make money in short time?",
                "What is the future trend of large language model?",
                "The famous tech companies in the world."]
        
        # send 16 chat requests at the same time
        tasks: List[asyncio.Task] = []
        for i in range(len(prompts)):
            tasks.append(
                asyncio.create_task(
                    chat_completion(model, max_tokens, prompts[i]))
            )

        # obtain the corresponding stream object for each request
        outputs: List[Stream[ChatCompletionChunk]] = await asyncio.gather(*tasks)

        # tasks for streaming chat responses
        tasks_stream: List[asyncio.Task] = []
        for i in range(len(outputs)):
            tasks_stream.append(
                asyncio.create_task(
                    stream_response(i, outputs[i]))
            )

        # gathering the response texts
        outputs: List[(int, str)] = await asyncio.gather(*tasks_stream)

        # print the results, you may find chat completion statistics in the backend server (i.e., candle-vllm)
        for idx, output in outputs:
            print("\n\n Response {}: \n\n {}".format(idx, output))

    asyncio.run(benchmark())
    ```
  </details>

## In-situ quantization
- **Loading unquantized models as gguf quantized or marlin format**
  <details>
    <summary>Show quantization config</summary>

    Candle-vllm supports in-situ quantization, allowing the transformation of default weights (F32/F16/BF16) into any GGML/GGUF format, or `4-bit GPTQ/AWQ` weights into `marlin format` during model loading. This feature helps conserve GPU memory and speedup inference performance, making it more efficient for consumer-grade GPUs (e.g., RTX 4090). To use this feature, simply supply the `isq` parameter when running candle-vllm.

    **For unquantized models:**

    ```
    cargo run --release --features cuda -- --p 2000 --w /home/Meta-Llama-3.1-8B-Instruct/ --isq q4k
    ```

    Options for `isq` parameters: ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k","q4k","q5k","q6k"]

    **For quantized 4-bit GPTQ model:**

    ```
    cargo run --release --features cuda -- --p 2000 --w /home/mistral_7b-int4/
    ```

    **Please note for marlin**:

    1) It may takes few minutes to load F32/F16/BF16 models into quantized;

    2) Marlin format in-situ conversion only support 4-bit GPTQ (with `sym=True`, `groupsize=128` or -1, `desc_act=False`) and 4-bit AWQ (after conversion using the given script, refer to `Other Usage`);

    3) Marlin format only supported in CUDA platform.
  </details>

## Other Usage
- KV Cache config, sampling parameter, etc.
  <details>
    <summary>Show details</summary>
    The `--mem` (kvcache-mem-gpu) parameter is used to control kv cache, default 4GB GPU memory, increase this for large batch and long-context inference. 

    For chat history settings, set `record_conversation` to `true` to let candle-vllm remember chat history. By `default`, candle-vllm `does not` record chat history; instead, the client sends both the messages and the contextual history to candle-vllm. If record_conversation is set to `true`, the client sends only new chat messages to candle-vllm, and candle-vllm is responsible for recording the previous chat messages. However, this approach requires per-session chat recording, which is not yet implemented, so the default approach `record_conversation=false` is recommended.

    For chat streaming, the `stream` flag in chat request need to be set to `True`.

    ```
    cargo run --release --features cuda -- --p 2000 --w /home/mistral_7b/
    ```

    `--max-gen-tokens` parameter is used to control the maximum output tokens per chat response. The value will be set to 1/5 of max_sequence_len by default.

    For `consumer GPUs`, it is suggested to run the models under GGML formats (or Marlin format), e.g.,

    ```
    cargo run --release --features cuda -- --p 2000 --w /home/Meta-Llama-3.1-8B-Instruct/ --isq q4k
    ```

    where `isq` is one of ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2k", "q3k","q4k","q5k","q6k", "awq", "gptq", "marlin", "gguf", "ggml"].
  </details>

- **Use Marlin kernel to speedup GPTQ/AWQ models**
  <details>
    <summary>Show details</summary>

    Candle-vllm now supports GPTQ/AWQ Marlin kernel, you can run these models directly, such as:

    ```shell
    cargo run --release --features cuda -- --dtype f16 --w /home/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-Marlin/
    ```

    or, convert existing AWQ 4bit model to marlin compatible format

    ```shell
    python3 examples/convert_awq_marlin.py --src /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --dst /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ --bits 4 --method awq --group 128 --nk False
    cargo run --release --features cuda,nccl -- --dtype f16 --d 0 --w /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/
    ```

    You may also use `GPTQModel` to transform a model to marlin-compatible format using the given script `examples/convert_marlin.py`. 

    **Note:** for using Marlin fast kernel, only 4-bit GPTQ quantization supported at the moment. 
  </details>

## Report issue
Installing `candle-vllm` is as simple as the following steps. If you have any problems, please create an
[issue](https://github.com/EricLBuehler/candle-vllm/issues).


## Contributing
The following features are planned to be implemented, but contributions are especially welcome:
- Sampling methods:
  - Beam search ([huggingface/candle#1319](https://github.com/huggingface/candle/issues/1319))
- More pipelines (from `candle-transformers`)

## Resources
- Python implementation: [`vllm-project`](https://github.com/vllm-project/vllm)
- [`vllm` paper](https://arxiv.org/abs/2309.06180)
