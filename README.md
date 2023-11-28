# candle-vllm
[![Continuous integration](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml)

Efficient, easy-to-use platform for inference and serving local LLMs including an OpenAI compatible API server.

## Features
- OpenAI compatible API server provided for serving LLMs.
- Highly extensible trait-based system to allow rapid implementation of new module pipelines,
- Streaming support in generation.

## Pipelines
- Llama
    - 7b
    - 13b
    - 70b
- Mistral
    - 7b

## Examples
See [this folder](examples/) for some examples.

### Example with Llama 7b
In your terminal, install the `openai` Python package by running `pip install openai`.

Then, create a new Python file and write the following code:
```python
import openai

openai.api_key = "EMPTY"

openai.base_url = "http://localhost:2000/v1/"

completion = openai.chat.completions.create(
    model="llama7b",
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
Next, launch a `candle-vllm` instance by running `HF_TOKEN=... cargo run --release -- --hf-token HF_TOKEN --port 2000 llama7b --repeat-last-n 64`.

After the `candle-vllm` instance is running, run the Python script and enjoy efficient inference with an OpenAI compatible API server!

## Installlation
1) Run `sudo apt install libssl-dev` (may need to run `sudo apt update`)
2) Run `sudo apt install pkg-config`

Be sure to install Rust here: https://www.rust-lang.org/tools/install
### Install with Pytorch (recommended)
3) Run `sudo find / -name libtorch_cpu.so`. Take note of the paths specified.
4) Install pytorch from https://pytorch.org/get-started/previous-versions/. Be sure that the correct CUDA version is used (`nvcc --version`).
5) Run `sudo find / -name libtorch_cpu.so`. Take note of the new path (not including the filename).
6) Add the following to .bashrc:
```bash
# candle-vllm
export LD_LIBRARY_PATH=/the/new/path/:$LD_LIBRARY_PATH
export LIBTORCH_USE_PYTORCH=1
```
7) Run `source .bashrc`

### Install manually
3) Download libtorch, the Pytorch C++ API from https://pytorch.org/get-started/locally/. Before executing the wget command, ensure the following:
    1) Be sure that you are downloading Pytorch 2.1.0 instead of Pytorch 2.1.1 (change the link, the number is near the end)
    2) If on Linux, use the cxx11 ABI
    3) The correct CUDA version is used (`nvcc --version`)

4) Unzip the directory.

5) Add the following line to your .bashrc:
```bash
# candle-lora
export LIBTORCH=/path/to/libtorch
```

6) Run `source .bashrc` or reload your terminal

#### Error loading shared libraries
If you get this error: `error while loading shared libraries: libtorch_cpu.so: cannot open shared object file: No such file or directory`,
Add the following to your .bashrc:
```bash
# For Linux
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
# For macOS
export DYLD_LIBRARY_PATH=/path/to/libtorch/lib:$DYLD_LIBRARY_PATH
```
Then, run "source .bashrc" or reload your terminal

## Contributing
The following features are planned to be implemented, but contributions are especially welcome:
- Sampling methods:
  - Beam search ([huggingface/candle#1319](https://github.com/huggingface/candle/issues/1319))
- Pipeline batching ([#3](https://github.com/EricLBuehler/candle-vllm/issues/3))
- PagedAttention ([#3](https://github.com/EricLBuehler/candle-vllm/issues/3))
    - See [this](https://github.com/EricLBuehler/candle-vllm/tree/paged_attention) branch.
- More pipelines (from `candle-transformers`)

## Resources
- Python implementation: [`vllm-project`](https://github.com/vllm-project/vllm)
- [`vllm` paper](https://arxiv.org/abs/2309.06180)
