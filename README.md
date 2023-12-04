<p align="center">
    <img src="./res/candle_vllm_logo.png" alt="candle vLLM" width=55%/>
</p>

[![Continuous integration](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml)

Efficient, easy-to-use platform for inference and serving local LLMs including an OpenAI compatible API server.

**candle-vllm is in active development and not currently stable.**

## Features
- OpenAI compatible API server provided for serving LLMs.
- Highly extensible trait-based system to allow rapid implementation of new module pipelines,
- Streaming support in generation.
- Efficient management of key-value cache with PagedAttention.

### Pipelines
- Llama
    - 7b
    - 13b
    - 70b
- Mistral
    - 7b

## Examples
See [this folder](examples/) for some examples.

### Example with Llama 7b
In your terminal, install the `openai` Python package by running `pip install openai`. I use version `1.3.5`.

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

## Installation
Installing `candle-vllm` is as simple as the following steps. If you have any problems, please create an
[issue](https://github.com/EricLBuehler/candle-lora/issues).

0) Be sure to install Rust here: https://www.rust-lang.org/tools/install
1) Run `sudo apt install libssl-dev` or equivalent install command
2) Run `sudo apt install pkg-config` or equivalent install command
3) See the "Compiling PagedAttention CUDA kernels" section.

Go to either the "Install with Pytorch" or "Install with libtorch" section to continue.

### Compiling PagedAttention CUDA kernels
1) Install `setuptools >= 49.4.0`: `pip install setuptools==49.4.0`
2) Run `python3 setup.py build` to compile the PagedAttention CUDA headers.
3) `todo!()`

### Install with Pytorch (recommended)
4) Run `sudo find / -name libtorch_cpu.so`, taking note of the paths returned.
5) Install Pytorch 2.1.0 from https://pytorch.org/get-started/previous-versions/. Be sure that the correct CUDA version is used (`nvcc --version`).
6) Run `sudo find / -name libtorch_cpu.so` again. Take note of the new path (not including the filename).
7) Add the following to `.bashrc` or equivalent:
```bash
# candle-vllm
export LD_LIBRARY_PATH=/the/new/path/:$LD_LIBRARY_PATH
export LIBTORCH_USE_PYTORCH=1
```
8) Either run `source .bashrc` (or equivalent) or reload the terminal.

### Install with libtorch (manual)
4) Download libtorch, the Pytorch C++ library, from https://pytorch.org/get-started/locally/. Before executing the `wget` command, ensure the following:
    1) Be sure that you are downloading Pytorch 2.1.0 instead of Pytorch 2.1.1 (change the link, the number is near the end).
    2) If on Linux, use the link corresponding to the CXX11 ABI.
    3) The correct CUDA version is used (`nvcc --version`).

5) Unzip the directory.

6) Add the following line to your `.bashrc` or equivalent:
```bash
# candle-lora
export LIBTORCH=/path/to/libtorch
```

7) Either run `source .bashrc` (or equivalent) or reload your terminal.

#### Error loading shared libraries
If you get this error: `error while loading shared libraries: libtorch_cpu.so: cannot open shared object file: No such file or directory`,
Add the following to your `.bashrc` or equivalent:
```bash
# For Linux
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
# For macOS
export DYLD_LIBRARY_PATH=/path/to/libtorch/lib:$DYLD_LIBRARY_PATH
```
Then, either run `source .bashrc` (or equivalent) or reload the terminal

## Contributing
The following features are planned to be implemented, but contributions are especially welcome:
- Sampling methods:
  - Beam search ([huggingface/candle#1319](https://github.com/huggingface/candle/issues/1319))
- Pipeline batching ([#3](https://github.com/EricLBuehler/candle-vllm/issues/3))
- More pipelines (from `candle-transformers`)

## Resources
- Python implementation: [`vllm-project`](https://github.com/vllm-project/vllm)
- [`vllm` paper](https://arxiv.org/abs/2309.06180)
