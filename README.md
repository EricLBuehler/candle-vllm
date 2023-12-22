<p align="center">
    <img src="./res/candle_vllm_logo.png" alt="candle vLLM" width=55%/>
</p>

[![Continuous integration](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml)
[![Discord server](https://dcbadge.vercel.app/api/server/Fp47vVj6)](https://discord.gg/Fp47vVj6)

Efficient, easy-to-use platform for inference and serving local LLMs including an OpenAI compatible API server.

**Development status: candle-vllm is currently unable to compile as the CUDA kernels are being developed.**
**See the `cudarc_backend` branch for an implementation from scratch, and the `master` branch, which links to the vLLM kernels.**

## Features
- OpenAI compatible API server provided for serving LLMs.
- Highly extensible trait-based system to allow rapid implementation of new module pipelines,
- Streaming support in generation.
- Efficient management of key-value cache with PagedAttention.
- Continuous batching.

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
3) Run `sudo apt-get install python3-dev` or equivalent install command
4) Install `torch 2.1.0` with `pip install torch==2.1.0`
5) Run `sudo find / -name libpython3.so`, taking note of it and adding it to line 2 of `build.rs`.
6) Install `setuptools >= 49.4.0`: `pip install setuptools==49.4.0`
7) Run `python3 setup.py build` to compile the vLLM CUDA kernels.
8) `cp build/lib<TAB>/<TAB>/<TAB> librustbind.so` to extract the compiled CUDA kernels. `<TAB>` will use your terminal's autocomplete.

Go to either the "Install with Pytorch" or "Install with libtorch" section to continue.

https://stackoverflow.com/a/3891372

### Install with Pytorch (recommended)
9) Run `python3 -c 'import torch;print(torch.__file__.replace("__init__.py", "lib/"))'`
10) Add the following to `.bashrc` or equivalent:
```bash
# candle-vllm
export LD_LIBRARY_PATH=/the/path/printed/:$LD_LIBRARY_PATH
export LIBTORCH_USE_PYTORCH=1
```
11) Either run `source .bashrc` (or equivalent) or reload the terminal.

### Install with libtorch (manual)
9) Download libtorch, the Pytorch C++ library, from https://pytorch.org/get-started/locally/. Before executing the `wget` command, ensure the following:
    1) Be sure that you are downloading Pytorch 2.1.0 instead of Pytorch 2.1.1 (change the link, the number is near the end).
    2) If on Linux, use the link corresponding to the CXX11 ABI.
    3) The correct CUDA version is used (`nvcc --version`).

10) Unzip the directory.

11) Add the following line to your `.bashrc` or equivalent:
```bash
# candle-lora
export LIBTORCH=/path/to/libtorch
```

12) Either run `source .bashrc` (or equivalent) or reload your terminal.

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
- More pipelines (from `candle-transformers`)

## Resources
- Python implementation: [`vllm-project`](https://github.com/vllm-project/vllm)
- [`vllm` paper](https://arxiv.org/abs/2309.06180)
