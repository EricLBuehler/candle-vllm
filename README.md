<p align="center">
    <img src="./res/candle_vllm_logo.png" alt="candle vLLM logo"/>
</p>

[![Continuous integration](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml)

Efficient, easy-to-use platform for inference and serving local LLMs including an OpenAI compatible API server.

## Features
- OpenAI compatible API server provided for serving LLMs.
- Highly extensible trait-based system to allow rapid implementation of new module pipelines,
- Streaming support in generation.

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
