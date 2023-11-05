# candle-vllm

Efficient platform for inference and serving local LLMs including an OpenAI compatible API server.

**This project is in active development**. I would appreciate any contributions, but contributions pertaining to the following features would
be especially welcome:
- Sampling methods:
  - Beam search
  - `presence_penalty` and `frequency_penalty`
- Pipeline batching ([#3](https://github.com/EricLBuehler/candle-vllm/issues/3))
- KV cache ([#3](https://github.com/EricLBuehler/candle-vllm/issues/3))
- PagedAttention ([#3](https://github.com/EricLBuehler/candle-vllm/issues/3))
- More pipelines (models)

Currently, I am actively working on the following features:
- Streaming: ([#2](https://github.com/EricLBuehler/candle-vllm/issues/2))
  - Streaming support in generation (expect)
- Top-K support ([candle/#1271](https://github.com/huggingface/candle/pull/1271))
- Beam search
- More pipelines (models)


## Overview
`candle-vllm` is designed to interface locally served LLMs using an OpenAI compatible API server. `candle-vllm` can serve a single model per instance
(multiple `candle-vllm`s could serve different ones). 

- During initial setup, the model and tokenizer are loaded and other parameters are initialized.

- When a request is received, it is parsed, verified, and converted to a prompt. Finally, the model runs on said prompt returning the 
output.

- This process is abstracted by a trait `ModulePipeline` which acts like the `Module` trait in `Candle`. It provides a clean interface for
new pipelines to be rapidly implemented.

## Features
- OpenAI compatible API server provided for serving LLMs.
- `ModulePipeline` trait acts like the `Module` trait in `Candle`. It provides a clean interface for
  new pipelines to be rapidly implemented.

## Resources
- Python implementation: [`vllm-project`](https://github.com/vllm-project/vllm)