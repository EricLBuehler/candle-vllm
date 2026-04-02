# Tool Call Parsing

`candle-vllm` uses model-specific tool parsers for both streaming and non-streaming responses. The goal is to keep tool-call handling consistent across model families while remaining robust to partial output.

## Parser selection

Parser selection follows this order:

1. `--enforce-parser` if provided and valid.
2. Model-based heuristics from model family and model id.
3. Fallback to `passthrough`.

If `--enforce-parser` is invalid, startup fails with the list of valid parser
names.

Common parser names:

- `passthrough`
- `json`
- `mistral`
- `qwen`
- `qwen_coder`
- `llama`
- `deepseek`
- `glm47_moe`

## Streaming parsing

Streaming tool parsing uses an internal state machine:

1. Normal content streams through.
2. When a tool-call start marker is detected, content is buffered.
3. Buffered content is incrementally parsed into tool call fragments.
4. On tool-call end detection, buffered fragments are finalized into full
   `tool_calls`.
5. If parsing fails, buffered content is released as normal text instead of
   being dropped.

Reasoning and code-block tracking are used to avoid false-positive tool parsing inside `<think>...</think>` blocks or fenced code blocks.

## Non-streaming parsing

Non-streaming responses reuse the same parser configuration and fallback logic, so parser behavior stays aligned between stream and non-stream paths.

## Reasoning routing environment variable

For tool-enabled requests, `CANDLE_VLLM_STREAM_AS_REASONING_CONTENT` controls whether reasoning is emitted as OpenAI-style `reasoning_content` instead of remaining inside `content` with reasoning markers.

```bash
export CANDLE_VLLM_STREAM_AS_REASONING_CONTENT=1
```

Values `0`, `false`, or `no` disable the split and keep reasoning text in `content`. The default is enabled.

## Recommended parser settings

- Qwen coder models: `--enforce-parser qwen_coder`
- Regular Qwen tool calling: use the default model-based parser unless you are debugging format drift
- Gemma and other JSON-style models: `json`
