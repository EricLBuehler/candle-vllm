# Prefix Cache (KV Reuse)

Prefix cache lets `candle-vllm` reuse KV cache blocks from prior requests when new prompts share a prefix.

## How it works

- Finished sequences contribute reusable full KV blocks.
- New requests find the longest cached block-aligned prefix.
- Remaining tokens are prefetched normally after the reused prefix.

Prefix cache is block-granular. If a shared prefix ends in the middle of a block, the remainder of that block is recomputed. When a prompt is fully cached at block boundaries, the last block is recomputed so the request still has a non-empty prefill step.

## Flags

- `--prefix-cache`: enable prefix cache
- `--prefix-cache-max-tokens <N>`: cap cache size in tokens, rounded down to block size

If `--prefix-cache-max-tokens` is omitted, the cache defaults to roughly 25% of GPU KV blocks in this project.

## Usage Reporting

OpenAI-compatible chat responses include prefix-cache and reasoning token details when they are non-zero:

```json
{
  "usage": {
    "prompt_tokens": 128,
    "completion_tokens": 64,
    "total_tokens": 192,
    "prompt_time_costs": 4,
    "completion_time_costs": 250,
    "prompt_tokens_details": {
      "cached_tokens": 64
    },
    "completion_tokens_details": {
      "reasoning_tokens": 32
    }
  }
}
```

`prompt_tokens_details.cached_tokens` reports the number of prompt tokens reused from the prefix cache. `completion_tokens_details.reasoning_tokens` reports generated tokens inside reasoning blocks such as `<think>...</think>`. Both detail objects are omitted when their count is zero.

## Hybrid Mamba snapshot stride

For hybrid Mamba models, prefix reuse also needs compatible snapshot boundaries.
Use `CANDLE_VLLM_MAMBA_SNAPSHOT_STRIDE_BLOCKS` to control sparse decode-time snapshot capture (larger stride size useful for limited GPU memory).

```bash
export CANDLE_VLLM_MAMBA_SNAPSHOT_STRIDE_BLOCKS=1
```

Behavior:

- Default: `1`
- Minimum valid value: `1`
- Effective token stride: `block_size * stride`

Example with `block_size=64` and stride `8`:

- snapshot boundary every `512` tokens
- hybrid prefix reuse aligns to the nearest captured boundary

This setting only affects decode-time sparse snapshot capture. Prompt/prefill
snapshot capture remains dense.

## Notes

- Prefix cache shares the same KV memory pool as active sequences.
- Larger caches reduce maximum concurrent live tokens.
- Sliding-window attention limits how much cached context is effectively reused.
