## Prefix Cache (KV Reuse)

Prefix cache lets candle-vLLM reuse KV cache blocks from prior requests when a new
prompt shares a prefix. This accelerates consecutive requests with overlapping
history (e.g., chat sessions that replay the same system + earlier turns).

### How it works
- The scheduler stores KV cache blocks for finished sequences as cacheable prefix blocks.
- For a new request, it finds the longest cached prefix (block-aligned) and reuses those blocks.
- The remaining tokens are prefetched as usual, with KV writes continuing after the cached prefix.

Prefix cache is block-granular: only full KV blocks are reused. If the common
prefix ends mid-block, the tail of that block is recomputed. This keeps the
implementation simple and safe, while still capturing most real-world reuse.
When a prompt is fully cached at block boundaries, the last block is recomputed
to ensure a non-empty prefill step for correct sampling.

### Flags
- `--prefix-cache`: enable prefix cache.
- `--prefix-cache-max-tokens <N>`: cap the cache size in tokens (rounded down to block size).

If `--prefix-cache-max-tokens` is not set, a default of ~25% of the GPU KV blocks
is reserved for cached prefixes.

### Notes
- Prefix cache uses the same KV memory pool as active sequences. A larger cache
  reduces the maximum number of concurrent tokens available for new requests.
- With prefix cache enabled, prefill can reuse cached KV even without
  `flash-decoding` as long as the backend kernels support it.
- Sliding window attention limits how much cached context is effectively used.
