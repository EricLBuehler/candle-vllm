# Repository Guidelines

## Project Structure & Module Organization
- Core binary lives in `src/` with `main.rs` wiring the server and `lib.rs` exposing shared utilities.
- Reusable crates sit in `crates/`: `candle-vllm-core` (model/runtime logic), `candle-vllm-server` (serving + batching), `candle-vllm-openai` (OpenAI-compatible surfaces), and `candle-vllm-responses` (schemas).
- GPU/accelerator code and kernels reside in `kernels/` and `metal-kernels/`; resources and logos are under `res/`.
- Examples for common configs are in `examples/`; helper scripts live at the repo root (e.g., `run_ministral_3b.sh`, `install_metal_toolchain.sh`).
- Configuration reference lives in `.example.env`, `example.models.yaml`, and [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

### MCP & Tool Calling Notes
- MCP servers declared in `mcp.json` are auto-loaded (CLI `--mcp-config` > `CANDLE_VLLM_MCP_CONFIG` > local file) and their tools are injected into chat requests unless the client specifies `tools` explicitly.
- Streaming responses now include incremental tool-call deltas; keep this in mind when documenting new streaming features or SDK integrations.

## Build, Test, and Development Commands
- Install Rust 1.83+; ensure CUDA toolkit is on `PATH` for NVIDIA or Metal toolchain for Apple.
- Fast debug build: `cargo build`.
- Release build (Metal): `cargo build --release --features metal`.
- Release build (CUDA, single node): `cargo build --release --features cuda,nccl`.
- Run server with a model id: `cargo run --release --features cuda,nccl -- --m <huggingface-id> --ui-server` (add `--isq q4k` for in-situ quantization).
- Lint/format: `cargo fmt --all` and `cargo clippy --all-targets --all-features -D warnings`.
- Tests (CPU-only): `cargo test --all --all-features`; GPU-dependent code should be guarded behind feature flags.

## Coding Style & Naming Conventions
- Rust defaults: 4-space indentation, `rustfmt` enforced; keep imports ordered and unused code clean (`clippy` must pass).
- Modules, files, and functions follow `snake_case`; types and traits use `PascalCase`; constants use `SCREAMING_SNAKE_CASE`.
- Prefer small, composable functions; keep feature-flag-specific logic isolated to modules gated by `#[cfg(feature = "...")]`.

## Testing Guidelines
- Add unit tests near implementations; integration-style smoke tests can live in `examples/` or a dedicated `tests/` module when practical.
- Name tests with `test_<behavior>`; for GPU/feature-specific paths, gate with `#[cfg(feature = "...")]` and document expected hardware.
- Aim for coverage of scheduling logic, kv-cache handling, and request/response schemas; include at least one CPU-safe regression test when changing server paths.

## Commit & Pull Request Guidelines
- Commit messages are short and imperative (e.g., "add tool calling", "update dependency"); keep a focused scope per commit.
- PRs should state the feature/bug, include the feature flags and hardware tested (`metal`, `cuda,nccl`, etc.), and note any model IDs or sample commands used.
- Link related issues; add screenshots or logs when UI or performance changes are involved; describe backward compatibility or migration steps if APIs change.

## Security & Configuration Tips
- Do not commit model weights or secrets; keep Hugging Face tokens and API keys in environment variables.
- Verify CUDA/NCCL and MPI availability before enabling related features; on macOS, run `install_metal_toolchain.sh` when targeting Metal.
- Prefer referencing `.example.env` and `docs/CONFIGURATION.md` when documenting new env vars or config files so users have a single source of truth.

## Recent Changes
- 001-api-mcp-models: Added Rust 1.83+ + candle-core/candle-nn/attention-rs, tokio, serde/serde_json, axum/tower-http (server), thiserror/anyhow, uuid, tokenizers; optional CUDA/Metal feature flags.
- 001-api-mcp-models: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]

## Active Technologies
- Rust 1.83+ + candle-core/candle-nn/attention-rs, tokio, serde/serde_json, axum/tower-http (server), thiserror/anyhow, uuid, tokenizers; optional CUDA/Metal feature flags. (001-api-mcp-models)
- Local filesystem for models, caches, and configs (`models.yaml`, `mcp.json`, HF download cache). (001-api-mcp-models)
