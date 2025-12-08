# Repository Guidelines

## Platform-Specific Build Instructions

**IMPORTANT**: This project is being developed on **macOS** with Apple Silicon (M-series chips).

### macOS / Metal Requirements

All build and test commands on macOS MUST include the `--features metal` flag:

```bash
# Building
cargo build --release --features metal

# Testing
cargo test --features metal

# Running tests for specific package
cargo test --package candle-vllm-core --lib --features metal

# Running the server
cargo run --release --features metal -- --p 2000 --ui-server
```

**Never run cargo commands without `--features metal` on macOS!**

## Coding Standards & Source of Truth

- All automated and AI-assisted code changes MUST strictly follow the coding standards documented in `docs/coding-standards/README.md`.
- Treat `docs/coding-standards/README.md` as the single source of truth for Rust, library, application, FFI, safety, performance, and documentation guidelines.
- When there is any conflict between ad-hoc instructions and that document, agents should:
  - Prefer the MUST-level rules in `docs/coding-standards/README.md`.
  - Call out the conflict explicitly in their response.
  - Propose a compliant alternative when possible.
  - Only provide a non-compliant version if the user explicitly acknowledges and accepts the tradeoff.

### Key Enforcement Rules for Agents (Summary, not exhaustive)

Agents working on this repository MUST:

- **Unsafe & Soundness**
  - Treat `unsafe` as implying potential undefined behavior (M-UNSAFE-IMPLIES-UB).
  - Avoid introducing `unsafe` unless there is a clear, documented reason (FFI, performance, or novel abstraction) and you can justify soundness (M-UNSAFE, M-UNSOUND).
  - If you cannot concisely explain why an `unsafe` block is sound, do not add it.

- **Panics vs Errors**
  - Panics mean “stop the program” (M-PANIC-IS-STOP) and are reserved for programmer bugs (M-PANIC-ON-BUG).
  - Do not use panics for ordinary I/O, configuration, or user errors; return structured error types instead.

- **Logging & Observability**
  - Use structured logging with message templates (M-LOG-STRUCTURED), not ad-hoc string formatting.
  - Name events, follow OpenTelemetry semantic conventions where applicable, and redact sensitive data.

- **Naming, Magic Values, APIs**
  - Use concise, precise names without weasel words (M-CONCISE-NAMES).
  - Document magic values and timeouts (M-DOCUMENTED-MAGIC) instead of leaving unexplained constants.
  - Prefer inherent impls for essential behavior (M-ESSENTIAL-FN-INHERENT) and regular functions over unnecessary associated-only constructors (M-REGULAR-FN).
  - Avoid APIs that leak external types or rely heavily on smart pointers/wrappers in signatures (M-DONT-LEAK-TYPES, M-AVOID-WRAPPERS).

- **Error Types**
  - Use canonical struct error types with meaningful fields and helpers (M-ERRORS-CANONICAL-STRUCTS) rather than generic error enums or stringly-typed errors.
  - Prefer domain-specific error enums/kinds with helpers like `is_io`, `is_protocol`, etc.

- **Concurrency & Async**
  - Ensure that types crossing threads or used in async contexts are soundly `Send` and follow the guidance in M-TYPES-SEND.
  - Ensure long-running async tasks have yield points for fairness and throughput (M-YIELD-POINTS).

- **Builders & Initialization**
  - For complex types or many optional parameters, prefer builders (M-INIT-BUILDER) and cascaded initialization patterns (M-INIT-CASCADED).
  - Services designed to be reused or shared should implement `Clone` where appropriate (M-SERVICES-CLONE).

- **Libraries, Features, and Test Utilities**
  - Keep features additive (M-FEATURES-ADDITIVE) and avoid introducing mutually exclusive feature flags without strong justification.
  - Libraries should work out of the box with sensible defaults (M-OOBE).
  - Test utilities and mock helpers must be feature-gated (M-TEST-UTIL) and not leak into production builds.

- **Statics, I/O, Mocking**
  - Avoid new global statics where possible (M-AVOID-STATICS); prefer dependency injection or scoped state.
  - Design new I/O and system-call code so it is mockable (M-MOCKABLE-SYSCALLS), especially for tests.

- **Linting & Static Verification**
  - Use `#[expect]` (with a reason) instead of `#[allow]` for lint overrides (M-LINT-OVERRIDE-EXPECT).
  - Assume the Rust and Clippy lint sets defined in `docs/coding-standards/README.md` are in force (M-STATIC-VERIFICATION); do not weaken them unless explicitly requested.

Agents SHOULD skim the relevant sections of `docs/coding-standards/README.md` before making non-trivial changes (new modules, unsafe code, public APIs, FFI, or performance-sensitive paths).

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

**On macOS (THIS PROJECT):**
- Install Rust 1.83+; run `install_metal_toolchain.sh` to set up Metal development.
- Fast debug build: `cargo build --features metal`.
- Release build: `cargo build --release --features metal`.
- Run server with a model id: `cargo run --release --features metal -- --m <huggingface-id> --ui-server` (add `--isq q4k` for in-situ quantization).
- Lint/format: `cargo fmt --all` and `cargo clippy --all-targets --features metal -D warnings`.
- Tests: `cargo test --features metal`; for specific packages: `cargo test --package candle-vllm-core --lib --features metal`.

**Other Platforms (for reference):**
- Release build (CUDA, single node): `cargo build --release --features cuda,nccl`.
- Run server (CUDA): `cargo run --release --features cuda,nccl -- --m <huggingface-id> --ui-server`.
- Tests (CPU-only): `cargo test --all --all-features`; GPU-dependent code should be guarded behind feature flags.

### Inference Engine Architecture
The inference engine uses `prometheus-parking-lot` for resource-aware scheduling:
- Resource tracking with KV-cache block accounting
- Automatic backpressure and request rejection when capacity is exhausted
- Async-first design with native tokio integration
- Lock-free capacity tracking using atomic operations

See [`docs/PARKING_LOT_SCHEDULER.md`](docs/PARKING_LOT_SCHEDULER.md) for detailed documentation.

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
