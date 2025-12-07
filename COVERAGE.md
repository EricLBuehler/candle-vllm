# Test Coverage Guide

This document describes how to run and achieve 100% test coverage for candle-vllm.

## Prerequisites

Install the required tools:

```bash
# Install llvm-tools component
rustup component add llvm-tools-preview

# Install cargo-llvm-cov
cargo install cargo-llvm-cov
```

## Running Coverage

### Generate HTML Coverage Report

```bash
# Run all tests with coverage and generate HTML report
cargo llvm-cov --all-features --workspace --html --output-dir coverage/report

# Open the report in your browser
open coverage/report/index.html
```

### Generate Text Summary

```bash
# Quick coverage summary
cargo llvm-cov --all-features --workspace --summary-only

# Detailed text report
cargo llvm-cov --all-features --workspace --text > coverage/report.txt
```

### Generate LCOV for CI/CD

```bash
# Generate lcov.info for codecov/coveralls
cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
```

## Coverage Goals

We aim for **100% coverage** across all modules:

- **Line coverage**: Every line executed
- **Branch coverage**: Every if/else/match arm tested
- **Function coverage**: Every function called
- **Edge case coverage**: Boundary conditions tested

## Coverage by Module

### Core Modules (crates/candle-vllm-core)

- Backend (`src/backend/`) - KV cache, device backends
- OpenAI API (`src/openai/`) - Conversation, logits, pipelines, tool parsing
- Parking Lot (`src/parking_lot/`) - Executor, worker pool, streaming registry
- Scheduler (`src/scheduler/`) - Cache engine, block manager

### Server Modules (crates/candle-vllm-server)

- Config (`src/config/`) - Configuration structs and parsing
- Handlers (`src/handlers/`) - HTTP endpoints, streaming, error handling
- Server - Initialization and lifecycle

### Other Crates

- `candle-vllm-openai` - OpenAI-compatible API surface
- `candle-vllm-responses` - Response schemas and serialization

## Identifying Coverage Gaps

1. Generate HTML report: `cargo llvm-cov --all-features --workspace --html --output-dir coverage/report`
2. Open `coverage/report/index.html` in browser
3. Look for red (uncovered) or yellow (partially covered) lines
4. Write tests to cover those lines
5. Re-run coverage to verify

## Coverage Best Practices

1. **Unit tests**: Test individual functions with various inputs
2. **Integration tests**: Test end-to-end workflows
3. **Error path tests**: Test all error conditions
4. **Edge case tests**: Test boundary values, empty inputs, max values
5. **Mock tests**: Use mocks for expensive operations (GPU, network)
6. **Property tests**: Use proptest for invariant testing

## CI/CD Integration

Coverage is automatically checked in CI via `.github/workflows/coverage.yml`.

PRs must maintain 100% coverage to pass.

## Troubleshooting

### "No coverage data found"

Make sure you're running tests with coverage instrumentation:
```bash
cargo llvm-cov test --all-features --workspace
```

### Feature-gated code not covered

Run with all features enabled:
```bash
cargo llvm-cov --all-features --workspace
```

### Uncoverable code

Use `#[cfg(not(coverage))]` for truly uncoverable code like debug-only paths.
