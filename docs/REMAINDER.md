# Remainder Plan for Model Switch Queue & Streaming

Context: Implement single-active-model scheduling with per-model queues, thread pool execution, OpenAI-aligned streaming status chunks, and telemetry. Phase 5 (US3) incomplete—requires request scheduling and streaming/busy semantics.

## Remaining Steps

1) Scheduling & Queue Core
- Add per-model request queues with max size (default 10, configurable); enforce 503 on enqueue overflow.
- Introduce worker pool abstraction for inference execution; enqueue when pool full, cancel on client disconnect if possible.
- Implement fairness: drain current active-model queue; only switch when empty or on explicit request; coalesce requests targeting the same pending model during switch.

2) Model Switch FSM Wiring
- Extend `ModelManager` to drive actual load/unload: states Ready → Switching → Loading → Ready/Error; one load at a time.
- Integrate loader hook to perform model load; on success, mark active and drain that model queue; on failure, fail queued requests with clear error.
- Prevent concurrent switches; serialize switches; respect idle_unload policy if present.

3) Chat/Responses Path Integration
- Route `/v1/chat/completions` / Responses to scheduler instead of direct engine call.
- Streaming requests: on enqueue, emit OpenAI-format chunk with `choices[*].delta.content` explaining queued status and extension metadata (e.g., `candle_metadata.status=queued|waiting_for_model|loading`, position, model, optional eta_ms).
- Non-streaming: wait in queue; if wait exceeds timeout (default 30s, configurable) return 503 busy with `{error:{message,type:"server_busy",queued:true}}`.
- When switching is required, emit streaming “waiting for model change” chunk; on load start, emit “model loading” chunk; then stream normal tokens. Apply same semantics for all waiting requests of that target model.

4) Status & Endpoints
- Enhance `/v1/models/status` to reflect FSM states (Ready/Idle/Switching/Loading/Error), active model, last_error, queue lengths, switch_requested_at.
- Consider exposing `/v1/models/select` to trigger switch and return immediate switch status; include 202 vs 409/404 semantics.

5) Configuration & Defaults
- Add config for queue size (default 10), non-streaming wait timeout (default 30s), heartbeat interval for streaming queue notices (~1s), and worker pool size.
- Document precedence between CLI flags and `models.yaml` aliases; log overrides.

6) Telemetry & Logging (hooks, optional)
- Track queue length per model, wait times, switch/load durations, failures; expose counters/gauges behind feature or minimal interface.
- Structured logs for switch start/success/fail and 503 busy events.

7) Tests & Validation
- Add unit tests for `ModelManager` FSM (enqueue, coalescing, single active, failure path).
- Add integration tests (CPU-safe) for queue overflow → 503, streaming queued chunk formatting, and model switch path with fake loader.
- Ensure `cargo fmt`, `cargo clippy --all-targets --all-features -D warnings`, and `cargo test --all --all-features` pass.

## Notes
- Streaming status chunks should align with OpenAI chat completion chunks; include extension field for metadata without breaking clients.
- Keep single active model invariant; no concurrent loads; switch requests serialize.
- Make all new defaults configurable but safe for small deployments.
