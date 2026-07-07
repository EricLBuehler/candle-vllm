import argparse
import asyncio
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import APIStatusError, AsyncOpenAI
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


# Run candle-vllm service:
#   cargo run --release -- --port 2000 --model-id <MODEL_ID> <MODEL_TYPE>
#
# Example benchmark:
#   python3 examples/benchmark.py --num-prompts 64 --input-lens 128,512,2048 --output-lens 128,512 --concurrency 8


PROMPT_SEEDS = [
    "Explain Rust ownership, borrowing, lifetimes, async runtimes, and practical debugging techniques.",
    "Discuss transformer inference, KV cache scheduling, batching, prefill, decode, and throughput tradeoffs.",
    "Write a detailed systems design review covering API shape, latency budgets, observability, and failure modes.",
    "Compare quantization formats, memory bandwidth, attention kernels, and deployment constraints for LLM serving.",
    "Describe a careful migration plan for a production service with tests, rollout gates, and rollback strategy.",
    "Analyze how to build reliable command line tools with clear errors, structured logs, and benchmark reporting.",
]


@dataclass
class RequestResult:
    request_id: int
    input_len_target: int
    output_len_target: int
    success: bool
    latency_s: float
    ttft_s: Optional[float] = None
    output_text: str = ""
    error: Optional[str] = None
    error_kind: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_time_ms: int = 0
    completion_time_ms: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: Optional[str] = None
    started_at_s: Optional[float] = None
    first_chunk_s: Optional[float] = None
    last_chunk_s: Optional[float] = None
    chunk_count: int = 0
    reasoning_text: str = ""
    reasoning_chunk_tokens: int = 0

    @property
    def output_tokens_per_s(self) -> float:
        return safe_div(self.completion_tokens, self.latency_s)

    @property
    def total_tokens_per_s(self) -> float:
        return safe_div(self.total_tokens, self.latency_s)

    @property
    def prefill_s(self) -> float:
        if self.prompt_time_ms:
            return self.prompt_time_ms / 1000.0
        return self.ttft_s or 0.0

    @property
    def input_tokens_per_prefill_s(self) -> float:
        return safe_div(self.prompt_tokens, self.prefill_s)


@dataclass
class BenchmarkCase:
    input_len: int
    output_len: int
    concurrency: int
    results: List[RequestResult] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def successful(self) -> List[RequestResult]:
        return [result for result in self.results if result.success]

    @property
    def failed(self) -> List[RequestResult]:
        return [result for result in self.results if not result.success]

    @property
    def effective_concurrency(self) -> int:
        if not self.successful:
            return 1
        return max(1, min(self.concurrency or len(self.successful), len(self.successful)))


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def fmt_float(value: float, digits: int = 2) -> str:
    if value == 0:
        return "0"
    return f"{value:.{digits}f}"


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.2f}"


def parse_lengths(value: Optional[str], fallback: int) -> List[int]:
    if not value:
        return [fallback]
    lengths = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            lengths.append(int(raw))
    return lengths or [fallback]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_prompt(target_tokens: int, request_id: int) -> str:
    """Build a deterministic synthetic prompt with roughly target_tokens words."""
    seed = PROMPT_SEEDS[request_id % len(PROMPT_SEEDS)]
    words = seed.split()
    filler = (
        "Use precise technical language, include concrete numbers, discuss tradeoffs, "
        "and keep the reasoning grounded in operational details."
    ).split()
    output = [
        "You are benchmarking an LLM serving endpoint.",
        f"Request id: {request_id}.",
        "Answer the user request directly.",
        "User request:",
    ]
    while len(" ".join(output).split()) < target_tokens:
        output.extend(words)
        output.extend(filler)
    return " ".join(" ".join(output).split()[:target_tokens])


def build_history_text(target_tokens: int, request_id: int, turn: int, label: str) -> str:
    seed = PROMPT_SEEDS[(request_id + turn) % len(PROMPT_SEEDS)]
    words = (
        f"{label} history turn {turn} for request {request_id}. "
        f"{seed} Keep the context deterministic for benchmarking."
    ).split()
    output = []
    while len(output) < target_tokens:
        output.extend(words)
    return " ".join(output[:target_tokens])


def build_messages(args: argparse.Namespace, prompt: str, request_id: int) -> List[Dict[str, str]]:
    messages = []
    for turn in range(args.history_turns):
        messages.append(
            {
                "role": "user",
                "content": build_history_text(args.history_tokens, request_id, turn, "User"),
            }
        )
        assistant_message = {
            "role": "assistant",
            "content": build_history_text(args.history_tokens, request_id, turn, "Assistant"),
        }
        if args.history_reasoning_tokens:
            assistant_message["reasoning_content"] = build_history_text(
                args.history_reasoning_tokens,
                request_id,
                turn,
                "Assistant reasoning",
            )
        messages.append(assistant_message)
    messages.append({"role": "user", "content": prompt})
    return messages


def usage_value(usage: Any, name: str, default: int = 0) -> int:
    if usage is None:
        return default
    if isinstance(usage, dict):
        return int(usage.get(name, default) or default)
    return int(getattr(usage, name, default) or default)


def nested_usage_value(usage: Any, section: str, name: str, default: int = 0) -> int:
    if usage is None:
        return default
    value = usage.get(section) if isinstance(usage, dict) else getattr(usage, section, None)
    if value is None:
        return default
    if isinstance(value, dict):
        return int(value.get(name, default) or default)
    return int(getattr(value, name, default) or default)


def response_body_text(exc: APIStatusError) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return ""
    try:
        text = response.text
    except Exception:
        return ""
    return text.strip()


def format_request_error(exc: Exception) -> Tuple[str, str]:
    if isinstance(exc, APIStatusError):
        body = response_body_text(exc)
        status = getattr(exc, "status_code", None) or getattr(exc.response, "status_code", None)
        if body:
            return "HTTP", f"HTTP {status}: {body[:300]}"
        return "HTTP", f"HTTP {status} from endpoint"
    return "Transport", str(exc)


def make_extra_body(args: argparse.Namespace) -> Dict[str, Any]:
    extra_body = {
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repeat_last_n": args.repeat_last_n,
        "thinking": args.thinking,
        "ignore_eos": args.ignore_eos,
    }
    return {key: value for key, value in extra_body.items() if value is not None}


def apply_usage_to_result(
    result: RequestResult,
    usage: Any,
    args: argparse.Namespace,
    input_len: int,
) -> None:
    result.prompt_tokens = usage_value(usage, "prompt_tokens")
    result.completion_tokens = usage_value(usage, "completion_tokens")
    result.total_tokens = usage_value(usage, "total_tokens")
    result.prompt_time_ms = usage_value(usage, "prompt_time_costs")
    result.completion_time_ms = usage_value(usage, "completion_time_costs")
    result.cached_tokens = nested_usage_value(usage, "prompt_tokens_details", "cached_tokens")
    result.reasoning_tokens = nested_usage_value(
        usage, "completion_tokens_details", "reasoning_tokens"
    )
    if result.reasoning_tokens == 0:
        result.reasoning_tokens = result.reasoning_chunk_tokens

    if usage:
        return

    fallback_reasoning_tokens = len(result.reasoning_text.split())
    fallback_content_tokens = len(result.output_text.split())
    result.prompt_tokens = input_len + args.history_turns * (
        2 * args.history_tokens + args.history_reasoning_tokens
    )
    result.reasoning_tokens = result.reasoning_chunk_tokens or fallback_reasoning_tokens
    result.completion_tokens = max(
        1, fallback_content_tokens + fallback_reasoning_tokens
    )
    result.total_tokens = result.prompt_tokens + result.completion_tokens


async def run_one_request(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    request_id: int,
    prompt: str,
    input_len: int,
    output_len: int,
    semaphore: asyncio.Semaphore,
    scheduled_at: float,
) -> RequestResult:
    await asyncio.sleep(max(0.0, scheduled_at - time.perf_counter()))

    async with semaphore:
        started = time.perf_counter()
        result = RequestResult(
            request_id=request_id,
            input_len_target=input_len,
            output_len_target=output_len,
            success=False,
            latency_s=0.0,
            started_at_s=started,
        )
        usage = None

        try:
            stream = await client.chat.completions.create(
                model=args.model,
                messages=build_messages(args, prompt, request_id),
                max_tokens=output_len,
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=make_extra_body(args),
                timeout=args.timeout,
            )

            async for chunk in stream:
                now = time.perf_counter()
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason:
                    result.finish_reason = finish_reason

                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                reasoning = (
                    getattr(delta, "reasoning_content", None) if delta is not None else None
                )
                text = content or reasoning or ""
                if text:
                    if result.ttft_s is None:
                        result.ttft_s = now - started
                        result.first_chunk_s = now
                    result.last_chunk_s = now
                    result.chunk_count += 1
                    if content:
                        result.output_text += content
                    if reasoning:
                        result.reasoning_text += reasoning
                        result.reasoning_chunk_tokens += 1

            ended = time.perf_counter()
            result.latency_s = ended - started
            apply_usage_to_result(result, usage, args, input_len)
            result.success = True
        except Exception as exc:
            result.latency_s = time.perf_counter() - started
            if usage is not None:
                apply_usage_to_result(result, usage, args, input_len)
                result.success = True
            else:
                result.error_kind, result.error = format_request_error(exc)

        return result


def scheduled_offsets(num_prompts: int, request_rate: float) -> List[float]:
    if request_rate <= 0:
        return [0.0] * num_prompts
    return [i / request_rate for i in range(num_prompts)]


async def run_case(
    console: Console,
    client: AsyncOpenAI,
    args: argparse.Namespace,
    input_len: int,
    output_len: int,
) -> BenchmarkCase:
    prompts = [build_prompt(input_len, i) for i in range(args.num_prompts)]
    semaphore = asyncio.Semaphore(args.concurrency or args.num_prompts)
    start = time.perf_counter()
    offsets = scheduled_offsets(args.num_prompts, args.request_rate)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]benchmark[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(
            f"input={input_len} output={output_len}", total=args.num_prompts
        )
        tasks = [
            asyncio.create_task(
                run_one_request(
                    client,
                    args,
                    request_id=i,
                    prompt=prompts[i],
                    input_len=input_len,
                    output_len=output_len,
                    semaphore=semaphore,
                    scheduled_at=start + offsets[i],
                )
            )
            for i in range(args.num_prompts)
        ]
        results = []
        for task in asyncio.as_completed(tasks):
            results.append(await task)
            progress.advance(task_id)

    duration = time.perf_counter() - start
    return BenchmarkCase(
        input_len=input_len,
        output_len=output_len,
        concurrency=args.concurrency or args.num_prompts,
        results=results,
        duration_s=duration,
    )


def summarize(case: BenchmarkCase) -> Dict[str, float]:
    ok = case.successful
    latencies = [r.latency_s for r in ok]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    prompt_times = [r.prompt_time_ms / 1000.0 for r in ok if r.prompt_time_ms]
    completion_times = [r.completion_time_ms / 1000.0 for r in ok if r.completion_time_ms]
    prefill_times = [r.prefill_s for r in ok if r.prefill_s]
    prompt_tokens = sum(r.prompt_tokens for r in ok)
    completion_tokens = sum(r.completion_tokens for r in ok)
    total_tokens = sum(r.total_tokens for r in ok)
    prompt_time_s = sum(prompt_times)
    completion_time_s = sum(completion_times)
    model_parallelism = case.effective_concurrency
    has_usage_timing = bool(prompt_time_s or completion_time_s)
    prefill_done_times = [
        (r.started_at_s or 0.0) + r.prefill_s
        for r in ok
        if r.started_at_s is not None and r.prefill_s
    ]
    prefill_start_times = [
        r.started_at_s for r in ok if r.started_at_s is not None and r.prefill_s
    ]
    prefill_window_s = (
        max(prefill_done_times) - min(prefill_start_times)
        if prefill_done_times and prefill_start_times
        else safe_div(prompt_time_s, model_parallelism) if prompt_time_s else 0.0
    )
    prefill_sum_s = sum(prefill_times)
    model_input_time_s = prefill_window_s
    model_output_time_s = safe_div(completion_time_s, model_parallelism) if completion_time_s else case.duration_s
    model_total_time_s = (
        model_input_time_s + model_output_time_s if has_usage_timing else case.duration_s
    )
    avg_request_output_model_throughput = (
        safe_div(completion_tokens, completion_time_s)
        if completion_time_s
        else statistics.mean([r.output_tokens_per_s for r in ok]) if ok else 0.0
    )
    request_input_throughputs = [r.input_tokens_per_prefill_s for r in ok if r.prefill_s]
    avg_request_input_model_throughput = (
        statistics.mean(request_input_throughputs) if request_input_throughputs else 0.0
    )
    avg_request_total_model_throughput = (
        safe_div(total_tokens, prompt_time_s + completion_time_s)
        if has_usage_timing
        else statistics.mean([r.total_tokens_per_s for r in ok]) if ok else 0.0
    )

    return {
        "successful": len(ok),
        "failed": len(case.failed),
        "duration_s": case.duration_s,
        "request_throughput": safe_div(len(ok), case.duration_s),
        "input_throughput": safe_div(prompt_tokens, model_input_time_s),
        "input_throughput_avg_request": safe_div(prompt_tokens, prefill_sum_s),
        "output_throughput": safe_div(completion_tokens, model_output_time_s),
        "total_throughput": safe_div(total_tokens, model_total_time_s),
        "avg_request_input_throughput": avg_request_input_model_throughput,
        "avg_request_output_model_throughput": avg_request_output_model_throughput,
        "avg_request_output_throughput": statistics.mean([r.output_tokens_per_s for r in ok]) if ok else 0.0,
        "avg_request_total_model_throughput": avg_request_total_model_throughput,
        "avg_request_total_throughput": statistics.mean([r.total_tokens_per_s for r in ok]) if ok else 0.0,
        "timing_source": 1.0 if has_usage_timing else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": sum(r.cached_tokens for r in ok),
        "reasoning_tokens": sum(r.reasoning_tokens for r in ok),
        "avg_input_tokens": statistics.mean([r.prompt_tokens for r in ok]) if ok else 0.0,
        "avg_output_tokens": statistics.mean([r.completion_tokens for r in ok]) if ok else 0.0,
        "max_input_tokens": max([r.prompt_tokens for r in ok], default=0),
        "max_output_tokens": max([r.completion_tokens for r in ok], default=0),
        "latency_avg_ms": statistics.mean(latencies) * 1000.0 if latencies else 0.0,
        "latency_p50_ms": percentile(latencies, 50) * 1000.0,
        "latency_p90_ms": percentile(latencies, 90) * 1000.0,
        "latency_p95_ms": percentile(latencies, 95) * 1000.0,
        "latency_p99_ms": percentile(latencies, 99) * 1000.0,
        "latency_max_ms": max(latencies, default=0.0) * 1000.0,
        "ttft_avg_ms": statistics.mean(ttfts) * 1000.0 if ttfts else 0.0,
        "ttft_p50_ms": percentile(ttfts, 50) * 1000.0,
        "ttft_p95_ms": percentile(ttfts, 95) * 1000.0,
        "ttft_p99_ms": percentile(ttfts, 99) * 1000.0,
        "prefill_window_ms": prefill_window_s * 1000.0,
        "prefill_avg_ms": statistics.mean(prefill_times) * 1000.0 if prefill_times else 0.0,
        "prefill_p95_ms": percentile(prefill_times, 95) * 1000.0,
        "server_prefill_avg_ms": statistics.mean(prompt_times) * 1000.0 if prompt_times else 0.0,
        "server_prefill_p95_ms": percentile(prompt_times, 95) * 1000.0,
        "server_decode_avg_ms": statistics.mean(completion_times) * 1000.0 if completion_times else 0.0,
        "server_decode_p95_ms": percentile(completion_times, 95) * 1000.0,
        "tpot_avg_ms": safe_div(sum(r.completion_time_ms for r in ok), completion_tokens),
    }


def render_case_summary(console: Console, case: BenchmarkCase) -> None:
    stats = summarize(case)
    highlight_style = "bold yellow"

    def highlighted(value: str) -> Text:
        return Text(value, style=highlight_style)

    table = Table(
        title=f"Benchmark: input={case.input_len}, output={case.output_len}",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    rows = [
        ("Successful requests", f"{int(stats['successful'])}", "Failed requests", f"{int(stats['failed'])}"),
        ("Benchmark duration (s)", fmt_float(stats["duration_s"]), "Request throughput (req/s)", fmt_float(stats["request_throughput"])),
        (highlighted("Total input tokens"), highlighted(str(stats['prompt_tokens'])), highlighted("Total output tokens"), highlighted(str(stats['completion_tokens']))),
        ("Max input tokens", f"{int(stats['max_input_tokens'])}", "Max output tokens", f"{int(stats['max_output_tokens'])}"),
        ("Avg input tokens", fmt_float(stats["avg_input_tokens"]), "Avg output tokens", fmt_float(stats["avg_output_tokens"])),
        (highlighted("Prefill input throughput (tok/s)"), highlighted(fmt_float(stats["input_throughput"])), highlighted("Output throughput (tok/s)"), highlighted(fmt_float(stats["output_throughput"]))),
        (highlighted("Avg prefill tok/s per request"), highlighted(fmt_float(stats["avg_request_input_throughput"])), highlighted("Avg output tok/s per request"), highlighted(fmt_float(stats["avg_request_output_model_throughput"]))),
        ("TTFT avg (ms)", fmt_float(stats["ttft_avg_ms"]), "TTFT p95 (ms)", fmt_float(stats["ttft_p95_ms"])),
        ("Prefill window (ms)", fmt_float(stats["prefill_window_ms"]), "Prefill avg (ms)", fmt_float(stats["prefill_avg_ms"])),
        ("Prefill p95 (ms)", fmt_float(stats["prefill_p95_ms"]), "TPOT decode (ms/token)", fmt_float(stats["tpot_avg_ms"])),
        ("Reasoning tokens", f"{int(stats['reasoning_tokens'])}", "Total tokens", f"{int(stats['total_tokens'])}"),
    ]
    for row in rows:
        table.add_row(*row)
    console.print(table)

    if case.failed:
        errors = Table(title="HTTP/transport failures", box=box.SIMPLE, header_style="bold red")
        errors.add_column("Request", justify="right")
        errors.add_column("Kind")
        errors.add_column("Error")
        for result in case.failed[:5]:
            errors.add_row(
                str(result.request_id),
                result.error_kind or "Error",
                result.error or "unknown error",
            )
        console.print(errors)


def render_comparison(console: Console, cases: Sequence[BenchmarkCase]) -> None:
    if len(cases) <= 1:
        return

    throughput = Table(title="Throughput Comparison", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    throughput.add_column("Input", justify="right")
    throughput.add_column("Output", justify="right")
    throughput.add_column("OK", justify="right")
    throughput.add_column("Req/s", justify="right")
    throughput.add_column("Prefill tok/s", justify="right")
    throughput.add_column("Output tok/s", justify="right")
    throughput.add_column("Total tok/s", justify="right")
    throughput.add_column("Avg req out tok/s", justify="right")

    latency = Table(title="Latency Comparison", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    latency.add_column("Input", justify="right")
    latency.add_column("Output", justify="right")
    latency.add_column("Avg ms", justify="right")
    latency.add_column("P50 ms", justify="right")
    latency.add_column("P95 ms", justify="right")
    latency.add_column("P99 ms", justify="right")
    latency.add_column("TTFT avg", justify="right")
    latency.add_column("TTFT p95", justify="right")
    latency.add_column("Max in/out", justify="right")

    for case in cases:
        stats = summarize(case)
        throughput.add_row(
            str(case.input_len),
            str(case.output_len),
            str(int(stats["successful"])),
            fmt_float(stats["request_throughput"]),
            fmt_float(stats["input_throughput"]),
            fmt_float(stats["output_throughput"]),
            fmt_float(stats["total_throughput"]),
            fmt_float(stats["avg_request_output_model_throughput"]),
        )
        latency.add_row(
            str(case.input_len),
            str(case.output_len),
            fmt_float(stats["latency_avg_ms"]),
            fmt_float(stats["latency_p50_ms"]),
            fmt_float(stats["latency_p95_ms"]),
            fmt_float(stats["latency_p99_ms"]),
            fmt_float(stats["ttft_avg_ms"]),
            fmt_float(stats["ttft_p95_ms"]),
            f"{int(stats['max_input_tokens'])}/{int(stats['max_output_tokens'])}",
        )
    console.print(throughput)
    console.print(latency)


def render_request_metrics(console: Console, case: BenchmarkCase, limit: int) -> None:
    if limit <= 0:
        return

    ok = sorted(case.successful, key=lambda result: result.request_id)[:limit]
    if not ok:
        return

    table = Table(title="Per-request prefill metrics", box=box.SIMPLE, header_style="bold cyan")
    table.add_column("Request", justify="right")
    table.add_column("Input tokens", justify="right")
    table.add_column("Prefill ms", justify="right")
    table.add_column("Input tok/s", justify="right")
    table.add_column("TTFT ms", justify="right")
    table.add_column("Timing")

    for result in ok:
        timing_source = "server" if result.prompt_time_ms else "ttft"
        table.add_row(
            str(result.request_id),
            str(result.prompt_tokens),
            fmt_float(result.prefill_s * 1000.0),
            fmt_float(result.input_tokens_per_prefill_s),
            fmt_ms(result.ttft_s or 0.0),
            timing_source,
        )
    console.print(table)


def render_samples(console: Console, cases: Sequence[BenchmarkCase], sample_count: int, chars: int) -> None:
    if sample_count <= 0:
        return
    for case in cases:
        ok = case.successful[:sample_count]
        for result in ok:
            text = result.output_text.strip()
            if len(text) > chars:
                text = text[:chars].rstrip() + "..."
            console.print(
                Panel(
                    text or "<empty>",
                    title=f"Sample output #{result.request_id} input={case.input_len} output={case.output_len}",
                    border_style="cyan",
                )
            )


async def main(args: argparse.Namespace) -> None:
    console = Console()
    random.seed(args.seed)
    base_url = args.base_url or f"http://localhost:{args.port}/v1/"
    client = AsyncOpenAI(api_key=args.api_key, base_url=base_url, max_retries=args.max_retries)
    input_lens = parse_lengths(args.input_lens, args.input_len)
    output_lens = parse_lengths(args.output_lens, args.output_len)

    console.print(
        Panel(
            "\n".join(
                [
                    f"Endpoint: {base_url}",
                    f"Model: {args.model}",
                    f"Requests per case: {args.num_prompts}",
                    f"Concurrency: {args.concurrency or args.num_prompts}",
                    f"Request rate: {'unlimited' if args.request_rate <= 0 else args.request_rate}",
                    f"Input lengths: {', '.join(map(str, input_lens))}",
                    f"Output lengths: {', '.join(map(str, output_lens))}",
                    f"History turns: {args.history_turns}",
                ]
            ),
            title="candle-vllm benchmark",
            border_style="cyan",
        )
    )

    cases = []
    for input_len in input_lens:
        for output_len in output_lens:
            case = await run_case(console, client, args, input_len, output_len)
            cases.append(case)
            render_case_summary(console, case)
            render_request_metrics(console, case, args.print_request_metrics)

    render_comparison(console, cases)
    render_samples(console, cases, args.print_samples, args.sample_chars)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a candle-vllm OpenAI-compatible chat endpoint.")
    parser.add_argument("--base-url", default=None, help="Full OpenAI-compatible base URL. Overrides --port.")
    parser.add_argument("--port", default=2000, type=int, help="candle-vllm server port.")
    parser.add_argument("--api-key", default="EMPTY", help="API key placeholder for local servers.")
    parser.add_argument("--model", default="any", help="Model name sent to the server.")
    parser.add_argument("--num-prompts", "--batch", dest="num_prompts", default=8, type=int)
    parser.add_argument("--concurrency", default=8, type=int, help="Maximum in-flight requests.")
    parser.add_argument("--request-rate", default=0.0, type=float, help="Requests per second; <=0 sends immediately.")
    parser.add_argument("--input-len", "--input_len", default=4096, type=int, help="Synthetic input length target.")
    parser.add_argument("--output-len", "--output_len", "--max-tokens", "--max_tokens", dest="output_len", default=1024, type=int)
    parser.add_argument("--input-lens", default=None, help="Comma-separated input length sweep, e.g. 128,512,2048.")
    parser.add_argument("--output-lens", default=None, help="Comma-separated output length sweep, e.g. 128,512.")
    parser.add_argument("--history-turns", "--history_turns", default=0, type=int, help="Synthetic user/assistant history turns to include before each benchmark prompt.")
    parser.add_argument("--history-tokens", "--history_tokens", default=64, type=int, help="Approximate tokens per synthetic history message.")
    parser.add_argument("--history-reasoning-tokens", "--history_reasoning_tokens", default=0, type=int, help="Approximate reasoning_content tokens per synthetic assistant history message.")
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top-p", "--top_p", default=None, type=float)
    parser.add_argument("--top-k", "--top_k", default=None, type=int)
    parser.add_argument("--min-p", "--min_p", default=None, type=float)
    parser.add_argument("--frequency-penalty", "--frequency_penalty", default=None, type=float)
    parser.add_argument("--presence-penalty", "--presence_penalty", default=None, type=float)
    parser.add_argument("--repeat-last-n", "--repeat_last_n", default=None, type=int)
    parser.add_argument("--thinking", default=None, type=str2bool, help="Enable thinking for reasoning models.")
    parser.add_argument("--ignore-eos", action="store_true", help="Ask server to ignore EOS and target max output length.")
    parser.add_argument("--timeout", default=600.0, type=float, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", default=2, type=int, help="HTTP retries per request for transient endpoint failures.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--print-request-metrics", default=0, type=int, help="Print per-request prefill metrics for the first N successful requests in each case.")
    parser.add_argument("--print-samples", default=0, type=int, help="Print N sample outputs per case.")
    parser.add_argument("--sample-chars", default=1200, type=int, help="Maximum characters per sample output.")
    return parser


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
