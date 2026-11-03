"""Benchmark the serving overhead of the RAG LoRA adapter on a vLLM server.

Pure HTTP client (openai.AsyncOpenAI + httpx); it does NOT import torch or
transformers and measures serving performance only. Answer *quality* is
measured by a separate eval-harness repo.

Workload: the 50 held-out validation rows (the last 50 rows of
$DATASET_DIR/squad-v2-rag/train.jsonl), cycled to --num-requests and sent as
the harness's one-message chat request. Every request uses temperature=0,
max_tokens=64 and ignore_eos=True, so each generates exactly 64 tokens and the
base-vs-adapter difference is purely LoRA serving overhead.

Run:
    uv run python serve_and_bench.py --tag base
    uv run python serve_and_bench.py --tag lora --adapter-name rag-lora-1b
    uv run python serve_and_bench.py --tag lora --adapter-name rag-lora-1b --load-only
    uv run python serve_and_bench.py --compare base.json lora.json

Env: DATASET_DIR (workload), BENCHMARK_DIR (results).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

BASE_MODEL = "google/gemma-3-1b-it"
DATA_SUBDIR = "squad-v2-rag"  # written by scripts/prepare_data.py
ADAPTER_MOUNT = "/adapters"  # $CHECKPOINT_DIR is mounted here in the vLLM container
VAL_SIZE = 50  # the last 50 rows form the validation workload
MAX_TOKENS = 64  # fixed output length (ignore_eos=True)
DEFAULT_BASE_URL = "http://localhost:8001"
SERVER_TIMEOUT_S = 60.0  # how long to poll for the server to come up
REQUEST_TIMEOUT_S = 300.0  # per-request HTTP timeout
WARMUP_MAX = 8  # warmup requests per level (capped by the level's concurrency)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def require_env(name: str) -> str:
    """Read a required env var, failing fast with a clear message."""
    val = os.environ.get(name)
    assert val, f"{name} is not set (is direnv loaded in this shell?)"
    return val


def load_val_messages() -> list[str]:
    """The user messages of the last VAL_SIZE rows (the validation workload)."""
    dataset_dir = require_env("DATASET_DIR")
    path = Path(dataset_dir) / DATA_SUBDIR / "train.jsonl"
    assert path.exists(), f"{path} is missing; run scripts/prepare_data.py first"
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) >= VAL_SIZE, f"need at least {VAL_SIZE} rows, got {len(rows)}"
    val = rows[-VAL_SIZE:]
    msgs: list[str] = []
    for r in val:
        content = r["prompt"][0]["content"]
        assert isinstance(content, str), f"{r['id']}: user message is not a string"
        msgs.append(content)
    return msgs


def gpu_name() -> str:
    """GPU name from nvidia-smi; 'unknown' if the query fails (tolerated)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return lines[0] if lines else "unknown"
    except Exception:
        return "unknown"


def percentile(values: list[float], p: float) -> float:
    """The p-th percentile (0-100) of latencies, using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def pct_delta(base: float, lora: float) -> str:
    """Signed percent change of lora relative to base, as a string."""
    if base == 0:
        return "n/a"
    return f"{(lora - base) / base * 100.0:+.1f}%"


def load_json(path: str) -> dict[str, Any]:
    """Read a bench JSON file written by this script."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# vLLM server endpoints (plain HTTP, not the OpenAI client)
# ---------------------------------------------------------------------------
async def poll_models(
    client: httpx.AsyncClient, base_url: str, timeout: float
) -> list[str]:
    """Poll GET /v1/models until the server answers, or fail with a clear error."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = await client.get(f"{base_url}/v1/models")
            r.raise_for_status()
            data = r.json()
            ids: list[str] = []
            for m in data.get("data", []):
                mid = m.get("id")
                assert isinstance(mid, str), f"unexpected /v1/models payload: {data!r}"
                ids.append(mid)
            return ids
        except Exception as e:  # keep polling through transient errors
            last_err = e
            await asyncio.sleep(1.0)
    raise RuntimeError(
        f"vLLM server at {base_url} did not become ready within {timeout:.0f}s; "
        f"last error: {last_err!r}. Is the container running?"
    )


async def get_version(client: httpx.AsyncClient, base_url: str) -> str:
    """GET /version, tolerating failure (always returns a readable string)."""
    try:
        r = await client.get(f"{base_url}/version")
        r.raise_for_status()
        text = r.text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text or "unknown"
        if isinstance(data, dict):
            v = data.get("version")
            return str(v) if v is not None else text
        return str(data)
    except Exception as e:
        return f"unknown ({type(e).__name__})"


async def unload_adapter(client: httpx.AsyncClient, base_url: str, name: str) -> None:
    """POST /v1/unload_lora_adapter, ignoring errors (it may not be loaded)."""
    try:
        await client.post(f"{base_url}/v1/unload_lora_adapter", json={"lora_name": name})
    except Exception:
        pass


async def load_adapter(client: httpx.AsyncClient, base_url: str, name: str) -> None:
    """Unload (ignoring errors), load from /adapters/<name>, confirm it's listed."""
    await unload_adapter(client, base_url, name)
    r = await client.post(
        f"{base_url}/v1/load_lora_adapter",
        json={"lora_name": name, "lora_path": f"{ADAPTER_MOUNT}/{name}"},
    )
    r.raise_for_status()
    models = await poll_models(client, base_url, timeout=30.0)
    assert name in models, f"adapter {name!r} not listed after load; models={models}"
    print(f"loaded adapter {name!r} (models now: {models})")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
async def one_request(
    client: AsyncOpenAI, model: str, message: str
) -> tuple[float, int]:
    """One chat completion; return (latency_seconds, output_tokens)."""
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        temperature=0,
        max_tokens=MAX_TOKENS,
        extra_body={"ignore_eos": True},  # vLLM honors ignore_eos in extra_body; the fixed 64-token workload depends on it
    )
    dt = time.perf_counter() - t0
    n = resp.usage.completion_tokens if resp.usage is not None else MAX_TOKENS
    return dt, n


async def run_level(
    client: AsyncOpenAI,
    model: str,
    messages: list[str],
    concurrency: int,
    num_requests: int,
) -> dict[str, float | int]:
    """Short warmup, then num_requests gated by a semaphore of size `concurrency`."""
    warmup_n = min(concurrency, WARMUP_MAX)
    await asyncio.gather(
        *(one_request(client, model, messages[i % len(messages)]) for i in range(warmup_n))
    )
    sem = asyncio.Semaphore(concurrency)

    async def gated(i: int) -> tuple[float, int]:
        async with sem:
            return await one_request(client, model, messages[i % len(messages)])

    t0 = time.perf_counter()
    results = await asyncio.gather(*(gated(i) for i in range(num_requests)))
    wall = time.perf_counter() - t0
    latencies = [r[0] for r in results]
    tokens = [r[1] for r in results]
    total_tokens = sum(tokens)
    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "req_per_s": round(num_requests / wall, 3),
        "out_tok_per_s": round(total_tokens / wall, 1),
        "p50_ms": round(percentile(latencies, 50) * 1000, 1),
        "p95_ms": round(percentile(latencies, 95) * 1000, 1),
        "total_out_tokens": total_tokens,
        "wall_s": round(wall, 2),
    }


def fmt_table(levels: list[dict[str, float | int]]) -> str:
    """A markdown table of per-level throughput and latency."""
    lines = [
        "| concurrency | req/s | out tok/s | p50 ms | p95 ms |",
        "|---|---|---|---|---|",
    ]
    for lv in levels:
        lines.append(
            f"| {lv['concurrency']} | {lv['req_per_s']} | {lv['out_tok_per_s']} "
            f"| {lv['p50_ms']} | {lv['p95_ms']} |"
        )
    return "\n".join(lines)


def compare(a_path: str, b_path: str) -> None:
    """Print a markdown comparison of two bench files (A=base, B=adapter)."""
    a = load_json(a_path)
    b = load_json(b_path)
    la = a.get("tag", "A")
    lb = b.get("tag", "B")
    la_levels: dict[int, dict[str, Any]] = {
        lv["concurrency"]: lv for lv in a.get("levels", [])
    }
    lb_levels: dict[int, dict[str, Any]] = {
        lv["concurrency"]: lv for lv in b.get("levels", [])
    }
    common = sorted(set(la_levels) & set(lb_levels))
    assert common, f"no common concurrency levels between {a_path} and {b_path}"
    print(f"Comparing **{la}** ({a_path}) vs **{lb}** ({b_path})")
    for c in common:
        la_lv = la_levels[c]
        lb_lv = lb_levels[c]
        print(f"\n### concurrency {c}")
        print(f"| metric | {la} | {lb} | delta % |")
        print("|---|---|---|---|")
        for metric, label in (
            ("req_per_s", "req/s"),
            ("out_tok_per_s", "out tok/s"),
            ("p50_ms", "p50 ms"),
            ("p95_ms", "p95 ms"),
        ):
            va = float(la_lv[metric])
            vb = float(lb_lv[metric])
            print(f"| {label} | {va} | {vb} | {pct_delta(va, vb)} |")


async def async_main(args: argparse.Namespace) -> None:
    """Poll the server, optionally load the adapter, then benchmark each level."""
    base_url: str = args.base_url
    tag: str = args.tag or (args.adapter_name or "base")
    model: str = args.adapter_name if args.adapter_name else args.base_model

    async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT_S)) as http_client:
        model_ids = await poll_models(http_client, base_url, SERVER_TIMEOUT_S)
        print(f"server ready at {base_url}; models={model_ids}")
        version = await get_version(http_client, base_url)
        gpu = gpu_name()

        if args.adapter_name:
            await load_adapter(http_client, base_url, args.adapter_name)
        if args.load_only:
            assert args.adapter_name, "--load-only requires --adapter-name"
            print(f"loaded adapter {args.adapter_name!r}; ready for the eval harness")
            return

        messages = load_val_messages()
        concurrencies = [int(c) for c in args.concurrency.split(",") if c.strip()]
        assert concurrencies, "no concurrency levels given"
        print(
            f"workload: {len(messages)} val messages cycled to {args.num_requests} requests; "
            f"model={model}; concurrency={concurrencies}; "
            f"max_tokens={MAX_TOKENS} (ignore_eos, temperature=0)"
        )

        levels: list[dict[str, float | int]] = []
        async with AsyncOpenAI(
            base_url=f"{base_url}/v1", api_key="unused"
        ) as openai_client:
            for c in concurrencies:
                lv = await run_level(openai_client, model, messages, c, args.num_requests)
                levels.append(lv)
                print(
                    f"  concurrency {c:>2}: {lv['req_per_s']} req/s, "
                    f"{lv['out_tok_per_s']} out tok/s, "
                    f"p50 {lv['p50_ms']} ms, p95 {lv['p95_ms']} ms"
                )

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        result = {
            "tag": tag,
            "base_url": base_url,
            "vllm_version": version,
            "gpu_name": gpu,
            "served_model": model,
            "adapter_name": args.adapter_name,
            "settings": {
                "base_model": args.base_model,
                "num_requests": args.num_requests,
                "concurrency": concurrencies,
                "max_tokens": MAX_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
                "val_messages": len(messages),
            },
            "levels": levels,
            "timestamp_utc": ts,
        }
        benchmark_dir = require_env("BENCHMARK_DIR")
        out_path = Path(benchmark_dir) / f"bench_{tag}_{ts}.json"
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {out_path}\n")
        print(fmt_table(levels))


def parse_args() -> argparse.Namespace:
    """Parse the command-line flags."""
    p = argparse.ArgumentParser(
        description="Benchmark RAG LoRA adapter serving overhead on vLLM."
    )
    p.add_argument("--tag", default=None, help="label for this run (e.g. base or lora)")
    p.add_argument(
        "--adapter-name", default=None, help="adapter to serve; omit for the base model"
    )
    p.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="vLLM server base URL"
    )
    p.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help=f"base model name on the server (default {BASE_MODEL})",
    )
    p.add_argument(
        "--concurrency", default="1,16", help="comma-separated concurrency levels"
    )
    p.add_argument(
        "--num-requests", type=int, default=200, help="requests per concurrency level"
    )
    p.add_argument(
        "--load-only", action="store_true", help="load the adapter and exit"
    )
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASE_JSON", "LORA_JSON"),
        default=None,
        help="compare two bench JSON files (no server needed)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare is not None:
        assert len(args.compare) == 2, "--compare needs exactly two files"
        compare(args.compare[0], args.compare[1])
        return
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
