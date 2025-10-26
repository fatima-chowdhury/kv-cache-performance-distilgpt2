# exp1_kvcache.py
# Experiment 1 — With and Without KV-Cache (distilgpt2)
# - Measures total generation time, tokens/sec, and memory usage.
# - Saves CSV logs and labeled plots for your report.

import os
import time
import json
import math
import csv
import argparse
from dataclasses import dataclass, asdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

try:
    import psutil
except ImportError:
    psutil = None  # CPU memory delta will be skipped if psutil is unavailable


@dataclass
class RunResult:
    use_cache: bool
    run_idx: int
    device: str
    total_time_s: float
    prompt_tokens: int
    new_tokens: int
    tokens_per_sec: float
    peak_mem_mb: float  # GPU peak if CUDA; CPU RSS delta if CPU (approx)
    note: str = ""


def setup_model(model_name: str = "distilgpt2", device: str | None = None):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # DistilGPT2 has no pad token; align pad with eos to avoid warnings
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()
    return tokenizer, model, device


def count_new_tokens(tokenizer, prompt_ids: torch.Tensor, output_ids: torch.Tensor) -> int:
    # new tokens = output length - prompt length
    return int(output_ids.shape[-1] - prompt_ids.shape[-1])


def measure_gpu_peak_mb() -> float:
    """Return CUDA peak memory in MB (since last reset)."""
    if not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def get_cpu_rss_mb() -> float:
    """Return current process RSS in MB (approx)."""
    if psutil is None:
        return float("nan")
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def run_one(
    tokenizer,
    model,
    device: str,
    prompt: str,
    max_new_tokens: int,
    use_cache: bool,
    run_idx: int,
) -> RunResult:
    # Prepare inputs
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Warmup for fair timing (compilation/cudnn/autotune/cold-cache)
    with torch.inference_mode():
        _ = model.generate(
            input_ids=input_ids,
            max_new_tokens=8,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Memory + timer setup
    note = ""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    else:
        if psutil is None:
            note = "psutil not installed; CPU mem delta unavailable."

    cpu_mem_mb_before = get_cpu_rss_mb() if device == "cpu" and psutil else float("nan")

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
        )
        if device == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    new_tokens = count_new_tokens(tokenizer, input_ids, output_ids)
    tps = (new_tokens / total_time) if total_time > 0 else float("nan")

    if device == "cuda":
        peak_mem_mb = measure_gpu_peak_mb()
    else:
        cpu_mem_mb_after = get_cpu_rss_mb() if psutil else float("nan")
        peak_mem_mb = (cpu_mem_mb_after - cpu_mem_mb_before) if (psutil and not math.isnan(cpu_mem_mb_before)) else float("nan")

    return RunResult(
        use_cache=use_cache,
        run_idx=run_idx,
        device=device,
        total_time_s=total_time,
        prompt_tokens=int(input_ids.shape[-1]),
        new_tokens=int(new_tokens),
        tokens_per_sec=float(tps),
        peak_mem_mb=float(peak_mem_mb),
        note=note,
    )


def aggregate(results: list[RunResult]) -> dict:
    def agg_for(flag: bool):
        rows = [r for r in results if r.use_cache == flag]
        n = len(rows)
        if n == 0:
            return {}
        mean = lambda xs: float(sum(xs) / len(xs)) if xs else float("nan")
        return {
            "use_cache": flag,
            "runs": n,
            "device": rows[0].device,
            "avg_total_time_s": mean([r.total_time_s for r in rows]),
            "avg_tokens_per_sec": mean([r.tokens_per_sec for r in rows]),
            "avg_peak_mem_mb": mean([r.peak_mem_mb for r in rows]),
            "avg_new_tokens": mean([r.new_tokens for r in rows]),
            "avg_prompt_tokens": mean([r.prompt_tokens for r in rows]),
        }

    return {
        "with_cache": agg_for(True),
        "without_cache": agg_for(False),
    }


def save_csv(path: str, rows: list[RunResult]):
    fieldnames = list(asdict(rows[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"Saved: {os.path.abspath(path)}")


def save_summary_csv(path: str, summary: dict):
    rows = []
    for key in ["with_cache", "without_cache"]:
        if summary.get(key):
            rows.append(summary[key])
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Saved: {os.path.abspath(path)}")


def plot_bars(summary: dict, out_prefix: str = "exp1_kvcache"):
    labels = ["use_cache=False", "use_cache=True"]
    # Ensure order: False, True
    no = summary.get("without_cache", {})
    yes = summary.get("with_cache", {})

    times = [no.get("avg_total_time_s", float("nan")), yes.get("avg_total_time_s", float("nan"))]
    tps = [no.get("avg_tokens_per_sec", float("nan")), yes.get("avg_tokens_per_sec", float("nan"))]
    mems = [no.get("avg_peak_mem_mb", float("nan")), yes.get("avg_peak_mem_mb", float("nan"))]

    # Latency
    plt.figure()
    plt.bar(labels, times)
    plt.ylabel("Average Total Time (s)")
    plt.title("Experiment 1: Latency vs. KV-Cache")
    plt.savefig(f"{out_prefix}_latency.png", bbox_inches="tight")
    print(f"Saved: {os.path.abspath(f'{out_prefix}_latency.png')}")

    # Throughput
    plt.figure()
    plt.bar(labels, tps)
    plt.ylabel("Average Tokens/sec")
    plt.title("Experiment 1: Throughput vs. KV-Cache")
    plt.savefig(f"{out_prefix}_tokens_per_sec.png", bbox_inches="tight")
    print(f"Saved: {os.path.abspath(f'{out_prefix}_tokens_per_sec.png')}")

    # Memory
    plt.figure()
    plt.bar(labels, mems)
    plt.ylabel("Average Peak Memory (MB)")
    plt.title("Experiment 1: Memory vs. KV-Cache")
    plt.savefig(f"{out_prefix}_peak_mem.png", bbox_inches="tight")
    print(f"Saved: {os.path.abspath(f'{out_prefix}_peak_mem.png')}")


def auto_discussion(summary: dict) -> str:
    """Generate a concise, ready-to-paste discussion paragraph."""
    w = summary.get("with_cache", {})
    n = summary.get("without_cache", {})
    if not w or not n:
        return "Insufficient data for discussion."

    def pct_delta(a, b):
        if (a is None) or (b is None) or (a != a) or (b != b) or b == 0:
            return float("nan")
        return 100.0 * (a - b) / b

    d_tps = pct_delta(w.get("avg_tokens_per_sec"), n.get("avg_tokens_per_sec"))
    d_time = pct_delta(n.get("avg_total_time_s"), w.get("avg_total_time_s"))  # time reduction positive if cache helps
    d_mem = pct_delta(w.get("avg_peak_mem_mb"), n.get("avg_peak_mem_mb"))

    lines = [
        "Discussion (Experiment 1 — KV-Cache):",
        f"- Device: {w.get('device', 'unknown')}",
        f"- With KV-cache, throughput changed by ~{d_tps:.1f}% vs. no-cache.",
        f"- Total latency changed by ~{d_time:.1f}% (positive means latency improved with cache).",
        f"- Peak memory changed by ~{d_mem:.1f}% (positive means more memory with cache).",
        "These trends match the expectation that KV-cache speeds up autoregressive decoding by reusing past keys/values,",
        "often increasing throughput and reducing per-token latency, at the cost of extra memory to store the cache.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="I have a dream")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed runs per condition")
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    args = parser.parse_args()

    torch.manual_seed(42)

    tokenizer, model, device = setup_model(args.model_name)

    results: list[RunResult] = []

    for use_cache in [False, True]:
        # Timed runs
        for i in range(args.repeats):
            r = run_one(
                tokenizer=tokenizer,
                model=model,
                device=device,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                use_cache=use_cache,
                run_idx=i + 1,
            )
            results.append(r)
            print(
                f"[use_cache={use_cache}] run {r.run_idx} | "
                f"time={r.total_time_s:.3f}s | new_tokens={r.new_tokens} | "
                f"toks/sec={r.tokens_per_sec:.2f} | peak_mem_mb={r.peak_mem_mb:.1f} | {r.note}"
            )

    # Save raw results
    save_csv("exp1_kvcache_results.csv", results)

    # Summary + plots
    summary = aggregate(results)
    save_summary_csv("exp1_kvcache_summary.csv", summary)
    plot_bars(summary, out_prefix="exp1_kvcache")

    # Auto discussion text (also saved as .txt for easy copy into report)
    discussion = auto_discussion(summary)
    print("\n" + discussion + "\n")
    with open("exp1_kvcache_discussion.txt", "w", encoding="utf-8") as f:
        f.write(discussion)
    print(f"Saved: {os.path.abspath('exp1_kvcache_discussion.txt')}")


if __name__ == "__main__":
    main()
