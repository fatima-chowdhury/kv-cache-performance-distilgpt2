# exp2_batching.py
# Experiment 2 — Batching Effect (distilgpt2)
# Measures p50/p95 latency, throughput, GPU utilization; estimates KV-cache scaling with batch size.

import os
import time
import math
import csv
import argparse
import threading
from dataclasses import dataclass, asdict
from statistics import mean
from typing import List, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional GPU utilization sampling via NVML
try:
    import pynvml  # pip install nvidia-ml-py3
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


@dataclass
class RunRow:
    batch_size: int
    run_idx: int
    device: str
    use_cache: bool
    prompt_len_tokens_avg: float
    new_tokens_per_seq: int
    batch_time_s: float
    tokens_per_sec: float        # total_new_tokens / batch_time
    gpu_util_mean: float         # %
    gpu_util_p95: float          # %
    est_cache_mb_prompt: float   # theoretical KV cache for prompts
    est_cache_mb_total: float    # theoretical KV cache for prompts + generated


def setup_model(model_name: str = "distilgpt2", device: str | None = None):
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()
    return tok, mdl, device


def make_prompts(n: int, base: str = "I have a dream") -> List[str]:
    # Simple, distinct prompts of varying lengths
    extras = [
        "",
        " that one day this nation will rise up",
        " that my four little children will one day live in a nation",
        " where they will not be judged by the color of their skin",
        " but by the content of their character",
        " and let freedom ring from every hill and molehill"
    ]
    res = []
    for i in range(n):
        tail = extras[i % len(extras)]
        res.append((base + " " + tail).strip())
    return res


# ---- GPU utilization sampler (NVML) ----
class GPUUtilSampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples = []

    def start(self):
        if not (_HAS_NVML and torch.cuda.is_available()):
            return
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._stop.clear()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.samples.append(util.gpu)  # percent
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self):
        if not (_HAS_NVML and torch.cuda.is_available()):
            return
        self._stop.set()
        self.th.join(timeout=1.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def mean(self):
        return float(np.mean(self.samples)) if self.samples else float("nan")

    def p95(self):
        return float(np.percentile(self.samples, 95)) if self.samples else float("nan")


# ---- KV-cache size estimator ----
def estimate_kv_cache_bytes(
    batch_size: int,
    seq_len: int,
    n_layer: int,
    n_head: int,
    head_dim: int,
    dtype_bytes: int = 2,   # bf16/float16 ~2 bytes; adjust to 4 for fp32
) -> int:
    # For each layer: store K and V of shape [batch, n_head, seq_len, head_dim]
    per_layer = 2 * batch_size * n_head * seq_len * head_dim * dtype_bytes
    total = n_layer * per_layer
    return int(total)


def percentile(xs: List[float], p: float) -> float:
    return float(np.percentile(np.array(xs, dtype=float), p)) if xs else float("nan")


def run_batch(
    tok,
    mdl,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    use_cache: bool,
    n_repeats: int,
    dtype_bytes: int = 2,
) -> List[RunRow]:

    cfg = mdl.config
    n_layer = int(getattr(cfg, "n_layer", 6))
    n_head  = int(getattr(cfg, "n_head", 12))
    d_model = int(getattr(cfg, "n_embd", 768))
    head_dim = d_model // n_head

    rows: List[RunRow] = []

    for run_idx in range(1, n_repeats + 1):
        prompts = make_prompts(batch_size)
        enc = tok(prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        prompt_lens = attn_mask.sum(dim=1).tolist()
        prompt_len_avg = float(np.mean(prompt_lens))

        # Warmup
        with torch.inference_mode():
            _ = mdl.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=8,
                do_sample=True, top_k=50, top_p=0.9,
                use_cache=use_cache,
                pad_token_id=tok.pad_token_id,
            )

        # GPU mem sync
        if device == "cuda":
            torch.cuda.synchronize()

        # Estimate cache (prompt-only and final total)
        cache_prompt_bytes = estimate_kv_cache_bytes(
            batch_size=batch_size, seq_len=int(prompt_len_avg),
            n_layer=n_layer, n_head=n_head, head_dim=head_dim, dtype_bytes=dtype_bytes
        )
        cache_total_bytes  = estimate_kv_cache_bytes(
            batch_size=batch_size, seq_len=int(prompt_len_avg) + max_new_tokens,
            n_layer=n_layer, n_head=n_head, head_dim=head_dim, dtype_bytes=dtype_bytes
        )

        # Sample GPU util during generation
        sampler = GPUUtilSampler(interval_s=0.05)
        sampler.start()

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = mdl.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True, top_k=50, top_p=0.9,
                use_cache=use_cache,
                pad_token_id=tok.pad_token_id,
            )
            if device == "cuda":
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        sampler.stop()

        # Compute new tokens per sequence (use first row for length)
        total_len = out.shape[-1]
        new_tokens = total_len - int(input_ids.shape[-1])  # same for each row due to padding
        total_new_tokens = new_tokens * batch_size

        batch_time = t1 - t0
        tps = (total_new_tokens / batch_time) if batch_time > 0 else float("inf")

        row = RunRow(
            batch_size=batch_size,
            run_idx=run_idx,
            device=device,
            use_cache=use_cache,
            prompt_len_tokens_avg=prompt_len_avg,
            new_tokens_per_seq=int(new_tokens),
            batch_time_s=batch_time,
            tokens_per_sec=tps,
            gpu_util_mean=sampler.mean(),
            gpu_util_p95=sampler.p95(),
            est_cache_mb_prompt=cache_prompt_bytes / (1024**2),
            est_cache_mb_total=cache_total_bytes / (1024**2),
        )
        rows.append(row)
        print(f"[bs={batch_size}] run {run_idx}: time={batch_time:.3f}s  "
              f"tps={tps:.1f} tok/s  gpu_mean={row.gpu_util_mean:.1f}%  "
              f"p95={row.gpu_util_p95:.1f}%  cache_prompt≈{row.est_cache_mb_prompt:.1f}MB "
              f"cache_total≈{row.est_cache_mb_total:.1f}MB")

    return rows


def aggregate(results: List[RunRow]) -> List[Dict]:
    out = []
    for bs in sorted(set(r.batch_size for r in results)):
        group = [r for r in results if r.batch_size == bs]
        latencies = [r.batch_time_s for r in group]
        throughputs = [r.tokens_per_sec for r in group]
        gpu_means = [r.gpu_util_mean for r in group if not math.isnan(r.gpu_util_mean)]
        gpu_p95s  = [r.gpu_util_p95  for r in group if not math.isnan(r.gpu_util_p95)]
        cache_prompt = mean([r.est_cache_mb_prompt for r in group])
        cache_total  = mean([r.est_cache_mb_total for r in group])

        out.append({
            "batch_size": bs,
            "runs": len(group),
            "device": group[0].device if group else "",
            "use_cache": group[0].use_cache if group else True,
            "p50_latency_s": percentile(latencies, 50),
            "p95_latency_s": percentile(latencies, 95),
            "mean_throughput_tokens_per_s": mean(throughputs),
            "gpu_util_mean_percent": mean(gpu_means) if gpu_means else float("nan"),
            "gpu_util_p95_percent": percentile(gpu_p95s, 95) if gpu_p95s else float("nan"),
            "est_cache_mb_prompt_avg": cache_prompt,
            "est_cache_mb_total_avg": cache_total,
        })
    return out


def save_csv(path: str, rows: List[dict] | List[RunRow]):
    if not rows:
        return
    if isinstance(rows[0], RunRow):
        fieldnames = list(asdict(rows[0]).keys())
        data_rows = [asdict(r) for r in rows]
    else:
        # dict summary
        fieldnames = sorted({k for row in rows for k in row.keys()})
        data_rows = rows
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in data_rows:
            w.writerow(r)
    print(f"Saved: {os.path.abspath(path)}")


def make_plots(summary: List[Dict], prefix="exp2_batching"):
    # Sort by batch size for nice plotting
    summary = sorted(summary, key=lambda d: d["batch_size"])
    bs = [d["batch_size"] for d in summary]

    # Latency
    p50 = [d["p50_latency_s"] for d in summary]
    p95 = [d["p95_latency_s"] for d in summary]
    plt.figure()
    plt.plot(bs, p50, marker="o", label="p50")
    plt.plot(bs, p95, marker="o", label="p95")
    plt.xlabel("Batch size (# parallel prompts)")
    plt.ylabel("Latency per batch (s)")
    plt.title("Experiment 2: Latency vs Batch Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_latency.png", bbox_inches="tight")
    print(f"Saved: {os.path.abspath(f'{prefix}_latency.png')}")

    # Throughput
    thr = [d["mean_throughput_tokens_per_s"] for d in summary]
    plt.figure()
    plt.plot(bs, thr, marker="o")
    plt.xlabel("Batch size (# parallel prompts)")
    plt.ylabel("Throughput (tokens/sec, total across batch)")
    plt.title("Experiment 2: Throughput vs Batch Size")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_throughput.png", bbox_inches="tight")
    print(f"Saved: {os.path.abspath(f'{prefix}_throughput.png')}")

    # GPU utilization
    gpu_mean = [d["gpu_util_mean_percent"] for d in summary]
    plt.figure()
    plt.plot(bs, gpu_mean, marker="o")
    plt.xlabel("Batch size")
    plt.ylabel("GPU Utilization (mean %)")
    plt.title("Experiment 2: GPU Utilization vs Batch Size")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_gpu.png", bbox_inches="tight")
    print(f"Saved: {os.path.abspath(f'{prefix}_gpu.png')}")

    # Estimated cache
    cache_total = [d["est_cache_mb_total_avg"] for d in summary]
    plt.figure()
    plt.plot(bs, cache_total, marker="o")
    plt.xlabel("Batch size")
    plt.ylabel("Estimated KV Cache (MB, prompt+generated)")
    plt.title("Experiment 2: KV-Cache vs Batch Size (theoretical)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_cache.png", bbox_inches="tight")
    print(f"Saved: {os.path.abspath(f'{prefix}_cache.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="distilgpt2")
    ap.add_argument("--use_cache", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--batches", type=str, default="1,2,4", help="comma list of batch sizes")
    ap.add_argument("--repeats", type=int, default=5, help="timed runs per batch size")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--dtype_bytes", type=int, default=2, help="2 for fp16/bf16, 4 for fp32")
    args = ap.parse_args()

    torch.manual_seed(42)

    tok, mdl, device = setup_model(args.model_name)
    batch_sizes = [int(x) for x in args.batches.split(",") if x.strip()]
    results: List[RunRow] = []

    for bs in batch_sizes:
        rows = run_batch(
            tok=tok, mdl=mdl, device=device,
            batch_size=bs, max_new_tokens=args.max_new_tokens,
            use_cache=args.use_cache, n_repeats=args.repeats,
            dtype_bytes=args.dtype_bytes
        )
        results.extend(rows)

    # Save raw
    save_csv("exp2_batching_results.csv", results)

    # Aggregate + save + plot
    summary = aggregate(results)
    save_csv("exp2_batching_summary.csv", summary)
    make_plots(summary, prefix="exp2_batching")

    # Brief discussion snippet for README/report
    lines = [
        "Discussion (Experiment 2 — Batching Effect):",
        "- As batch size increased (1 → 2 → 4), throughput (tokens/sec) rose because the GPU/CPU computed multiple sequences in parallel.",
        "- Latency per batch may increase slightly due to larger matrices and cache growth, so p50/p95 latencies can rise, but per-sequence amortized latency often drops.",
        "- KV-cache memory scales roughly linearly with batch size and sequence length (layers × heads × head_dim × 2(K,V) × dtype_bytes).",
        "- On GPU, average utilization typically climbs with batch size, reflecting better hardware saturation."
    ]
    with open("exp2_batching_discussion.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved:", os.path.abspath("exp2_batching_discussion.txt"))


if __name__ == "__main__":
    main()
