import os, time, math, csv, argparse, threading, warnings
from dataclasses import dataclass, asdict
from statistics import mean
from typing import List, Dict

# Quiet common noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional GPU util via NVML
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

@dataclass
class RunRow:
    batch_size:int; run_idx:int; device:str; use_cache:bool
    prompt_len_tokens_avg:float; new_tokens_per_seq:int
    batch_time_s:float; tokens_per_sec:float
    gpu_util_mean:float; gpu_util_p95:float
    est_cache_mb_prompt:float; est_cache_mb_total:float

def setup_model(name:str="distilgpt2"):
    tok = AutoTokenizer.from_pretrained(name)
    # Ensure left padding for decoder-only architectures
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    return tok, model, device

def make_prompts(n:int, base:str="I have a dream"):
    tails = [
        "",
        " that one day this nation will rise up",
        " that my four little children will one day live in a nation",
        " where they will not be judged by the color of their skin",
        " but by the content of their character",
        " and let freedom ring from every hill and molehill",
    ]
    return [ (base + " " + tails[i % len(tails)]).strip() for i in range(n) ]

class GPUUtilSampler:
    def __init__(self, interval_s:float=0.05):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples = []
        self._handle = None

    def start(self):
        if not (_HAS_NVML and torch.cuda.is_available()):
            return
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.samples.append(float(util.gpu))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self):
        if not (_HAS_NVML and torch.cuda.is_available()):
            return
        self._stop.set()
        try:
            self._th.join(timeout=1.0)
        except Exception:
            pass
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")

    def p95(self) -> float:
        return float(np.percentile(self.samples, 95)) if self.samples else float("nan")

def estimate_kv_cache_bytes(B:int, T:int, n_layer:int, n_head:int, head_dim:int, dtype_bytes:int=2) -> int:
    # Per layer we store K and V: [B, n_head, T, head_dim] each → ×2
    return int(n_layer * 2 * B * n_head * T * head_dim * dtype_bytes)

def percentile(xs:List[float], p:float) -> float:
    return float(np.percentile(np.array(xs, dtype=float), p)) if xs else float("nan")

def run_batch(tok, model, device:str, batch_size:int, max_new_tokens:int, use_cache:bool, repeats:int, dtype_bytes:int) -> List[RunRow]:
    cfg = model.config
    n_layer = int(getattr(cfg, "n_layer", 6))
    n_head  = int(getattr(cfg, "n_head", 12))
    d_model = int(getattr(cfg, "n_embd", 768))
    head_dim = d_model // n_head

    rows: List[RunRow] = []

    for run_idx in range(1, repeats + 1):
        prompts = make_prompts(batch_size)
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Average prompt length as float (avoid integer mean error)
        prompt_len_avg = float(attention_mask.sum(dim=1).float().mean().item())

        # Warmup to stabilize kernels
        with torch.inference_mode():
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=True, top_k=50, top_p=0.9,
                use_cache=use_cache,
                pad_token_id=tok.pad_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()

        # Theoretical KV cache (prompt-only and total)
        cache_prompt_mb = estimate_kv_cache_bytes(batch_size, int(prompt_len_avg), n_layer, n_head, head_dim, dtype_bytes) / (1024**2)
        cache_total_mb  = estimate_kv_cache_bytes(batch_size, int(prompt_len_avg) + max_new_tokens, n_layer, n_head, head_dim, dtype_bytes) / (1024**2)

        # Sample GPU util during timed generation
        sampler = GPUUtilSampler(0.05)
        sampler.start()

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True, top_k=50, top_p=0.9,
                use_cache=use_cache,
                pad_token_id=tok.pad_token_id,
            )
            if device == "cuda":
                torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        sampler.stop()

        new_tokens = int(out.shape[-1] - input_ids.shape[-1])  # same across batch when padding used
        total_new = new_tokens * batch_size
        tps = (total_new / dt) if dt > 0 else float("inf")

        row = RunRow(
            batch_size=batch_size, run_idx=run_idx, device=device, use_cache=use_cache,
            prompt_len_tokens_avg=prompt_len_avg, new_tokens_per_seq=new_tokens,
            batch_time_s=dt, tokens_per_sec=tps,
            gpu_util_mean=sampler.mean(), gpu_util_p95=sampler.p95(),
            est_cache_mb_prompt=cache_prompt_mb, est_cache_mb_total=cache_total_mb
        )
        rows.append(row)

        print(f"[bs={batch_size}] run {run_idx}: time={dt:.3f}s  tps={tps:.1f} tok/s  "
              f"gpu_mean={row.gpu_util_mean:.1f}%  p95={row.gpu_util_p95:.1f}%  "
              f"cache_prompt≈{cache_prompt_mb:.1f}MB  cache_total≈{cache_total_mb:.1f}MB")

    return rows

def aggregate(results: List[RunRow]) -> List[Dict]:
    out = []
    for bs in sorted({r.batch_size for r in results}):
        grp = [r for r in results if r.batch_size == bs]
        latencies = [r.batch_time_s for r in grp]
        throughputs = [r.tokens_per_sec for r in grp]
        gpu_means = [r.gpu_util_mean for r in grp if not math.isnan(r.gpu_util_mean)]
        gpu_p95s  = [r.gpu_util_p95  for r in grp if not math.isnan(r.gpu_util_p95)]
        out.append({
            "batch_size": bs,
            "runs": len(grp),
            "p50_latency_s": percentile(latencies, 50),
            "p95_latency_s": percentile(latencies, 95),
            "mean_throughput_tokens_per_s": mean(throughputs),
            "gpu_util_mean_percent": mean(gpu_means) if gpu_means else float("nan"),
            "gpu_util_p95_percent": percentile(gpu_p95s, 95) if gpu_p95s else float("nan"),
            "est_cache_mb_prompt_avg": mean([r.est_cache_mb_prompt for r in grp]),
            "est_cache_mb_total_avg":  mean([r.est_cache_mb_total  for r in grp]),
        })
    return out

def save_csv(path:str, rows):
    if not rows:
        return
    if hasattr(rows[0], "__dataclass_fields__"):
        fields = list(asdict(rows[0]).keys())
        data = [asdict(r) for r in rows]
    else:
        fields = sorted({k for row in rows for k in row.keys()})
        data = rows
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in data:
            w.writerow(r)
    print("Saved:", os.path.abspath(path))

def make_plots(summary: List[Dict], prefix:str="exp2_batching"):
    summary = sorted(summary, key=lambda d: d["batch_size"])
    bs   = [d["batch_size"] for d in summary]
    p50  = [d["p50_latency_s"] for d in summary]
    p95  = [d["p95_latency_s"] for d in summary]
    thr  = [d["mean_throughput_tokens_per_s"] for d in summary]
    gpu  = [d["gpu_util_mean_percent"] for d in summary]
    ctot = [d["est_cache_mb_total_avg"] for d in summary]

    plt.figure(); plt.plot(bs, p50, marker="o", label="p50"); plt.plot(bs, p95, marker="o", label="p95")
    plt.xlabel("Batch size (# prompts)"); plt.ylabel("Latency per batch (s)")
    plt.title("Experiment 2: Latency vs Batch Size"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_p50_latency_s.png", bbox_inches="tight"); print("Saved:", os.path.abspath(f"{prefix}_p50_latency_s.png"))
    # Save p95 separately for clarity
    plt.figure(); plt.plot(bs, p95, marker="o")
    plt.xlabel("Batch size (# prompts)"); plt.ylabel("Latency p95 (s)")
    plt.title("Experiment 2: p95 Latency vs Batch Size"); plt.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_p95_latency_s.png", bbox_inches="tight"); print("Saved:", os.path.abspath(f"{prefix}_p95_latency_s.png"))

    plt.figure(); plt.plot(bs, thr, marker="o")
    plt.xlabel("Batch size (# prompts)"); plt.ylabel("Throughput (tokens/sec, total)")
    plt.title("Experiment 2: Throughput vs Batch Size"); plt.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_throughput.png", bbox_inches="tight"); print("Saved:", os.path.abspath(f"{prefix}_throughput.png"))

    plt.figure(); plt.plot(bs, gpu, marker="o")
    plt.xlabel("Batch size"); plt.ylabel("GPU Utilization (mean %)")
    plt.title("Experiment 2: GPU Utilization vs Batch Size"); plt.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_gpu_mean.png", bbox_inches="tight"); print("Saved:", os.path.abspath(f"{prefix}_gpu_mean.png"))

    plt.figure(); plt.plot(bs, ctot, marker="o")
    plt.xlabel("Batch size"); plt.ylabel("Estimated KV Cache (MB, prompt+generated)")
    plt.title("Experiment 2: KV-Cache vs Batch Size (theoretical)"); plt.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_cache_MB.png", bbox_inches="tight"); print("Saved:", os.path.abspath(f"{prefix}_cache_MB.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=str, default="1,2,4")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--use_cache", type=lambda x: str(x).lower()=="true", default=True)
    ap.add_argument("--dtype_bytes", type=int, default=2)  # 2 for fp16/bf16; 4 for fp32
    args = ap.parse_args()

    torch.manual_seed(42)
    tok, model, device = setup_model("distilgpt2")

    results: List[RunRow] = []
    batch_sizes = [int(x) for x in args.batches.split(",") if x.strip()]
    for bs in batch_sizes:
        results += run_batch(tok, model, device, bs, args.max_new_tokens, args.use_cache, args.repeats, args.dtype_bytes)

    save_csv("exp2_batching_results.csv", results)
    summary = aggregate(results)
    save_csv("exp2_batching_summary.csv", summary)
    make_plots(summary, "exp2_batching")

if __name__ == "__main__":
    main()
