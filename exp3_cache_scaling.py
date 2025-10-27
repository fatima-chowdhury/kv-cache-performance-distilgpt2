import os, time, math, csv, argparse, warnings
from dataclasses import dataclass, asdict
from statistics import mean
from typing import List, Dict

# Quiet common noise (TF preload messages in Colab)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
try:
    import psutil
except Exception:
    psutil = None

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class Row:
    seq_len:int
    run_idx:int
    device:str
    use_cache:bool
    prompt_tokens:int
    new_tokens:int
    total_time_s:float
    latency_per_token_ms:float
    tokens_per_sec:float
    peak_mem_measured_mb:float
    theo_cache_prompt_mb:float
    theo_cache_total_mb:float


def setup_model(name="distilgpt2"):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # decoder-only; left pad to align context
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


def make_exact_len_ids(tok, target_len:int) -> torch.Tensor:
    """
    Create a 1 x target_len input_ids tensor EXACTLY target_len tokens long.
    We generate text tokens by repeating a base phrase and slicing to target_len.
    """
    base = "I have a dream that one day freedom will ring from every hill and valley. "
    text = base * 200  # ensure plenty of tokens
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if ids.shape[0] < target_len:
        # fallback: repeat again if needed (unlikely)
        reps = (target_len // ids.shape[0]) + 1
        ids = tok(base * reps, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    ids = ids[:target_len].unsqueeze(0)  # shape [1, target_len]
    return ids


def estimate_kv_cache_bytes(B:int, T:int, n_layer:int, n_head:int, head_dim:int, dtype_bytes:int=2) -> int:
    # For decoder-only attention: store K and V per layer: [B, n_head, T, head_dim] each → ×2
    return int(n_layer * 2 * B * n_head * T * head_dim * dtype_bytes)


def measure_cpu_rss_mb() -> float:
    if psutil is None: return float("nan")
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024**2)


def run_once(tok, model, device:str, seq_len:int, max_new_tokens:int, use_cache:bool, dtype_bytes:int) -> Row:
    cfg = model.config
    n_layer = int(getattr(cfg, "n_layer", 6))
    n_head  = int(getattr(cfg, "n_head", 12))
    d_model = int(getattr(cfg, "n_embd", 768))
    head_dim = d_model // n_head

    # Build exact-length prompt (B=1)
    ids = make_exact_len_ids(tok, seq_len).to(device)
    attn = torch.ones_like(ids, dtype=torch.long, device=device)

    # Warmup for stable kernels & caches
    with torch.inference_mode():
        _ = model.generate(
            input_ids=ids, attention_mask=attn,
            max_new_tokens=8, use_cache=use_cache,
            do_sample=True, top_k=50, top_p=0.9,
            pad_token_id=tok.pad_token_id,
        )
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    else:
        mem_before = measure_cpu_rss_mb()

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids, attention_mask=attn,
            max_new_tokens=max_new_tokens, use_cache=use_cache,
            do_sample=True, top_k=50, top_p=0.9,
            pad_token_id=tok.pad_token_id,
        )
        if device == "cuda":
            torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    new_tokens = int(out.shape[-1] - ids.shape[-1])
    lat_per_token_ms = (dt / max(1, new_tokens)) * 1000.0
    tps = (new_tokens / dt) if dt > 0 else float("inf")

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        mem_after = measure_cpu_rss_mb()
        peak_mb = (mem_after - mem_before) if (not math.isnan(mem_after)) else float("nan")

    theo_prompt_mb = estimate_kv_cache_bytes(1, seq_len, n_layer, n_head, head_dim, dtype_bytes) / (1024**2)
    theo_total_mb  = estimate_kv_cache_bytes(1, seq_len + max_new_tokens, n_layer, n_head, head_dim, dtype_bytes) / (1024**2)

    return Row(
        seq_len=seq_len, run_idx=0, device=device, use_cache=use_cache,
        prompt_tokens=seq_len, new_tokens=new_tokens, total_time_s=dt,
        latency_per_token_ms=lat_per_token_ms, tokens_per_sec=tps,
        peak_mem_measured_mb=float(peak_mb),
        theo_cache_prompt_mb=float(theo_prompt_mb),
        theo_cache_total_mb=float(theo_total_mb),
    )


def aggregate(rows: List[Row]) -> List[Dict]:
    out=[]
    for T in sorted({r.seq_len for r in rows}):
        G = [r for r in rows if r.seq_len == T]
        out.append(dict(
            seq_len=T,
            runs=len(G),
            device=G[0].device if G else "",
            use_cache=G[0].use_cache if G else True,
            avg_latency_per_token_ms=mean([r.latency_per_token_ms for r in G]),
            p50_latency_per_token_ms=float(np.percentile([r.latency_per_token_ms for r in G], 50)),
            p95_latency_per_token_ms=float(np.percentile([r.latency_per_token_ms for r in G], 95)),
            avg_tokens_per_sec=mean([r.tokens_per_sec for r in G]),
            avg_peak_mem_measured_mb=mean([r.peak_mem_measured_mb for r in G]),
            theo_cache_prompt_mb=G[0].theo_cache_prompt_mb if G else float("nan"),
            theo_cache_total_mb =G[0].theo_cache_total_mb  if G else float("nan"),
        ))
    return out


def save_csv(path:str, rows):
    if not rows: return
    if hasattr(rows[0], "__dataclass_fields__"):
        fields=list(asdict(rows[0]).keys()); data=[asdict(r) for r in rows]
    else:
        fields=sorted({k for row in rows for k in row.keys()}); data=rows
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in data: w.writerow(r)
    print("Saved:", os.path.abspath(path))


def make_plots(summary: List[Dict], prefix="exp3_cache_scaling"):
    summary=sorted(summary, key=lambda d:d["seq_len"])
    x=[d["seq_len"] for d in summary]

    # Theoretical cache memory (prompt and total) vs sequence length
    y_prompt=[d["theo_cache_prompt_mb"] for d in summary]
    y_total =[d["theo_cache_total_mb"]  for d in summary]
    plt.figure()
    plt.plot(x, y_prompt, marker="o", label="Theoretical cache (prompt)")
    plt.plot(x, y_total,  marker="o", label="Theoretical cache (prompt+generated)")
    plt.xlabel("Prompt length (tokens)"); plt.ylabel("KV cache (MB)")
    plt.title("Experiment 3: Theoretical KV Cache vs Sequence Length (B=1)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(f"{prefix}_cache_vs_seq.png", bbox_inches="tight"); print("Saved:", os.path.abspath(f"{prefix}_cache_vs_seq.png"))

    # Latency per token vs sequence length (empirical)
    y_lat=[d["avg_latency_per_token_ms"] for d in summary]
    plt.figure()
    plt.plot(x, y_lat, marker="o")
    plt.xlabel("Prompt length (tokens)"); plt.ylabel("Latency per token (ms)")
    plt.title("Experiment 3: Latency per token vs Sequence Length (B=1)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_latency_per_token.png", bbox_inches="tight"); print("Saved:", os.path.abspath(f"{prefix}_latency_per_token.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_lens", type=str, default="32,128,256", help="comma-separated prompt lengths")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--use_cache", type=lambda x: str(x).lower()=="true", default=True)
    ap.add_argument("--dtype_bytes", type=int, default=2, help="2 for fp16/bf16, 4 for fp32")
    args = ap.parse_args()

    torch.manual_seed(42)
    tok, model, device = setup_model("distilgpt2")

    rows: List[Row] = []
    seqs = [int(s) for s in args.seq_lens.split(",") if s.strip()]
    for T in seqs:
        for i in range(args.repeats):
            r = run_once(tok, model, device, T, args.max_new_tokens, args.use_cache, args.dtype_bytes)
            r.run_idx = i+1
            rows.append(r)
            print(f"[T={T}] run {r.run_idx}: time={r.total_time_s:.3f}s  "
                  f"lat/token={r.latency_per_token_ms:.2f} ms  tps={r.tokens_per_sec:.1f} tok/s  "
                  f"meas_peak={r.peak_mem_measured_mb:.1f} MB  "
                  f"theo_prompt≈{r.theo_cache_prompt_mb:.1f} MB  theo_total≈{r.theo_cache_total_mb:.1f} MB")

    save_csv("exp3_cache_scaling_results.csv", rows)

    summary = aggregate(rows)
    save_csv("exp3_cache_scaling_summary.csv", summary)

    make_plots(summary, "exp3_cache_scaling")

   

if __name__ == "__main__":
    main()
