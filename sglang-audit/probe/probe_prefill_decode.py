"""SGLang prefill/decode time-distribution probe (C1).

Drives a running SGLang server's /start_profile + /stop_profile endpoints
(profile_by_stage=True, which records separate prefill and decode chrome
traces), runs a synthetic workload, then parses the traces and prints a
GPU-time breakdown by op class per stage.

Usage:
  python probe_prefill_decode.py --base-url http://127.0.0.1:30000 \
      --out ./probe-out --prefill-tokens 4096 --decode-steps 64 --num-steps 200

No sglang import: works against any server version exposing ProfileReq.
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict
from urllib import request as urlreq

# ---------------------------------------------------------------- HTTP helpers


def http_json(url, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urlreq.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urlreq.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def generate(base, prompt, max_tokens, timeout=600):
    return http_json(
        f"{base}/generate",
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_tokens,
                "ignore_eos": True,
            },
        },
        timeout=timeout,
    )


# ---------------------------------------------------------------- workload


def fake_prompt(n_tokens):
    # deterministic filler; tokenizer length is approximate, which is fine
    return "the quick brown fox jumps over the lazy dog. " * max(1, n_tokens // 10)


def run_workload(base, prefill_tokens, decode_steps, rounds):
    """rounds x (one long prefill + decode_steps decode iterations)."""
    t0 = time.perf_counter()
    for r in range(rounds):
        generate(base, fake_prompt(prefill_tokens), max_tokens=decode_steps)
    dt = time.perf_counter() - t0
    print(f"[probe] workload done in {dt:.1f}s ({rounds} rounds)")


# ---------------------------------------------------------------- trace parse

OP_CLASSES = [
    ("GEMM", r"gemm|gemv|cutlass|cublas|nvjet|matmul|mat_mul|w4a16|mxfp4|mmq|sgemm|hgemm|einsum|mmvq|mmv"),
    ("MoE", r"moe|topk|top_k|swiglu|silu|expert|align_block|moe_sum"),
    ("Attention", r"mla|flash|attn|attention|mqa|fa[234]|rope|softmax|sinkhorn"),
    ("Comm", r"nccl|all[_-]?reduce|reduce[_-]?scatter|all[_-]?gather|broadcast|p2p|custom[_-]?all"),
    ("Memcpy", r"memcpy|memset|copy[_-]?engine|DtoD|HtoD|DtoH"),
    ("Norm/Elem", r"norm|rms|elementwise|vectorized|unrolled|cast|quant|dequant|fill|index|scatter|gather"),
]


def classify(name):
    low = name.lower()
    for label, pat in OP_CLASSES:
        if re.search(pat, low):
            return label
    return "Other"


def parse_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    by_class = defaultdict(float)
    by_kernel = defaultdict(float)
    total = 0.0
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime"):
            if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
        dur = ev.get("dur", 0) / 1000.0  # us -> ms
        name = ev.get("name", "?")
        cls = classify(name)
        by_class[cls] += dur
        by_kernel[(cls, name[:90])] += dur
        total += dur
    return total, by_class, by_kernel


def report(path):
    total, by_class, by_kernel = parse_trace(path)
    stage = "decode" if "-decode" in os.path.basename(path) else (
        "prefill" if "-prefill" in os.path.basename(path) else "mixed"
    )
    print(f"\n=== {os.path.basename(path)}  [stage={stage}]  GPU total {total:.1f} ms")
    for cls, ms in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls:12s} {ms:9.1f} ms  {100*ms/max(total,1e-9):5.1f}%")
    print("  top kernels:")
    top = sorted(by_kernel.items(), key=lambda x: -x[1])[:12]
    for (cls, name), ms in top:
        print(f"    {ms:8.1f} ms {100*ms/max(total,1e-9):5.1f}%  [{cls}] {name}")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--out", default="./probe-out")
    ap.add_argument("--prefill-tokens", type=int, default=4096)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--num-steps", type=int, default=400,
                    help="forward steps the server records before auto-stop")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--traces-only", action="store_true",
                    help="skip workload; just parse traces in --out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not args.traces_only:
        base = args.base_url.rstrip("/")
        # warmup (also warms Triton JIT so compile time is outside the window)
        for _ in range(args.warmup):
            generate(base, fake_prompt(256), max_tokens=8)
        started = time.time()
        http_json(f"{base}/start_profile", {
            "output_dir": os.path.abspath(args.out),
            "num_steps": args.num_steps,
            "activities": ["GPU"],
            "profile_by_stage": True,
            "profile_prefix": "probe",
            "detailed_annotations": True,
        })
        try:
            run_workload(base, args.prefill_tokens, args.decode_steps, args.rounds)
        finally:
            try:
                http_json(f"{base}/stop_profile", {})
            except Exception as e:
                print(f"[probe] stop_profile: {e}")
        # wait for traces to land
        deadline = time.time() + 180
        while time.time() < deadline:
            traces = [f for f in os.listdir(args.out) if f.endswith(".trace.json.gz")]
            fresh = [f for f in traces
                     if os.path.getmtime(os.path.join(args.out, f)) >= started - 5]
            if fresh:
                break
            time.sleep(5)

    traces = sorted(f for f in os.listdir(args.out) if f.endswith(".trace.json.gz"))
    if not traces:
        print("[probe] no traces found in", args.out)
        return 1
    for t in traces:
        report(os.path.join(args.out, t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
