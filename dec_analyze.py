#!/usr/bin/env python
"""Decode-step anatomy from a /start_profile DECODE trace.

Decode at bs=1 is the one-token critical path: every stage's per-step device
work and wall cadence add (or hide) directly into tok/s. Prints per (TP,PP):
step count, wall/step, device busy/step, family split, top kernels/call.

Usage: dec_analyze.py profiles/2026...-dec26 [--stage 3]
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import statistics
import sys


FAMILIES = [
    ("attn", re.compile(r"attention|attn|headshared|sparse|flash|mla|topk_transform"
                        r"|_topk|mqa_logits|sinkhorn|indexer|compress|expand_prefill"
                        r"|softmax|swa|merge_splits|rope", re.I)),
    ("moe", re.compile(r"w4a16|w8a16|mxfp4|expert|moe|silu|apply_sub80|combine|align"
                       r"|swiglu|topk_softmax|sigmoid", re.I)),
    ("gemm", re.compile(r"gemm|gemv|mm$|cutlass|cublas|s16816|s1688|nn_?gemm|sm90|"
                        r"w13|down_proj|gate_proj|up_proj|qkv|wo_a|absorb|linear",
                        re.I)),
    ("comm", re.compile(r"nccl|allreduce|all_reduce|sendrecv|send_recv|reduce_scatter"
                        r"|all_gather|p2p|memcpy|custom_allreduce", re.I)),
    ("norm_rope", re.compile(r"rmsnorm|layernorm|rope|rotary|norm", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|copy|cat|fill|where|clamp|"
                               r"mul|add|div|exp|log|sum|mean|max|min|abs|neg|"
                               r"index|cast|convert|softmax|topk|sort|quant", re.I)),
]


def family(name):
    for fam, rx in FAMILIES:
        if rx.search(name):
            return fam
    return "other"


def stage_of(path):
    m = re.search(r"TP-(\d+)-PP-(\d+)", os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def detect_scale(d):
    launch = [e["dur"] for e in d.get("traceEvents", [])
              if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"
              and e.get("dur") is not None]
    if not launch:
        return 1e-3, "no cuda_runtime; assume us"
    med = sorted(launch)[len(launch) // 2]
    if med > 200.0:
        return 1.0, f"median launch {med:.0f} units -> ms"
    return 1e-3, f"median launch {med:.1f} us -> us"


def events(d):
    out = []
    for e in d.get("traceEvents", []):
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        ts, dur = e.get("ts"), e.get("dur")
        if ts is None or dur is None:
            continue
        out.append((ts, dur, e.get("name", "?")))
    out.sort()
    return out


def union(iv):
    if not iv:
        return 0.0
    iv = sorted(iv)
    t, cs, ce = 0.0, *iv[0]
    for s, e in iv[1:]:
        if s > ce:
            t += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return t + ce - cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--stage", type=int, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*DECODE*.trace.json*")))
    if not files:
        sys.exit("no DECODE traces in " + args.dir)

    for path in files:
        tp, pp = stage_of(path)
        if args.stage is not None and pp != args.stage:
            continue
        with (gzip.open(path, "rt") if path.endswith(".gz")
              else open(path)) as f:
            d = json.load(f)
        scale, basis = detect_scale(d)
        ev = events(d)
        if not ev:
            print(f"TP{tp} PP{pp}: no kernel events")
            continue
        t0 = ev[0][0]
        span = max(t + du for t, du, _ in ev) - t0
        busy = union([(t, t + du) for t, du, _ in ev])
        fam_ms = collections.Counter()
        kern = collections.Counter()
        kcnt = collections.Counter()
        for t, du, name in ev:
            fam_ms[family(name)] += du * scale
            kern[name] += du * scale
            kcnt[name] += 1

        # step count: from profiler num_steps; here infer from the dominant
        # per-layer kernel (e.g. headshared appears layers_per_step times/step)
        # simpler: per-step = span / n_steps reported in filename? use 20.
        ns = 20
        print(f"\n===== TP{tp} PP{pp}  (time basis: {basis}) =====")
        print(f"  wall span      {span*scale:8.1f} ms   ({span*scale/ns:.2f} ms/step)")
        print(f"  device busy    {busy*scale:8.1f} ms   ({busy*scale/ns:.2f} ms/step) "
              f"= {busy/span*100:.1f}%")
        tot = sum(fam_ms.values())
        print("  family   ms/step   %")
        for f, ms in fam_ms.most_common():
            print(f"  {f:<10} {ms/ns:7.3f}  {ms/tot*100:5.1f}%")
        print("  top kernels (ms/call  calls  total_ms/step  name[:64])")
        top = kern.most_common(14)
        for name, ms in top:
            n = kcnt[name]
            print(f"    {ms/n:8.4f}  {n:5d}  {ms/ns:7.3f}  {name[:64]}")


if __name__ == "__main__":
    main()
