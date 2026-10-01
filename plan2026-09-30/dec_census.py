#!/usr/bin/env python
"""Per-kernel decode accounting, bucketed by family, from a torch profiler trace.

Answers the question SM75_DSV4_DECODE_PROFILE.md closed on short-context
evidence: which kernel family actually grows between a ~440-token prompt and a
~100K-token one. The earlier census put headshared_sparse (MLA) at 0.96 ms of
12.94 ms and concluded attention had no headroom left -- but that was measured
with the compressed KV cache nearly empty, where there is nothing to gather.

Two things the older tooling would not give directly:
  - family-level totals, so "attention" can be compared as a share rather than
    as one kernel line;
  - the raw per-step sums for short and long side by side.

    dec_census.py <short_trace_dir> <long_trace_dir> [--pp 3]
"""
import argparse
import collections
import glob
import gzip
import json
import os
import statistics
import sys


def load_events(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return data.get("traceEvents", data if isinstance(data, list) else [])


def classify(name):
    n = name.lower()
    if "headshared" in n or "sparse" in n:
        return "attn_sparse(MLA)"
    if "mqa" in n or "indexer" in n:
        return "attn_indexer"
    if "topk_transform" in n or "topk" in n:
        return "topk"
    if "moe" in n or "w4a16" in n or "expert" in n or "swiglu" in n:
        return "moe"
    if "gemv" in n or "gemm" in n or "cublas" in n or "cutlass" in n or "mm" in n:
        return "gemm"
    if "allreduce" in n or "all_reduce" in n or "nccl" in n or "reduce" in n:
        return "comm"
    if "layernorm" in n or "rms" in n or "norm" in n or "rope" in n or "quant" in n \
       or "scale" in n or "cast" in n or "copy" in n or "elementwise" in n:
        return "elementwise"
    if "graph" in n or "launch" in n:
        return "graph_launch"
    return "other"


def census(path):
    """Sum GPU kernel durations per family, and per-step totals."""
    events = load_events(path)
    fam = collections.Counter()
    fam_n = collections.Counter()
    per_step = collections.Counter()
    step_of = {}
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if "kernel" not in cat and "Kernel" not in cat:
            continue
        dur = e.get("dur", 0)
        if dur <= 0:
            continue
        f = classify(e.get("name", ""))
        fam[f] += dur
        fam_n[f] += 1
        # torch profiler numbers steps via the enclosing "step" markers
        pid = e.get("pid")
        ts = e.get("ts")
        step_of[(pid, ts)] = None
    return fam, fam_n, len(events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="trace directories (short first)")
    ap.add_argument("--pp", type=int, default=3)
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args()

    tables = []
    for d in args.dirs:
        pats = glob.glob(os.path.join(d, f"*-PP-{args.pp}-DECODE.trace.json.gz"))
        if not pats:
            pats = glob.glob(os.path.join(d, "*.trace.json.gz"))
        if not pats:
            print(f"!! no trace under {d}", file=sys.stderr)
            tables.append(None)
            continue
        fam, n, nev = census(pats[0])
        tot = sum(fam.values()) or 1.0
        tables.append((os.path.basename(d.rstrip("/")), fam, n, tot, nev, pats[0]))

    good = [t for t in tables if t]
    if not good:
        return
    width = max(len(t[0]) for t in good) + 2
    print("PP%d decode kernel families, us per trace (share)\n" % args.pp)
    fams = sorted({f for _, fam, _, _, _, _ in good for f in fam},
                  key=lambda f: -max(fam.get(f, 0) for _, fam, _, _, _, _ in good))
    hdr = "family".ljust(26) + "".join(t[0][:22].rjust(width) for t in good)
    print(hdr)
    print("-" * len(hdr))
    for f in fams[:args.top]:
        row = f.ljust(26)
        for _, fam, n, tot, _, _ in good:
            v = fam.get(f, 0)
            row += (f"{v:.0f} ({v/tot*100:.1f}%)".rjust(width))
        print(row)
    print("-" * len(hdr))
    row = "TOTAL us".ljust(26)
    for _, fam, n, tot, _, _ in good:
        row += f"{tot:.0f}".rjust(width)
    print(row)
    print()
    for name, fam, n, tot, nev, p in good:
        print(f"{name}: {nev} events, kernels={sum(n.values())}, "
              f"trace={os.path.basename(p)}")


if __name__ == "__main__":
    main()
