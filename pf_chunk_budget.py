#!/usr/bin/env python
"""Per-chunk compute budget for a prefill trace dir, the number that decides where
to optimize.

This is the part of pf_analyze.py worth running every time. It answers one
question: per 512-token chunk, how much REAL kernel work does each PP stage do,
and which stage is the ceiling?

Method, and why each step is needed:

  * Chunk boundaries. A prefill chunk is one pass over a stage's layers, so the
    head-shared attention kernel fires twice per layer per chunk. Counting those
    calls and dividing by 2*layers gives the chunk stride. (Verified against
    the config: 43 layers over PP4 = 10/11/11/11.)
  * Union, not sum. Kernels on different streams overlap; summing durations can
    exceed the wall span. Intervals are merged before totalling.
  * Non-nccl only. The nccl SendRecv kernel runs 100% concurrent with compute --
    it is a spin absorbing pipeline skew, not a transfer. Counting it as work
    would tell you PP0/PP3 are comm-bound when they are not.
  * Microseconds. Kineto writes us while advertising displayTimeUnit=ms.

Usage: pf_chunk_budget.py profiles/pfprod-XXXX [--tp 0]
"""
import argparse
import glob
import gzip
import json
import os
import re
import statistics


def load(p):
    o = gzip.open if p.endswith(".gz") else open
    with o(p, "rt") as f:
        return json.load(f)


def scale_of(d):
    launches = [e["dur"] for e in d.get("traceEvents", [])
                if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"
                and e.get("dur") is not None]
    if not launches:
        return 1e-3
    return 1.0 if sorted(launches)[len(launches) // 2] > 200 else 1e-3


def union_ms(iv):
    if not iv:
        return 0.0
    iv = sorted(iv)
    tot, cs, ce = 0.0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tp", type=int, default=0)
    ap.add_argument("--layers", default="", help="override e.g. 10,11,11,11")
    ap.add_argument("--tokens-per-chunk", type=int, default=512)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.trace.json.gz")))
    if not files:
        raise SystemExit("no traces")
    stages = {}
    for f in files:
        m = re.search(r"TP-(\d+)-PP-(\d+)", os.path.basename(f))
        if not m or int(m.group(1)) != a.tp:
            continue
        d = load(f)
        s = scale_of(d)
        K = sorted((e["ts"] * s, e["dur"] * s, e["name"])
                   for e in d["traceEvents"]
                   if e.get("cat") == "kernel" and e.get("ph") == "X"
                   and e.get("ts") is not None and e.get("dur") is not None)
        hs = [(ts, du) for ts, du, n in K if "headshared" in n]
        nq = len([1 for _, _, n in K if "fused_q_norm_rope" in n])
        stages[int(m.group(2))] = (K, hs, nq, f)

    if not stages:
        raise SystemExit("no traces for that tp")

    nl_override = [int(x) for x in a.layers.split(",")] if a.layers else None
    print(f"time units: {'ms' if scale_of(load(files[0])) == 1.0 else 'us'}"
          f" (kineto default is us)")
    print()
    print("%-4s %6s %7s %12s %12s %9s" %
          ("PP", "layers", "chunks", "comp/chunk", "cadence", "util%"))
    comp = {}
    for pp in sorted(stages):
        K, hs, nq, f = stages[pp]
        nl = nl_override[pp] if nl_override else max(1, round(nq / 3))
        per = nl * 2
        nch = len(hs) // per
        if nch == 0:
            print(f"{pp:<4d} {nl:6d} {'--':>7s}  (fewer than one full chunk captured)")
            continue
        cs, cad = [], []
        for i in range(nch):
            a0 = hs[i * per][0]
            a1 = max(hs[i * per + j][0] + hs[i * per + j][1] for j in range(per))
            cs.append(union_ms([(t, t + du) for t, du, n in K
                                if a0 <= t <= a1 and "nccl" not in n.lower()]))
        for i in range(nch - 1):
            cad.append(hs[(i + 1) * per][0] - hs[i * per][0])
        comp[pp] = statistics.median(cs)
        print("%-4d %6d %7d %12.1f %12.1f %8.1f%%" %
              (pp, nl, nch, comp[pp], statistics.median(cad) if cad else 0,
               100 * comp[pp] / statistics.median(cad) if cad else 0))

    if comp:
        b = max(comp, key=lambda p: comp[p])
        tpc = a.tokens_per_chunk
        print()
        print(f"ceiling stage = PP{b} at {comp[b]:.0f} ms of compute per chunk")
        print(f"  perfect-pipeline ceiling : {tpc/(comp[b]/1000):.0f} tok/s")
        print(f"  balanced-across-stages   : {tpc/(sum(comp.values())/1000/len(comp)*1000/1000):.0f}"
              f" tok/s  (mean {sum(comp.values())/len(comp):.0f} ms)")
        print(f"  if every stage were as fast as the fastest "
              f"(PP{min(comp, key=lambda p: comp[p])} = "
              f"{comp[min(comp, key=lambda p: comp[p])]:.0f} ms): "
              f"{tpc/(comp[min(comp, key=lambda p: comp[p])]/1000):.0f} tok/s")


if __name__ == "__main__":
    main()
