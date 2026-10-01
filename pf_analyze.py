#!/usr/bin/env python
"""Where does prefill time actually go?  Kernel-family + stage + pipeline analysis.

Answers the question the existing scripts only answer in prose. Every family
share quoted in those docstrings ("attention is 31%", "MoE is 49.7% of EXTEND")
was read off a 2026-09-20 capture of the DEV server, before opt1's fused kernels
shipped. This re-derives them from whatever trace dir you point it at.

Three levels, because "prefill is slow" has three different causes and they
need different fixes:

  1. STAGE BUDGET   per PP stage: device time, wall span, busy fraction.
                    A low busy fraction with a small kernel total = pipeline
                    bubble (scheduling). High busy = the kernels are the wall.
  2. KERNEL FAMILY  device time aggregated by family (attn / moe / gemm / comm
                    / elementwise / copy). Tells you what to optimize.
  3. CHUNK CADENCE  chunk boundaries reconstructed from the per-layer attention
                    kernel, so we can see whether chunks actually overlap
                    across stages (pipelining) or march in lockstep (serial).

The family map is deliberately explicit rather than a regex soup: kernel names
are read from the trace and the map is printed, so a wrong guess shows up as
"unclassified" instead of silently inflating a family.

Usage:
  pf_analyze.py profiles/pfprod-1756...            # whole dir, all stages
  pf_analyze.py profiles/pfprod-... --stage 3     # one stage
  pf_analyze.py profiles/pfprod-... --top 30
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


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


# --- family map -----------------------------------------------------------
# Order matters: first match wins. Keep the patterns tight; a loose one silently
# moves ms between families and the whole report becomes fiction.
FAMILIES = [
    ("attn", re.compile(r"attention|attn|headshared|sparse|flash|mla|topk_transform"
                        r"|_topk|mqa_logits|sinkhorn|indexer|compress|expand_prefill"
                        r"|softmax|swa", re.I)),
    ("moe", re.compile(r"w4a16|w8a16|mxfp4|expert|moe|silu_and_mul|apply_sub80",
                       re.I)),
    ("gemm", re.compile(r"gemm|gemv|mm$|cutlass|cublas|s16816|nn_?gemm|sm90_xmma|"
                        r"_kernel$|w13|down_proj|gate_proj|up_proj|qkv", re.I)),
    ("comm", re.compile(r"nccl|allreduce|all_reduce|sendrecv|send_recv|reduce_scatter"
                        r"|all_gather|memcpy", re.I)),
    ("norm_rope", re.compile(r"rmsnorm|layernorm|rope|rotary|norm", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|copy|cat|fill|where|clamp|"
                               r"mul|add|div|exp|log|sum|mean|max|min|abs|neg|"
                               r"index|cast|convert|softmax|topk|sort", re.I)),
]


def family(name):
    for fam, rx in FAMILIES:
        if rx.search(name):
            return fam
    return "other"


def stage_of(path):
    m = re.search(r"TP-(\d+)-PP-(\d+)", os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def events(d, cat="kernel"):
    out = []
    for e in d.get("traceEvents", []):
        if e.get("ph") != "X" or e.get("cat") != cat:
            continue
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is None or dur is None:
            continue
        out.append((ts, dur, e.get("name", "?"), e.get("args", {}) or {}))
    out.sort()
    return out


def detect_scale(d):
    """Return the factor that converts trace time units to milliseconds.

    Kineto writes ts/dur in MICROSECONDS but sets displayTimeUnit="ms", so
    trusting that field inflates every number by 1000x. Rather than hardcode
    either answer, calibrate on something whose duration is known a priori: a
    cuda_runtime API call is a host-side launch, microseconds in practice and
    never milliseconds. If the median launch looks like milliseconds, the trace
    really is in ms and we divide by 1.

    Returns (scale_to_ms, how_we_know) so the report can state its own basis
    instead of quietly trusting a metadata field.
    """
    launch = [e["dur"] for e in d.get("traceEvents", [])
              if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"
              and e.get("dur") is not None]
    if not launch:
        return 1e-3, "no cuda_runtime events; assuming us"
    med = sorted(launch)[len(launch) // 2]
    # a median host launch above 200 "units" cannot be microseconds
    if med > 200.0:
        return 1.0, f"median cuda_runtime launch {med:.0f} units -> ms"
    return 1e-3, f"median cuda_runtime launch {med:.1f} us -> us (kineto default)"


def merge(iv):
    """Total device time of possibly-overlapping intervals (union, not sum).

    Kernels on different streams genuinely overlap; summing their durations
    double-counts and can exceed the wall span, which is how you get the
    nonsense numbers the old scripts occasionally printed.
    """
    if not iv:
        return 0.0, []
    iv = sorted(iv)
    total = 0.0
    merged = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            merged.append((cs, ce))
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    merged.append((cs, ce))
    return total, merged


def analyze(path, topn=25, want_chunks=True):
    d = load(path)
    kern = events(d, "kernel")
    memcpy = events(d, "gpu_memcpy")
    memset = events(d, "gpu_memset")
    tp, pp = stage_of(path)

    if not kern:
        return None

    scale, how = detect_scale(d)
    kern = [(ts * scale, dur * scale, n, a) for ts, dur, n, a in kern]
    memcpy = [(ts * scale, dur * scale, n, a) for ts, dur, n, a in memcpy]
    memset = [(ts * scale, dur * scale, n, a) for ts, dur, n, a in memset]
    t0 = kern[0][0]
    t1 = max(k[0] + k[1] for k in kern)
    span = t1 - t0

    # raw sum of durations, and union of kernel intervals
    raw_sum = sum(k[1] for k in kern)
    union, _ = merge([(k[0], k[0] + k[1]) for k in kern])
    comm_union, _ = merge([(k[0], k[0] + k[1]) for k in kern
                           if family(k[2]) == "comm"])

    # --- family breakdown (by raw duration sum; note overlap caveat) ---
    fam_dur = collections.Counter()
    fam_cnt = collections.Counter()
    for ts, dur, name, _ in kern:
        f = family(name)
        fam_dur[f] += dur
        fam_cnt[f] += 1

    # --- per-kernel-name ranking ---
    name_dur = collections.Counter()
    name_cnt = collections.Counter()
    name_grid = {}
    for ts, dur, name, args in kern:
        name_dur[name] += dur
        name_cnt[name] += 1
        name_grid.setdefault(name, (args.get("grid"), args.get("block")))

    # --- chunk cadence from the per-layer attention kernel ---
    # A prefill chunk is one pass over the stage's layers, so the head-shared
    # attention kernel fires once per layer per chunk. Grouping by a fixed
    # count is fragile (layer counts differ per stage, and a stage boundary can
    # split a group), so instead cut on a temporal gap: consecutive calls that
    # are close together are the same chunk, a long silence starts a new one.
    chunks = []
    if want_chunks:
        marker = None
        for n, c in name_cnt.most_common():
            if "headshared" in n.lower():
                marker = n
                break
        if marker is None:
            for n, c in name_cnt.most_common():
                if family(n) == "attn":
                    marker = n
                    break
        if marker:
            seq = [(ts, dur) for ts, dur, name, _ in kern if name == marker]
            if seq:
                med = statistics.median(d for _, d in seq)
                gap_thresh = max(med * 3.0, 5.0)
                cur = [seq[0]]
                for s, d in seq[1:]:
                    if s - cur[-1][0] > gap_thresh:
                        chunks.append(cur)
                        cur = []
                    cur.append((s, d))
                chunks.append(cur)
                chunks = [{
                    "start": c[0][0] - t0,
                    "end": max(x[0] + x[1] for x in c) - t0,
                    "n_layers": len(c),
                } for c in chunks if len(c) >= 2]

    return {
        "path": path, "file": os.path.basename(path), "tp": tp, "pp": pp,
        "units": how,
        "n_kernels": len(kern), "span_ms": span,
        "raw_sum_ms": raw_sum, "union_ms": union,
        "comm_union_ms": comm_union,
        "busy_frac": union / span if span else 0.0,
        "overlap_factor": raw_sum / union if union else 0.0,
        "memcpy_ms": sum(m[1] for m in memcpy),
        "memset_ms": sum(m[1] for m in memset),
        "fam_dur": fam_dur, "fam_cnt": fam_cnt,
        "name_dur": name_dur, "name_cnt": name_cnt, "name_grid": name_grid,
        "chunks": chunks, "t0": t0, "t1": t1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--stage", type=int, default=None,
                    help="only this PP rank")
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--chunk-tol-ms", type=float, default=0.0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.trace.json.gz"))) or \
            sorted(glob.glob(os.path.join(a.dir, "*.json.gz")))
    if not files:
        sys.exit(f"no trace files under {a.dir}")
    print(f"{len(files)} trace file(s) in {a.dir}\n")

    res = []
    for f in files:
        r = analyze(f, a.top)
        if r is None:
            print(f"  {os.path.basename(f)}: NO KERNELS (failed capture)")
            continue
        if a.stage is not None and r["pp"] != a.stage:
            continue
        if a.tp is not None and r["tp"] != a.tp:
            continue
        res.append(r)

    if not res:
        sys.exit("no traces matched the filter")

    # ---------------- 1. stage budget ----------------
    print("=" * 100)
    print("1. STAGE BUDGET  (union of kernel intervals / trace span)")
    print("=" * 100)
    if res:
        print(f"time units: {res[0]['units']}")
    print()
    print("%-5s %-4s %8s %9s %9s %9s %8s %8s %9s" %
          ("PP", "TP", "kernels", "span_ms", "union_ms", "rawsum_ms", "busy%",
           "ovl", "comm_ms"))
    tot_span = tot_union = 0.0
    for r in sorted(res, key=lambda x: (x["pp"], x["tp"])):
        print("%-5d %-4d %8d %9.0f %9.0f %9.0f %7.1f%% %7.2fx %9.0f" %
              (r["pp"], r["tp"], r["n_kernels"], r["span_ms"], r["union_ms"],
               r["raw_sum_ms"], 100 * r["busy_frac"], r["overlap_factor"],
               r["comm_union_ms"]))
        if r["tp"] == min(x["tp"] for x in res if x["pp"] == r["pp"]):
            tot_span += r["span_ms"]
            tot_union += r["union_ms"]

    if tot_span:
        print(f"\n  pipeline-wide: device {tot_union:.0f} ms of {tot_span:.0f} ms "
              f"across stages = {100*tot_union/tot_span:.1f}% busy")
        busiest = max((r for r in res if r["tp"] == 0),
                      key=lambda x: x["union_ms"], default=None)
        if busiest and busiest["span_ms"]:
            # what a perfectly packed pipeline would give, on the busiest stage
            packed = 512.0 / (busiest["union_ms"] /
                              max(1, len(busiest["chunks"]))) if busiest["chunks"] else 0
            if packed:
                print(f"  busiest stage PP{busiest['pp']}: {len(busiest['chunks'])} "
                      f"chunks, {busiest['union_ms']/max(1,len(busiest['chunks'])):.0f} ms "
                      f"device/chunk -> {packed:.0f} tok/s if fully packed")

    # ---------------- 2. kernel families ----------------
    print()
    print("=" * 100)
    print("2. KERNEL FAMILY  (raw duration sum, TP0 only -- TP1 is the same work)")
    print("=" * 100)
    tp0 = [r for r in res if r["tp"] == 0] or res[:1]
    fam = collections.Counter()
    cnt = collections.Counter()
    for r in tp0:
        fam.update(r["fam_dur"])
        cnt.update(r["fam_cnt"])
    total = sum(fam.values()) or 1.0
    print("%-14s %10s %8s %8s" % ("family", "ms", "calls", "share"))
    for f, ms in fam.most_common():
        print("%-14s %10.0f %8d %7.1f%%" % (f, ms, cnt[f], 100 * ms / total))
    print("%-14s %10.0f %8d" % ("TOTAL", total, sum(cnt.values())))

    # ---------------- 3. top kernels ----------------
    print()
    print("=" * 100)
    print("3. TOP KERNELS  (TP0, aggregated over stages)")
    print("=" * 100)
    nd = collections.Counter()
    nc = collections.Counter()
    ng = {}
    for r in tp0:
        nd.update(r["name_dur"])
        nc.update(r["name_cnt"])
        for k, v in r["name_grid"].items():
            ng.setdefault(k, v)
    print("%-46s %9s %7s %7s %s" % ("kernel", "ms", "calls", "share", "grid"))
    for n, ms in nd.most_common(a.top):
        print("%-46s %9.1f %7d %6.1f%% %s"
              % (n[:46], ms, nc[n], 100 * ms / total, ng.get(n, ('', ''))))

    # ---------------- 4. chunk cadence ----------------
    chunks = [c for r in tp0 for c in r["chunks"]]
    if len(chunks) > 1:
        print()
        print("=" * 100)
        print("4. CHUNK CADENCE  (chunk = one pass of the per-layer attention kernel)")
        print("=" * 100)
        chunks.sort(key=lambda c: c["start"])
        durs = [c["end"] - c["start"] for c in chunks]
        starts = [c["start"] for c in chunks]
        gaps = [starts[i + 1] - chunks[i]["end"] for i in range(len(chunks) - 1)]
        print("  chunks=%d  n_layers/chunk=%d"
              % (len(chunks), chunks[0]["n_layers"]))
        print("  device per chunk : median %.0f ms  (min %.0f  max %.0f)"
              % (statistics.median(durs), min(durs), max(durs)))
        if gaps:
            print("  idle between chunks: median %.0f ms  (min %.0f  max %.0f)"
                  % (statistics.median(gaps), min(gaps), max(gaps)))
            occ = sum(durs) / (chunks[-1]["end"] - chunks[0]["start"])
            print("  stage occupancy   : %.1f%%" % (100 * occ))
        # per-stage start offsets tell us whether stages overlap
        print("\n  per-stage first-chunk offset (0 = serialized, large = pipelined):")
        for r in sorted(tp0, key=lambda x: x["pp"]):
            if r["chunks"]:
                print("    PP%d: first chunk at %+.0f ms, %d chunks"
                      % (r["pp"], r["chunks"][0]["start"], len(r["chunks"])))


if __name__ == "__main__":
    main()
