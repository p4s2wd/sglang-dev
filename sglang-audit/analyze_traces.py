"""Aggregate sglang Chrome traces into a prefill/decode time breakdown.

The server writes one .trace.json.gz per (rank, stage): the stage is the
"-EXTEND" (prefill) or "-DECODE" suffix in the filename. Each is a Chrome trace
whose events carry cat in {kernel, cuda_runtime, gpu_memcpy, gpu_memset, cpu_op,
user_annotation, python_function}.

What matters for optimization is *where the wall time goes*, so the report
separates:
  - device time: sum of kernel durations, and the top kernels by that sum
  - host time:   the CPU ops that dominate, which is what a CUDA graph cannot
                 remove if they are inside the captured region
  - coverage:    device busy / wall. Low coverage means launch/CPU bound.

Usage: analyze_traces.py <dir-or-trace...> [--top N] [--min-dur-us US]
"""
import argparse
import gzip
import io
import json
import os
import re
from collections import defaultdict

STAGE_RE = re.compile(r"-(EXTEND|DECODE|SPLIT|DRAFT_EXTEND)\b")


def load_events(path):
    opener = gzip.open if path.endswith(".gz") else io.open
    with opener(path, "rt") as f:
        data = json.load(f)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    return events


def stage_of(path):
    m = STAGE_RE.search(os.path.basename(path))
    if m:
        return m.group(1)
    return "PREFILL" if "EXTEND" in path else ("DECODE" if "DECODE" in path else "?")


def rank_of(path):
    m = re.search(r"TP-(\d+)", path)
    p = re.search(r"PP-(\d+)", path)
    return (f"TP{m.group(1)}" if m else "") + (f"/PP{p.group(1)}" if p else "")


def fmt(us):
    if us >= 1e6:
        return f"{us/1e6:8.3f}s"
    if us >= 1e3:
        return f"{us/1e3:8.2f}ms"
    return f"{us:8.1f}us"


# Semantic buckets. Kernel names from cuBLAS and ATen are long templates, so
# grouping is by substring rather than by the trimmed name -- trimming to the
# first '<' collapses every cuBLAS GEMV into "std::enable_if", which hides
# exactly the distinction that matters (GEMV vs GEMM).
GROUPS = [
    ("moe_w4a16", ("w4a16_ptx_kernel", "mxfp4_w4a16")),
    ("cublas_gemv", ("gemvx::kernel", "gemv2T_kernel", "gemv1T_kernel", "dot_kernel")),
    ("cublas_gemm", ("sgemm", "gemm_fp16", "s1688gemm", "h16816gemm", "cutlass")),
    ("allreduce", ("all_reduce", "nccl", "cross_device")),
    ("index_gather", ("index_elementwise", "index_kernel_impl", "gather")),
    ("elementwise_copy", ("direct_copy_kernel", "bfloat16_copy", "half_copy",
                          "copy_kernel_cuda")),
    ("elementwise_math", ("MulFunctor", "CUDAFunctor_add", "AddFunctor",
                          "masked_fill", "exp_kernel", "softmax")),
    ("reduce", ("reduce_kernel", "sum_functor")),
    ("sinkhorn_hc", ("sinkhorn", "hc_split", "mhc")),
    ("rope_norm", ("rope", "norm_kernel", "rmsnorm")),
    ("topk", ("topk", "top_")),
]


def group_of(name):
    for label, needles in GROUPS:
        if any(n in name for n in needles):
            return label
    return "other"


def analyze(paths, top, min_dur_us):
    # stage -> category -> name -> [count, total_us]
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0.0])))
    wall = defaultdict(float)
    first_ts = {}
    last_end = {}

    for path in paths:
        stage = stage_of(path)
        rank = rank_of(path)
        # Keep ranks apart: summing both TP ranks into one span makes the duty
        # cycle read ~2x too high, which is the number that decides whether a
        # stage is compute-bound or launch-bound.
        key = f"{stage} {rank}".strip()
        for ev in load_events(path):
            if ev.get("ph") != "X":
                continue
            dur = ev.get("dur")
            if not dur:
                continue
            cat = ev.get("cat", "?")
            name = ev.get("name", "?")
            # Keep full names: grouping and display both need the template
            # body (cuBLAS GEMV vs GEMM only differs after the first '<').
            row = agg[key][cat][name]
            row[0] += 1
            row[1] += dur
            ts = ev.get("ts", 0)
            if key not in first_ts or ts < first_ts[key]:
                first_ts[key] = ts
            end = ts + dur
            if key not in last_end or end > last_end[key]:
                last_end[key] = end
        print(f"loaded {rank or '?':>9}  {os.path.basename(path)[:70]}")

    for key in sorted(agg):
        span = last_end[key] - first_ts[key]
        print("\n" + "=" * 78)
        print(f"STAGE {key}   wall span {fmt(span)}")
        print("=" * 78)

        # Device busy: kernels are per-stream, so summing overstates a single
        # stream's busy time but is the right proxy for aggregate device work.
        cats = agg[key]
        dev_cats = [c for c in cats if c in ("kernel", "gpu_memcpy", "gpu_memset")]
        dev_total = sum(sum(v[1] for v in cats[c].values()) for c in dev_cats)
        cpu_total = sum(sum(v[1] for v in cats[c].values())
                        for c in cats if c in ("cpu_op", "cuda_runtime"))

        if span > 0:
            print(f"device work {fmt(dev_total)}  "
                  f"({100*dev_total/span:.1f}% of span)   "
                  f"host ops {fmt(cpu_total)}")
        else:
            print(f"device work {fmt(dev_total)}   host ops {fmt(cpu_total)}")

        print("\ncategory totals:")
        for c in sorted(cats, key=lambda c: -sum(v[1] for v in cats[c].values())):
            tot = sum(v[1] for v in cats[c].values())
            n = sum(v[0] for v in cats[c].values())
            print(f"  {c:>18}  {fmt(tot)}  {n:>7} events")

        # Grouped device time: what kind of work, not which template.
        groups = defaultdict(lambda: [0, 0.0])
        for name, (n, t) in ((k, v) for k, v in cats.get("kernel", {}).items()):
            g = groups[group_of(name)]
            g[0] += n
            g[1] += t
        ktot = sum(g[1] for g in groups.values()) or 1.0
        print(f"\ndevice time by kind ({fmt(ktot)} of kernels):")
        for label, (n, t) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
            print(f"  {100*t/ktot:5.1f}%  {fmt(t)}  x{n:<6} {label}")

        for c in ("kernel", "cuda_runtime", "cpu_op", "gpu_memcpy"):
            if c not in cats:
                continue
            rows = [(n, v[0], v[1]) for n, v in cats[c].items() if v[1] >= min_dur_us]
            rows.sort(key=lambda r: -r[2])
            tot = sum(r[2] for r in rows) or 1.0
            print(f"\ntop {c} by total time ({fmt(tot)} accounted):")
            for name, cnt, t in rows[:top]:
                print(f"  {100*t/tot:5.1f}%  {fmt(t)}  x{cnt:<6} {name[:88]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="trace files or a directory")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--min-dur-us", type=float, default=0.0)
    ap.add_argument("--stage", default=None, help="only this stage (EXTEND|DECODE)")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in names
                          if n.endswith(".trace.json.gz") or n.endswith(".trace.json")]
        else:
            files.append(p)
    files = sorted(set(files))
    if args.stage:
        files = [f for f in files if args.stage in os.path.basename(f).upper()]
    if not files:
        raise SystemExit("no trace files found")
    print(f"{len(files)} trace file(s)")
    analyze(files, args.top, args.min_dur_us)


main()
