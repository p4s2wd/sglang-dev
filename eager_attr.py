"""Attribute decode elementwise work to sglang call sites, from an eager decode trace.

Decode normally runs under CUDA graphs, so its kernels carry no External id and cannot
be linked to a CPU op or a Python frame. The join against an eager PREFILL trace only
recovers ops whose (name, grid, block) triple also appears in prefill, which left
5.73 of the 9.50 ms/token elementwise family UNMATCHED.

This trace was captured with --disable-cuda-graph, so decode kernels do have CPU ops
and stacks. Attribute directly: kernel -> cudaLaunchKernel (by correlation) -> the
enclosing aten op (by External id) -> the innermost sglang python_function frame.
Report ms per token using the sinkhorn call count (2 per layer per step, 11 layers).
"""
import bisect, gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
f = max(glob.glob(d + "/*TP-0-PP-3*"))
ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]

py = sorted([e for e in ev if e.get("cat") == "python_function" and ".py(" in e.get("name", "")],
            key=lambda e: e["ts"])
starts = [e["ts"] for e in py]
cpu = sorted([e for e in ev if e.get("cat") == "cpu_op"], key=lambda e: e["ts"])
cstarts = [e["ts"] for e in cpu]
rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
corr2ext = {}
for e in rt:
    a = e.get("args", {})
    if "correlation" in a and "External id" in a:
        corr2ext[a["correlation"]] = a["External id"]
ext2cpu = {}
for c in cpu:
    eid = c.get("args", {}).get("External id")
    if eid is not None:
        ext2cpu[eid] = c


def innermost(ts):
    i = bisect.bisect_right(starts, ts)
    best = None
    for e in py[max(0, i - 2000):i]:
        if e["ts"] <= ts <= e["ts"] + e.get("dur", 0):
            n = e["name"]
            if "/site-packages/" in n or "/torch/" in n:
                continue
            if "/sglang/" in n:
                best = n
    return best


ks = [e for e in ev if e.get("cat") == "kernel"]
steps = max(1, sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]) / 22.0)
print("PP3 eager decode: %d kernels, %.1f steps" % (len(ks), steps))

agg = defaultdict(lambda: [0.0, 0])
matched = unmatched = 0.0
for e in ks:
    n = e["name"]
    if not ("elementwise" in n or "reduce_kernel" in n or "fill" in n or "CatArray" in n):
        continue
    eid = corr2ext.get(e["args"].get("correlation"))
    c = ext2cpu.get(eid) if eid is not None else None
    if c is not None:
        fr = innermost(c["ts"])
        lab = ((fr + " [aten:" + c["name"] + "]") if fr
               else ("(no sglang frame) [aten:" + c["name"] + "]"))
        matched += e["dur"] / 1e3
    else:
        lab = "UNLINKED " + n[:48]
        unmatched += e["dur"] / 1e3
    agg[lab][0] += e["dur"] / 1e3 / steps
    agg[lab][1] += 1

tot = sum(v[0] for v in agg.values())
print("elementwise family %.2f ms/token; linked %.1f ms, unlinked %.1f ms\n"
      % (tot, matched / steps, unmatched / steps))
print("%9s %9s  call site" % ("ms/tok", "calls/tok"))
for k, (ms, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:18]:
    print("%9.3f %9d  %s" % (ms, c, k[:120]))
