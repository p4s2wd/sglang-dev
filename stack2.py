"""Attribute elementwise device time to sglang Python call sites, correctly.

The previous attempt read args.module_path, which this profiler does not emit; the
frames are in the event NAME as "file.py(line): func", and there are 167k of them.
Elementwise work is 13.6% of prefill device time, and the norm probe showed eager
norms cost 3-9x the fused ones, so naming the eager sites is the difference between
a real fix and a guess.

Linkage: kernels carry args.correlation_id, which matches the cuda_runtime launch
event with the same id; that launch sits inside the CPU op's time window, and the
CPU op window contains the python frames. So walk the python frames sorted by start
and take the innermost sglang frame containing the cpu_op's start.
"""
import bisect, gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
f = max(glob.glob(d + "/*PP-3*EXTEND*"))
ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]

py = sorted([e for e in ev if e.get("cat") == "python_function" and ".py(" in e.get("name", "")],
            key=lambda e: e["ts"])
starts = [e["ts"] for e in py]
cpu = sorted([e for e in ev if e.get("cat") == "cpu_op"], key=lambda e: e["ts"])
cpu_starts = [e["ts"] for e in cpu]

# sglang-internal frames only; torch internals are noise.
def sglang_frame(ts):
    i = bisect.bisect_right(starts, ts)
    best = None
    for e in py[max(0, i - 900):i]:
        if e["ts"] <= ts <= e["ts"] + e.get("dur", 0):
            n = e["name"]
            if "/sglang/" in n or n.startswith(("deepseek", "layernorm", "fp8", "attention")):
                if "/site-packages/" in n:
                    continue
                best = n
    return best

# kernel -> launch -> cpu op window
launch = {}
for e in ev:
    if e.get("cat") in ("cuda_runtime", "cuda_driver"):
        cid = e.get("args", {}).get("correlation_id")
        if cid is not None:
            launch[cid] = e

pat = ("elementwise", "reduce_kernel", "pow_", "copy_kernel", "index_", "fill_", "Cat")
agg = defaultdict(lambda: [0.0, 0])
tot = 0.0
for e in ev:
    if e.get("cat") != "kernel":
        continue
    tot += e["dur"] / 1e3
    n = e["name"]
    if not any(p in n for p in pat):
        continue
    cid = e.get("args", {}).get("correlation_id")
    L = launch.get(cid)
    if L is None:
        agg["<no launch event>"][0] += e["dur"] / 1e3
        agg["<no launch event>"][1] += 1
        continue
    i = bisect.bisect_right(cpu_starts, L["ts"])
    fr = None
    for c in cpu[max(0, i - 6):i + 1]:
        if c["ts"] <= L["ts"] <= c["ts"] + c.get("dur", 0):
            fr = sglang_frame(c["ts"])
            if fr:
                break
    key = fr or ("cpu:" + (cpu[i - 1]["name"][:40] if i else "?"))
    agg[key][0] += e["dur"] / 1e3
    agg[key][1] += 1

print("total device %.1f ms" % tot)
print("%8s %6s %8s  sglang call site" % ("ms", "calls", "us/call"))
for k, (ms, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:18]:
    print("%8.1f %6d %8.1f  %s" % (ms, n, ms * 1e3 / n, k[:100]))
