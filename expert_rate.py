"""What rate does the production expert kernel actually run at?

The isolated benchmark (bench_w4a16_ptx.py) reported the repacked v3 kernel at
137 GB/s, and route (2) of the objective is "W4A16 from 112 GB/s toward ~500 GB/s",
so 137 reads like 3.6x short. But the production DECODE trace shows
w4a16_v3_kernel at 0.088 ms per call, and a call covers the layer's active experts,
so the two numbers may not describe the same amount of work.

Compute the production rate directly: read the kernel's launch arguments from the
trace to get the real tensor shapes and grid, derive the bytes it must read, and
divide by its measured duration. That is the number route (2) should be judged on.
"""
import gzip, glob, json
from collections import defaultdict

d = "profiles/exact-1790060350"
f = max(glob.glob(d + "/*TP-0-PP-3.trace.json.gz"))
ev = json.load(gzip.open(f))["traceEvents"]
ks = [e for e in ev if e.get("cat") == "kernel"]
steps = max(1, sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]) / 22.0)
ex = [e for e in ks if "w4a16" in e["name"]]
print("stage PP3: %d expert calls over %.1f steps = %.1f calls/step" % (len(ex), steps, len(ex) / steps))
print("mean duration %.4f ms" % (sum(e["dur"] for e in ex) / 1e3 / len(ex)))
print("\nsample launch args:")
a = ex[0]["args"]
for k in sorted(a):
    print("   %-28s %s" % (k, a[k]))
print("\ngrid/block: %s / %s  registers %s  smem %s"
      % (a.get("grid"), a.get("block"), a.get("registers per thread"),
         a.get("shared memory")))
