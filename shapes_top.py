"""What are the actual shapes behind decode's biggest elementwise kernels?

The graphed decode trace leaves ~5.1 ms/token of elementwise work UNMATCHED (its
(name,grid,block) triple never occurs in the eager prefill trace, because decode
shapes are batch 1-2 rows). An eager decode trace attributed 2.0 ms/token to the MoE
epilogue, but measuring that epilogue in isolation at the real shapes gives 0.046
ms/token/stage -- 43x less -- so the eager attribution is inflated and cannot be
trusted for magnitude either.

The eager capture has record_shapes=True, so read the Input shape of the aten op each
kernel belongs to and report the biggest elementwise ops by (bytes moved, duration).
Bytes moved is what decides whether an op is worth fusing: a 96x2048 fp16 tensor is
393 KB and should cost ~1 us at 616 GB/s, so anything costing tens of microseconds on
a tensor that small is launch-bound, not bandwidth-bound, and fusing it buys little.
"""
import bisect, gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
f = max(glob.glob(d + "/*TP-0-PP-3*"))
ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]

cpu = sorted([e for e in ev if e.get("cat") == "cpu_op"], key=lambda e: e["ts"])
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

ks = [e for e in ev if e.get("cat") == "kernel"]
steps = max(1, sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]) / 22.0)
print("steps %.1f, kernels %d" % (steps, len(ks)))

agg = defaultdict(lambda: [0.0, 0])
for e in ks:
    n = e["name"]
    if not ("elementwise" in n or "reduce_kernel" in n or "fill" in n or "CatArray" in n):
        continue
    eid = corr2ext.get(e["args"].get("correlation"))
    c = ext2cpu.get(eid) if eid is not None else None
    if c is None:
        key = ("UNLINKED", n[:40], "")
    else:
        shapes = c.get("args", {}).get("Input shape", "")
        key = (c["name"], n[:40], shapes if isinstance(shapes, str) else str(shapes)[:60])
    agg[key][0] += e["dur"] / 1e3 / steps
    agg[key][1] += 1

print("\n%9s %8s  %s" % ("ms/tok", "calls", "aten op / kernel / input shapes"))
for (op, kn, sh), (ms, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:16]:
    print("%9.3f %8d  %-22s %-34s %s" % (ms, c, op[:22], kn[:34], sh[:52]))
