"""Why is PP0 the slowest stage? Compare kernel families across the 4 stages.

Normalizing the DECODE traces by the attention call count (88 per stage = 11 layers x
2 attention calls x 4 steps; PP0 shows 72, i.e. 9 layers x 2 x 4) gives 4 decode steps
in the capture and per-stage busy of PP0 22.65, PP1 15.82, PP2 17.55, PP3 19.67 ms per
step. Divided by layer count that is 2.52 / 1.44 / 1.60 / 1.79 ms per layer, so PP0 is
1.75x the per-layer cost of PP1 while owning the FEWEST layers.

In a pipeline the cadence is set by the slowest stage, so a 43% skew on the stage with
the fewest layers is a direct throughput loss and it is fixable by moving layers. Find
which kernel families account for PP0's excess: report each family's ms per step per
stage, normalized by that stage's layer count, so a family that is not per-layer (an
embedding, a sinkhorn, a logits op) stands out as PP0-only excess.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
# layers per stage inferred from attention calls / (2 calls x 4 steps)
STEPS = 4
fams = {}
layers = {}
for f in sorted(glob.glob(d + "/*DECODE*.trace.json.gz")):
    ev = json.load(gzip.open(f))["traceEvents"]
    pp = int(f.split("PP-")[1].split("-")[0])
    tp = int(f.split("TP-")[1].split("-")[0])
    if tp != 0:
        continue
    ks = [e for e in ev if e.get("cat") == "kernel"]
    na = sum(1 for e in ks if "sparse" in e["name"] or "headshared" in e["name"])
    layers[pp] = na / (2 * STEPS)
    agg = defaultdict(float)
    for e in ks:
        n = e["name"]
        key = ("attn" if ("sparse" in n or "headshared" in n) else
               "expert w4a16" if "w4a16" in n else
               "w8a16 gemv" if "w8a16" in n else
               "dequant fp8" if "dequant_block_fp8" in n else
               "nccl SendRecv" if "SendRecv" in n else
               "allreduce" if "all_reduce" in n else
               "cutlass/turing gemm" if ("cutlass" in n or "turing_fp16" in n or "gemvx" in n or "volta_sgemm" in n) else
               "hc/sinkhorn" if ("hc_" in n or "sinkhorn" in n) else
               "topk/indexer" if ("topk" in n.lower() or "index" in n) else
               "elementwise" if ("elementwise" in n or "reduce_kernel" in n or "fill" in n) else
               "other")
        agg[key] += e["dur"] / 1e3
    fams[pp] = agg

keys = sorted({k for a in fams.values() for k in a})
print("layers per stage: %s" % {k: round(v, 1) for k, v in sorted(layers.items())})
print("\nms per step per stage, and in brackets ms per layer:")
print("%-20s" % "family" + "".join("%18s" % ("PP%d" % p) for p in sorted(fams)))
for k in keys:
    row = "%-20s" % k
    for p in sorted(fams):
        ms = fams[p].get(k, 0.0) / STEPS
        per = ms / max(layers[p], 1)
        row += "%18s" % ("%.2f [%.3f]" % (ms, per))
    print(row)
print("%-20s" % "TOTAL" + "".join(
    "%18s" % ("%.2f [%.3f]" % (sum(fams[p].values()) / STEPS,
                              sum(fams[p].values()) / STEPS / max(layers[p], 1)))
    for p in sorted(fams)))
