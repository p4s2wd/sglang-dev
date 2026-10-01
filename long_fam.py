"""Family ranking for decode at a topk-saturating context, split kernel live.

PP-2's trace came out empty (1116 bytes, zero kernels) so this analyses the three
stages that captured and normalizes per stage rather than per token across four, which
avoids the missing stage silently deflating the total.

The point of the capture: attention just got 1.57x faster and the previous ranking came
from short-context decode, where attention gathered a handful of tiles instead of 32.
The family balance the next optimization should target has changed.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
FAMS = [
    ("attn", ("_tiled_sparse_decode_kernel", "_headshared_sparse_kernel", "_merge_partial")),
    ("expert w4a16", ("w4a16",)),
    ("w8a16 gemv", ("_w8a16_gemv_kernel",)),
    ("topk/indexer", ("topk", "indexer", "moe_fused_gate", "sinkhorn")),
    ("gemm", ("cutlass", "gemm", "Kernel_", "splitK", "nvjet", "sgemm", "hgemm")),
    ("elementwise/reduce", ("elementwise", "reduce_kernel", "fill", "CatArray", "copy_device")),
]


def fam(n):
    for name, pats in FAMS:
        if any(p in n for p in pats):
            return name
    return "other"


per_stage = {}
for pp in (0, 1, 3):
    f = max(glob.glob(d + "/*TP-0-PP-%d.trace.json.gz" % pp))
    ev = json.load(gzip.open(f))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    steps = max(1, sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]) / 22.0)
    agg = defaultdict(float)
    cnt = defaultdict(int)
    comm = 0.0
    for e in ks:
        n = e["name"]
        if "SendRecv" in n or "AllGather" in n or "AllReduce" in n or "all_reduce" in n:
            comm += e["dur"]
            continue
        agg[fam(n)] += e["dur"] / 1e3 / steps
        cnt[fam(n)] += 1
    per_stage[pp] = (steps, agg, cnt, comm / 1e3 / steps)

print("steps per stage: " + ", ".join("PP%d %.1f" % (p, per_stage[p][0]) for p in per_stage))
tot = sum(sum(per_stage[p][1].values()) for p in per_stage) / 3
print("\nper stage per token (mean of 3 captured stages), compute excluding comm:")
print("%22s %9s %8s %10s" % ("family", "ms/stage", "share", "calls/stage"))
for name, _ in FAMS + [("other", ())]:
    ms = sum(per_stage[p][1].get(name, 0.0) for p in per_stage) / 3
    c = sum(per_stage[p][2].get(name, 0) for p in per_stage) / 3
    print("%22s %9.2f %7.1f%% %10.0f" % (name, ms, 100 * ms / tot, c))
print("%22s %9.2f" % ("TOTAL compute", tot))
comm = sum(per_stage[p][3] for p in per_stage) / 3
print("%22s %9.2f  (spin at bs=1, not work)" % ("comm", comm))
print("\nx4 stages serial: %.2f ms/token -> %.1f tok/s ceiling" % (4 * tot, 1000 / (4 * tot)))
