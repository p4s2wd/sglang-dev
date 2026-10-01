"""What topk does production attention actually run, and what is its headroom?

Attention is now the largest compute family (12.77 ms/token summed over the 4 stages,
25.9%). Its bandwidth floor is set by the bytes it must gather: topk tokens x 576 B per
token per call. The isolated probe used topk=512 and measured 0.279 ms of device time
for the per-head kernel, which works out at 8.7 us per 16-token tile -- but production
splits attention into a small SWA window plus a c4/c128 compressed cache, so the topk
per call is smaller than 512 and the probe's absolute number does not apply.

Read the launch geometry from the production trace: grid is (B, H) for the per-head
kernel, and the call count per step tells how many attention calls a step makes. Then
compute the bytes each call must gather and the achieved rate, which is the number that
says whether attention is bandwidth-bound (leave it alone) or latency-bound (worth a
new kernel).
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
f = max(glob.glob(d + "/*TP-0-PP-3.trace.json.gz"))
ev = json.load(gzip.open(f))["traceEvents"]
ks = [e for e in ev if e.get("cat") == "kernel"]
steps = max(1, sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]) / 22.0)
LAY = 11

per = [e for e in ks if "_tiled_sparse_decode_kernel" in e["name"]]
hs = [e for e in ks if "_headshared_sparse_kernel" in e["name"]]
print("steps %.1f, layers %d" % (steps, LAY))
print("per-head calls/step %.1f   headshared calls/step %.1f" %
      (len(per) / steps, len(hs) / steps))
print("per-head ms/step %.2f   headshared ms/step %.2f" %
      (sum(e["dur"] for e in per) / 1e3 / steps, sum(e["dur"] for e in hs) / 1e3 / steps))
if per:
    print("grid %s block %s regs %s smem %s occ %s%%" %
          (per[0]["args"].get("grid"), per[0]["args"].get("block"),
           per[0]["args"].get("registers per thread"), per[0]["args"].get("shared memory"),
           per[0]["args"].get("est. achieved occupancy %")))

# calls per layer per step: 2 (SWA + compressed) for layers that have both
calls_layer = (len(per) + len(hs)) / steps / LAY
ms_layer = (sum(e["dur"] for e in per) + sum(e["dur"] for e in hs)) / 1e3 / steps / LAY
print("\nper layer per step: %.1f attention calls, %.3f ms" % (calls_layer, ms_layer))

# The c4/c128 compressed cache: at ratio 128 a 256K context compresses to 2048 tokens,
# and index_topk is 512, so a compressed-cache call gathers min(512, compressed_len).
# The SWA window is sliding_window=128.
for name, topk in (("SWA window", 128), ("compressed (index_topk)", 512)):
    b = topk * 576
    print("  %-22s topk %4d -> %6.1f KB per head, x64 heads = %5.2f MB per call"
          % (name, topk, b / 1e3, b * 64 / 1e6))
