"""Print the kernel sequence of one decode layer, in launch order.

CUDA graph replay decouples kernels from their originating aten ops (every
kernel joins to cudaGraphLaunch), so the op->kernel join is useless inside a
captured region. The time-ordered kernel stream still shows the pattern: with a
4-layer dummy server, the kernels between two consecutive w4a16 anchors are
exactly one layer's work, which is what a fusion pass would rewrite.
"""
import gzip
import json
import re
import sys

path = sys.argv[1]
ev = json.load(gzip.open(path)).get("traceEvents", [])
ks = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e.get("ts", 0))


def short(n):
    n = re.sub(r"\s*\(.*", "", n)
    n = re.sub(r"<.*", "", n).strip()
    n = n.replace("void at::native::", "at::")
    n = n.replace("std::enable_if", "gemv")
    return n[:44]


anch = [i for i, e in enumerate(ks) if "w4a16" in e.get("name", "")]
print(f"{len(ks)} kernels, {len(anch)} w4a16 anchors")
if len(anch) < 2:
    raise SystemExit("need two anchors")
# One layer is everything between the same anchor two layers apart; with a
# 4-layer server the anchors repeat every 2 gemms x 4 layers, so take the span
# between anchors that are 8 apart to cover a whole layer (gemm1, act, gemm2,
# combine, attention, gate) rather than just gemm1->gemm2.
step = 2
a, b = anch[0], anch[min(step, len(anch) - 1)]
unit = ks[a:b]
tot = sum(e.get("dur", 0) for e in unit)
print(f"span = {len(unit)} kernels, {tot/1e3:.3f} ms device")

# Aggregate first: which names dominate, so the long tail is visible at a glance.
agg = {}
for e in unit:
    n = short(e.get("name", "?"))
    c, d = agg.get(n, (0, 0.0))
    agg[n] = (c + 1, d + e.get("dur", 0))
print("\nby name:")
for n, (c, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:14]:
    print(f"  {d/1e3:7.3f}ms x{c:<4d} avg {d/c:6.1f}us  {n}")

if "--seq" in sys.argv:
    print("\nin launch order:")
    for i, e in enumerate(unit):
        d = e.get("dur", 0)
        print(f"  {i:3d} {d:8.1f}us  {short(e.get('name', '?'))}")
