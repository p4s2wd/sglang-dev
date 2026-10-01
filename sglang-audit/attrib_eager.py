"""Attribute decode's elementwise/reduce/index kernels to the aten op that launched them.

In eager mode (no CUDA graph) the profiler keeps the cpu_op -> cuda_runtime ->
kernel correlation chain, which graph replay collapses. Kernels join to a
cuda_runtime event by "correlation"; that runtime event carries the "External id"
of the aten op that issued it. This is the map a fusion pass needs: which op
launches the ~93 small kernels a layer spends 20 ms/token on.

Run against the eager trace (serve with --disable-cuda-graph).
"""
import gzip
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
ev = json.load(gzip.open(path)).get("traceEvents", [])

ext2op = {}
for e in ev:
    if e.get("cat") == "cpu_op":
        eid = (e.get("args") or {}).get("External id")
        if eid is not None:
            ext2op[eid] = e.get("name", "?")

corr2ext = {}
for e in ev:
    if e.get("cat") == "cuda_runtime":
        a = e.get("args") or {}
        c = a.get("correlation")
        if c is not None:
            corr2ext[c] = a.get("External id")


def small_kernel(name):
    n = re.sub(r"\s*\(.*", "", name)
    n = re.sub(r"<.*", "", n).strip()
    if "elementwise" in n or "vectorized_elementwise" in n:
        return "elementwise"
    if "reduce_kernel" in n or "reduce_1Block" in n or n.endswith("_reduce"):
        return "reduce"
    if "index_elementwise" in n or "index_kernel" in n or "index_put" in n:
        return "index"
    if "copy_kernel" in n or n == "void at::native::copy_":
        return "copy"
    return None


rows = defaultdict(lambda: [0, 0.0])
unlinked = [0, 0.0]
for e in ev:
    if e.get("cat") != "kernel":
        continue
    kind = small_kernel(e.get("name", ""))
    if kind is None:
        continue
    c = (e.get("args") or {}).get("correlation")
    op = ext2op.get(corr2ext.get(c)) if c is not None else None
    if op is None:
        unlinked[0] += 1
        unlinked[1] += e.get("dur", 0)
        op = "(unlinked)"
    rows[(op, kind)][0] += 1
    rows[(op, kind)][1] += e.get("dur", 0)

tot = sum(v[1] for v in rows.values()) + unlinked[1]
print(f"{tot/1e3:.2f}ms of small kernels; unlinked {unlinked[0]} / {unlinked[1]/1e3:.2f}ms\n")
# Aggregate by op across kinds, then break the top ops down.
by_op = defaultdict(lambda: [0, 0.0])
for (op, kind), (c, d) in rows.items():
    by_op[op][0] += c
    by_op[op][1] += d
print("by aten op (launches, device ms):")
for op, (c, d) in sorted(by_op.items(), key=lambda kv: -kv[1][1])[:16]:
    print(f"  {d/1e3:7.2f}ms x{c:<5d} {op[:60]}")
