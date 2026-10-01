"""Find the contiguous runs of tiny kernels a fusion pass would collapse.

The layer dump shows the small kernels are not scattered one at a time: they come
in runs of 10-25 launches with nothing expensive between them, which means a
single fused kernel could replace each run. A run is broken by any kernel over
BREAK_US (a real GEMM/GEMV), so the clusters are the ones where fusion is
mechanically possible rather than merely desirable.

Reports each run's launch count, device time, and how much of it is the 2 us
per-kernel floor (i.e. what fusion would actually recover).
"""
import gzip
import json
import re
import sys

path = sys.argv[1]
BREAK_US = 15.0
ev = json.load(gzip.open(path)).get("traceEvents", [])
ks = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e.get("ts", 0))


def short(n):
    n = re.sub(r"\s*\(.*", "", n)
    n = re.sub(r"<.*", "", n).strip()
    n = n.replace("void at::native::", "at::")
    n = n.replace("std::enable_if", "gemv")
    return n[:38]


anch = [i for i, e in enumerate(ks) if "w4a16" in e.get("name", "")]
unit = ks[anch[0]:anch[2]] if len(anch) > 2 else ks

runs = []
cur = []
for e in unit:
    d = e.get("dur", 0)
    if d > BREAK_US:
        if len(cur) >= 4:
            runs.append(cur)
        cur = []
    else:
        cur.append(e)
if len(cur) >= 4:
    runs.append(cur)

tot_small = sum(e.get("dur", 0) for r in runs for e in r)
print(f"layer span {len(unit)} kernels; {len(runs)} runs of >=4 tiny kernels")
print(f"tiny kernels inside runs: {sum(len(r) for r in runs)} launches, "
      f"{tot_small/1e3:.3f} ms\n")
FLOOR = 2.0
for i, r in enumerate(runs):
    d = sum(e.get("dur", 0) for e in r)
    floor = len(r) * FLOOR
    names = {}
    for e in r:
        n = short(e.get("name", "?"))
        names[n] = names.get(n, 0) + 1
    top = sorted(names.items(), key=lambda kv: -kv[1])[:3]
    tops = ", ".join(f"{n}x{c}" for n, c in top)
    print(f"  run {i}: x{len(r):<3d} {d:7.1f}us  floor {floor:5.1f}us "
          f"({floor/d*100:4.0f}%)  {tops[:78]}")
