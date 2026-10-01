"""Compare decode kernel count and device time between sparse-MLA backends.

Two eager traces of the same 4-layer dummy server, one per backend. The
comparison excludes the TP allreduce kernels: all_reduce_1shot_push spins
waiting for the peer rank, so in eager mode (no graph, no overlap scheduler) it
measures how out-of-step the two ranks are, not any communication cost. That
number swings by 100x between runs and would swamp the real difference.
"""
import gzip
import json
import re
import sys
from collections import defaultdict


def stats(path):
    ev = json.load(gzip.open(path)).get("traceEvents", [])
    n_all = d_all = 0
    n_x = d_x = 0
    per = defaultdict(lambda: [0, 0.0])
    for e in ev:
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        d = e.get("dur", 0)
        n_all += 1
        d_all += d
        if "all_reduce" in name or "nccl" in name.lower():
            n_x += 1
            d_x += d
            continue
        short = re.sub(r"\s*\(.*", "", re.sub(r"<.*", "", name)).strip()
        short = short.replace("void at::native::", "at::").replace("std::enable_if", "gemv")
        per[short[:44]][0] += 1
        per[short[:44]][1] += d
    return n_all, d_all, n_x, d_x, per


a, b = sys.argv[1], sys.argv[2]
na, da, nax, dax, pa = stats(a)
nb, db, nbx, dbx, pb = stats(b)
print(f"backend A: {na} kernels, {da/1e3:.2f}ms device ({nax} allreduce/{dax/1e3:.1f}ms excl)")
print(f"backend B: {nb} kernels, {db/1e3:.2f}ms device ({nbx} allreduce/{dbx/1e3:.1f}ms excl)")
print(f"\nexcluding allreduce:  A {na-nax} kernels {(da-dax)/1e3:.2f}ms"
      f"   B {nb-nbx} kernels {(db-dbx)/1e3:.2f}ms")
print(f"per decode step (10 steps): A {(na-nax)/10:.0f} kernels  B {(nb-nbx)/10:.0f} kernels")

print("\nkernels only in A (eliminated by B):")
for k in sorted(pa, key=lambda x: -pa[x][1]):
    if k not in pb and pa[k][1] > 2000:
        print(f"  {pa[k][1]/1e3:7.2f}ms x{pa[k][0]:<5d} {k}")
print("\nkernels only in B (introduced):")
for k in sorted(pb, key=lambda x: -pb[x][1]):
    if k not in pa and pb[k][1] > 2000:
        print(f"  {pb[k][1]/1e3:7.2f}ms x{pb[k][0]:<5d} {k}")
