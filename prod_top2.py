"""Rank EXTEND (prefill) kernels after the v3 expert kernel landed.

The user's active priority is prefill >= 1000 tok/s; the head-shared attention
kernel and now the expert GEMM have both moved, so the old ranking -- w4a16 at
21.2%, allreduce at 35.6% -- is stale. Re-measure before picking the next lever.
Aggregated per kernel over the whole capture, plus the share of the aggregate
that each represents, so the ranking says what to attack next rather than what
was worth attacking last round.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
f = max(glob.glob(d + "/*EXTEND*"))
ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]
ks = [e for e in ev if e.get("cat") == "kernel"]
agg = defaultdict(lambda: [0.0, 0])
for e in ks:
    agg[e["name"][:58]][0] += e["dur"] / 1e3
    agg[e["name"][:58]][1] += 1
tot = sum(v[0] for v in agg.values())
print("%s\n%d kernels, %.1f ms aggregate device time" % (f.split("/")[-1], len(ks), tot))
print("%-60s %8s %6s %7s %6s" % ("kernel", "ms", "share", "calls", "ms/call"))
for n, (ms, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:14]:
    print("%-60s %8.1f %5.1f%% %7d %6.3f" % (n, ms, 100 * ms / tot, c, ms / c))
