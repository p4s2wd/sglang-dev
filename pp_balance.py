"""Per-stage device time from the prefill profile traces.

Prefill runs 880 tok/s while the busiest pipeline stage spends 345 ms of device
time per 512-token chunk, which alone would allow 1484 tok/s. Either the stages
are unbalanced (so the quoted 345 ms understates the real critical path) or there
is ~220 ms/chunk of bubble between them. The traces name the stage, so summing
kernel time per PP rank separates the two: a big spread means rebalancing layers
is a free throughput win, an even spread means the loss is pipeline fill and only
faster kernels or more concurrency will help.
"""
import gzip, glob, json, sys, re
from collections import defaultdict

d = sys.argv[1]
tot = defaultdict(float)
for f in sorted(glob.glob(d + "/*EXTEND*")):
    m = re.search(r"PP-(\d+)", f)
    if not m: continue
    ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]
    s = sum(e["dur"] for e in ev if e.get("cat") == "kernel") / 1e3
    tot[int(m.group(1))] += s
if not tot:
    print("no PP-tagged traces"); sys.exit(1)
n = max(len(glob.glob(d + "/*EXTEND*")) // 2, 1)
print("aggregate device ms per PP stage (same capture window):")
for pp in sorted(tot):
    print("  PP%d  %8.1f ms" % (pp, tot[pp]))
vals = sorted(tot.values())
print("spread: max/min = %.2fx   busiest stage share of sum = %.0f%%"
      % (vals[-1] / vals[0], 100 * vals[-1] / sum(vals)))
