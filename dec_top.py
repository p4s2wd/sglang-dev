"""Rank DECODE kernels. prod_top2.py hardcodes glob("/*EXTEND*"), so it silently
ranks PREFILL when pointed at a decode profile directory -- which is how a
"decode ranking" ended up reporting 299.9 ms of device time with 22 allreduce
calls, numbers that cannot describe a 64 ms/token decode step.

This selects DECODE traces, reports per-stage totals, and normalizes to ms per
token so the ranking can be compared against the 64 ms/token measured wall time
and against the 10.7 ms/token bandwidth floor.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
files = sorted(glob.glob(d + "/*DECODE*.trace.json.gz"))
if not files:
    print("no DECODE traces in", d)
    sys.exit(0)

# Aggregate across stages: one token passes through all of them.
agg = defaultdict(lambda: [0.0, 0])
tot = 0.0
nstages = 0
for f in files:
    ev = json.load(gzip.open(f))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    nstages += 1
    tot += sum(e["dur"] for e in ks) / 1e3
    for e in ks:
        agg[e["name"][:60]][0] += e["dur"] / 1e3
        agg[e["name"][:60]][1] += 1

# Decode steps captured: infer from the per-layer kernel call count.
attn = [v for k, v in agg.items() if "sparse" in k or "headshared" in k]
steps = max((v[1] for v in attn), default=0) / 11.0 / nstages
print("%d DECODE stages, %.1f ms aggregate device time, ~%.1f decode steps"
      % (nstages, tot, steps))
print("=> %.1f ms of device time per token across all %d stages" % (tot / max(steps, 1e-9), nstages))
print("\n%8s %6s %9s %8s  kernel" % ("ms", "calls", "us/call", "share"))
for k, (ms, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:14]:
    print("%8.1f %6d %9.1f %7.1f%%  %s" % (ms, n, ms * 1e3 / n, 100 * ms / tot, k))
