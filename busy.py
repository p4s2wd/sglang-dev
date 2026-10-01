"""Device-busy fraction of one PP stage during prefill, from the trace span.

Aggregate kernel time said 345 ms of device work per 512-token chunk, which alone
would allow 1484 tok/s, while wall time is 582 ms/chunk (880 tok/s). The PP stages
are balanced to 1.07x, so an unbalanced stage is not the explanation. This
measures the gap directly: for one stage's trace, the wall span from the first
kernel start to the last kernel end against the sum of kernel durations, plus the
largest individual idle gaps. If busy is ~60% and the gaps are many small ones,
the loss is launch/scheduler overhead and the fix is fewer, bigger kernels. If
busy is ~60% but a few gaps dominate, the fix is whatever blocks on those.
"""
import gzip, glob, json, sys, re
from collections import defaultdict

d = sys.argv[1]
f = max(glob.glob(d + "/*PP-3*EXTEND*"))
ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]
ks = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
if not ks:
    print("no kernels"); sys.exit(1)
span = (ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]) / 1e3
busy = sum(e["dur"] for e in ks) / 1e3
print("stage PP3: %d kernels, span %.1f ms, busy %.1f ms (%.0f%%)"
      % (len(ks), span, busy, 100 * busy / span))

# Largest idle gaps between consecutive kernels.
gaps = []
for a, b in zip(ks, ks[1:]):
    g = (b["ts"] - (a["ts"] + a["dur"])) / 1e3
    if g > 0:
        gaps.append((g, a["name"][:38], b["name"][:38]))
gaps.sort(reverse=True)
tot_gap = sum(g for g, _, _ in gaps)
print("total idle %.1f ms in %d gaps" % (tot_gap, len(gaps)))
print("top gaps:")
for g, a, b in gaps[:8]:
    print("  %7.2f ms  after %-38s before %s" % (g, a, b))
buckets = defaultdict(float)
for g, _, _ in gaps:
    k = ">=10ms" if g >= 10 else "1-10ms" if g >= 1 else "0.1-1ms" if g >= 0.1 else "<0.1ms"
    buckets[k] += g
print("idle by size bucket:")
for k in (">=10ms", "1-10ms", "0.1-1ms", "<0.1ms"):
    if k in buckets:
        print("  %-8s %7.1f ms (%.0f%% of idle)" % (k, buckets[k], 100 * buckets[k] / tot_gap))
