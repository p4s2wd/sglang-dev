"""Per-stage device work per decode token, from a capture with a known step count.

prof_exact.sh drove exactly one request for exactly N tokens at bs=1, so the number of
decode steps is exactly N. Count the steps from the trace itself (sinkhorn calls /
layers, and SendRecv count) to confirm N, then divide each stage's kernel time by it.
This replaces the inferred step count that made the report's budget wrong by up to 2x.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
N = int(sys.argv[2])
print("driven tokens (steps at bs=1): %d" % N)
tot = 0.0
print("%4s %10s %10s %10s %10s" % ("PP", "busy ms", "ms/token", "SendRecv", "SR ms/tok"))
for pp in range(4):
    busy = sr = 0.0
    nsk = 0
    for tp in (0, 1):
        f = glob.glob(d + "/*TP-%d-PP-%d-DECODE*" % (tp, pp))
        if not f:
            continue
        ev = json.load(gzip.open(max(f)))["traceEvents"]
        ks = [e for e in ev if e.get("cat") == "kernel"]
        b = sum(e["dur"] for e in ks) / 1e3
        s = sum(e["dur"] for e in ks if "SendRecv" in e["name"]) / 1e3
        nsk = max(nsk, sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]))
        if tp == 0:      # TP ranks split the same work; count one rank per stage
            busy += b
            sr += s
    tot += busy
    print("%4d %10.2f %10.2f %10.2f %10.2f   (sinkhorn calls %d -> %.1f steps if 1/layer, %.1f if 2/layer)"
          % (pp, busy, busy / N, sr, sr / N, nsk, nsk / 11.0, nsk / 22.0))
print("\nsum over 4 stages (TP0 only): %.2f ms/token" % tot)
print("sinkhorn-derived step count check below against driven N=%d" % N)
