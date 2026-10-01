"""Per-stage device busy fraction during a bs=1 decode step.

The budget says a bs=1 step takes 63.7 ms wall while the summed device work across
the 4 PP stages is 39.0 ms, leaving 24.7 ms unexplained. Two very different causes:

  - stages are IDLE waiting (pipeline bubble / token dependency): the fix is
    structural (fewer stages, or overlap), and no kernel work helps.
  - stages are BUSY but slow: the fix is kernels, and the 3.6x gap to the bandwidth
    floor is the real target.

Compute it directly from the DECODE traces: busy = sum of kernel durations, span =
last kernel end minus first kernel start, per stage. Also report the largest gaps
between consecutive kernels on each stage, since a stage that is idle shows up as a
few long gaps rather than general thinness.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
for f in sorted(glob.glob(d + "/*DECODE*.trace.json.gz")):
    ev = json.load(gzip.open(f))["traceEvents"]
    pp = f.split("PP-")[1].split("-")[0]
    tp = f.split("TP-")[1].split("-")[0]
    ks = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    if len(ks) < 2:
        continue
    busy = sum(e["dur"] for e in ks) / 1e3
    span = (ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]) / 1e3
    gaps = []
    for a, b in zip(ks, ks[1:]):
        g = (b["ts"] - (a["ts"] + a["dur"])) / 1e3
        if g > 0:
            gaps.append((g, a["name"][:34], b["name"][:34]))
    gaps.sort(reverse=True)
    print("PP%s TP%s  busy %6.1f ms  span %6.1f ms  busy %5.1f%%  kernels %d  gaps %d ms"
          % (pp, tp, busy, span, 100 * busy / span, len(ks), sum(g for g, _, _ in gaps)))
    for g, a, b in gaps[:3]:
        print("      gap %6.2f ms  after %-34s before %s" % (g, a, b))
