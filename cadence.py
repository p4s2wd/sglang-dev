"""Chunk cadence per PP stage, from the server's own log timestamps.

The trace says one stage does 172 ms of device work per 512-token chunk while wall
time is 581 ms per chunk, and the four stages together do 689 ms -- so the stages
overlap only about 19%, close to fully serial. Perfect pipelining would give 172
ms/chunk (2976 tok/s) against the measured 857, so the prize is 3.4x and it is all
scheduling, not kernels. But the trace window is short and profile_by_stage can
truncate it, so confirm with an independent measurement: the server logs every
prefill batch with a timestamp and the stage that ran it. The spacing between one
stage's consecutive chunks IS that stage's cycle time. If PP0 logs a chunk every
~172 ms it is saturated and the 581 ms figure is wrong somewhere; if every ~581 ms
it is idle 70% of the time and the pipeline really is not overlapping.
"""
import re, sys, subprocess
from collections import defaultdict
from datetime import datetime

log = sys.argv[1]
txt = subprocess.run(["grep", "-aE", "Prefill batch", log], capture_output=True, text=True).stdout
pat = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\.(\d+))?\s+PP(\d+) TP\d+\] Prefill batch, #new-seq: (\d+), #new-token: (\d+)")
seen = defaultdict(list)
for m in pat.finditer(txt):
    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    frac = float("0." + m.group(2)) if m.group(2) else 0.0
    ts = ts.timestamp() + frac
    seen[int(m.group(3))].append((ts, int(m.group(4)), int(m.group(5))))

print("per-stage chunk cadence (seconds between that stage's consecutive chunks):")
for pp in sorted(seen):
    ev = seen[pp]
    if len(ev) < 3:
        continue
    gaps = [b[0] - a[0] for a, b in zip(ev, ev[1:])]
    tok = sum(g * 512 for g in gaps)
    # one-second log resolution: report the distribution, not just the mean
    hist = defaultdict(int)
    for g in gaps:
        hist[round(g, 1)] += 1
    top = sorted(hist.items(), key=lambda x: -x[1])[:4]
    span = (ev[-1][0] - ev[0][0]) or 1
    print("  PP%d: %3d chunks, span %.0fs -> %.0f ms/chunk; gaps %s"
          % (pp, len(ev), span, 1000 * span / (len(ev) - 1),
             ", ".join("%.0fms x%d" % (g * 1e3, c) for g, c in top)))
    print("       stage throughput %.0f tok/s" % (sum(t for _, _, t in ev) / span))
