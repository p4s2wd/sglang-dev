"""Per-stage busy fraction WITHIN a decode step, not across the capture window.

The whole-window busy fraction is 3-4%, but that is meaningless: the traces contain
single gaps of 1083 ms, 1653 ms and 1894 ms between two nccl SendRecv kernels, which
is the capture window sitting idle, not a decode step.

Segment each stage's kernels into steps by cutting at gaps > 20 ms (a bs=1 step is
63.7 ms wall, so a real step's kernels are contiguous well inside that), then report
busy and span per step. That answers the question the budget left open: the summed
device work is 39.0 ms of a 63.7 ms step, so is the missing 24.7 ms stages sitting
idle (structural, fix by pipeline shape) or stages running kernels slowly (fix with
kernels)?
"""
import gzip, glob, json, sys

d = sys.argv[1]
CUT = 20e3  # us
rows = []
for f in sorted(glob.glob(d + "/*DECODE*.trace.json.gz")):
    ev = json.load(gzip.open(f))["traceEvents"]
    pp = int(f.split("PP-")[1].split("-")[0])
    tp = int(f.split("TP-")[1].split("-")[0])
    ks = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    if len(ks) < 2:
        continue
    steps, cur = [], [ks[0]]
    for a, b in zip(ks, ks[1:]):
        if (b["ts"] - (a["ts"] + a["dur"])) > CUT:
            steps.append(cur)
            cur = [b]
        else:
            cur.append(b)
    steps.append(cur)
    for s in steps:
        busy = sum(e["dur"] for e in s) / 1e3
        span = (s[-1]["ts"] + s[-1]["dur"] - s[0]["ts"]) / 1e3
        if span > 150:  # a step is ~64 ms; longer runs are merged windows
            continue
        rows.append((pp, tp, busy, span, len(s)))

print("%d step-windows under 150 ms across %d stage traces" % (len(rows), len(glob.glob(d + '/*DECODE*'))))
print("\n%4s %4s %10s %10s %8s %8s" % ("PP", "TP", "busy ms", "span ms", "busy%", "kernels"))
for pp, tp, busy, span, n in sorted(rows)[:24]:
    print("%4d %4d %10.2f %10.2f %7.1f%% %8d" % (pp, tp, busy, span, 100 * busy / span, n))

import statistics as st
by_stage = {}
for pp, tp, busy, span, n in rows:
    by_stage.setdefault((pp, tp), []).append((busy, span))
print("\nper (PP,TP) median over windows:")
tot_busy = tot_span = 0.0
for (pp, tp), v in sorted(by_stage.items()):
    b = st.median(x[0] for x in v)
    s = st.median(x[1] for x in v)
    tot_busy += b
    tot_span = max(tot_span, s)
    print("  PP%d TP%d  busy %6.2f ms  span %6.2f ms  busy %5.1f%%  (%d windows)"
          % (pp, tp, b, s, 100 * b / s, len(v)))
print("\nsum of per-stage busy across the 4 PP stages (TP medians): %.2f ms" % tot_busy)
print("TP0+TP1 are parallel halves, so per-stage work = sum over PP of TP-median:")
tp0 = sum(st.median(x[0] for x in v) for (pp, tp), v in by_stage.items() if tp == 0)
print("  PP-sum (TP0) = %.2f ms  -> 4-stage serial floor %.2f ms" % (tp0, tp0))
