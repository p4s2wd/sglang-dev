"""Decode step anatomy on one shared clock: per-stage busy vs wall per step.

step_busy.py segmented each stage trace independently, so its windows were not the
same interval on different stages and the numbers could not be added. All 8 traces
come from one machine and one clock, so segment GLOBALLY: find gaps > 20 ms in the
union of all kernels, giving activity bursts, then within each burst report each
stage's busy time, the burst span, and the implied per-step cost.

Use nccl SendRecv as the step marker: each stage sends its activations to the next
stage once per decode step, so the count of SendRecv launches per stage is the number
of steps captured. That converts busy time into ms/step per stage, which can be
compared against the measured 63.7 ms wall per step to say how much of a step is
serial stage work versus overlap.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
stages = {}
for f in sorted(glob.glob(d + "/*DECODE*.trace.json.gz")):
    ev = json.load(gzip.open(f))["traceEvents"]
    pp = int(f.split("PP-")[1].split("-")[0])
    tp = int(f.split("TP-")[1].split("-")[0])
    ks = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    stages[(pp, tp)] = ks

allk = sorted([e for ks in stages.values() for e in ks], key=lambda e: e["ts"])
# global bursts
bursts, cur = [], [allk[0]]
for a, b in zip(allk, allk[1:]):
    if (b["ts"] - (a["ts"] + a["dur"])) > 20e3:
        bursts.append(cur); cur = [b]
    else:
        cur.append(b)
bursts.append(cur)
print("%d global bursts from %d kernels" % (len(bursts), len(allk)))

# steps = SendRecv count per stage
sr = {}
for k, ks in stages.items():
    sr[k] = sum(1 for e in ks if "SendRecv" in e["name"])
print("SendRecv launches per stage: %s" % dict(sorted(sr.items())))

print("\n%6s %8s %s" % ("burst", "span ms", "busy ms per (PP,TP)  [TP0/TP1]"))
for i, b in enumerate(bursts):
    lo, hi = b[0]["ts"], max(e["ts"] + e["dur"] for e in b)
    span = (hi - lo) / 1e3
    if span < 30:
        continue
    parts = []
    tot = 0.0
    for (pp, tp), ks in sorted(stages.items()):
        busy = sum(e["dur"] for e in ks if lo <= e["ts"] < hi) / 1e3
        if tp == 0:
            tot += busy
        parts.append("PP%dT%d %5.1f" % (pp, tp, busy))
    print("%6d %8.1f  %s   PP-sum(TP0) %5.1f" % (i, span, " ".join(parts), tot))

# per-step: total busy / steps
print("\nper-step (using SendRecv steps per stage):")
for (pp, tp), ks in sorted(stages.items()):
    n = max(sr[(pp, tp)], 1)
    busy = sum(e["dur"] for e in ks) / 1e3
    print("  PP%d TP%d  %6.2f ms busy / %2d steps = %5.2f ms/step" % (pp, tp, busy, n, busy / n))
tp0 = [sum(e["dur"] for e in stages[(pp, 0)]) / 1e3 / max(sr[(pp, 0)], 1) for pp in range(4)]
print("\n  sum over 4 PP stages (TP0): %.2f ms/step" % sum(tp0))
print("  measured wall per step at bs=1: 63.7 ms")
print("  -> serial stage work %.2f ms = %.0f%% of the step" % (sum(tp0), 100 * sum(tp0) / 63.7))
