"""How much of a decode step has NO kernel running on the stage?

Settling the step count first. Two per-layer kernels disagree unless attention is
counted correctly: sinkhorn is 1 per layer and shows 88 calls / 11 layers = 8, while
attention shows 88 calls. Attention is called twice per layer only on layers that
have a compressed cache; compress_ratios is [0,0,4,128,...,0,0,0], so the last three
layers and the first two have a single call. 88 calls over 11 layers across N steps
fits N=8 with a mixed per-layer count, and 4 only if every layer made 2 calls. So the
capture holds 8 decode steps per stage, which makes per-stage device work
PP0 11.3, PP1 7.9, PP2 8.8, PP3 9.8 ms per token -- 37.8 ms summed against a 63.7 ms
wall step, matching the budget in the report.

Given that, split each stage's step time into (a) kernel time, (b) time inside
SendRecv, which is a spin waiting on a peer rather than work, and (c) gaps where no
kernel is resident at all. (c) is host-bound: launch, sync, and eager Python between
graph replays. Cut at gaps > 20 ms to exclude the capture window sitting idle.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
STEPS = 8
for f in sorted(glob.glob(d + "/*DECODE*.trace.json.gz")):
    ev = json.load(gzip.open(f))["traceEvents"]
    pp = int(f.split("PP-")[1].split("-")[0])
    tp = int(f.split("TP-")[1].split("-")[0])
    if tp != 0:
        continue
    ks = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    bursts, cur = [], [ks[0]]
    for a, b in zip(ks, ks[1:]):
        if (b["ts"] - (a["ts"] + a["dur"])) > 20e3:
            bursts.append(cur); cur = [b]
        else:
            cur.append(b)
    bursts.append(cur)
    kern = sr = gap = 0.0
    n = 0
    for b in bursts:
        span = (b[-1]["ts"] + b[-1]["dur"] - b[0]["ts"]) / 1e3
        if span < 30 or span > 400:
            continue
        n += 1
        kern += sum(e["dur"] for e in b) / 1e3
        sr += sum(e["dur"] for e in b if "SendRecv" in e["name"]) / 1e3
        gap += sum(max(0.0, (y["ts"] - (x["ts"] + x["dur"]))) / 1e3
                   for x, y in zip(b, b[1:]))
    if not n:
        continue
    k, s, g = kern / n, sr / n, gap / n
    print("PP%d  per burst(%d): kernels %6.2f ms  of which SendRecv %5.2f  "
          "no-kernel gaps %6.2f ms  span %6.2f" % (pp, n, k, s, g, (k + g)))
    print("      per token (/%d): kernels %5.2f  SendRecv-spin %5.2f  host-gap %5.2f"
          % (STEPS, k / STEPS, s / STEPS, g / STEPS))
