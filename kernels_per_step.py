"""How many kernels does one decode step launch per stage, and what do they cost?

Objective item (3) is "fuse the ~98 small kernels per layer". Whether that is worth
attacking depends on the real count and the real cost, which the graphed trace gives
directly -- no eager capture, no attribution needed. Count kernels per step per stage,
bucket by duration, and report how much of the stage's device time sits in kernels
below the launch-latency floor. Those are the ones fusing can remove: a kernel that
costs 0.4 us on a 393 KB tensor is paying launch and teardown, not bandwidth, so
merging ten of them into one recovers most of the ten.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
for pp in (3,):
    f = max(glob.glob(d + "/*TP-0-PP-%d.trace.json.gz" % pp))
    ev = json.load(gzip.open(f))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    steps = max(1, sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]) / 22.0)
    tot = sum(e["dur"] for e in ks) / 1e3
    print("PP%d: %d kernels over %.1f steps = %.0f kernels/step, %.2f ms/step"
          % (pp, len(ks), steps, len(ks) / steps, tot / steps))

    # bucket by duration
    buckets = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 50), (50, 10000)]
    print("\n  %10s %10s %9s %9s" % ("us range", "kernels/step", "ms/step", "share"))
    for lo, hi in buckets:
        sel = [e for e in ks if lo <= e["dur"] / 1e3 < hi]
        ms = sum(e["dur"] for e in sel) / 1e3 / steps
        print("  %10s %10.0f %9.2f %8.1f%%"
              % ("%d-%d" % (lo, hi) if hi < 10000 else ">=%d" % lo,
                 len(sel) / steps, ms, 100 * ms / (tot / steps)))

    tiny = [e for e in ks if e["dur"] / 1e3 < 2]
    ms_tiny = sum(e["dur"] for e in tiny) / 1e3 / steps
    print("\n  kernels under 2 us: %.0f per step, %.2f ms/step (%.0f%% of stage time)"
          % (len(tiny) / steps, ms_tiny, 100 * ms_tiny / (tot / steps)))
    print("  if 4 such kernels fuse into 1, saving ~0.6 us each: %.2f ms/step"
          % (len(tiny) / steps * 0.75 * 0.6e-3))

    # per-layer count
    lay = sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]) / steps / 2
    print("  layers per stage: %.0f -> %.0f kernels per layer per step"
          % (lay, len(ks) / steps / lay))
