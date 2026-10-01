"""Kernel count and cost per decode step, correct duration buckets.

The previous version bucketed on dur/1e3 < 1, which means "under 1000 us", so almost
every kernel landed in the first bucket and the table was nonsense. dur is already in
microseconds (median 2.3, p99 115, max 6496), so bucket on it directly.

Objective item (3) is "fuse the ~98 small kernels per layer". This gives the real
count and the real cost: how many kernels per step per stage, how much device time
sits in the sub-5 us population, and what fusing them could recover.
"""
import gzip, glob, json, sys

d = sys.argv[1]
f = max(glob.glob(d + "/*TP-0-PP-3.trace.json.gz"))
ev = json.load(gzip.open(f))["traceEvents"]
ks = [e for e in ev if e.get("cat") == "kernel"]
steps = max(1, sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]) / 22.0)
tot = sum(e["dur"] for e in ks) / 1e3 / steps
LAY = 11
print("PP3: %.0f kernels/step (%.0f per layer), %.2f ms/step" %
      (len(ks) / steps, len(ks) / steps / LAY, tot))
print("\n%10s %12s %10s %8s %12s" % ("us range", "kernels/step", "ms/step", "share", "per layer"))
for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 50), (50, 200), (200, 10 ** 9)):
    sel = [e for e in ks if lo <= e["dur"] < hi]
    ms = sum(e["dur"] for e in sel) / 1e3 / steps
    lbl = "%d-%d" % (lo, hi) if hi < 10 ** 9 else ">=%d" % lo
    print("%10s %12.1f %10.2f %7.1f%% %12.1f"
          % (lbl, len(sel) / steps, ms, 100 * ms / tot, len(sel) / steps / LAY))

for CUT in (2, 3, 5):
    sel = [e for e in ks if e["dur"] < CUT]
    ms = sum(e["dur"] for e in sel) / 1e3 / steps
    print("\nunder %d us: %.0f kernels/step (%.0f%% of count), %.2f ms/step (%.0f%% of time)"
          % (CUT, len(sel) / steps, 100 * len(sel) / len(ks), ms, 100 * ms / tot))
    # what perfect fusion of groups of 4 would recover: 3 launches removed, each
    # costing at least the measured median of that population
    import statistics
    med = statistics.median([e["dur"] for e in sel])
    print("   median %.2f us; fusing 4->1 removes 3/4 of the launches: %.2f ms/step best case"
          % (med, ms * 0.75))
