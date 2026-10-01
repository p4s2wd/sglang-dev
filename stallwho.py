"""Which stages are idle during the two stalls?

The PP-3 trace has two real stalls (500 ms and 1506 ms) between chunk 6 and chunk
7. Excluding them the stage is 94% busy, so the whole 30%-busy figure is those two
events. What they are depends on who else is stalled at the same wall-clock moment:

  - all four stages stalled together  -> a global event (profiler flush, a barrier,
    or the scheduler doing O(prompt) work for every rank at once)
  - only the last stage stalled       -> the "last PP rank staggler" the loop
    docstring names, i.e. CPU post-processing on the drain rank
  - a staggered cascade               -> genuine pipeline back-pressure

All four traces share one clock, so this is a direct comparison.
"""
import gzip, glob, json, sys

d = sys.argv[1]
files = sorted(glob.glob(d + "/*EXTEND*"))
data = {}
for f in files:
    pp = int(f.split("PP-")[1].split("-")[0])
    ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]
    data[pp] = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])

# Common time base: earliest kernel across all stages.
t0 = min(ks[0]["ts"] for ks in data.values() if ks)
print("stall windows taken from the largest stage; busy%% = kernel time in window / window")

# Find the stalls on the reference stage, then test every stage in those windows.
ref = max(data, key=lambda p: len(data[p]))
ks = data[ref]
cur = ks[0]["ts"] + ks[0]["dur"]
stalls = []
for e in ks[1:]:
    if e["ts"] - cur > 200e3:
        stalls.append((cur, e["ts"]))
    cur = max(cur, e["ts"] + e["dur"])

print("reference stage PP%d, %d stalls > 200 ms" % (ref, len(stalls)))
for a, b in stalls:
    print("\n  window t=%.0f..%.0f ms (%.0f ms)" % ((a - t0) / 1e3, (b - t0) / 1e3, (b - a) / 1e3))
    for pp in sorted(data):
        k = [e for e in data[pp] if e["ts"] >= a and e["ts"] < b]
        busy = sum(e["dur"] for e in k) / (b - a) * 100
        names = {}
        for e in k:
            names[e["name"][:34]] = names.get(e["name"][:34], 0) + e["dur"]
        top = sorted(names.items(), key=lambda x: -x[1])[:2]
        print("    PP%d busy %5.1f%%  %d kernels  %s"
              % (pp, busy, len(k), ", ".join("%s %.0fms" % (n, v / 1e3) for n, v in top)))
