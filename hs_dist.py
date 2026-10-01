import gzip, glob, json, sys
from collections import defaultdict
d = sys.argv[1]
f = max(glob.glob(d + "/*TP-0-PP-3*EXTEND*"))
ev = json.load(gzip.open(f))["traceEvents"]
hs = [e for e in ev if e.get("cat") == "kernel" and "headshared" in e["name"]]
by_grid = defaultdict(list)
for e in hs:
    by_grid[tuple(e["args"].get("grid", []))].append(e["dur"] / 1e3)
for g, durs in sorted(by_grid.items(), key=lambda x: -sum(x[1])):
    durs.sort()
    n = len(durs)
    print("grid %-14s n=%3d  min %.2f  med %.2f  p90 %.2f  max %.2f ms  total %.0f ms"
          % (str(g), n, durs[0], durs[n//2], durs[int(n*0.9)], durs[-1], sum(durs)))
# what else runs alongside: index kernels?
oth = defaultdict(lambda: [0.0, 0])
for e in ev:
    if e.get("cat") != "kernel": continue
    n = e["name"]
    if any(k in n for k in ("index", "topk", "sink", "rope", "quant", "gather")):
        oth[n[:52]][0] += e["dur"] / 1e3; oth[n[:52]][1] += 1
print("\nrelated kernels in the same stage:")
for n, (ms, c) in sorted(oth.items(), key=lambda x: -x[1][0])[:8]:
    print("  %-54s %7.1f ms  %5d calls" % (n, ms, c))
