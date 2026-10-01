"""Rank kernels in one trace file, with the phase stated explicitly.

The previous version hardcoded glob("/*EXTEND*"), so pointing it at a directory that
also contained DECODE traces silently ranked PREFILL and printed it as if it were the
answer. That produced two wrong conclusions this session: a "decode ranking" claiming
per-head attention cost 1.338 ms/call (that was prefill's head-shared kernel), and a
claim that a gate change had cut decode attention by 6.2x. Neither was true.

Fix: take the phase from the filename, print it in the header, and refuse to guess --
if the pattern matches nothing, say so instead of falling back to another phase.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
phase = sys.argv[2] if len(sys.argv) > 2 else "EXTEND"
hits = glob.glob(d + "/*%s*" % phase)
if not hits:
    print("no *%s* traces in %s" % (phase, d))
    sys.exit(1)
f = max(hits)
ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]
ks = [e for e in ev if e.get("cat") == "kernel"]
agg = defaultdict(lambda: [0.0, 0])
for e in ks:
    agg[e["name"][:58]][0] += e["dur"] / 1e3
    agg[e["name"][:58]][1] += 1
tot = sum(v[0] for v in agg.values())
print("PHASE=%s  file=%s" % (phase, f.split("/")[-1]))
print("%d kernels, %.1f ms aggregate device time" % (len(ks), tot))
print("%-60s %7s %6s %8s" % ("kernel", "ms", "calls", "ms/call"))
for k, (ms, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:15]:
    print("%-60s %7.1f %6d %8.3f" % (k, ms, n, ms / n))
