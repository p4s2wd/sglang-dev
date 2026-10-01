"""Count both attention kernels in both traces, per stage and per phase.

prod_top2.py hardcodes glob("/*EXTEND*"), so every "decode kernel ranking" I have
quoted from it was actually a PREFILL ranking. That explains the numbers that never
reconciled (299.9 ms of device time with 22 allreduce calls cannot describe a
64 ms/token decode step). Before correcting the record, establish what is actually
in each trace: count the per-head and head-shared attention kernels separately, in
both the EXTEND and DECODE traces, before and after the gate change.
"""
import gzip, glob, json, sys
from collections import defaultdict

for d in sys.argv[1:]:
    print("=== %s" % d.split("/")[-1])
    for f in sorted(glob.glob(d + "/*.trace.json.gz")):
        pp = f.split("PP-")[1].split("-")[0]
        ph = "EXTEND" if "EXTEND" in f else "DECODE"
        ev = json.load(gzip.open(f))["traceEvents"]
        ks = [e for e in ev if e.get("cat") == "kernel"]
        c = defaultdict(lambda: [0, 0.0])
        for e in ks:
            n = e["name"]
            if "tiled_sparse" in n:
                c["per-head"][0] += 1; c["per-head"][1] += e["dur"] / 1e3
            elif "headshared" in n:
                c["headshared"][0] += 1; c["headshared"][1] += e["dur"] / 1e3
        tot = sum(e["dur"] for e in ks) / 1e3
        parts = ", ".join("%s %dx %.1fms (%.3f/call)"
                          % (k, v[0], v[1], v[1] / v[0] if v[0] else 0)
                          for k, v in sorted(c.items()))
        print("  PP%s %-6s total %7.1f ms  %s" % (pp, ph, tot, parts or "no attn kernel"))
