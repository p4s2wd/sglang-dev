"""Per-stage DECODE device work per token, from a capture with a known step count.

prof_exact.sh drove one request for N tokens at bs=1, so decode steps = N. Two things
had to be settled first, and both were wrong in earlier analysis:

1. File semantics. With profile_by_stage=True this capture wrote a plain file and an
   -EXTEND file, no -DECODE file. The plain file is the UNION of both phases, so its
   kernel time double-counts prefill; decode-only = plain minus EXTEND.

2. Calls per layer per step. sinkhorn and attention both show 682 calls for a 31-step
   capture over 11 layers per stage: 682 / 31 = 22 = 2 per layer. So both are called
   TWICE per layer per step, which means the earlier trace's 88 calls is 88/22 = 4
   steps, not 8. The report's budget used 8, halving every per-token figure.
"""
import gzip, glob, json, sys

d = sys.argv[1]
N = int(sys.argv[2])
print("%4s %10s %10s %10s %10s %10s" % ("PP", "plain ms", "extend ms", "decode ms",
                                        "ms/token", "SR ms/tok"))
tot = 0.0
steps_seen = []
for pp in range(4):
    f = glob.glob(d + "/*TP-0-PP-%d.trace.json.gz" % pp)
    fe = glob.glob(d + "/*TP-0-PP-%d-EXTEND*" % pp)
    ev = json.load(gzip.open(max(f)))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    plain = sum(e["dur"] for e in ks) / 1e3
    sk = sum(1 for e in ks if "hc_split_sinkhorn" in e["name"])
    steps = sk / 22.0
    steps_seen.append(steps)
    ext = 0.0
    if fe:
        ev2 = json.load(gzip.open(max(fe)))["traceEvents"]
        ext = sum(e["dur"] for e in ev2 if e.get("cat") == "kernel") / 1e3
    sr = sum(e["dur"] for e in ks if "SendRecv" in e["name"]) / 1e3
    dec = plain - ext
    tot += dec
    print("%4d %10.2f %10.2f %10.2f %10.2f %10.2f  (steps %.1f)"
          % (pp, plain, ext, dec, dec / max(steps, 1), sr / max(steps, 1), steps))
st = sum(steps_seen) / 4
print("\ndriven tokens %d, steps seen per stage %s" % (N, [round(x, 1) for x in steps_seen]))
print("per-stage decode work per token: %s" % " ".join("%.1f" % x for x in []))
print("sum over 4 stages = %.2f ms/token (one token's full traversal)" % (tot / st))
print("max single stage  = %.2f ms/token (pipeline cadence floor at bs>1)" % 0)
