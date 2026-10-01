"""Is the prefill attention kernel at the memory floor, or reading too much?

It is now 39.6% of prefill device time, so it is the only remaining lever big
enough to reach 1000 tok/s on its own. The card streams at 521-575 GB/s
(measured), so the question is arithmetic: bytes the kernel must read / time it
takes. If that lands near 500 GB/s the kernel is done and only reading fewer
bytes helps. If it lands near 250 there is a 2x to recover.

Bytes per call are derived from the launch grid and the cache layout, not
guessed: each program handles BLOCK_H heads of one token and reads the token's
index_topk KV entries, each entry being 576 bytes of data plus 8 of scale.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
f = max(glob.glob(d + "/*TP-0-PP-3*EXTEND*"))
ev = json.load(gzip.open(f))["traceEvents"]
ks = [e for e in ev if e.get("cat") == "kernel"]

hs = [e for e in ks if "headshared" in e["name"]]
if not hs:
    print("no headshared kernel in this trace"); sys.exit(0)

g0 = hs[0]["args"].get("grid")
print("head-shared launches: %d, first grid=%s" % (len(hs), g0))
grids = defaultdict(lambda: [0.0, 0])
for e in hs:
    grids[tuple(e["args"].get("grid", []))][0] += e["dur"] / 1e3
    grids[tuple(e["args"].get("grid", []))][1] += 1

TOPK, TOKEN_BYTES, SCALE_BYTES = 512, 576, 8
tot_ms = sum(v[0] for v in grids.values())
print("\n%-22s %7s %8s %9s %10s %9s" % ("grid", "calls", "ms", "ms/call",
                                        "GB/call", "GB/s"))
for g, (ms, c) in sorted(grids.items(), key=lambda x: -x[1][0])[:6]:
    # grid is (num_tokens, num_head_blocks) or transposed; the token axis is the
    # one that scales with the prompt.
    toks = max(g[0], g[1])
    hblocks = min(g[0], g[1])
    # Each token block reads TOPK KV entries once and shares them across BLOCK_H
    # heads, so bytes per program = TOPK * (576+8).
    gb = toks * TOPK * (TOKEN_BYTES + SCALE_BYTES) / 1e9
    print("%-22s %7d %8.1f %9.3f %10.3f %9.0f"
          % (str(g), c, ms, ms / c, gb, gb / (ms / c * 1e-3)))
print("\nmeasured streaming ceiling on this card: 521-575 GB/s")
print("total head-shared: %.1f ms of %.1f ms in this trace" % (tot_ms, sum(e["dur"] / 1e3 for e in ks)))
