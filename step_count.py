"""How many decode steps are in the capture? Settle it from several per-layer kernels.

step_busy/global_step analysis says each DECODE trace holds 4 decode steps, which
contradicts dec_top.py's "~7.6 decode steps". dec_top computed
    steps = max(attn call count) / 11.0 / nstages
but its `agg` was already summed over all 8 stage traces, so the numerator was 8x too
large and it divided by nstages again -- the two errors do not cancel, and the result
understated ms per token by about 2x. That error propagated into the report's budget
("9.75 ms device work per stage per token", "39.0 ms serial"), so the step count has to
be pinned down from kernels whose per-step count is known independently.

Per decode step, per stage: attention is called twice per layer (SWA cache plus the
c4/c128 compressed cache, flash_mla_sm120_triton.py:371 and :377), the expert kernel
twice per MoE layer (gate and up, then down), and the W8A16 GEMV once per dense
projection. With 11 layers per stage those give 22, 22 and ~44 calls per step.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
f = max(glob.glob(d + "/*TP-1-PP-3-DECODE*.trace.json.gz"))
ev = json.load(gzip.open(f))["traceEvents"]
ks = [e for e in ev if e.get("cat") == "kernel"]
c = defaultdict(int)
for e in ks:
    n = e["name"]
    if "sparse" in n or "headshared" in n:
        c["attn (2 per layer)"] += 1
    elif "w4a16" in n:
        c["expert w4a16 (2 per MoE layer)"] += 1
    elif "w8a16" in n:
        c["w8a16 gemv (per dense proj)"] += 1
    elif "SendRecv" in n:
        c["nccl SendRecv (1 per step)"] += 1
    elif "hc_split_sinkhorn" in n:
        c["sinkhorn (1 per layer)"] += 1

LAY = 11
print("stage PP3 TP1, %d kernels" % len(ks))
for k, v in sorted(c.items()):
    if "attn" in k:
        n = v / (2 * LAY)
    elif "expert" in k:
        n = v / (2 * LAY)
    elif "sinkhorn" in k:
        n = v / LAY
    elif "SendRecv" in k:
        n = v
    else:
        n = None
    print("  %-34s %5d calls  -> %s steps"
          % (k, v, ("%.2f" % n) if n else "(count dense projs/layer to use)"))
