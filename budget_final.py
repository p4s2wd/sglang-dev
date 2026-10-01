"""The authoritative decode budget: per-stage device work per token, one clean pass.

Two earlier scripts disagreed (54.09 vs 35.3 ms/token of compute) because they counted
steps differently per stage and one of them subtracted an EXTEND file that the plain
file does not contain. Verified: the plain and -EXTEND traces have DISJOINT timestamp
ranges, so the plain file is already decode-only and subtracting EXTEND is wrong.

Ground truth for the step count: prof_exact.sh drove one request for 32 tokens at
bs=1, so there are 32 decode steps (31 observed plus one absorbed by prefill), and
sinkhorn runs twice per layer per step (682 calls / 11 layers / 31 steps = 2).

Report per stage: total device time, time inside nccl collectives (a spin at bs=1, not
work), and real compute. The serial floor for one token is the sum of per-stage
compute, because a token must traverse all 4 stages in order.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
LAY = 11
print("%4s %8s %9s %9s %9s %9s" % ("PP", "steps", "total ms", "comm ms", "comp ms",
                                   "comp/tok"))
tot_comp = tot_comm = 0.0
per_stage = []
for pp in range(4):
    f = glob.glob(d + "/*TP-0-PP-%d.trace.json.gz" % pp)
    ev = json.load(gzip.open(max(f)))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    sk = sum(1 for e in ks if "hc_split_sinkhorn" in e["name"])
    steps = sk / (2 * LAY)
    comm = sum(e["dur"] for e in ks if ("SendRecv" in e["name"] or "AllGather" in e["name"]
                                        or "all_reduce" in e["name"]
                                        or "AllReduce" in e["name"])) / 1e3
    total = sum(e["dur"] for e in ks) / 1e3
    comp = total - comm
    tot_comp += comp
    tot_comm += comm
    per_stage.append(comp / steps)
    print("%4d %8.1f %9.1f %9.1f %9.1f %9.2f"
          % (pp, steps, total, comm, comp, comp / steps))

print("\nsteps per stage should all be ~31 (32 driven tokens at bs=1)")
print("sum of per-stage COMPUTE per token = %.2f ms  <- serial floor for one token"
      % sum(per_stage))
print("  => single-stream ceiling with zero bubble: %.1f tok/s"
      % (1000.0 / sum(per_stage)))
print("comm (spin) per token summed over stages: %.2f ms" % (tot_comm / 31.0))
print("\nmeasured wall step at bs=1: 63.7 ms -> 15.7 tok/s")
print("  compute floor %.2f ms = %.0f%% of the measured step"
      % (sum(per_stage), 100 * sum(per_stage) / 63.7))
