"""Per-layer COMPUTE per token, excluding communication spin, and the elementwise split.

The corrected step count changes the whole budget. prof_exact.sh drove one request for
32 tokens at bs=1, so steps are known exactly, and both sinkhorn and attention make 2
calls per layer per step (682 calls / 11 layers / 31 steps). Earlier analysis assumed 1
call per layer and so counted 8 steps where there were 4, halving every per-token
figure it produced.

With steps known, per-stage device time per token is PP0 23.23, PP1 13.74, PP2 13.95,
PP3 16.23 ms. Most of PP0's excess is nccl SendRecv, which at bs=1 is a spin waiting on
the pipeline rather than work. Subtracting it leaves real compute of 7.72 / 9.48 / 8.46
/ 9.66 = 35.3 ms per token, against a 63.7 ms measured step: the remaining ~28 ms is
pipeline bubble, inherent to PP=4 at batch 1.

That gives the honest ceiling for single-stream decode: 1000/35.3 = 28.3 tok/s even
with the bubble removed entirely, so 50 tok/s needs compute at or below 20 ms/token.
Rank the compute families per layer to see what could plausibly deliver that, and break
the elementwise family into individual kernels with their launch geometry so the top
sites can be attributed.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
LAY = 11
fams = defaultdict(lambda: [0.0, 0])
tot_comp = 0.0
for pp in range(4):
    f = glob.glob(d + "/*TP-0-PP-%d.trace.json.gz" % pp)
    fe = glob.glob(d + "/*TP-0-PP-%d-EXTEND*" % pp)
    ev = json.load(gzip.open(max(f)))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    steps = max(1, sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]) / 22.0)
    # EXTEND is a subset of the plain file; drop it to keep decode only
    ext_ts = set()
    if fe:
        ev2 = json.load(gzip.open(max(fe)))["traceEvents"]
        ext_ts = {(e["ts"], e["dur"]) for e in ev2 if e.get("cat") == "kernel"}
    for e in ks:
        if (e["ts"], e["dur"]) in ext_ts:
            continue
        n = e["name"]
        if "SendRecv" in n or "AllGather" in n or "all_reduce" in n:
            continue  # communication / spin, not compute
        key = ("attn" if ("sparse" in n or "headshared" in n) else
               "expert w4a16" if "w4a16" in n else
               "w8a16 gemv" if "w8a16" in n else
               "dequant fp8" if "dequant_block_fp8" in n else
               "gemm (cutlass/turing/cublas)" if ("cutlass" in n or "turing_fp16" in n
                                                  or "gemvx" in n or "volta_sgemm" in n) else
               "hc/sinkhorn" if ("hc_" in n or "sinkhorn" in n) else
               "topk/indexer" if ("topk" in n.lower() or "index_" in n) else
               "elementwise/reduce" if ("elementwise" in n or "reduce_kernel" in n
                                        or "fill" in n or "triton_" in n) else
               "other")
        fams[key][0] += e["dur"] / 1e3 / steps
        fams[key][1] += 1
        tot_comp += e["dur"] / 1e3 / steps

print("compute (no comm) summed over the 4 PP stages: %.2f ms/token" % tot_comp)
print("=> single-stream ceiling with zero bubble: %.1f tok/s\n" % (1000.0 / tot_comp))
print("%-32s %9s %7s %9s %9s" % ("family", "ms/token", "share", "calls/tok", "us/call"))
for k, (ms, n) in sorted(fams.items(), key=lambda x: -x[1][0]):
    print("%-32s %9.2f %6.1f%% %9d %9.1f" % (k, ms, 100 * ms / tot_comp, n, ms * 1e3 / n))

# elementwise detail
print("\ntop elementwise kernels (all 4 stages, per token):")
det = defaultdict(lambda: [0.0, 0])
for pp in range(4):
    f = glob.glob(d + "/*TP-0-PP-%d.trace.json.gz" % pp)
    fe = glob.glob(d + "/*TP-0-PP-%d-EXTEND*" % pp)
    ev = json.load(gzip.open(max(f)))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    steps = max(1, sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]) / 22.0)
    ext_ts = set()
    if fe:
        ev2 = json.load(gzip.open(max(fe)))["traceEvents"]
        ext_ts = {(e["ts"], e["dur"]) for e in ev2 if e.get("cat") == "kernel"}
    for e in ks:
        if (e["ts"], e["dur"]) in ext_ts:
            continue
        n = e["name"]
        if not ("elementwise" in n or "reduce_kernel" in n or "fill" in n or "triton_" in n):
            continue
        det[(n[:64], tuple(e["args"].get("grid", [])))][0] += e["dur"] / 1e3 / steps
        det[(n[:64], tuple(e["args"].get("grid", [])))][1] += 1
for (n, g), (ms, c) in sorted(det.items(), key=lambda x: -x[1][0])[:12]:
    print("  %6.3f ms/tok %6d calls grid=%-14s %s" % (ms, c, str(g)[:14], n[:56]))
