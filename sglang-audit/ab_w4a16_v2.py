"""W4A16 expert GEMM v2: multi-warp blocks + prefetch ring, vs the shipped kernel.

Objective item (2) asks for the expert kernel to move from ~112 GB/s of weight
traffic toward 500. The shipped repacked kernel runs at 87-116 GB/s, and the
reason is visible in the profile: one warp per block, SM75 caps resident blocks
at 16 per SM, so an SM runs 16 of its 32 warps, and each keeps a single 128-byte
weight load in flight. That is ~2 KB outstanding per SM, and Little's law caps a
latency-bound kernel at in-flight-bytes / latency, which for 2 KB over 68 SMs at
600 ns is about 232 GB/s. The kernel is at half of even its own ceiling because
the scale byte is loaded inside the loop with no prefetch, so every iteration
pays a dependent load.

v2 changes only the launch shape and the load schedule: WARPS warps per block
(each owning a different n tile, so the block-count cap buys WARPS times the
resident warps) and a DEPTH-deep prefetch ring for both the weight word and the
scale byte. The mma sequence per 16x8 tile is untouched, so results must be
bit-identical -- which is what the correctness pass here checks against the
shipped kernel, not against a tolerance.

Run: python ab_w4a16_v2.py
"""
import os
import sys
import time
import types

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

import torch

from sglang.kernels.ops.moe.mxfp4_w4a16_kernels import (  # noqa: E402
    get_ptx_module,
    get_ptx_v2_module,
    mxfp4_w4a16_gemm_ptx,
    mxfp4_w4a16_gemm_ptx_v2,
    repack_mxfp4_for_ptx,
)

torch.manual_seed(0)
dev = "cuda:0"
E, HIDDEN, INTER, TOPK, BLOCK_M = 256, 4096, 2048, 6, 16

assert get_ptx_module() is not None, "shipped repacked kernel unavailable"
assert get_ptx_v2_module() is not None, "v2 module failed to build"


def bench(fn, iters=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def build(n_act, n_rank):
    """n_act experts, one 16-row block each; weights sharded across n_rank ranks."""
    num_slots = n_act * BLOCK_M
    ids = torch.full((num_slots,), TOPK, dtype=torch.int32, device=dev)
    for e in range(n_act):
        ids[e * BLOCK_M] = e % TOPK
    eids = torch.arange(n_act, dtype=torch.int32, device=dev)
    num_valid = torch.full((1,), num_slots, dtype=torch.int32, device=dev)

    n13 = 2 * INTER // n_rank
    k13 = HIDDEN
    n2 = HIDDEN // n_rank
    k2 = INTER

    a1 = (torch.randn(TOPK, k13, device=dev) * 0.05).half()
    w13 = torch.randint(-128, 127, (n_act, n13, k13 // 2), dtype=torch.int8, device=dev)
    s13 = torch.randint(118, 127, (n_act, n13, k13 // 32), dtype=torch.uint8, device=dev)
    out1 = torch.zeros(num_slots, n13, dtype=torch.float16, device=dev)

    a2 = (torch.randn(num_slots, k2, device=dev) * 0.05).half()
    w2 = torch.randint(-128, 127, (n_act, n2, k2 // 2), dtype=torch.int8, device=dev)
    s2 = torch.randint(118, 127, (n_act, n2, k2 // 32), dtype=torch.uint8, device=dev)
    out2 = torch.zeros(num_slots, n2, dtype=torch.float16, device=dev)

    slot_ids = torch.arange(num_slots, dtype=torch.int32, device=dev)
    # The repacked kernel consumes the permuted layout the loader produces.
    r13 = repack_mxfp4_for_ptx(w13)
    r2 = repack_mxfp4_for_ptx(w2)
    return dict(a1=a1, w13=r13, s13=s13, out1=out1, a2=a2, w2=r2, s2=s2, out2=out2,
                ids=ids, slot_ids=slot_ids, eids=eids, num_valid=num_valid,
                bytes13=w13.numel(), bytes2=w2.numel(), n_act=n_act, n_rank=n_rank)


CFGS = [(1, 1), (1, 4), (1, 8), (2, 4), (2, 8), (4, 1), (4, 4), (4, 8), (8, 4), (8, 8)]

print("correctness: v2 vs the shipped repacked kernel (must be bit-identical)")
bad = []
for (n_rank, n_act) in ((2, 6), (2, 48), (1, 6)):
    d = build(n_act, n_rank)
    num_slots = n_act * BLOCK_M
    ref1 = mxfp4_w4a16_gemm_ptx(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"],
                                torch.zeros_like(d["out1"]),
                                sentinel=TOPK, num_valid=d["num_valid"]).clone()
    ref2 = mxfp4_w4a16_gemm_ptx(d["a2"], d["w2"], d["s2"], d["slot_ids"], d["eids"],
                                torch.zeros_like(d["out2"]),
                                sentinel=num_slots, num_valid=d["num_valid"]).clone()
    for (w, dep) in CFGS:
        o1 = torch.zeros_like(d["out1"])
        o2 = torch.zeros_like(d["out2"])
        try:
            mxfp4_w4a16_gemm_ptx_v2(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"],
                                    o1, sentinel=TOPK, num_valid=d["num_valid"],
                                    warps=w, depth=dep)
            mxfp4_w4a16_gemm_ptx_v2(d["a2"], d["w2"], d["s2"], d["slot_ids"], d["eids"],
                                    o2, sentinel=num_slots, num_valid=d["num_valid"],
                                    warps=w, depth=dep)
            torch.cuda.synchronize()
        except Exception as e:
            print("  w%d d%d  BUILD/RUN FAIL: %s" % (w, dep, str(e).splitlines()[0][:56]))
            bad.append((n_rank, n_act, w, dep))
            continue
        eq1 = torch.equal(o1, ref1)
        eq2 = torch.equal(o2, ref2)
        if not (eq1 and eq2):
            m1 = (o1.float() - ref1.float()).abs().max().item()
            m2 = (o2.float() - ref2.float()).abs().max().item()
            print("  w%d d%d  TP%d n_act=%d  MISMATCH gemm1 %s (%.3g) gemm2 %s (%.3g)"
                  % (w, dep, n_rank, n_act, eq1, m1, eq2, m2))
            bad.append((n_rank, n_act, w, dep))
print("  all configs bit-identical" if not bad else "  FAILURES: %s" % bad[:6])

print("\nspeed at the decode shape (TP2 shard), median of 5 interleaved rounds:")
d = build(6, 2)
num_slots = 6 * BLOCK_M


def run_shipped():
    mxfp4_w4a16_gemm_ptx(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"], d["out1"],
                         sentinel=TOPK, num_valid=d["num_valid"])
    mxfp4_w4a16_gemm_ptx(d["a2"], d["w2"], d["s2"], d["slot_ids"], d["eids"], d["out2"],
                         sentinel=num_slots, num_valid=d["num_valid"])


def run_v2(w, dep):
    def f():
        mxfp4_w4a16_gemm_ptx_v2(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"],
                                d["out1"], sentinel=TOPK, num_valid=d["num_valid"],
                                warps=w, depth=dep)
        mxfp4_w4a16_gemm_ptx_v2(d["a2"], d["w2"], d["s2"], d["slot_ids"], d["eids"],
                                d["out2"], sentinel=num_slots, num_valid=d["num_valid"],
                                warps=w, depth=dep)
    return f


wb = (d["bytes13"] + d["bytes2"]) / 1e6
res = {"shipped": []}
for (w, dep) in CFGS:
    res["w%d d%d" % (w, dep)] = []
ROUNDS = 5
for r in range(ROUNDS):
    res["shipped"].append(bench(run_shipped, iters=20, warmup=5))
    for (w, dep) in CFGS:
        res["w%d d%d" % (w, dep)].append(bench(run_v2(w, dep), iters=20, warmup=5))

base = sorted(res["shipped"])[ROUNDS // 2]
print("  %-12s %8s %10s %9s" % ("config", "ms", "GB/s", "vs shipped"))
for name, ts in res.items():
    ts = sorted(ts)
    med = ts[ROUNDS // 2]
    print("  %-12s %8.3f %10.0f %8.3fx  spread %.0f%%"
          % (name, med, wb / med, base / med, 100 * (ts[-1] - ts[0]) / med))
