"""Is the W4A16 expert kernel the prefill bottleneck, and what would a
tensor-core path cost instead?

In the EXTEND trace w4a16_ptx_kernel is 631.2 of 1270.2 ms (49.7%) -- the largest
single item now that attention is fixed. The kernel is SIMT: it unpacks fp4
nibbles and multiplies on CUDA cores, which is right for decode (M=1 per expert,
bandwidth-bound) but leaves SM75's fp16 tensor cores unused once M grows.

Prefill shape: chunk 512 tokens x topk 6 = 3072 token-expert pairs over 256
experts (128 per TP2 rank), so ~24 rows per expert = 2 blocks of 16.

Compares three paths at that shape:
  1. the shipped repacked W4A16 kernel
  2. dequantize the expert to fp16, then torch.mm (cublas tensor cores)
  3. the theoretical floor from weight bytes at 616 GB/s
"""
import json
import pathlib
import sys
import time

import torch

repo = "/data/nvme/sglang-codex/sglang"
sys.path.insert(0, repo + "/python")

from sglang.kernels.ops.moe.mxfp4_w4a16_kernels import (  # noqa: E402
    mxfp4_w4a16_gemm,
    mxfp4_w4a16_gemm_ptx,
    repack_mxfp4_for_ptx,
)

torch.manual_seed(0)
dev = "cuda:0"
cfg = json.loads(pathlib.Path(
    "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731/config.json").read_text())
HIDDEN = cfg["hidden_size"]
INTER = cfg["moe_intermediate_size"]
TOPK, BLOCK_M, N_RANK = 6, 16, 2


def bench(fn, iters=15, warmup=4):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def build(n_experts, blocks_per_expert):
    """Grouped-GEMM inputs for n_experts, each with blocks_per_expert 16-row blocks."""
    n_blocks = n_experts * blocks_per_expert
    num_slots = n_blocks * BLOCK_M
    ids = torch.full((num_slots,), TOPK, dtype=torch.int32, device=dev)
    # fill each block's first row with a real token id, rest are pad slots
    for blk in range(n_blocks):
        for r in range(BLOCK_M):
            ids[blk * BLOCK_M + r] = (blk * BLOCK_M + r) % TOPK
    eids = torch.arange(n_experts, dtype=torch.int32, device=dev).repeat_interleave(
        blocks_per_expert)
    num_valid = torch.full((1,), num_slots, dtype=torch.int32, device=dev)

    n13, k13 = 2 * INTER // N_RANK, HIDDEN
    n2, k2 = HIDDEN // N_RANK, INTER

    a1 = (torch.randn(num_slots, k13, device=dev) * 0.05).half()
    w13 = torch.randint(-128, 127, (n_experts, n13, k13 // 2), dtype=torch.int8,
                        device=dev)
    s13 = torch.randint(118, 127, (n_experts, n13, k13 // 32), dtype=torch.uint8,
                        device=dev)
    out1 = torch.zeros(num_slots, n13, dtype=torch.float16, device=dev)

    a2 = (torch.randn(num_slots, k2, device=dev) * 0.05).half()
    w2 = torch.randint(-128, 127, (n_experts, n2, k2 // 2), dtype=torch.int8,
                       device=dev)
    s2 = torch.randint(118, 127, (n_experts, n2, k2 // 32), dtype=torch.uint8,
                       device=dev)
    out2 = torch.zeros(num_slots, n2, dtype=torch.float16, device=dev)

    slot_ids = torch.arange(num_slots, dtype=torch.int32, device=dev)
    return dict(a1=a1, w13=w13, s13=s13, out1=out1, a2=a2, w2=w2, s2=s2,
                out2=out2, ids=ids, slot_ids=slot_ids, eids=eids,
                num_valid=num_valid, n_experts=n_experts,
                bytes13=w13.numel(), bytes2=w2.numel(),
                n13=n13, k13=k13, n2=n2, k2=k2)


def run_case(label, n_experts, blocks):
    d = build(n_experts, blocks)
    sentinel = TOPK
    w13r = repack_mxfp4_for_ptx(d["w13"].view(torch.uint8))
    w2r = repack_mxfp4_for_ptx(d["w2"].view(torch.uint8))

    def ptx1():
        mxfp4_w4a16_gemm_ptx(d["a1"], w13r, d["s13"], d["ids"], d["eids"],
                             d["out1"], sentinel, num_valid=d["num_valid"])

    def ptx2():
        mxfp4_w4a16_gemm_ptx(d["a2"], w2r, d["s2"], d["slot_ids"], d["eids"],
                             d["out2"], d["out2"].shape[0],
                             num_valid=d["num_valid"])

    t1, t2 = bench(ptx1), bench(ptx2)
    tot = t1 + t2
    wbytes = d["bytes13"] + d["bytes2"]

    # tensor-core alternative: one fp16 copy of each expert + cublas over all rows
    w13_16 = (torch.randn(d["n13"], d["k13"], device=dev) * 0.02).half()
    w2_16 = (torch.randn(d["n2"], d["k2"], device=dev) * 0.02).half()
    rows = d["a1"].shape[0]

    def cublas1():
        return torch.mm(d["a1"], w13_16.t())

    def cublas2():
        return torch.mm(d["a2"], w2_16.t())

    c1, c2 = bench(cublas1), bench(cublas2)
    ctot = c1 + c2
    floor = wbytes / 616e9 * 1e3
    print(f"{label}  experts={n_experts:3d} x{blocks} blk  rows/expert={blocks*BLOCK_M}")
    print(f"   w4a16 repacked : gemm1 {t1:7.3f}  gemm2 {t2:7.3f}  total {tot:7.3f} ms"
          f"   weight-bw {wbytes/tot/1e6:5.0f} GB/s   ({tot/floor:5.1f}x BW floor)")
    print(f"   cublas fp16    : gemm1 {c1:7.3f}  gemm2 {c2:7.3f}  total {ctot:7.3f} ms"
          f"   -> {tot / ctot:5.2f}x vs w4a16")
    print(f"   BW floor for these weight bytes: {floor:.3f} ms")

    # The Triton grouped GEMM already exists and lowers to tl.dot, so it uses
    # SM75's fp16 tensor cores. If it is near cublas at this shape, dispatching
    # to it above some M is a routing change, not new kernel work.
    try:
        def tri1():
            mxfp4_w4a16_gemm(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"],
                             d["out1"], sentinel=sentinel, num_valid=d["num_valid"])

        def tri2():
            mxfp4_w4a16_gemm(d["a2"], d["w2"], d["s2"], d["slot_ids"], d["eids"],
                             d["out2"], sentinel=d["out2"].shape[0],
                             num_valid=d["num_valid"])

        u1, u2 = bench(tri1), bench(tri2)
        utot = u1 + u2
        print(f"   triton tl.dot  : gemm1 {u1:7.3f}  gemm2 {u2:7.3f}  total {utot:7.3f} ms"
              f"   -> {tot / utot:5.2f}x vs w4a16, {utot / ctot:5.2f}x of cublas")
    except Exception as e:
        print(f"   triton tl.dot  : FAILED {type(e).__name__}: {str(e)[:80]}")
    del w13r, w2r
    print()


print(f"HIDDEN={HIDDEN} INTER={INTER} TP{N_RANK}  (per-rank expert weights)")
run_case("decode  ", 6, 1)
run_case("prefill ", 128, 2)
run_case("prefill ", 128, 1)
