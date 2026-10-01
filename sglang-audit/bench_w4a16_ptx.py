"""Baseline for the kernel decode actually uses: mxfp4_w4a16_gemm_ptx_direct.

bench_w4a16.py measures the Triton kernel (8-11 GB/s). Production picks the PTX
mma.sync kernel when the JIT module loads (fp8.py: `use_ptx`), which the server
profile puts at ~112 GB/s of weight traffic -- 4.5x below what these cards reach
on a pure weight stream. This measures the kernel that is actually on the hot
path, at the shapes and slot layout a decode step produces.

Decode layout: topk=6 slots spread over 6 experts, block_m=16, so each expert
occupies one 16-row block holding one real row. TP2 shards the intermediate
dimension, so both the full and the per-rank shapes are reported.
"""
import os as _os
import pathlib as _pb
import time

import torch


def _find_repo():
    r = _os.environ.get("SGLANG_REPO")
    if r:
        return r
    d = _pb.Path(__file__).resolve().parent
    for _ in range(8):
        for cand in (d, d / "sglang"):
            if (cand / "python" / "sglang").is_dir():
                return str(cand)
        d = d.parent
    raise RuntimeError("set SGLANG_REPO")


_PY = _find_repo() + "/python"

from sglang.kernels.ops.moe.mxfp4_w4a16_kernels import (  # noqa: E402
    get_ptx_direct_module,
    mxfp4_w4a16_gemm_ptx_direct,
)

torch.cuda.set_device(0)
dev = "cuda:0"
E, HIDDEN, INTER, TOPK, BLOCK_M = 256, 4096, 2048, 6, 16


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def build(n_act, n_rank):
    """n_act experts, each with one 16-row block; weights sharded by n_rank."""
    num_slots = n_act * BLOCK_M
    ids = torch.full((num_slots,), TOPK, dtype=torch.int32, device=dev)
    for e in range(n_act):
        ids[e * BLOCK_M] = e % TOPK
    eids = torch.arange(n_act, dtype=torch.int32, device=dev)
    num_valid = torch.full((1,), num_slots, dtype=torch.int32, device=dev)

    n13 = 2 * INTER // n_rank          # gate|up, sharded on the intermediate dim
    k13 = HIDDEN                       # gemm1 reduces over hidden
    n2 = HIDDEN // n_rank              # gemm2 output, sharded on hidden
    k2 = INTER                         # gemm2 reduces over the intermediate dim

    a1 = (torch.randn(TOPK, k13, device=dev) * 0.05).half()
    w13 = torch.randint(-128, 127, (n_act, n13, k13 // 2), dtype=torch.int8,
                        device=dev)
    s13 = torch.randint(118, 127, (n_act, n13, k13 // 32), dtype=torch.uint8,
                        device=dev)
    out1 = torch.zeros(num_slots, n13, dtype=torch.float16, device=dev)

    a2 = (torch.randn(num_slots, k2, device=dev) * 0.05).half()
    w2 = torch.randint(-128, 127, (n_act, n2, k2 // 2), dtype=torch.int8,
                       device=dev)
    s2 = torch.randint(118, 127, (n_act, n2, k2 // 32), dtype=torch.uint8,
                       device=dev)
    out2 = torch.zeros(num_slots, n2, dtype=torch.float16, device=dev)

    slot_ids = torch.arange(num_slots, dtype=torch.int32, device=dev)
    bytes13 = w13.numel()
    bytes2 = w2.numel()
    return (a1, w13, s13, out1, a2, w2, s2, out2, ids, slot_ids, eids,
            num_valid, bytes13, bytes2)


assert get_ptx_direct_module() is not None or True
if get_ptx_direct_module() is None:
    raise SystemExit("PTX direct module unavailable; production would use Triton")

for n_rank in (1, 2):
    for n_act in (6, 48):
        (a1, w13, s13, out1, a2, w2, s2, out2, ids, slot_ids, eids, num_valid,
         b13, b2) = build(n_act, n_rank)
        num_slots = n_act * BLOCK_M
        sentinel = TOPK
        # The wrapper takes sentinel positionally before num_valid; pass both by
        # name so the order cannot silently become a type error.
        t1 = bench(lambda: mxfp4_w4a16_gemm_ptx_direct(
            a1, w13, s13, ids, eids, out1,
            sentinel=sentinel, num_valid=num_valid))
        t2 = bench(lambda: mxfp4_w4a16_gemm_ptx_direct(
            a2, w2, s2, slot_ids, eids, out2,
            sentinel=num_slots, num_valid=num_valid))
        tot = t1 + t2
        wbytes = b13 + b2
        print(f"TP{n_rank+0} n_act={n_act:3d}  gemm1 {t1:6.3f}ms "
              f"({b13/t1/1e6:4.0f} GB/s)  gemm2 {t2:6.3f}ms "
              f"({b2/t2/1e6:4.0f} GB/s)  total {tot:6.3f}ms  "
              f"weight-bw {wbytes/tot/1e6:5.0f} GB/s")
        if n_rank == 2:
            # what the server profile implies: 43 layers, this cost per layer
            print(f"           -> x43 layers = {tot*43:6.1f} ms/token")

        # Same shapes through the repacked kernel, which reads one coalesced u32
        # per lane per k-step instead of four strided bytes. If this is a large
        # multiple of the direct number, repacking at load time (and dropping the
        # raw layout) is worth the loader surgery: steady-state memory is then the
        # same, because only one layer is raw at a time.
        from sglang.kernels.ops.moe.mxfp4_w4a16_kernels import (
            mxfp4_w4a16_gemm_ptx, repack_mxfp4_for_ptx)
        w13r = repack_mxfp4_for_ptx(w13.view(torch.uint8))
        w2r = repack_mxfp4_for_ptx(w2.view(torch.uint8))
        # The kernel now takes the raw UE8M0 byte, so the repacked path needs no
        # fp32 copy of the scales: that copy was 1.6 GiB/card and was the whole
        # memory objection to repacking.
        try:
            r1 = bench(lambda: mxfp4_w4a16_gemm_ptx(
                a1, w13r, s13, ids, eids, out1, sentinel, num_valid=num_valid))
            r2 = bench(lambda: mxfp4_w4a16_gemm_ptx(
                a2, w2r, s2, slot_ids, eids, out2, num_slots,
                num_valid=num_valid))
            rt = r1 + r2
            print(f"           repacked: gemm1 {r1:6.3f}ms gemm2 {r2:6.3f}ms "
                  f"total {rt:6.3f}ms  weight-bw {wbytes/rt/1e6:5.0f} GB/s  "
                  f"= {tot/rt:.2f}x direct")
        except Exception as e:
            print(f"           repacked: FAILED {type(e).__name__}: {str(e)[:90]}")
        del w13r, w2r
