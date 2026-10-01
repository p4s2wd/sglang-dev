"""Is the W4A16 kernel ALU-bound or memory-bound? The decisive measurement.

Route (2) wants 112 -> 500 GB/s. Before rewriting anything, find out what the
115 GB/s ceiling actually is. Two candidates, opposite fixes:

  * memory/access-bound: the per-lane stride and the 16-warp occupancy cap mean
    the loads cannot go faster. Then no kernel change reaches 500 and the target
    is wrong for this layout.
  * ALU-bound on nibble dequant: each weight byte costs a shift, two 4-bit
    decodes and a multiply before the mma can use it. Then the fix is fewer ops
    per byte (a PRMT/LUT-based dequant), not more memory parallelism.

The read-only probe streams exactly the same bytes with the same addresses and
the same launch shape but does no dequant and no mma. Its rate is the access
pattern's ceiling. If it is ~500 the shipped kernel is ALU-bound; if it is ~115
the ceiling is memory and 500 is unreachable.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
from sglang.kernels.ops.moe.mxfp4_w4a16_kernels import (
    get_ptx_module, get_ptx_v3_module, mxfp4_w4a16_gemm_ptx,
    mxfp4_w4a16_gemm_ptx_v3, repack_mxfp4_for_ptx)

torch.manual_seed(0); dev = "cuda:0"
HIDDEN, INTER, TOPK, BLOCK_M = 4096, 2048, 6, 16
assert get_ptx_module() is not None and get_ptx_v3_module() is not None


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def build(n_act, n_rank):
    num_slots = n_act * BLOCK_M
    ids = torch.full((num_slots,), TOPK, dtype=torch.int32, device=dev)
    for e in range(n_act): ids[e * BLOCK_M] = e % TOPK
    eids = torch.arange(n_act, dtype=torch.int32, device=dev)
    nv = torch.full((1,), num_slots, dtype=torch.int32, device=dev)
    n13, k13 = 2 * INTER // n_rank, HIDDEN
    n2, k2 = HIDDEN // n_rank, INTER
    a1 = (torch.randn(TOPK, k13, device=dev) * .05).half()
    w13 = torch.randint(-128, 127, (n_act, n13, k13 // 2), dtype=torch.int8, device=dev)
    s13 = torch.randint(118, 127, (n_act, n13, k13 // 32), dtype=torch.uint8, device=dev)
    o1 = torch.zeros(num_slots, n13, dtype=torch.float16, device=dev)
    a2 = (torch.randn(num_slots, k2, device=dev) * .05).half()
    w2 = torch.randint(-128, 127, (n_act, n2, k2 // 2), dtype=torch.int8, device=dev)
    s2 = torch.randint(118, 127, (n_act, n2, k2 // 32), dtype=torch.uint8, device=dev)
    o2 = torch.zeros(num_slots, n2, dtype=torch.float16, device=dev)
    sid = torch.arange(num_slots, dtype=torch.int32, device=dev)
    return dict(a1=a1, w13=repack_mxfp4_for_ptx(w13), s13=s13, o1=o1,
                a2=a2, w2=repack_mxfp4_for_ptx(w2), s2=s2, o2=o2,
                ids=ids, sid=sid, eids=eids, nv=nv,
                b13=w13.numel(), b2=w2.numel(), ns=num_slots)


for n_act in (6, 48):
    d = build(n_act, 2)
    wb = (d["b13"] + d["b2"]) / 1e6
    def shipped():
        mxfp4_w4a16_gemm_ptx(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"], d["o1"],
                             sentinel=TOPK, num_valid=d["nv"])
        mxfp4_w4a16_gemm_ptx(d["a2"], d["w2"], d["s2"], d["sid"], d["eids"], d["o2"],
                             sentinel=d["ns"], num_valid=d["nv"])
    def scalepf():
        mxfp4_w4a16_gemm_ptx_v2(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"], d["o1"],
                                sentinel=TOPK, num_valid=d["nv"], warps=0, depth=1)
        mxfp4_w4a16_gemm_ptx_v2(d["a2"], d["w2"], d["s2"], d["sid"], d["eids"], d["o2"],
                                sentinel=d["ns"], num_valid=d["nv"], warps=0, depth=1)
    def mk(cfg):
        def f():
            mxfp4_w4a16_gemm_ptx_v3(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"],
                                    d["o1"], sentinel=TOPK, num_valid=d["nv"], cfg=cfg)
            mxfp4_w4a16_gemm_ptx_v3(d["a2"], d["w2"], d["s2"], d["sid"], d["eids"],
                                    d["o2"], sentinel=d["ns"], num_valid=d["nv"], cfg=cfg)
        return f
    def v3full():
        mxfp4_w4a16_gemm_ptx_v3(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"], d["o1"],
                                sentinel=TOPK, num_valid=d["nv"], cfg=3)
        mxfp4_w4a16_gemm_ptx_v3(d["a2"], d["w2"], d["s2"], d["sid"], d["eids"], d["o2"],
                                sentinel=d["ns"], num_valid=d["nv"], cfg=3)
    R = 5
    res = {"shipped (full)": [], "v3 NT4 (committed)": [],
           "read-only (no compute)": [],
           "v3 NT4 no-activation": [], "v3 NT4 no-dequant": []}
    fns = {"shipped (full)": shipped, "v3 NT4 (committed)": v3full,
           "read-only (no compute)": mk(-1),
           "v3 NT4 no-activation": mk(20), "v3 NT4 no-dequant": mk(21)}
    for r in range(R):
        for name, f in fns.items():
            res[name].append(bench(f))
    print("n_act=%d  weight bytes %.1f MB" % (n_act, wb))
    for name, ts in res.items():
        ts = sorted(ts); med = ts[R // 2]
        print("  %-18s %7.3f ms  %6.0f GB/s" % (name, med, wb / med))
    print()
