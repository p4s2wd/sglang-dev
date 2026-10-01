"""W4A16 v3: activation reuse + PRMT dequant, measured against the shipped kernel.

The attribution that motivated this (probe_bw.py) is:

    read-only, no compute                  521-575 GB/s   memory ceiling
    activation loads replaced by a const   169-174 GB/s   -> loads cost ~34%
    nibble dequant replaced by a move      146-150 GB/s   -> dequant costs ~21%
    shipped                                115-122 GB/s

v3 attacks both: NT consecutive n tiles per warp so the m16n8k8 A fragments are
loaded once per k step instead of once per tile, and a PRMT byte lookup in place
of the arithmetic nibble decode. cfg sweeps them independently:

    0 (NT=1, arith)   1 (NT=1, PRMT)   2 (NT=2, PRMT)   3 (NT=4, PRMT)
    4 (NT=2, arith)   5 (NT=4, arith)  6 (NT=8, PRMT)

Correctness is bit-equality with the shipped kernel, not a tolerance. The one
permitted difference is negative zero: e2m1 nibble 8 is -0.0, the arithmetic
decoder discards its sign and yields +0.0 while PRMT keeps it, and those are
equal under every fp16 operation the mma performs. So the comparison adds 0.0 to
both sides first, which normalises -0.0 to +0.0 without hiding a real error.

Run: python ab_w4a16_v3.py
"""
import sys
import time

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

import torch

from sglang.kernels.ops.moe.mxfp4_w4a16_kernels import (  # noqa: E402
    W4A16_V3_CFGS,
    get_ptx_module,
    get_ptx_v3_module,
    mxfp4_w4a16_gemm_ptx,
    mxfp4_w4a16_gemm_ptx_v3,
    repack_mxfp4_for_ptx,
)

torch.manual_seed(0)
dev = "cuda:0"
E, HIDDEN, INTER, TOPK, BLOCK_M = 256, 4096, 2048, 6, 16

assert get_ptx_module() is not None, "shipped kernel unavailable"
assert get_ptx_v3_module() is not None, "v3 module failed to build"


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
    num_slots = n_act * BLOCK_M
    ids = torch.full((num_slots,), TOPK, dtype=torch.int32, device=dev)
    for e in range(n_act):
        ids[e * BLOCK_M] = e % TOPK
    eids = torch.arange(n_act, dtype=torch.int32, device=dev)
    nv = torch.full((1,), num_slots, dtype=torch.int32, device=dev)

    n13, k13 = 2 * INTER // n_rank, HIDDEN
    n2, k2 = HIDDEN // n_rank, INTER

    a1 = (torch.randn(TOPK, k13, device=dev) * 0.05).half()
    w13 = torch.randint(-128, 127, (n_act, n13, k13 // 2), dtype=torch.int8, device=dev)
    s13 = torch.randint(118, 127, (n_act, n13, k13 // 32), dtype=torch.uint8, device=dev)
    o1 = torch.zeros(num_slots, n13, dtype=torch.float16, device=dev)

    a2 = (torch.randn(num_slots, k2, device=dev) * 0.05).half()
    w2 = torch.randint(-128, 127, (n_act, n2, k2 // 2), dtype=torch.int8, device=dev)
    s2 = torch.randint(118, 127, (n_act, n2, k2 // 32), dtype=torch.uint8, device=dev)
    o2 = torch.zeros(num_slots, n2, dtype=torch.float16, device=dev)

    sid = torch.arange(num_slots, dtype=torch.int32, device=dev)
    return dict(a1=a1, w13=repack_mxfp4_for_ptx(w13), s13=s13, o1=o1,
                a2=a2, w2=repack_mxfp4_for_ptx(w2), s2=s2, o2=o2,
                ids=ids, sid=sid, eids=eids, nv=nv, ns=num_slots,
                b13=w13.numel(), b2=w2.numel())


def norm(x):
    # -0.0 and +0.0 are equal in every fp16 op the mma performs; adding zero
    # folds them together so bit-equality is a meaningful check.
    return (x + 0.0).view(torch.int16)


CFGS = [c for c in sorted(W4A16_V3_CFGS) if c in (0, 1, 2, 3, 10, 11, 12, 13)]

print("correctness: v3 vs shipped, bit-equal after normalising signed zero")
bad = []
for (n_rank, n_act) in ((2, 6), (2, 48), (1, 6)):
    d = build(n_act, n_rank)
    ref1 = mxfp4_w4a16_gemm_ptx(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"],
                                torch.zeros_like(d["o1"]),
                                sentinel=TOPK, num_valid=d["nv"]).clone()
    ref2 = mxfp4_w4a16_gemm_ptx(d["a2"], d["w2"], d["s2"], d["sid"], d["eids"],
                                torch.zeros_like(d["o2"]),
                                sentinel=d["ns"], num_valid=d["nv"]).clone()
    r1n, r2n = norm(ref1), norm(ref2)
    for cfg in CFGS:
        o1 = torch.zeros_like(d["o1"])
        o2 = torch.zeros_like(d["o2"])
        try:
            mxfp4_w4a16_gemm_ptx_v3(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"],
                                    o1, sentinel=TOPK, num_valid=d["nv"], cfg=cfg)
            mxfp4_w4a16_gemm_ptx_v3(d["a2"], d["w2"], d["s2"], d["sid"], d["eids"],
                                    o2, sentinel=d["ns"], num_valid=d["nv"], cfg=cfg)
            torch.cuda.synchronize()
        except Exception as e:
            print("  cfg%d TP%d n_act=%d  FAIL %s"
                  % (cfg, n_rank, n_act, str(e).splitlines()[0][:52]))
            bad.append((cfg, n_rank, n_act))
            continue
        ok1 = torch.equal(norm(o1), r1n)
        ok2 = torch.equal(norm(o2), r2n)
        if not (ok1 and ok2):
            m = max((o1.float() - ref1.float()).abs().max().item(),
                    (o2.float() - ref2.float()).abs().max().item())
            print("  cfg%d TP%d n_act=%d  MISMATCH gemm1=%s gemm2=%s maxdiff=%.4g"
                  % (cfg, n_rank, n_act, ok1, ok2, m))
            bad.append((cfg, n_rank, n_act))
print("  all configs bit-identical to shipped" if not bad else "  FAILURES: %s" % bad[:8])

print("\nspeed, median of 5 interleaved rounds:")
for n_act in (6, 48):
    d = build(n_act, 2)
    wb = (d["b13"] + d["b2"]) / 1e6

    def shipped():
        mxfp4_w4a16_gemm_ptx(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"], d["o1"],
                             sentinel=TOPK, num_valid=d["nv"])
        mxfp4_w4a16_gemm_ptx(d["a2"], d["w2"], d["s2"], d["sid"], d["eids"], d["o2"],
                             sentinel=d["ns"], num_valid=d["nv"])

    def mk(cfg):
        def f():
            mxfp4_w4a16_gemm_ptx_v3(d["a1"], d["w13"], d["s13"], d["ids"], d["eids"],
                                    d["o1"], sentinel=TOPK, num_valid=d["nv"], cfg=cfg)
            mxfp4_w4a16_gemm_ptx_v3(d["a2"], d["w2"], d["s2"], d["sid"], d["eids"],
                                    d["o2"], sentinel=d["ns"], num_valid=d["nv"], cfg=cfg)
        return f

    def label(cfg):
        c = W4A16_V3_CFGS[cfg]
        s = "cfg%d NT%d %s" % (cfg, c[0], "prmt" if c[1] else "arith")
        if len(c) > 2:
            s += " d%d" % c[2]
        return s
    res = {"shipped": []}
    fns = {"shipped": shipped}
    for cfg in CFGS:
        fns[label(cfg)] = mk(cfg)
    R = 5
    for name in fns:
        res[name] = []
    for r in range(R):
        for name, f in fns.items():
            res[name].append(bench(f, iters=20, warmup=5))
    base = sorted(res["shipped"])[R // 2]
    print("  n_act=%d  %.1f MB weights" % (n_act, wb))
    for name in fns:
        ts = sorted(res[name])
        med = ts[R // 2]
        print("    %-20s %7.3f ms  %5.0f GB/s  %6.3fx  spread %.0f%%"
              % (name, med, wb / med, base / med, 100 * (ts[-1] - ts[0]) / med))
