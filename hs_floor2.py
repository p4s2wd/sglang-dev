"""Attention prefill: gather floor vs full kernel, using the faithful standalone.

hs_floor.py's hand-written gather kernel measured 0.24 ms for the same bytes the
production kernel reads in 19 ms. That 80x cannot be the dot products (the early
calls in the trace run the same FLOPs in 2 ms), so the floor probe must be
missing something the real kernel does. bh32_nohoist.py is a faithful copy of the
production kernel that already compiles, so its DEQ knob isolates the decode
stage exactly: DEQ=2 is the real path (byte gather + scale load + exp2 + LUT
gather), DEQ=1 drops only the LUT gather, DEQ=0 drops scale and LUT and is a pure
gather of the same addresses. If DEQ=0 is near 0.24 ms and DEQ=2 near 19 ms, the
gap is the decode stage and the fix is a cheaper decode. If DEQ=0 is also ~19 ms,
the gather itself is the wall and the floor probe was wrong.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import ab_headshared_scale as A
import bh32_nohoist as B
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

POOL = 236800
Bt, H, D, TOPK = 512, 32, 512, 512
q, kc, flat_idx = A.build(Bt, H, D, TOPK, POOL)
out = torch.zeros(Bt, H, D, dtype=torch.float16, device="cuda")
lse = torch.zeros(Bt, H, dtype=torch.float32, device="cuda")
print("pool %.0f MB, B=%d topk=%d" % (POOL * 584 / 1e6, Bt, TOPK))


def bench(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


mb = Bt * TOPK * 576 / 1e6
res = {}
res["production"] = bench(lambda: M._run_headshared_sparse_decode(
    q, kc, flat_idx, torch.full((Bt,), TOPK, dtype=torch.int32, device="cuda"), 0.088))
for deq, name in ((2, "standalone DEQ=2 (real)"), (1, "standalone DEQ=1 (no LUT)"),
                  (0, "standalone DEQ=0 (gather)")):
    res[name] = bench(lambda d=deq: B.run(q, kc, flat_idx, 0.088, out, lse,
                                          bh=16, ncol=64, warps=4, hoist=1,
                                          stages=2, deq=d))
print("\n%.0f MB useful bytes per call" % mb)
for name, ms in res.items():
    print("%-28s %7.2f ms  %6.1f GB/s" % (name, ms, 2 * mb / ms))
