"""Can a tuned head-shared kernel beat the per-head kernel at bs=1?

Production decode is bs=1 and runs the per-head kernel at 0.273 ms of device time
(topk=512). The head-shared kernel costs 0.710 ms there, which is why the gate is at
bs>=2. But the block-overlap measurement shows head-shared is FLAT from 4 to 32 blocks
(0.710 -> 0.710 ms for 8x the work), so at bs=1 its 4 blocks leave most of the machine
idle and its 32 serial tile iterations are pure exposed latency.

Its tile shape and launch config are env-tunable (SGLANG_SM75_HS_NCOL / _WARPS /
_STAGES) except BLOCK_T, which is hardcoded at 16. A wider BLOCK_T means fewer serial
iterations; more stages means more gathers in flight. Sweep them and compare against
the per-head kernel's 0.273 ms: if a config beats it, the gate can drop to 1 and decode
attention improves without writing a new kernel. If nothing beats it, the only route is
an in-kernel topk split plus a register reduction, which is a rewrite.
"""
import importlib, os, sys
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
from torch.profiler import ProfilerActivity, profile
import ab_headshared_scale as A

H, D, TOPK = 64, 512, 512
q, kc, idx = A.build(1, H, D, TOPK, 236800)
qb = q.expand(1, 1, H, D).contiguous()
full = idx.reshape(1, -1).contiguous()


def dev_ms(M, fn, iters=25):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in pr.key_averages()
               if e.device_time_total and ("sparse" in e.key or "headshared" in e.key)
               ) / 1e3 / iters


import sglang.kernels.ops.attention.flash_mla_sm120_triton as M
base = dev_ms(M, lambda: M._run_triton_sparse_decode(qb, kc, full, None, 0.088))
print("per-head (production at bs=1): %.4f ms  <- target to beat" % base)

results = []
for ncol in (64, 128, 256):
    for warps in (4, 8):
        for stages in (1, 2, 3, 4):
            os.environ["SGLANG_SM75_HS_NCOL"] = str(ncol)
            os.environ["SGLANG_SM75_HS_WARPS"] = str(warps)
            os.environ["SGLANG_SM75_HS_STAGES"] = str(stages)
            importlib.reload(M)
            try:
                t = dev_ms(M, lambda: M._run_headshared_sparse_decode(qb, kc, full, None, 0.088))
                results.append((t, ncol, warps, stages))
                print("  NCOL=%-4d warps=%d stages=%d  %.4f ms  (%.2fx vs per-head)"
                      % (ncol, warps, stages, t, base / t))
            except Exception as e:
                print("  NCOL=%-4d warps=%d stages=%d  FAIL %s" % (ncol, warps, stages, str(e)[:50]))
results.sort()
print("\nbest: NCOL=%d warps=%d stages=%d at %.4f ms (%.2fx vs per-head)"
      % (results[0][1], results[0][2], results[0][3], results[0][0], base / results[0][0]))
print("production per-head remains better" if results[0][0] > base
      else "HEADSHARED WINS at bs=1 -> lower the gate to 1")
