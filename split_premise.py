"""Do head-shared attention blocks overlap? The premise for an in-kernel topk split.

Attention is now the largest decode compute family (12.77 ms/token over the 4 stages,
25.9%). Production runs the per-head kernel: grid (1,64), 255 registers per thread, so
exactly 1 block per SM and 8 of 32 warp slots used, and each block walks its topk tiles
SERIALLY. That is why it costs 0.281 ms per layer per step against a bandwidth floor of
0.038 ms -- latency-bound, not bandwidth-bound.

The fix would be a flash-decoding style split: add a grid dimension over topk chunks so
each block walks fewer tiles and the chunks run concurrently, merged with the existing
_merge_partial_attn. That only works if blocks actually overlap. The head-shared kernel
is the clean test because it is already flat-ish in batch: at bs=1 it runs 4 blocks and
at bs=16 it runs 64 blocks doing 16x the work. If 16x work costs ~1x time, blocks
overlap for free and a split will pay; if it costs ~4x (16x work / 4x more blocks), the
machine is already saturated and a split buys nothing.

Measure device-only with the profiler -- host wall time has a ~0.155 ms wrapper floor
that made an earlier version of this probe useless.
"""
import sys
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
from torch.profiler import ProfilerActivity, profile
import ab_headshared_scale as A
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

H, D, TOPK = 64, 512, 512
q, kc, idx = A.build(1, H, D, TOPK, 236800)
full = idx.reshape(1, -1).contiguous()


def dev_ms(fn, iters=30):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in pr.key_averages()
               if e.device_time_total and ("sparse" in e.key or "headshared" in e.key)
               ) / 1e3 / iters


print("head-shared kernel: blocks = B * (H/16) = B * 4")
print("%6s %10s %10s %12s %12s" % ("bs", "blocks", "ms", "ms per block", "work ratio"))
base = None
for bs in (1, 2, 4, 8, 16, 32):
    qb = q.expand(bs, 1, H, D).contiguous()
    ib = full.expand(bs, -1, -1).contiguous()
    t = dev_ms(lambda: M._run_headshared_sparse_decode(qb, kc, ib, None, 0.088))
    if base is None:
        base = t
    print("%6d %10d %10.4f %12.4f %12.1fx" % (bs, bs * 4, t, t / (bs * 4), t / base))

print("\nper-head kernel: blocks = B * 64")
print("%6s %10s %10s %12s" % ("bs", "blocks", "ms", "ms per block"))
b0 = None
for bs in (1, 2, 4):
    qb = q.expand(bs, 1, H, D).contiguous()
    ib = full.expand(bs, -1, -1).contiguous()
    t = dev_ms(lambda: M._run_triton_sparse_decode(qb, kc, ib, None, 0.088))
    if b0 is None:
        b0 = t
    print("%6d %10d %10.4f %12.4f" % (bs, bs * 64, t, t / (bs * 64)))
