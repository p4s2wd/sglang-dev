"""Is the head-shared kernel latency-bound per tile, or fixed-cost per call?

The previous probe split topk across n SEQUENTIAL kernel launches and found time
scaling linearly with n. That does not test the hypothesis: real flash-decoding
splits inside ONE launch so the n_splits blocks run concurrently, whereas sequential
launches serialize and pay any per-call cost n times. The linear result was an
artifact of the simulation, not evidence.

The decisive measurement is a single call with a shrinking topk, which shortens each
block's sequential tile loop while leaving everything else fixed:
  - time falls with chunk  -> the tile loop dominates and it is latency-bound per
    tile, so raising in-flight parallelism (in-kernel topk split, or a wider BLOCK_T)
    is the lever.
  - time stays flat        -> a genuine per-call fixed cost, and no split helps.

Cross-check from the batch sweep already measured: bs=1 (4 blocks) costs 0.513 ms and
bs=16 (64 blocks, 16x the work) costs 0.646 ms, i.e. 1.26x. Blocks parallelize almost
for free, which already implies a single block's 32-tile loop takes ~0.5 ms on its own
-- about 16 us per tile for 16 tokens x 576 B, which is pure exposed latency.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import ab_headshared_scale as A
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

H, D, TOPK = 64, 512, 512
POOL = 236800
q, kc, idx = A.build(1, H, D, TOPK, POOL)
qb = q.expand(1, 1, H, D).contiguous()
full = idx.reshape(1, -1).contiguous()


def bench(fn, iters=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


print("single head-shared call, topk shrinking (tile loop shortens, all else fixed)")
print("%8s %8s %10s %10s %12s" % ("topk", "tiles", "ms", "us/tile", "GB/s (useful)"))
base = None
for tk in (512, 256, 128, 64, 32, 16):
    sl = full[:, :tk].contiguous()
    tl_ = torch.full((1,), tk, dtype=torch.int32, device="cuda")
    t = sorted(bench(lambda: M._run_headshared_sparse_decode(qb, kc, sl, tl_, 0.088))
               for _ in range(3))[1]
    tiles = tk // 16
    gb = tk * 576 / 1e9 / (t / 1e3)
    if base is None:
        base = t
    print("%8d %8d %10.4f %10.2f %12.1f" % (tk, tiles, t, t * 1e3 / tiles, gb))

print("\nsame for the per-head kernel production uses at bs=1")
print("%8s %10s %10s" % ("topk", "ms", "vs headshared"))
for tk in (512, 256, 128, 64, 32, 16):
    sl = full[:, :tk].contiguous()
    tl_ = torch.full((1,), tk, dtype=torch.int32, device="cuda")
    t = sorted(bench(lambda: M._run_triton_sparse_decode(qb, kc, sl, tl_, 0.088))
               for _ in range(3))[1]
    print("%8d %10.4f %10.2fx" % (tk, t, t / base if base else 0))
