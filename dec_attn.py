"""Decode attention: per-head kernel vs headshared kernel at real decode batch.

Decode's #1 kernel at 190 W is _tiled_sparse_decode_kernel, 49.7% of stage device
time (1.338 ms/call). It is NOT the kernel the transposed-gather work optimized --
that went to _headshared_sparse_kernel, which the dispatcher only selects when
batch >= SGLANG_SM75_HEADSHARED_MIN_BATCH (default 16, flash_mla_sm120_triton.py:408,
:865). Decode runs at bs=1-2, so it always takes the per-head path.

The gate exists for a reason: this is MLA, all 64 heads share one KV entry per
token, so the per-head kernel gathers the same 512x576 B block 64 times. The
headshared kernel gathers it once per 16-head group, cutting gather 16x -- but its
grid is (B, H/16), so at bs=1 it launches 4 blocks on a 68-SM GPU and leaves 94% of
the machine idle. If the kernel is compute-bound, that is 16x slower; if it is
gather-bound, it wins.

Measured both ways at bs=1,2,4,8,16,24 with the production cache layout, so the
gate's threshold is set by measurement rather than assumption. Also reports the
gather bytes each variant actually moves.
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
print("cache %d tokens = %.0f MB; topk=%d, %d heads" % (POOL, POOL * 584 / 1e6, TOPK, H))


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


print("\n%4s %12s %12s %9s %10s %10s" %
      ("bs", "per-head ms", "headshared", "ratio", "ph blocks", "hs blocks"))
for bs in (1, 2, 4, 8, 16, 24):
    qb = q.expand(bs, 1, H, D).contiguous()
    # topk indices per batch element
    ib = idx.expand(bs, -1, -1).contiguous() if idx.shape[0] == 1 else idx[:bs]
    tlen = torch.full((bs,), TOPK, dtype=torch.int32, device="cuda")
    try:
        a = sorted(bench(lambda: M._run_triton_sparse_decode(qb, kc, ib, tlen, 0.088))
                   for _ in range(3))[1]
    except Exception as e:
        a = float("nan")
        print("  per-head FAILED bs=%d %s" % (bs, str(e)[:60]))
    try:
        b = sorted(bench(lambda: M._run_headshared_sparse_decode(qb, kc, ib, tlen, 0.088))
                   for _ in range(3))[1]
    except Exception as e:
        b = float("nan")
        print("  headshared FAILED bs=%d %s" % (bs, str(e)[:60]))
    print("%4d %12.4f %12.4f %9.2fx %10d %10d"
          % (bs, a, b, a / b if b == b and b > 0 else float("nan"), bs * H, bs * (H // 16)))
