"""NCOL sweep at PRODUCTION pool size — the measurement the old sweep lacked.

The prior NCOL sweep picked 64 over 128, but it ran against a small KV cache that
fits in L2, where a 64-byte-per-token gather and a 512-byte-per-token gather cost
the same because everything is already resident. In production the pool is 138 MB
(DRAM), and the two differ completely: 576-byte tokens at random addresses, read
64 bytes at a time, touch one 64-byte slice per 576-byte stride, so each DRAM
fetch of a 128-byte line wastes most of it. The hand-written gather that hit
558 GB/s read all 512 columns in one wide load, i.e. 512 contiguous bytes per
token. This sweeps NCOL at the real pool size to find where coalescing stops
being the limit. DEQ=0 throughout, so this is purely the gather shape.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import ab_headshared_scale as A
import bh32_nohoist as B

POOL = 236800
Bt, H, D, TOPK = 512, 32, 512, 512
q, kc, flat_idx = A.build(Bt, H, D, TOPK, POOL)
out = torch.zeros(Bt, H, D, dtype=torch.float16, device="cuda")
lse = torch.zeros(Bt, H, dtype=torch.float32, device="cuda")
mb = Bt * TOPK * 576 / 1e6
print("pool %.0f MB; %.0f MB useful/call; DEQ=0 (pure gather shape)" % (POOL*584/1e6, mb))

def bench(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

print("%-22s %8s %9s" % ("config", "ms", "GB/s"))
for ncol in (64, 128, 256, 512):
    for warps in (4, 8):
        try:
            ms = bench(lambda n=ncol, w=warps: B.run(q, kc, flat_idx, 0.088, out, lse,
                     bh=16, ncol=n, warps=w, hoist=1, stages=2, deq=0))
            print("ncol=%3d warps=%d    %8.2f %9.1f" % (ncol, warps, ms, 2*mb/ms))
        except Exception as e:
            print("ncol=%3d warps=%d    FAIL %s" % (ncol, warps, str(e).splitlines()[-1][:50]))
