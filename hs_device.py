"""Device-only time for the two attention kernels at bs=1, measured with the profiler.

Every number in the previous probes was host wall-clock around a wrapper call, and the
wrapper does real host work per call: torch.zeros, torch.full, indices.contiguous(),
as_strided, two .view() calls, fp8_payload_lut, plus the Triton launch. That floor is
~0.155 ms, which is why topk=16 (1 tile, trivially little GPU work) and topk=128
(8 tiles) both measured ~0.16 ms. Host timing therefore cannot rank these kernels at
all -- and the production DECODE trace proves it, showing per-head attention at
0.159 ms of DEVICE time, already at the host floor.

Production decode runs under CUDA graphs, so the wrapper's host cost is captured once
and never repeats per step. The only quantity that matters is kernel duration. Read it
with the profiler, for both kernels, at the production shape.
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
qb = q.expand(1, 1, H, D).contiguous()
full = idx.reshape(1, -1).contiguous()


def device_ms(fn, iters=30):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = 0.0
    for e in pr.key_averages():
        if e.device_time_total and ("sparse" in e.key or "headshared" in e.key):
            tot += e.device_time_total
    return tot / 1e3 / iters


for tk in (512, 256, 128, 64):
    sl = full[:, :tk].contiguous()
    tl_ = torch.full((1,), tk, dtype=torch.int32, device="cuda")
    d_ph = device_ms(lambda: M._run_triton_sparse_decode(qb, kc, sl, tl_, 0.088))
    d_hs = device_ms(lambda: M._run_headshared_sparse_decode(qb, kc, sl, tl_, 0.088))
    print("topk=%4d  per-head device %7.4f ms   head-shared device %7.4f ms   hs/ph %.2fx"
          % (tk, d_ph, d_hs, d_hs / d_ph if d_ph else 0))
