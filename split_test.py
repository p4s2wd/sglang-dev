"""Correctness and speed of the topk-split head-shared kernel.

The split is a strict generalisation: N_SPLIT=1 must reproduce the unsplit kernel
bit-for-bit, and any N_SPLIT must reproduce the softmax of the full topk set, because
each partial carries its own max and denominator and the combine is the exact
LSE-weighted mean.

Speed matters at B=1, which is where production decode runs. Unsplit head-shared costs
0.710 ms there against the per-head kernel's 0.272 ms, which is why the gate is at B>=2.
If splitting 8 ways brings it under 0.272 ms, the gate can drop to 1 and production
attention gets faster without touching the per-head kernel.
"""
import importlib, os, sys
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
from torch.profiler import ProfilerActivity, profile
import ab_headshared_scale as A

H, D, TOPK = 64, 512, 512


def dev_ms(fn, iters=25):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in pr.key_averages()
               if e.device_time_total and ("sparse" in e.key or "headshared" in e.key
                                           or "merge" in e.key or "copy" in e.key
                                           or "sum" in e.key or "exp" in e.key
                                           or "max" in e.key or "where" in e.key
                                           or "log" in e.key or "clamp" in e.key)
               ) / 1e3 / iters


import sglang.kernels.ops.attention.flash_mla_sm120_triton as M

for bs in (1, 2, 4):
    q, kc, idx = A.build(bs, H, D, TOPK, 236800)
    qb = q.expand(bs, 1, H, D).contiguous() if q.shape[0] == 1 else q.contiguous()
    full = idx.reshape(bs, -1).contiguous()
    if q.shape[0] != bs:
        full = full[:1].expand(bs, -1).contiguous()
    print("\n=== bs=%d ===" % bs)
    os.environ["SGLANG_SM75_HS_TOPK_SPLIT"] = "1"
    importlib.reload(M)
    ref, ref_lse = M._run_headshared_sparse_decode(qb, kc, full, None, 0.088)
    t1 = dev_ms(lambda: M._run_headshared_sparse_decode(qb, kc, full, None, 0.088))
    tph = dev_ms(lambda: M._run_triton_sparse_decode(qb, kc, full, None, 0.088))
    print("  unsplit headshared %.4f ms   per-head %.4f ms" % (t1, tph))
    for s in (2, 4, 8, 16, 32):
        os.environ["SGLANG_SM75_HS_TOPK_SPLIT"] = str(s)
        importlib.reload(M)
        try:
            o, l = M._run_headshared_sparse_decode(qb, kc, full, None, 0.088)
            t = dev_ms(lambda: M._run_headshared_sparse_decode(qb, kc, full, None, 0.088))
            do = (o.float() - ref.float()).abs().max().item()
            dl = (l - ref_lse).abs().max().item()
            rel = do / max(ref.float().abs().max().item(), 1e-6)
            print("  split=%-3d %.4f ms (%.2fx unsplit, %.2fx per-head)  "
                  "max|dO| %.2e (rel %.2e)  max|dLSE| %.2e"
                  % (s, t, t1 / t, tph / t, do, rel, dl))
        except Exception as e:
            print("  split=%-3d FAIL %s" % (s, str(e)[:70]))
