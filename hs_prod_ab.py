"""Production head-shared kernel at the real prefill shape, one build per process.

SGLANG_SM75_HS_QK_TRANS is read at import time into a tl.constexpr, so the two
variants cannot be swapped inside one process (the JITFunction is autotune-wrapped
and its compiled cache is not cleanly resettable). Run it twice instead and
compare; the machine's thermal drift is ~6% within a session, so a 1.6x effect is
far outside that band and a cross-process comparison is sound here.
"""
import os, sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import ab_headshared_scale as A
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

POOL = 236800
B, H, D, TOPK = 512, 32, 512, 512
q, kc, idx = A.build(B, H, D, TOPK, POOL)
tlen = torch.full((B,), TOPK, dtype=torch.int32, device="cuda")
mb = B * TOPK * 576 / 1e6

def bench(fn, iters=10, warmup=4):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

f = lambda: M._run_headshared_sparse_decode(q, kc, idx, tlen, 0.088)
ts = sorted(bench(f) for _ in range(5))
print("QK_TRANS=%s  median %.2f ms  %.1f GB/s  (min %.2f max %.2f)"
      % (os.environ.get("SGLANG_SM75_HS_QK_TRANS", "1"), ts[2], mb / ts[2], ts[0], ts[-1]))
