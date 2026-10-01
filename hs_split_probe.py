"""Is a topk-split worth implementing in the head-shared attention kernel?

The budget decomposition says each (rank,stage) does 9.75 ms of device work per token
against a 2.68 ms bandwidth floor, and attention is the largest single kernel at
1.75 ms/token/stage (16.4% of decode). At bs=1 production uses the per-head kernel,
which re-gathers the same 512x576 B KV block once per head: 64x redundant, so it
moves ~236 MB per token per stage to do 3.7 MB of useful work.

The head-shared kernel amortizes that over BLOCK_H=16 heads (4x the traffic, not 64x)
but its grid is (B, H/16) = 4 blocks at bs=1, so 4 blocks on 68 SMs is why it loses
(0.514 vs 0.288 ms). The fix is a flash-decoding style split: add a grid dimension
over topk chunks so the block count becomes B x (H/16) x n_splits, which fills the
machine AND keeps the amortized gather.

Before writing that kernel, test whether the split pays at all using only existing
code: call the head-shared kernel n_splits times over slices of the index tensor and
merge with the existing _merge_partial_attn. That reproduces the split kernel's
behaviour exactly (same blocks, same bytes, plus a merge), so its timing is the
prediction. Compare against the per-head kernel production uses today.
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
BS = 1
qb = q.expand(BS, 1, H, D).contiguous()
ib = idx.reshape(BS, -1)[:BS].contiguous()


def bench(fn, iters=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


scale = 0.088
base = bench(lambda: M._run_triton_sparse_decode(qb, kc, ib, None, scale))
print("per-head kernel (production at bs=1): %.4f ms" % base)
print("\nhead-shared split over topk, merged with the existing _merge_partial_attn:")
print("%6s %10s %10s %10s %10s" % ("splits", "chunk", "ms total", "vs prod", "blocks"))


def run_split(ns):
    chunk = (TOPK + ns - 1) // ns
    outs, lses = [], []
    for s in range(ns):
        lo, hi = s * chunk, min((s + 1) * chunk, TOPK)
        if lo >= hi:
            continue
        sl = ib[:, lo:hi].contiguous()
        tl_ = torch.full((BS,), hi - lo, dtype=torch.int32, device="cuda")
        o, l = M._run_headshared_sparse_decode(qb, kc, sl, tl_, scale)
        outs.append(o); lses.append(l)
    out, lse = outs[0], lses[0]
    for o, l in zip(outs[1:], lses[1:]):
        out, lse = M._merge_partial_attn(out, lse, o, l)
    return out


for ns in (1, 2, 4, 8, 16):
    try:
        t = sorted(bench(lambda: run_split(ns)) for _ in range(3))[1]
        print("%6d %10d %10.4f %9.2fx %10d"
              % (ns, (TOPK + ns - 1) // ns, t, base / t, BS * (H // 16) * ns))
    except Exception as e:
        print("%6d  FAILED %s" % (ns, str(e)[:70]))
