"""What does _merge_splits itself cost, and is it why elementwise grew at long context?

The long-context family ranking puts elementwise/reduce at 5.99 ms per stage per step,
up from 2.53 in the short-context capture. The split kernel's own time is attributed to
the "attn" family, but the merge I wrote is plain torch (max, exp, where, sum, mul, div,
copy, log), so its kernels land in the elementwise bucket and are invisible in the attn
line. At 22 attention calls per step per stage and roughly 10 ops per merge, that is
~220 extra launches per step -- plausible as most of the growth.

Measure the merge alone at the production shape, device time and launch count, and
compare against the split kernel it serves. If the merge is a large fraction of the
attention cost, fusing it into one Triton kernel is the obvious next step.
"""
import os, sys
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
from torch.profiler import ProfilerActivity, profile
import sglang.kernels.ops.attention.flash_mla_sm120_triton as M

DEV = "cuda:0"
H, D, TOPK, BS = 64, 512, 512, 1
PAGE, STRIDE, NUM_TOK = 64, 576, 8192


def build(B, H, D, topk, num_tokens, page=PAGE, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    num_pages = -(-num_tokens // page)
    page_bytes = -(-584 * page // STRIDE) * STRIDE
    raw = torch.zeros(num_pages, page_bytes, dtype=torch.uint8, device=DEV)
    q8 = (torch.randn(num_tokens, 448, generator=g, device=DEV) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8)
    rope = (torch.randn(num_tokens, 64, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    sb = (127 + torch.randint(-6, 0, (num_tokens, 7), device=DEV, generator=g)).to(torch.uint8)
    pg = torch.arange(num_tokens, device=DEV) // page
    po = torch.arange(num_tokens, device=DEV) % page
    raw[pg[:, None], (po * STRIDE)[:, None] + torch.arange(448, device=DEV)[None, :]] = q8
    raw[pg[:, None], (po * STRIDE + 448)[:, None] + 2 * torch.arange(64, device=DEV)[None, :]] = rope.view(torch.uint8)[:, ::2]
    raw[pg[:, None], (page * STRIDE + po * 8)[:, None] + torch.arange(8, device=DEV)[None, :]] = torch.cat([sb, torch.zeros(num_tokens, 1, dtype=torch.uint8, device=DEV)], 1)
    kc = raw.as_strided((num_pages, page, 1, STRIDE), (page_bytes, STRIDE, STRIDE, 1)).view(torch.float8_e4m3fn)
    q = (torch.randn(B, 1, H, D, generator=g, device=DEV, dtype=torch.float32) * 0.5).half()
    idx = torch.randint(0, num_tokens, (B, 1, topk), generator=g, device=DEV, dtype=torch.int32)
    return q, kc, idx.reshape(B, -1).contiguous()


def prof(fn, iters=30):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    ka = [e for e in p.key_averages() if e.device_time_total]
    return (sum(e.device_time_total for e in ka) / 1e3 / iters,
            sum(e.count for e in ka) / iters)


q, kc, full = build(BS, H, D, TOPK, NUM_TOK)
S, B = 16, BS
part = torch.randn(S, B, H, D, dtype=torch.float16, device=DEV)
part_lse = torch.randn(S, B, H, dtype=torch.float32, device=DEV) + 6.0
out = torch.empty(B, H, D, dtype=torch.float16, device=DEV)
lse = torch.empty(B, H, dtype=torch.float32, device=DEV)

tm, nm = prof(lambda: M._merge_splits(part, part_lse, out, lse))
print("merge alone: %.4f ms in %.0f kernels" % (tm, nm))
print("  x22 attention calls/step/stage = %.2f ms/step/stage, %.0f launches/step"
      % (tm * 22, nm * 22))

os.environ["SGLANG_SM75_HS_TOPK_SPLIT"] = "16"
import importlib
importlib.reload(M)
tt, nt = prof(lambda: M._run_headshared_sparse_decode(q, kc, full, None, 0.088))
print("\nsplit=16 full path: %.4f ms in %.0f kernels" % (tt, nt))
print("  merge is %.0f%% of it" % (100 * tm / tt))

os.environ["SGLANG_SM75_HS_TOPK_SPLIT"] = "1"
importlib.reload(M)
tu, nu = prof(lambda: M._run_headshared_sparse_decode(q, kc, full, None, 0.088))
print("unsplit: %.4f ms in %.0f kernels" % (tu, nu))
print("\nnet vs unsplit: %.4f ms (%.2fx), merge overhead %.4f ms"
      % (tt, tu / tt, tm))
