"""Topk-split correctness and speed, sized to fit the 0.8 GiB still free per GPU.

The box is in a degraded state: six scheduler processes are zombies stuck in driver
teardown (dmesg shows NVRM rpcSendMessage 0x62 and NVLink P2P assertion failures) and
they still pin ~20.5 GiB on every GPU, so nothing large can be allocated.

Pool size does not change what this kernel does: the tile loop runs ceil(topk/BLOCK_T)
iterations over random indices regardless of how many tokens exist, so a small pool
exercises the identical code path and the same number of dots and gathers. Timing is
therefore still meaningful for the split's effect on the critical path, even though the
absolute number will differ from a 236800-token pool (L2 hits more often here).

Correctness is the point of this run: N_SPLIT=1 must match the shipped behaviour, and
every split must reproduce the full-topk softmax.
"""
import importlib, os, sys
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
from torch.profiler import ProfilerActivity, profile

DEV = "cuda:0"
H, D, TOPK, BS = 64, 512, 512, 1
PAGE, STRIDE, NUM_TOK = 64, 576, 8192


def build(B, H, D, topk, num_tokens, page=PAGE, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    num_pages = -(-num_tokens // page)
    page_bytes = -(-584 * page // STRIDE) * STRIDE
    raw = torch.zeros(num_pages, page_bytes, dtype=torch.uint8, device=DEV)
    nope = torch.randn(num_tokens, 448, generator=g, device=DEV) * 0.3
    q8 = nope.to(torch.float8_e4m3fn).view(torch.uint8)
    rope = (torch.randn(num_tokens, 64, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    sb = (127 + torch.randint(-6, 0, (num_tokens, 7), device=DEV, generator=g)).to(torch.uint8)
    pg = torch.arange(num_tokens, device=DEV) // page
    po = torch.arange(num_tokens, device=DEV) % page
    cols = torch.arange(448, device=DEV)
    raw[pg[:, None], (po * STRIDE)[:, None] + cols[None, :]] = q8
    raw[pg[:, None], (po * STRIDE + 448)[:, None] + 2 * torch.arange(64, device=DEV)[None, :]] \
        = rope.view(torch.uint8)[:, ::2]
    raw[pg[:, None], (page * STRIDE + po * 8)[:, None] + torch.arange(8, device=DEV)[None, :]] \
        = torch.cat([sb, torch.zeros(num_tokens, 1, dtype=torch.uint8, device=DEV)], 1)
    kc = raw.as_strided((num_pages, page, 1, STRIDE),
                        (page_bytes, STRIDE, STRIDE, 1)).view(torch.float8_e4m3fn)
    q = (torch.randn(B, 1, H, D, generator=g, device=DEV, dtype=torch.float32) * 0.5).half()
    idx = torch.randint(0, num_tokens, (B, 1, topk), generator=g, device=DEV, dtype=torch.int32)
    return q, kc, idx.reshape(B, -1).contiguous()


def dev_ms(fn, iters=30):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in pr.key_averages()
               if e.device_time_total) / 1e3 / iters


q, kc, full = build(BS, H, D, TOPK, NUM_TOK)
import sglang.kernels.ops.attention.flash_mla_sm120_triton as M

os.environ["SGLANG_SM75_HS_TOPK_SPLIT"] = "1"
importlib.reload(M)
ref, ref_lse = M._run_headshared_sparse_decode(q, kc, full, None, 0.088)
t1 = dev_ms(lambda: M._run_headshared_sparse_decode(q, kc, full, None, 0.088))
tph = dev_ms(lambda: M._run_triton_sparse_decode(q, kc, full, None, 0.088))
print("unsplit headshared %.4f ms   per-head %.4f ms" % (t1, tph))
print("ref out absmax %.4f  lse range %.3f..%.3f"
      % (ref.float().abs().max().item(), ref_lse.min().item(), ref_lse.max().item()))

for s in (2, 4, 8, 16, 32):
    os.environ["SGLANG_SM75_HS_TOPK_SPLIT"] = str(s)
    importlib.reload(M)
    try:
        o, l = M._run_headshared_sparse_decode(q, kc, full, None, 0.088)
        t = dev_ms(lambda: M._run_headshared_sparse_decode(q, kc, full, None, 0.088))
        do = (o.float() - ref.float()).abs().max().item()
        dl = (l - ref_lse).abs().max().item()
        rel = do / max(ref.float().abs().max().item(), 1e-6)
        print("split=%-3d %.4f ms (%.2fx unsplit, %.2fx per-head)  "
              "max|dO| %.2e (rel %.2e)  max|dLSE| %.2e  nan %d"
              % (s, t, t1 / t, tph / t, do, rel, dl, int(torch.isnan(o).sum().item())))
    except Exception as e:
        print("split=%-3d FAIL %s" % (s, str(e)[:80]))
