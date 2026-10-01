"""Does the topk split stay correct when topk_length truncates the index list?

Production always passes topk_length: the SWA and compressed caches carry a variable
number of valid entries per request, and the kernel masks tiles past it. The split
computes its range from n_tiles = ceil(topk/BLOCK_T), NOT from valid_len, so a program
can be assigned a range that is entirely masked out. That must produce l_i=0, store
-inf, and be dropped by the combine -- and programs that are partially valid must merge
exactly. This is the one production code path the previous test did not exercise.

Runs on GPU 0, which still has ~974 MiB free after the driver leaked the rest.
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
    raw[pg[:, None], (po * STRIDE)[:, None] + torch.arange(448, device=DEV)[None, :]] = q8
    raw[pg[:, None], (po * STRIDE + 448)[:, None] + 2 * torch.arange(64, device=DEV)[None, :]] \
        = rope.view(torch.uint8)[:, ::2]
    raw[pg[:, None], (page * STRIDE + po * 8)[:, None] + torch.arange(8, device=DEV)[None, :]] \
        = torch.cat([sb, torch.zeros(num_tokens, 1, dtype=torch.uint8, device=DEV)], 1)
    kc = raw.as_strided((num_pages, page, 1, STRIDE),
                        (page_bytes, STRIDE, STRIDE, 1)).view(torch.float8_e4m3fn)
    q = (torch.randn(B, 1, H, D, generator=g, device=DEV, dtype=torch.float32) * 0.5).half()
    idx = torch.randint(0, num_tokens, (B, 1, topk), generator=g, device=DEV, dtype=torch.int32)
    return q, kc, idx.reshape(B, -1).contiguous()


q, kc, full = build(BS, H, D, TOPK, NUM_TOK)
import sglang.kernels.ops.attention.flash_mla_sm120_triton as M

print("%10s %8s %12s %12s %8s" % ("topk_len", "split", "max|dO|", "max|dLSE|", "nan"))
for tkl in (1, 7, 16, 33, 128, 511, 512):
    tl = torch.tensor([tkl], dtype=torch.int32, device=DEV)
    os.environ["SGLANG_SM75_HS_TOPK_SPLIT"] = "1"
    importlib.reload(M)
    ref, ref_lse = M._run_headshared_sparse_decode(q, kc, full, tl, 0.088)
    line = []
    for s in (2, 4, 8, 16):
        os.environ["SGLANG_SM75_HS_TOPK_SPLIT"] = str(s)
        importlib.reload(M)
        o, l = M._run_headshared_sparse_decode(q, kc, full, tl, 0.088)
        do = (o.float() - ref.float()).abs().max().item()
        dl = (l - ref_lse).abs().max().item()
        line.append("s=%-2d %.2e/%.2e nan%d" % (s, do, dl, int(torch.isnan(o).sum())))
    print("%10d      -    %s" % (tkl, "  ".join(line)))
