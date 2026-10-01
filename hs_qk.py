"""Confirm the attention bottleneck is Triton's register->SMEM round-trip for tl.dot.

The gather is free (0.24 ms = 100% of peak bandwidth), so the production kernel's
18 ms is the mma path running at 1.5% of the 58 TFLOP/s measured peak. Triton on
sm_75 must stage register-computed operands through shared memory before
ldmatrix/mma -- the same hard limit that forced the hand-written W4A16 kernel.
This minimal QK-only kernel isolates it: same gather, one dot per tile, no
softmax/PV/LSE. The TRANS=1 variant gathers straight into [NCOL, BLOCK_T] so the
dot needs no tl.trans. If TRANS is much faster, the trans staging is the cost and
a transposed gather is a cheap fix in the real kernel. If both are ~the same, the
cost is Triton's operand staging itself and only hand-written mma reaches the
floor -- which is the conclusion the W4A16 kernel already documented.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch, triton
import triton.language as tl
import ab_headshared_scale as A
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

TOKB = tl.constexpr(576); SB = tl.constexpr(8); GRP = tl.constexpr(64)
NOPE = tl.constexpr(448)

@triton.jit
def qk(Q, cu8, lut, idx, out, topk, page_size, page_bytes, soff, sqb, sqh,
       BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr, NCOL: tl.constexpr,
       NCHUNK: tl.constexpr, TRANS: tl.constexpr):
    bid = tl.program_id(0); hblk = tl.program_id(1)
    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_t = tl.arange(0, BLOCK_T); offs_c = tl.arange(0, NCOL)
    qb = bid * sqb + offs_h * sqh
    q0 = tl.load(Q + qb[:, None] + offs_c[None, :])
    acc = tl.zeros([BLOCK_H, BLOCK_T], tl.float32)
    for ts in range(0, topk, BLOCK_T):
        t_idx = ts + offs_t
        raw = tl.load(idx + bid * topk + t_idx, mask=t_idx < topk, other=0).to(tl.int64)
        pg = raw // page_size; po = raw % page_size
        base = pg * page_bytes + po * TOKB
        soff_t = pg * page_bytes + soff + po * SB
        scores = tl.zeros([BLOCK_H, BLOCK_T], tl.float32)
        for c in tl.static_range(NCHUNK):
            col = c * NCOL + offs_c
            if TRANS:
                m = (t_idx < topk)[None, :] & (col < NOPE)[:, None]
                byte = tl.load(cu8 + base[None, :] + col[:, None], mask=m, other=0)
                sa = soff_t[None, :] + (col // GRP)[:, None]
                sc = tl.math.exp2(tl.load(cu8 + sa, mask=m, other=127).to(tl.float32) - 127.0)
                kv = (tl.load(lut + byte) * sc).to(tl.float16)   # [NCOL, BLOCK_T]
                scores += tl.dot(q0, kv)
            else:
                m = (t_idx < topk)[:, None] & (col < NOPE)[None, :]
                byte = tl.load(cu8 + base[:, None] + col[None, :], mask=m, other=0)
                sa = soff_t[:, None] + (col // GRP)[None, :]
                sc = tl.math.exp2(tl.load(cu8 + sa, mask=m, other=127).to(tl.float32) - 127.0)
                kv = (tl.load(lut + byte) * sc).to(tl.float16)   # [BLOCK_T, NCOL]
                scores += tl.dot(q0, tl.trans(kv))
        acc += scores
    if tl.sum(acc) == 12345.678:
        tl.store(out + bid + hblk, 1.0)

POOL = 236800
Bt, H, D, TOPK = 512, 32, 512, 512
q, kc, idx = A.build(Bt, H, D, TOPK, POOL)
flat = kc.as_strided((kc.numel(),), (1,)).view(torch.uint8)
lut = M.fp8_payload_lut(torch.device("cuda"), torch.float32)
q3 = q.squeeze(1).contiguous()
out = torch.zeros(Bt * 2, dtype=torch.float32, device="cuda")
ps, pb = kc.shape[1], kc.stride(0)
mb = Bt * TOPK * 576 / 1e6

def bench(fn, iters=10, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

print("QK-only kernel. %.0f MB gathered/call; gather floor 0.24 ms" % mb)
print("%-28s %8s %9s" % ("config", "ms", "GB/s"))
for ncol, warps, trans in ((64, 4, 0), (64, 4, 1), (128, 4, 1), (64, 8, 1), (128, 8, 1)):
    try:
        ms = bench(lambda n=ncol, w=warps, t=trans: qk[(Bt, 2)](
            q3, flat, lut, idx, out, TOPK, ps, int(pb), ps*576,
            q3.stride(0), q3.stride(1), BLOCK_H=16, BLOCK_T=16, NCOL=n,
            NCHUNK=512//n, TRANS=t, num_warps=w, num_stages=2))
        print("%-28s %8.2f %9.1f" % ("ncol=%d w%d trans=%d" % (ncol, warps, trans), ms, mb/ms))
    except Exception as e:
        print("%-28s FAIL %s" % ("ncol=%d w%d trans=%d" % (ncol, warps, trans),
                                 str(e).splitlines()[-1][:50]))
