"""Where do the prefill attention kernel's 19.8 ms go?

The gather-only probe reads the same 134 MB with the same token addressing in
0.24 ms (558 GB/s, under the card's 616 GB/s peak, so the number is credible).
The production kernel does the same reads plus a per-byte fp8 LUT lookup and the
QK/PV dots, and takes 19.84 ms. That is an 83x gap, which is far too large to be
"the dots" -- so one of the three stages of this reasoning is wrong and I need to
find which. Each variant below is the production loop with exactly one cost
removed, so the 19.8 ms is attributed rather than guessed. Every sink is gated on
the accumulator so no variant can be dead-code eliminated (the trap that made the
w4a16 read-only probe report an impossible 928 GB/s).
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch, triton
import triton.language as tl
import ab_headshared_scale as A
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

LOG2E = tl.constexpr(1.4426950408889634)
TOKEN_BYTES = tl.constexpr(576)
SCALE_BYTES = tl.constexpr(8)
GROUP = tl.constexpr(64)
NOPE = tl.constexpr(448)
ROPE = tl.constexpr(64)
PAD = tl.constexpr(512)


@triton.jit
def variant(idx_ptr, cache_u8, cache_bf16, lut_ptr, q_ptr, out_ptr,
            topk, page_size, page_bytes, scale_off, stride_qb, stride_qh,
            BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr, NCOL: tl.constexpr,
            DO_LUT: tl.constexpr, DO_DOT: tl.constexpr):
    """Production addressing. DO_LUT=0 uses the raw byte as the value (skips the
    fp8 table lookup); DO_DOT=0 skips the QK and PV dots."""
    bid = tl.program_id(0)
    hblk = tl.program_id(1)
    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_t = tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, NCOL)
    offs_r = tl.arange(0, ROPE)
    NCHUNK: tl.constexpr = PAD // NCOL

    q_base = bid * stride_qb + offs_h * stride_qh
    q0 = tl.load(q_ptr + q_base[:, None] + offs_c[None, :])
    q_r = tl.load(q_ptr + q_base[:, None] + NOPE + offs_r[None, :])

    acc = tl.zeros([BLOCK_H, NCOL], dtype=tl.float32)
    acc_r = tl.zeros([BLOCK_H, ROPE], dtype=tl.float32)
    scores_sum = tl.zeros([BLOCK_H], dtype=tl.float32)

    for ts in range(0, topk, BLOCK_T):
        t_idx = ts + offs_t
        raw = tl.load(idx_ptr + bid * topk + t_idx, mask=t_idx < topk, other=0).to(tl.int64)
        page_ids = raw // page_size
        page_offs = raw % page_size
        base = page_ids * page_bytes + page_offs * TOKEN_BYTES
        soff = page_ids * page_bytes + scale_off + page_offs * SCALE_BYTES

        scores = tl.zeros([BLOCK_H, BLOCK_T], dtype=tl.float32)
        for c in tl.static_range(NCHUNK):
            col = c * NCOL + offs_c
            b = tl.load(cache_u8 + base[:, None] + col[None, :],
                        mask=(t_idx < topk)[:, None], other=0)
            if DO_LUT:
                sc = tl.load(cache_u8 + soff[:, None] + (col // GROUP)[None, :],
                             mask=(t_idx < topk)[:, None], other=127).to(tl.float32)
                v = (tl.load(lut_ptr + b) * sc[:, None]).to(tl.float16)
            else:
                v = b.to(tl.float16)
            if DO_DOT:
                scores += tl.dot(q0, tl.trans(v))
            else:
                scores += tl.sum(v.to(tl.float32), axis=1)[None, :].broadcast_to(BLOCK_H, BLOCK_T)
        # rope half
        br = tl.load(cache_bf16 + (base[:, None] + NOPE) // 2,
                     mask=(t_idx < topk)[:, None], other=0.0)
        if DO_DOT:
            acc_r += tl.dot(scores.to(tl.float16), br.to(tl.float16))
        else:
            acc_r += tl.sum(scores, axis=1)[:, None].broadcast_to(BLOCK_H, ROPE)
        scores_sum += tl.sum(scores, axis=1)
        acc += scores[:, 0:1].broadcast_to(BLOCK_H, NCOL) * 0.0001

    if tl.sum(acc) + tl.sum(acc_r) + tl.sum(scores_sum) == 12345.678:
        tl.store(out_ptr + bid + hblk, 1.0)


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


POOL = 236800
B, H, D, TOPK = 512, 32, 512, 512
q, kc, indices = A.build(B, H, D, TOPK, POOL)
flat = kc.as_strided((kc.numel(),), (1,)).view(torch.uint8)
flat_bf16 = flat.view(torch.bfloat16)
lut = M.fp8_payload_lut(torch.device("cuda"), torch.float32)
q3 = q.view(B, H, D).contiguous()
out = torch.zeros(B * 2, dtype=torch.float32, device="cuda")
idx = indices.view(B, -1).contiguous()
ps, pb = kc.shape[1], kc.stride(0)
NCOL = 64
grid = (B, 2)

def run(do_lut, do_dot):
    variant[grid](idx, flat, flat_bf16, lut, q3, out, TOPK, ps, int(pb),
                  ps * 576, q3.stride(0), q3.stride(1),
                  BLOCK_H=16, BLOCK_T=16, NCOL=NCOL, DO_LUT=do_lut, DO_DOT=do_dot,
                  num_warps=4, num_stages=2)

tlen = torch.full((B,), TOPK, dtype=torch.int32, device="cuda")
t_prod = bench(lambda: M._run_headshared_sparse_decode(
    q.view(B, 1, H, D), kc, indices.view(B, 1, TOPK), tlen, 0.088))

res = {}
for name, (l, d) in {"full (lut+dot)": (1, 1), "no-dot (lut+scale)": (1, 0),
                     "no-lut (dot only)": (0, 1), "neither (gather only)": (0, 0)}.items():
    try:
        res[name] = bench(lambda: run(l, d))
    except Exception as e:
        res[name] = None
        lines = [l for l in str(e).splitlines() if l.strip() and not l.strip().startswith(("for ", "b = ", "scores ", "col =", "mask", "else", "if ", "@"))]
        print("%-22s FAIL %s" % (name, " || ".join(lines[-2:])[:220]))

mb = B * TOPK * 576 / 1e6
print("\n%d MB gathered per call (x2 with the head-block redundancy)" % mb)
print("production kernel    %7.2f ms  %6.1f GB/s" % (t_prod, 2 * mb / t_prod))
for name, ms in res.items():
    if ms:
        print("%-22s %7.2f ms  %6.1f GB/s" % (name, ms, 2 * mb / ms))
