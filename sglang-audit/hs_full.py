"""Head-shared sparse attention with a transposed KV gather: no tl.trans anywhere.

WHY. The carried-over verdict was "attention is at its gather floor: 2.524 ms vs
2.368 ms for a bare gather". That was measured against a KV cache small enough to
live in L2. At the production pool (236800 tokens = 138 MB, DRAM) the picture is
different. A bare gather of the same 151 MB runs in 0.24 ms = 614 GB/s, which is
100% of the card's 616 GB/s peak -- the gather is free. The production kernel
takes 19 ms for the same bytes, so ~18 ms is the mma path, running at 1.5% of the
58 TFLOP/s measured tensor-core peak. The one structurally expensive thing there
is tl.trans: Triton stages every register-computed operand through shared memory
before ldmatrix/mma, and the kernel transposes eight KV chunks per tile. A QK-only
probe measured it directly: transposed gather 3.45 ms vs tl.trans 13.47 ms, 3.9x.

HOW. Gather KV as [NCOL, BLOCK_T] (transposed) instead of [BLOCK_T, NCOL]. Then:
  QK:  scores[H,T] = dot(q[H,NCOL], kv_t[NCOL,T])          -- no trans
  PV:  acc_t[NCOL,H] += dot(kv_t[NCOL,T], p_t[T,H])         -- no trans
The probabilities p_t = trans(p) are transposed ONCE per tile and shared by all
eight nope chunks plus rope, replacing eight per-chunk KV transposes. The output
accumulator is [NCOL, BLOCK_H] and transposed once at the store, outside the loop.

This file is a testbed: it must match the production kernel's output before any of
it is ported. Correctness is checked against the production kernel on the same
inputs, not against a tolerance the author picked.

Run: python hs_full.py [pool_tokens]
"""
import sys
import time

sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

import torch
import triton
import triton.language as tl

import ab_headshared_scale as A
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

LOG2E = tl.constexpr(1.4426950408889634)
TOKB = tl.constexpr(576)
SB = tl.constexpr(8)
GRP = tl.constexpr(64)
NOPE = tl.constexpr(448)
ROPE = tl.constexpr(64)
PAD = tl.constexpr(512)


@triton.jit
def _kv_t(cu8, lut, tok_base, cols, pg, po, valid, page_bytes, soff):
    """Gather one [NCOL, BLOCK_T] nope slice, already transposed.

    Same bytes and same addresses as the production _kv, indexed transposed so
    the QK dot consumes it directly.
    """
    m = valid[None, :] & (cols < NOPE)[:, None]
    byte = tl.load(cu8 + tok_base[None, :] + cols[:, None], mask=m, other=0)
    sa = soff[None, :] + (cols // GRP)[:, None]
    scale = tl.math.exp2(tl.load(cu8 + sa, mask=m, other=127).to(tl.float32) - 127.0)
    return (tl.load(lut + byte) * scale).to(tl.float16)


@triton.jit
def _k(Q_ptr, cu8, cu16, lut, idx_ptr, O_ptr, LSE_ptr, softmax_scale,
       page_size, page_bytes, soff, H, topk, stride_qb, stride_qh,
       stride_ob, stride_oh,
       BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr, NCOL: tl.constexpr):
    bid = tl.program_id(0)
    hblk = tl.program_id(1)
    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_t = tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, NCOL)
    offs_r = tl.arange(0, ROPE)
    NCHUNK: tl.constexpr = PAD // NCOL

    h_valid = offs_h < H
    q_base = bid * stride_qb + offs_h * stride_qh
    # Query tiles stay [BLOCK_H, NCOL]; they are the left operand of the QK dot.
    q0 = tl.load(Q_ptr + q_base[:, None] + (0 * NCOL + offs_c)[None, :], mask=h_valid[:, None], other=0.0)
    q1 = tl.load(Q_ptr + q_base[:, None] + (1 * NCOL + offs_c)[None, :], mask=h_valid[:, None], other=0.0)
    q2 = tl.load(Q_ptr + q_base[:, None] + (2 * NCOL + offs_c)[None, :], mask=h_valid[:, None], other=0.0)
    q3 = tl.load(Q_ptr + q_base[:, None] + (3 * NCOL + offs_c)[None, :], mask=h_valid[:, None], other=0.0)
    q4 = tl.load(Q_ptr + q_base[:, None] + (4 * NCOL + offs_c)[None, :], mask=h_valid[:, None], other=0.0)
    q5 = tl.load(Q_ptr + q_base[:, None] + (5 * NCOL + offs_c)[None, :], mask=h_valid[:, None], other=0.0)
    q6 = tl.load(Q_ptr + q_base[:, None] + (6 * NCOL + offs_c)[None, :], mask=h_valid[:, None], other=0.0)
    q7 = tl.load(Q_ptr + q_base[:, None] + (7 * NCOL + offs_c)[None, :], mask=h_valid[:, None], other=0.0)
    q_r = tl.load(Q_ptr + q_base[:, None] + NOPE + offs_r[None, :], mask=h_valid[:, None], other=0.0)

    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    # PV accumulators are transposed [NCOL, BLOCK_H]; store transposes once.
    a0 = tl.zeros([NCOL, BLOCK_H], tl.float32)
    a1 = tl.zeros([NCOL, BLOCK_H], tl.float32)
    a2 = tl.zeros([NCOL, BLOCK_H], tl.float32)
    a3 = tl.zeros([NCOL, BLOCK_H], tl.float32)
    a4 = tl.zeros([NCOL, BLOCK_H], tl.float32)
    a5 = tl.zeros([NCOL, BLOCK_H], tl.float32)
    a6 = tl.zeros([NCOL, BLOCK_H], tl.float32)
    a7 = tl.zeros([NCOL, BLOCK_H], tl.float32)
    a_r = tl.zeros([ROPE, BLOCK_H], tl.float32)

    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + offs_t
        raw = tl.load(idx_ptr + bid * topk + t_idx, mask=t_idx < topk, other=-1)
        valid = (t_idx < topk) & (raw >= 0)
        safe = tl.where(valid, raw, 0).to(tl.int64)
        pg = safe // page_size
        po = safe % page_size
        tok_base = pg * page_bytes + po * TOKB
        soff_t = pg * page_bytes + soff + po * SB

        scores = tl.zeros([BLOCK_H, BLOCK_T], tl.float32)
        kv0 = _kv_t(cu8, lut, tok_base, 0 * NCOL + offs_c, pg, po, valid, page_bytes, soff_t)
        scores += tl.dot(q0, kv0)
        kv1 = _kv_t(cu8, lut, tok_base, 1 * NCOL + offs_c, pg, po, valid, page_bytes, soff_t)
        scores += tl.dot(q1, kv1)
        kv2 = _kv_t(cu8, lut, tok_base, 2 * NCOL + offs_c, pg, po, valid, page_bytes, soff_t)
        scores += tl.dot(q2, kv2)
        kv3 = _kv_t(cu8, lut, tok_base, 3 * NCOL + offs_c, pg, po, valid, page_bytes, soff_t)
        scores += tl.dot(q3, kv3)
        kv4 = _kv_t(cu8, lut, tok_base, 4 * NCOL + offs_c, pg, po, valid, page_bytes, soff_t)
        scores += tl.dot(q4, kv4)
        kv5 = _kv_t(cu8, lut, tok_base, 5 * NCOL + offs_c, pg, po, valid, page_bytes, soff_t)
        scores += tl.dot(q5, kv5)
        kv6 = _kv_t(cu8, lut, tok_base, 6 * NCOL + offs_c, pg, po, valid, page_bytes, soff_t)
        scores += tl.dot(q6, kv6)
        kv7 = _kv_t(cu8, lut, tok_base, 7 * NCOL + offs_c, pg, po, valid, page_bytes, soff_t)
        scores += tl.dot(q7, kv7)
        # rope, transposed [ROPE, BLOCK_T]; bf16 so byte offset /2.
        rb = ((tok_base + NOPE) // 2).to(tl.int64)
        kv_r = tl.load(cu16 + rb[None, :] + offs_r[:, None], mask=valid[None, :], other=0.0).to(tl.float16)
        scores += tl.dot(q_r, kv_r)

        s = tl.where(valid[None, :], scores * (softmax_scale * LOG2E), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - m_safe))
        p = tl.where(valid[None, :], tl.math.exp2(s - m_safe[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pf = p.to(tl.float16)
        # ONE transpose of the shared probabilities, used by all 9 PV dots.
        pf_t = tl.trans(pf)

        a0 = a0 * alpha[None, :] + tl.dot(kv0, pf_t)
        a1 = a1 * alpha[None, :] + tl.dot(kv1, pf_t)
        a2 = a2 * alpha[None, :] + tl.dot(kv2, pf_t)
        a3 = a3 * alpha[None, :] + tl.dot(kv3, pf_t)
        a4 = a4 * alpha[None, :] + tl.dot(kv4, pf_t)
        a5 = a5 * alpha[None, :] + tl.dot(kv5, pf_t)
        a6 = a6 * alpha[None, :] + tl.dot(kv6, pf_t)
        a7 = a7 * alpha[None, :] + tl.dot(kv7, pf_t)
        a_r = a_r * alpha[None, :] + tl.dot(kv_r, pf_t)
        m_i = m_new

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    o_base = bid * stride_ob + offs_h * stride_oh
    # Transpose each accumulator back once, outside the loop.
    if NCHUNK >= 1:
        c = 0 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 (tl.trans(a0) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 2:
        c = 1 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 (tl.trans(a1) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 3:
        c = 2 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 (tl.trans(a2) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 4:
        c = 3 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 (tl.trans(a3) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 5:
        c = 4 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 (tl.trans(a4) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 6:
        c = 5 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 (tl.trans(a5) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 7:
        c = 6 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 (tl.trans(a6) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 8:
        c = 7 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 (tl.trans(a7) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    tl.store(O_ptr + o_base[:, None] + NOPE + offs_r[None, :],
             (tl.trans(a_r) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None])
    tl.store(LSE_ptr + bid * H + offs_h,
             tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf")),
             mask=h_valid)


def run(q, kc, flat, softmax_scale, out, lse, bh=16, ncol=64, warps=4, stages=2, bt=16):
    B, _, H, D = q.shape
    page = kc.shape[1]
    page_bytes = kc.stride(0)
    total = kc.shape[0] * page_bytes
    raw_u8 = kc.as_strided((total,), (1,)).view(torch.uint8)
    raw_bf16 = raw_u8.view(torch.bfloat16)
    lut = M.fp8_payload_lut(q.device, torch.float32)
    q3 = q.squeeze(1).contiguous()
    grid = (B, triton.cdiv(H, bh))
    _k[grid](q3, raw_u8, raw_bf16, lut, flat, out, lse, softmax_scale,
             page, int(page_bytes), int(page * 576),
             H, flat.shape[1], q3.stride(0), q3.stride(1),
             out.stride(0), out.stride(1),
             BLOCK_H=bh, BLOCK_T=bt, NCOL=ncol,
             num_warps=warps, num_stages=stages)
    return out, lse


def bench(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 236800
    B, H, D, TOPK = 512, 32, 512, 512
    q, kc, idx = A.build(B, H, D, TOPK, POOL)
    out = torch.zeros(B, H, D, dtype=torch.float16, device="cuda")
    lse = torch.zeros(B, H, dtype=torch.float32, device="cuda")
    mb = B * TOPK * 576 / 1e6
    print("pool %.0f MB; %.0f MB gathered/call; gather floor 0.24 ms" % (POOL * 584 / 1e6, mb))

    # correctness against the production kernel
    p_out, p_lse = M._run_headshared_sparse_decode(
        q, kc, idx, torch.full((B,), TOPK, dtype=torch.int32, device="cuda"), 0.088)
    p_out = p_out.squeeze(1); p_lse = p_lse.squeeze(1)
    run(q, kc, idx, 0.088, out, lse, bh=16, ncol=32, warps=4, stages=1)
    torch.cuda.synchronize()
    do = (out.float() - p_out.float()).abs().max().item()
    dl = (lse - p_lse).abs().max().item()
    print("correctness vs production: max|dO|=%.3e max|dLSE|=%.3e  %s"
          % (do, dl, "OK" if do < 2e-2 and dl < 2e-2 else "MISMATCH"))

    t_prod = bench(lambda: M._run_headshared_sparse_decode(
        q, kc, idx, torch.full((B,), TOPK, dtype=torch.int32, device="cuda"), 0.088))
    print("production kernel : %7.2f ms  %5.1f GB/s" % (t_prod, mb / t_prod))
    print("%-24s %8s %9s" % ("transposed config", "ms", "GB/s"))
    for ncol, warps, stages, bt in ((32, 4, 1, 16), (32, 8, 1, 16), (32, 4, 2, 16),
                                    (32, 4, 3, 16), (32, 4, 4, 16), (32, 8, 2, 16),
                                    (32, 4, 2, 32), (16, 4, 2, 16)):
        try:
            ms = bench(lambda n=ncol, w=warps, s=stages, b=bt: run(
                q, kc, idx, 0.088, out, lse, bh=16, ncol=n, warps=w, stages=s, bt=b))
            print("ncol=%3d w%d st%d bt%d   %8.2f %9.1f" % (ncol, warps, stages, bt, ms, mb / ms))
        except Exception as e:
            print("ncol=%3d w%d st%d bt%d   FAIL %s" % (ncol, warps, stages, bt,
                                                        str(e).splitlines()[-1][:44]))


if __name__ == "__main__":
    main()
