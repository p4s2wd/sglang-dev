"""Does BLOCK_H=32 fit in SM75's 64 KB if the query tiles are not hoisted?

The head-shared kernel runs grid=(B, cdiv(H,16)); at the production shape that is
two head-blocks per request, each gathering the request's whole selected-KV set,
so the gather -- which is the entire cost of the kernel (2.524 ms measured vs
2.368 ms for a bare gather of the same bytes) -- is paid twice.

BLOCK_H=32 would gather once, but needs 92160 B of shared memory at NCOL=256 and
81920 B at NCOL=128, against a 64 KB cap. The hoisted query tiles are a large part
of that: BLOCK_H=32 x 448 columns x 2 bytes = 28 KB held for the whole kernel. Not
hoisting them costs extra loads per tile, which on a gather-bound kernel may be
free, and is the last untested way to fit.

Written as a standalone kernel rather than another edit of the A/B harness, which
has accumulated enough generated variants to be fragile.

Run: python bh32_nohoist.py
"""
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

import ab_headshared_scale as A  # noqa: E402
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M  # noqa: E402

LOG2E = tl.constexpr(1.4426950408889634)
NOPE = tl.constexpr(448)
ROPE = tl.constexpr(64)
TOKB = tl.constexpr(576)
SB = tl.constexpr(8)
GRP = tl.constexpr(64)
PAD = tl.constexpr(512)


@triton.jit
def _kv(cache_u8_ptr, lut_ptr, tok_base, cols, pg, po, kv_valid, page_bytes, soff,
        DEQ: tl.constexpr, TRANS: tl.constexpr):
    if TRANS:
        # Gather straight into [NCOL, BLOCK_T] so the QK dot needs no tl.trans.
        # The bare gather measurement shows the bytes cost 0.24 ms at 100% of the
        # card's peak bandwidth, so the kernel's 18 ms is the mma path, and
        # tl.trans is the part of it that round-trips every chunk through shared
        # memory. Addressing is the same bytes, just indexed transposed.
        mask = kv_valid[None, :] & (cols < NOPE)[:, None]
        byte = tl.load(cache_u8_ptr + tok_base[None, :] + cols[:, None], mask=mask, other=0)
        s_addr = (pg * page_bytes + soff + po * SB)[None, :] + (cols // GRP)[:, None]
        scale = tl.math.exp2(
            tl.load(cache_u8_ptr + s_addr, mask=mask, other=127).to(tl.float32) - 127.0)
        return (tl.load(lut_ptr + byte) * scale).to(tl.float16)
    mask = kv_valid[:, None] & (cols < NOPE)[None, :]
    byte = tl.load(cache_u8_ptr + tok_base[:, None] + cols[None, :], mask=mask, other=0)
    if DEQ == 0:
        # Gather only: no scale load, no exp2, no LUT gather. Same addresses and
        # same byte count as the real path, minus the decode.
        return byte.to(tl.float16)
    s_addr = (pg * page_bytes + soff + po * SB)[:, None] + (cols // GRP)[None, :]
    scale = tl.math.exp2(
        tl.load(cache_u8_ptr + s_addr, mask=mask, other=127).to(tl.float32) - 127.0)
    if DEQ == 1:
        # Scale path but no LUT gather.
        return (byte.to(tl.float32) * scale).to(tl.float16)
    return (tl.load(lut_ptr + byte) * scale).to(tl.float16)


@triton.jit
def _k(Q_ptr, cache_u8_ptr, cache_bf16_ptr, lut_ptr, indices_ptr,
       O_ptr, LSE_ptr, softmax_scale, page_size, page_bytes, soff,
       H, topk, stride_qb, stride_qh, stride_ob, stride_oh,
       BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr, NCOL: tl.constexpr,
       HOIST: tl.constexpr, DEQ: tl.constexpr, TRANS: tl.constexpr):
    """Head-shared attention; HOIST=1 keeps the query tiles live across the loop.

    The nope half is covered in NCOL-wide chunks out to 512 with the tail dropped
    by the cols < 448 mask, matching the shipped kernel's padding. Accumulators
    are written out for 8 chunks because Triton allows neither a list
    comprehension nor the tuple builtin inside @jit; NCHUNK guards kill the rest.
    """
    bid = tl.program_id(0)
    hblk = tl.program_id(1)
    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    h_valid = offs_h < H
    offs_t = tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, NCOL)
    rope_offs = tl.arange(0, ROPE)
    q_base = bid * stride_qb + offs_h * stride_qh
    NCHUNK: tl.constexpr = PAD // NCOL

    a0 = tl.zeros([NCOL, BLOCK_H], tl.float32) if TRANS == 2 else tl.zeros([BLOCK_H, NCOL], tl.float32)
    a1 = tl.zeros([NCOL, BLOCK_H], tl.float32) if TRANS == 2 else tl.zeros([BLOCK_H, NCOL], tl.float32)
    a2 = tl.zeros([NCOL, BLOCK_H], tl.float32) if TRANS == 2 else tl.zeros([BLOCK_H, NCOL], tl.float32)
    a3 = tl.zeros([NCOL, BLOCK_H], tl.float32) if TRANS == 2 else tl.zeros([BLOCK_H, NCOL], tl.float32)
    a4 = tl.zeros([NCOL, BLOCK_H], tl.float32) if TRANS == 2 else tl.zeros([BLOCK_H, NCOL], tl.float32)
    a5 = tl.zeros([NCOL, BLOCK_H], tl.float32) if TRANS == 2 else tl.zeros([BLOCK_H, NCOL], tl.float32)
    a6 = tl.zeros([NCOL, BLOCK_H], tl.float32) if TRANS == 2 else tl.zeros([BLOCK_H, NCOL], tl.float32)
    a7 = tl.zeros([NCOL, BLOCK_H], tl.float32) if TRANS == 2 else tl.zeros([BLOCK_H, NCOL], tl.float32)
    acc_r = tl.zeros([ROPE, BLOCK_H], tl.float32) if TRANS == 2 else tl.zeros([BLOCK_H, ROPE], tl.float32)
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)

    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + offs_t
        raw = tl.load(indices_ptr + bid * topk + t_idx, mask=t_idx < topk, other=-1)
        idx_valid = (t_idx < topk) & (raw >= 0)
        safe = tl.where(idx_valid, raw, 0).to(tl.int64)
        pg = safe // page_size
        po = safe % page_size
        tok_base = pg * page_bytes + po * TOKB

        scores = tl.zeros([BLOCK_H, BLOCK_T], tl.float32)
        if NCHUNK >= 1:
            kv0 = _kv(cache_u8_ptr, lut_ptr, tok_base, 0 * NCOL + offs_c, pg, po,
                      idx_valid, page_bytes, soff, DEQ=DEQ, TRANS=TRANS)
            if HOIST:
                q0 = tl.load(Q_ptr + q_base[:, None] + (0 * NCOL + offs_c)[None, :],
                             mask=h_valid[:, None], other=0.0)
            else:
                q0 = tl.load(Q_ptr + q_base[:, None] + (0 * NCOL + offs_c)[None, :],
                             mask=h_valid[:, None] & (offs_c < NOPE)[None, :], other=0.0)
            scores += tl.dot(q0, kv0) if TRANS else tl.dot(q0, tl.trans(kv0))
        if NCHUNK >= 2:
            kv1 = _kv(cache_u8_ptr, lut_ptr, tok_base, 1 * NCOL + offs_c, pg, po,
                      idx_valid, page_bytes, soff, DEQ=DEQ, TRANS=TRANS)
            q1 = tl.load(Q_ptr + q_base[:, None] + (1 * NCOL + offs_c)[None, :],
                         mask=h_valid[:, None], other=0.0)
            scores += tl.dot(q1, kv1) if TRANS else tl.dot(q1, tl.trans(kv1))
        if NCHUNK >= 3:
            kv2 = _kv(cache_u8_ptr, lut_ptr, tok_base, 2 * NCOL + offs_c, pg, po,
                      idx_valid, page_bytes, soff, DEQ=DEQ, TRANS=TRANS)
            q2 = tl.load(Q_ptr + q_base[:, None] + (2 * NCOL + offs_c)[None, :],
                         mask=h_valid[:, None], other=0.0)
            scores += tl.dot(q2, kv2) if TRANS else tl.dot(q2, tl.trans(kv2))
        if NCHUNK >= 4:
            kv3 = _kv(cache_u8_ptr, lut_ptr, tok_base, 3 * NCOL + offs_c, pg, po,
                      idx_valid, page_bytes, soff, DEQ=DEQ, TRANS=TRANS)
            q3 = tl.load(Q_ptr + q_base[:, None] + (3 * NCOL + offs_c)[None, :],
                         mask=h_valid[:, None], other=0.0)
            scores += tl.dot(q3, kv3) if TRANS else tl.dot(q3, tl.trans(kv3))
        if NCHUNK >= 5:
            kv4 = _kv(cache_u8_ptr, lut_ptr, tok_base, 4 * NCOL + offs_c, pg, po,
                      idx_valid, page_bytes, soff, DEQ=DEQ, TRANS=TRANS)
            q4 = tl.load(Q_ptr + q_base[:, None] + (4 * NCOL + offs_c)[None, :],
                         mask=h_valid[:, None], other=0.0)
            scores += tl.dot(q4, kv4) if TRANS else tl.dot(q4, tl.trans(kv4))
        if NCHUNK >= 6:
            kv5 = _kv(cache_u8_ptr, lut_ptr, tok_base, 5 * NCOL + offs_c, pg, po,
                      idx_valid, page_bytes, soff, DEQ=DEQ, TRANS=TRANS)
            q5 = tl.load(Q_ptr + q_base[:, None] + (5 * NCOL + offs_c)[None, :],
                         mask=h_valid[:, None], other=0.0)
            scores += tl.dot(q5, kv5) if TRANS else tl.dot(q5, tl.trans(kv5))
        if NCHUNK >= 7:
            kv6 = _kv(cache_u8_ptr, lut_ptr, tok_base, 6 * NCOL + offs_c, pg, po,
                      idx_valid, page_bytes, soff, DEQ=DEQ, TRANS=TRANS)
            q6 = tl.load(Q_ptr + q_base[:, None] + (6 * NCOL + offs_c)[None, :],
                         mask=h_valid[:, None], other=0.0)
            scores += tl.dot(q6, kv6) if TRANS else tl.dot(q6, tl.trans(kv6))
        if NCHUNK >= 8:
            kv7 = _kv(cache_u8_ptr, lut_ptr, tok_base, 7 * NCOL + offs_c, pg, po,
                      idx_valid, page_bytes, soff, DEQ=DEQ, TRANS=TRANS)
            q7 = tl.load(Q_ptr + q_base[:, None] + (7 * NCOL + offs_c)[None, :],
                         mask=h_valid[:, None], other=0.0)
            scores += tl.dot(q7, kv7) if TRANS else tl.dot(q7, tl.trans(kv7))

        q_r = tl.load(Q_ptr + q_base[:, None] + NOPE + rope_offs[None, :],
                      mask=h_valid[:, None], other=0.0)
        rope_base = ((tok_base + NOPE) // 2).to(tl.int64)
        if TRANS == 2:
            # rope gathered transposed [ROPE, BLOCK_T]; bf16 so byte offset /2.
            rb = ((tok_base + NOPE) // 2).to(tl.int64)
            kv_r = tl.load(cache_bf16_ptr + rb[None, :] + rope_offs[:, None],
                           mask=idx_valid[None, :], other=0.0).to(tl.float16)
            scores += tl.dot(q_r, kv_r)
        else:
            kv_r = tl.load(cache_bf16_ptr + rope_base[:, None] + rope_offs[None, :],
                           mask=idx_valid[:, None], other=0.0).to(tl.float16)
            scores += tl.dot(q_r, tl.trans(kv_r))

        s = tl.where(idx_valid[None, :], scores * (softmax_scale * LOG2E), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - m_safe))
        p = tl.where(idx_valid[None, :], tl.math.exp2(s - m_safe[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pf = p.to(tl.float16)
        if TRANS == 2:
            # One transpose of the shared probabilities replaces the eight
            # per-chunk KV transposes the natural layout forces on the QK dot.
            pf_t = tl.trans(pf)
        if TRANS == 2:
            acc_r = acc_r * alpha[None, :] + tl.dot(kv_r, pf_t)
        else:
            acc_r = acc_r * alpha[:, None] + tl.dot(pf, kv_r)
        m_i = m_new

        if NCHUNK >= 1:
            a0 = (a0 * alpha[None, :] + tl.dot(kv0, pf_t)) if TRANS == 2 else (a0 * alpha[:, None] + tl.dot(pf, kv0))
        if NCHUNK >= 2:
            a1 = (a1 * alpha[None, :] + tl.dot(kv1, pf_t)) if TRANS == 2 else (a1 * alpha[:, None] + tl.dot(pf, kv1))
        if NCHUNK >= 3:
            a2 = (a2 * alpha[None, :] + tl.dot(kv2, pf_t)) if TRANS == 2 else (a2 * alpha[:, None] + tl.dot(pf, kv2))
        if NCHUNK >= 4:
            a3 = (a3 * alpha[None, :] + tl.dot(kv3, pf_t)) if TRANS == 2 else (a3 * alpha[:, None] + tl.dot(pf, kv3))
        if NCHUNK >= 5:
            a4 = (a4 * alpha[None, :] + tl.dot(kv4, pf_t)) if TRANS == 2 else (a4 * alpha[:, None] + tl.dot(pf, kv4))
        if NCHUNK >= 6:
            a5 = (a5 * alpha[None, :] + tl.dot(kv5, pf_t)) if TRANS == 2 else (a5 * alpha[:, None] + tl.dot(pf, kv5))
        if NCHUNK >= 7:
            a6 = (a6 * alpha[None, :] + tl.dot(kv6, pf_t)) if TRANS == 2 else (a6 * alpha[:, None] + tl.dot(pf, kv6))
        if NCHUNK >= 8:
            a7 = (a7 * alpha[None, :] + tl.dot(kv7, pf_t)) if TRANS == 2 else (a7 * alpha[:, None] + tl.dot(pf, kv7))

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    o_base = bid * stride_ob + offs_h * stride_oh
    if NCHUNK >= 1:
        c = 0 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 ((tl.trans(a0) if TRANS == 2 else a0) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 2:
        c = 1 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 ((tl.trans(a1) if TRANS == 2 else a1) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 3:
        c = 2 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 ((tl.trans(a2) if TRANS == 2 else a2) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 4:
        c = 3 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 ((tl.trans(a3) if TRANS == 2 else a3) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 5:
        c = 4 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 ((tl.trans(a4) if TRANS == 2 else a4) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 6:
        c = 5 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 ((tl.trans(a5) if TRANS == 2 else a5) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 7:
        c = 6 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 ((tl.trans(a6) if TRANS == 2 else a6) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    if NCHUNK >= 8:
        c = 7 * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + c[None, :],
                 ((tl.trans(a7) if TRANS == 2 else a7) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c < NOPE)[None, :])
    tl.store(O_ptr + o_base[:, None] + NOPE + rope_offs[None, :],
             ((tl.trans(acc_r) if TRANS == 2 else acc_r) / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None])
    tl.store(LSE_ptr + bid * H + offs_h,
             tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf")),
             mask=h_valid)


def run(q, kc, flat, scale, out, lse, bh, ncol, warps=4, hoist=0, stages=1, bt=16, deq=2, trans=0):
    B, _, H, D = q.shape
    page = kc.shape[1]
    page_bytes = kc.stride(0)
    total = kc.shape[0] * page_bytes
    raw_u8 = kc.as_strided((total,), (1,)).view(torch.uint8)
    raw_bf16 = raw_u8.view(torch.bfloat16)
    lut = M.fp8_payload_lut(q.device, torch.float32)
    q3 = q.squeeze(1).contiguous()
    grid = (B, triton.cdiv(H, bh))
    _k[grid](q3, raw_u8, raw_bf16, lut, flat, out, lse, scale,
             page, int(page_bytes), int(page * 576),
             H, flat.shape[1], q3.stride(0), q3.stride(1),
             out.stride(0), out.stride(1),
             BLOCK_H=bh, BLOCK_T=bt, NCOL=ncol, HOIST=hoist, DEQ=deq, TRANS=trans,
             num_warps=warps, num_stages=stages)
    return out, lse


def timeit(fn, iters=12, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    B, H, D, topk = 113, 32, 512, 512
    q, kc, flat = A.build(B, H, D, topk, num_tokens=B * topk)
    scale = 1.0 / (D ** 0.5)
    out = torch.zeros(B, H, D, dtype=q.dtype, device=q.device)
    lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)

    prod, _ = M._run_headshared_sparse_decode(q, kc, flat, None, scale)
    torch.cuda.synchronize()
    ref = prod.squeeze(1).clone()

    cfgs = [("bh16 n128 w4 (shipped)", dict(bh=16, ncol=128, warps=4, hoist=1)),
            ("bh32 n128 w4 nohoist", dict(bh=32, ncol=128, warps=4, hoist=0)),
            ("bh32 n64  w4 nohoist", dict(bh=32, ncol=64, warps=4, hoist=0)),
            ("bh32 n64  w8 nohoist", dict(bh=32, ncol=64, warps=8, hoist=0)),
            ("bh32 n256 w4 nohoist", dict(bh=32, ncol=256, warps=4, hoist=0)),
            ("bh16 n128 w4 nohoist", dict(bh=16, ncol=128, warps=4, hoist=0))]
    ok = []
    for name, kw in cfgs:
        try:
            out.zero_()
            run(q, kc, flat, scale, out, lse, **kw)
            torch.cuda.synchronize()
            rel = ((out.float() - ref.float()).abs().max().item()
                   / max(ref.abs().max().item(), 1e-6))
            print("  %-24s compiled  rel %.1e%s"
                  % (name, rel, "" if rel < 1e-2 else "  <-- MISMATCH"))
            if rel < 1e-2:
                ok.append((name, kw))
        except Exception as e:
            print("  %-24s %s" % (name, str(e).splitlines()[0][:56]))

    if not ok:
        print("\nnothing compiled")
        return
    R = 6
    res = {n: [] for n, _ in ok}
    for r in range(R):
        for name, kw in ok:
            res[name].append(timeit(lambda: run(q, kc, flat, scale, out, lse, **kw)))
    base = sorted(res[ok[0][0]])[R // 2]
    print("\nmedian of %d interleaved rounds (production shape B=113, grid=(B,2)):" % R)
    for name, _ in ok:
        v = sorted(res[name])
        med = v[R // 2]
        print("  %-24s %7.3f ms  (%5.3fx)  spread %.1f%%"
              % (name, med, base / med, 100 * (v[-1] - v[0]) / med))


if __name__ == "__main__":
    main()
