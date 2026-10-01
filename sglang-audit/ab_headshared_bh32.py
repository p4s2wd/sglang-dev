"""Can BLOCK_H=32 fit in SM75's 64 KB if the dots are narrowed?

The head-shared kernel runs grid=(B, cdiv(H,16)); at the real prefill shape H=32
that is two programs per request, each gathering the request's whole selected-KV
set, so the gather is paid twice -- measured at 1.95x (H=32 vs H=16 at B=173).
It is not a bandwidth problem (the kernel sits at 1.6% of DRAM); the second
program issues the same gather instructions again.

BLOCK_H=32 would fix it and needs 92160 B of shared memory against a 64 KB cap.
The obvious suspicion is that Triton stages a full [32,256] fp16 q tile per dot
and keeps both nope halves live at once. Narrowing the dot to 128 columns should
halve the largest live tile, at the cost of twice as many (smaller) mma ops.

Variants, all numerically identical to the shipped kernel:
  bh32_t16          -- plain BLOCK_H=32 (expected to fail, the control)
  bh32_n128         -- nope dot split into four 128-column dots
  bh32_n128_t8      -- same with BLOCK_T=8 (needs M>=16 for tl.dot, so likely rejected)
  bh16_n128         -- split applied at BLOCK_H=16, to price the split itself

Run: python ab_headshared_bh32.py
"""
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

from sglang.kernels.ops.attention import flash_mla_sm120_triton as M  # noqa: E402
import ab_headshared_scale as A  # noqa: E402

LOG2E = tl.constexpr(1.4426950408889634)
# Names the production kernel's body uses, mirrored here as plain ints so the
# copied kernel text compiles unchanged inside this module's globals.
_HS_NOPE = tl.constexpr(448)
_HS_ROPE = tl.constexpr(64)
_HS_TOKEN_BYTES = tl.constexpr(576)
_HS_SCALE_BYTES = tl.constexpr(8)
_HS_GROUP = tl.constexpr(64)
_HS_NOPE_PAD = tl.constexpr(512)

NOPE = tl.constexpr(448)
ROPE = tl.constexpr(64)
TOKB = tl.constexpr(576)
SB = tl.constexpr(8)
GRP = tl.constexpr(64)


@triton.jit
def _load_kv(cache_u8_ptr, lut_ptr, tok_base, cols, n_cols, pg, po, kv_valid,
             page_bytes, scale_off):
    """Gather and dequant one [BLOCK_T, n_cols] slice of the nope half."""
    mask = kv_valid[:, None] & (cols < _HS_NOPE)[None, :]
    byte = tl.load(cache_u8_ptr + tok_base[:, None] + cols[None, :], mask=mask, other=0)
    s_addr = (pg * page_bytes + scale_off + po * _HS_SCALE_BYTES)[:, None] + (cols // _HS_GROUP)[None, :]
    scale = tl.math.exp2(
        tl.load(cache_u8_ptr + s_addr, mask=mask, other=127).to(tl.float32) - 127.0)
    return (tl.load(lut_ptr + byte) * scale).to(tl.float16)


@triton.jit
def _bh_kernel(Q_ptr, cache_u8_ptr, cache_bf16_ptr, lut_ptr, indices_ptr,
               topk_len_ptr, O_ptr, LSE_ptr, softmax_scale, page_size, page_bytes,
               scale_off, H, topk, HAS_TOPK_LEN,
               stride_qb, stride_qh, stride_ob, stride_oh,
               BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr, NCOL: tl.constexpr):
    """Head-shared attention whose nope dots are NCOL wide.

    The 448-wide nope half is covered in NCOL-wide chunks out to 512, with the
    tail dropped by the cols < 448 mask -- the same padding the shipped two-half
    form used, so NCOL=256 reproduces that structure exactly. Narrowing NCOL
    shrinks the largest tile live at once, which is the only way to fit a wider
    BLOCK_H under SM75's 64 KB shared-memory cap; a wider BLOCK_H is what removes
    the kernel's 2x redundant gather.

    Accumulators and query tiles are written out for up to 8 chunks because
    Triton supports neither a list comprehension nor the tuple builtin in @jit;
    the NCHUNK guards make the unused ones dead code.
    """
    bid = tl.program_id(0)
    hblk = tl.program_id(1)
    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    h_valid = offs_h < H
    offs_t = tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, NCOL)
    rope_offs = tl.arange(0, _HS_ROPE)
    q_base = bid * stride_qb + offs_h * stride_qh

    NCHUNK: tl.constexpr = _HS_NOPE_PAD // NCOL
    c0 = 0 * NCOL + offs_c
    q0 = tl.load(Q_ptr + q_base[:, None] + c0[None, :],
                   mask=h_valid[:, None] & (c0 < _HS_NOPE)[None, :], other=0.0)
    c1 = 1 * NCOL + offs_c
    q1 = tl.load(Q_ptr + q_base[:, None] + c1[None, :],
                   mask=h_valid[:, None] & (c1 < _HS_NOPE)[None, :], other=0.0)
    c2 = 2 * NCOL + offs_c
    q2 = tl.load(Q_ptr + q_base[:, None] + c2[None, :],
                   mask=h_valid[:, None] & (c2 < _HS_NOPE)[None, :], other=0.0)
    c3 = 3 * NCOL + offs_c
    q3 = tl.load(Q_ptr + q_base[:, None] + c3[None, :],
                   mask=h_valid[:, None] & (c3 < _HS_NOPE)[None, :], other=0.0)
    c4 = 4 * NCOL + offs_c
    q4 = tl.load(Q_ptr + q_base[:, None] + c4[None, :],
                   mask=h_valid[:, None] & (c4 < _HS_NOPE)[None, :], other=0.0)
    c5 = 5 * NCOL + offs_c
    q5 = tl.load(Q_ptr + q_base[:, None] + c5[None, :],
                   mask=h_valid[:, None] & (c5 < _HS_NOPE)[None, :], other=0.0)
    c6 = 6 * NCOL + offs_c
    q6 = tl.load(Q_ptr + q_base[:, None] + c6[None, :],
                   mask=h_valid[:, None] & (c6 < _HS_NOPE)[None, :], other=0.0)
    c7 = 7 * NCOL + offs_c
    q7 = tl.load(Q_ptr + q_base[:, None] + c7[None, :],
                   mask=h_valid[:, None] & (c7 < _HS_NOPE)[None, :], other=0.0)
    q_r = tl.load(Q_ptr + q_base[:, None] + _HS_NOPE + rope_offs[None, :],
                  mask=h_valid[:, None], other=0.0)

    valid_len = topk
    if HAS_TOPK_LEN:
        valid_len = tl.load(topk_len_ptr + bid).to(tl.int32)

    a0 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a1 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a2 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a3 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a4 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a5 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a6 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a7 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    acc_r = tl.zeros([BLOCK_H, _HS_ROPE], tl.float32)
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)

    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + offs_t
        raw = tl.load(indices_ptr + bid * topk + t_idx,
                      mask=t_idx < topk, other=-1)
        idx_valid = (t_idx < valid_len) & (raw >= 0)
        safe = tl.where(idx_valid, raw, 0).to(tl.int64)
        pg = safe // page_size
        po = safe % page_size
        tok_base = pg * page_bytes + po * _HS_TOKEN_BYTES

        scores = tl.zeros([BLOCK_H, BLOCK_T], tl.float32)
        if NCHUNK >= 1:
            kv0 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 0 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q0, tl.trans(kv0))
        if NCHUNK >= 2:
            kv1 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 1 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q1, tl.trans(kv1))
        if NCHUNK >= 3:
            kv2 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 2 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q2, tl.trans(kv2))
        if NCHUNK >= 4:
            kv3 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 3 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q3, tl.trans(kv3))
        if NCHUNK >= 5:
            kv4 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 4 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q4, tl.trans(kv4))
        if NCHUNK >= 6:
            kv5 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 5 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q5, tl.trans(kv5))
        if NCHUNK >= 7:
            kv6 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 6 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q6, tl.trans(kv6))
        if NCHUNK >= 8:
            kv7 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 7 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q7, tl.trans(kv7))
        rope_base = ((tok_base + _HS_NOPE) // 2).to(tl.int64)
        kv_r = tl.load(cache_bf16_ptr + rope_base[:, None] + rope_offs[None, :],
                       mask=idx_valid[:, None], other=0.0).to(tl.float16)
        scores += tl.dot(q_r, tl.trans(kv_r))

        s = tl.where(idx_valid[None, :],
                     scores * (softmax_scale * LOG2E), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - m_safe))
        p = tl.where(idx_valid[None, :], tl.math.exp2(s - m_safe[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pf = p.to(tl.float16)
        acc_r = acc_r * alpha[:, None] + tl.dot(pf, kv_r)
        m_i = m_new

        if NCHUNK >= 1:
            a0 = a0 * alpha[:, None] + tl.dot(pf, kv0)
        if NCHUNK >= 2:
            a1 = a1 * alpha[:, None] + tl.dot(pf, kv1)
        if NCHUNK >= 3:
            a2 = a2 * alpha[:, None] + tl.dot(pf, kv2)
        if NCHUNK >= 4:
            a3 = a3 * alpha[:, None] + tl.dot(pf, kv3)
        if NCHUNK >= 5:
            a4 = a4 * alpha[:, None] + tl.dot(pf, kv4)
        if NCHUNK >= 6:
            a5 = a5 * alpha[:, None] + tl.dot(pf, kv5)
        if NCHUNK >= 7:
            a6 = a6 * alpha[:, None] + tl.dot(pf, kv6)
        if NCHUNK >= 8:
            a7 = a7 * alpha[:, None] + tl.dot(pf, kv7)
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    o_base = bid * stride_ob + offs_h * stride_oh
    if NCHUNK >= 1:
        tl.store(O_ptr + o_base[:, None] + c0[None, :],
                 (a0 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c0 < _HS_NOPE)[None, :])
    if NCHUNK >= 2:
        tl.store(O_ptr + o_base[:, None] + c1[None, :],
                 (a1 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c1 < _HS_NOPE)[None, :])
    if NCHUNK >= 3:
        tl.store(O_ptr + o_base[:, None] + c2[None, :],
                 (a2 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c2 < _HS_NOPE)[None, :])
    if NCHUNK >= 4:
        tl.store(O_ptr + o_base[:, None] + c3[None, :],
                 (a3 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c3 < _HS_NOPE)[None, :])
    if NCHUNK >= 5:
        tl.store(O_ptr + o_base[:, None] + c4[None, :],
                 (a4 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c4 < _HS_NOPE)[None, :])
    if NCHUNK >= 6:
        tl.store(O_ptr + o_base[:, None] + c5[None, :],
                 (a5 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c5 < _HS_NOPE)[None, :])
    if NCHUNK >= 7:
        tl.store(O_ptr + o_base[:, None] + c6[None, :],
                 (a6 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c6 < _HS_NOPE)[None, :])
    if NCHUNK >= 8:
        tl.store(O_ptr + o_base[:, None] + c7[None, :],
                 (a7 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c7 < _HS_NOPE)[None, :])
    tl.store(O_ptr + o_base[:, None] + _HS_NOPE + rope_offs[None, :],
             (acc_r / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None])
    tl.store(LSE_ptr + bid * H + offs_h,
             tl.where(l_i > 0.0, m_i / tl.constexpr(1.4426950408889634) + tl.math.log(safe_l), float("-inf")),
             mask=h_valid)


@triton.jit
def _bh_kernel_nohoist(Q_ptr, cache_u8_ptr, cache_bf16_ptr, lut_ptr, indices_ptr,
               topk_len_ptr, O_ptr, LSE_ptr, softmax_scale, page_size, page_bytes,
               scale_off, H, topk, HAS_TOPK_LEN,
               stride_qb, stride_qh, stride_ob, stride_oh,
               BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr, NCOL: tl.constexpr):
    """Head-shared attention whose nope dots are NCOL wide.

    The 448-wide nope half is covered in NCOL-wide chunks out to 512, with the
    tail dropped by the cols < 448 mask -- the same padding the shipped two-half
    form used, so NCOL=256 reproduces that structure exactly. Narrowing NCOL
    shrinks the largest tile live at once, which is the only way to fit a wider
    BLOCK_H under SM75's 64 KB shared-memory cap; a wider BLOCK_H is what removes
    the kernel's 2x redundant gather.

    Accumulators and query tiles are written out for up to 8 chunks because
    Triton supports neither a list comprehension nor the tuple builtin in @jit;
    the NCHUNK guards make the unused ones dead code.
    """
    bid = tl.program_id(0)
    hblk = tl.program_id(1)
    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    h_valid = offs_h < H
    offs_t = tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, NCOL)
    rope_offs = tl.arange(0, _HS_ROPE)
    q_base = bid * stride_qb + offs_h * stride_qh

    NCHUNK: tl.constexpr = _HS_NOPE_PAD // NCOL
    c0 = 0 * NCOL + offs_c
    q0 = tl.load(Q_ptr + q_base[:, None] + c0[None, :],
                   mask=h_valid[:, None] & (c0 < _HS_NOPE)[None, :], other=0.0)
    c1 = 1 * NCOL + offs_c
    q1 = tl.load(Q_ptr + q_base[:, None] + c1[None, :],
                   mask=h_valid[:, None] & (c1 < _HS_NOPE)[None, :], other=0.0)
    c2 = 2 * NCOL + offs_c
    q2 = tl.load(Q_ptr + q_base[:, None] + c2[None, :],
                   mask=h_valid[:, None] & (c2 < _HS_NOPE)[None, :], other=0.0)
    c3 = 3 * NCOL + offs_c
    q3 = tl.load(Q_ptr + q_base[:, None] + c3[None, :],
                   mask=h_valid[:, None] & (c3 < _HS_NOPE)[None, :], other=0.0)
    c4 = 4 * NCOL + offs_c
    q4 = tl.load(Q_ptr + q_base[:, None] + c4[None, :],
                   mask=h_valid[:, None] & (c4 < _HS_NOPE)[None, :], other=0.0)
    c5 = 5 * NCOL + offs_c
    q5 = tl.load(Q_ptr + q_base[:, None] + c5[None, :],
                   mask=h_valid[:, None] & (c5 < _HS_NOPE)[None, :], other=0.0)
    c6 = 6 * NCOL + offs_c
    q6 = tl.load(Q_ptr + q_base[:, None] + c6[None, :],
                   mask=h_valid[:, None] & (c6 < _HS_NOPE)[None, :], other=0.0)
    c7 = 7 * NCOL + offs_c
    q7 = tl.load(Q_ptr + q_base[:, None] + c7[None, :],
                   mask=h_valid[:, None] & (c7 < _HS_NOPE)[None, :], other=0.0)
    valid_len = topk
    if HAS_TOPK_LEN:
        valid_len = tl.load(topk_len_ptr + bid).to(tl.int32)

    a0 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a1 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a2 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a3 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a4 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a5 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a6 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a7 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    acc_r = tl.zeros([BLOCK_H, _HS_ROPE], tl.float32)
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)

    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + offs_t
        raw = tl.load(indices_ptr + bid * topk + t_idx,
                      mask=t_idx < topk, other=-1)
        idx_valid = (t_idx < valid_len) & (raw >= 0)
        safe = tl.where(idx_valid, raw, 0).to(tl.int64)
        pg = safe // page_size
        po = safe % page_size
        tok_base = pg * page_bytes + po * _HS_TOKEN_BYTES

        scores = tl.zeros([BLOCK_H, BLOCK_T], tl.float32)
            q0 = tl.load(Q_ptr + q_base[:, None] + (0 * NCOL + offs_c)[None, :],
                           mask=h_valid[:, None] & (0 * NCOL + offs_c < _HS_NOPE)[None, :],
                           other=0.0)
            q1 = tl.load(Q_ptr + q_base[:, None] + (1 * NCOL + offs_c)[None, :],
                           mask=h_valid[:, None] & (1 * NCOL + offs_c < _HS_NOPE)[None, :],
                           other=0.0)
            q2 = tl.load(Q_ptr + q_base[:, None] + (2 * NCOL + offs_c)[None, :],
                           mask=h_valid[:, None] & (2 * NCOL + offs_c < _HS_NOPE)[None, :],
                           other=0.0)
            q3 = tl.load(Q_ptr + q_base[:, None] + (3 * NCOL + offs_c)[None, :],
                           mask=h_valid[:, None] & (3 * NCOL + offs_c < _HS_NOPE)[None, :],
                           other=0.0)
            q4 = tl.load(Q_ptr + q_base[:, None] + (4 * NCOL + offs_c)[None, :],
                           mask=h_valid[:, None] & (4 * NCOL + offs_c < _HS_NOPE)[None, :],
                           other=0.0)
            q5 = tl.load(Q_ptr + q_base[:, None] + (5 * NCOL + offs_c)[None, :],
                           mask=h_valid[:, None] & (5 * NCOL + offs_c < _HS_NOPE)[None, :],
                           other=0.0)
            q6 = tl.load(Q_ptr + q_base[:, None] + (6 * NCOL + offs_c)[None, :],
                           mask=h_valid[:, None] & (6 * NCOL + offs_c < _HS_NOPE)[None, :],
                           other=0.0)
            q7 = tl.load(Q_ptr + q_base[:, None] + (7 * NCOL + offs_c)[None, :],
                           mask=h_valid[:, None] & (7 * NCOL + offs_c < _HS_NOPE)[None, :],
                           other=0.0)
        if NCHUNK >= 1:
            kv0 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 0 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q0, tl.trans(kv0))
        if NCHUNK >= 2:
            kv1 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 1 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q1, tl.trans(kv1))
        if NCHUNK >= 3:
            kv2 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 2 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q2, tl.trans(kv2))
        if NCHUNK >= 4:
            kv3 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 3 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q3, tl.trans(kv3))
        if NCHUNK >= 5:
            kv4 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 4 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q4, tl.trans(kv4))
        if NCHUNK >= 6:
            kv5 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 5 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q5, tl.trans(kv5))
        if NCHUNK >= 7:
            kv6 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 6 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q6, tl.trans(kv6))
        if NCHUNK >= 8:
            kv7 = _load_kv(cache_u8_ptr, lut_ptr, tok_base, 7 * NCOL + offs_c,
                             NCOL, pg, po, idx_valid, page_bytes, scale_off)
            scores += tl.dot(q7, tl.trans(kv7))
        rope_base = ((tok_base + _HS_NOPE) // 2).to(tl.int64)
        kv_r = tl.load(cache_bf16_ptr + rope_base[:, None] + rope_offs[None, :],
                       mask=idx_valid[:, None], other=0.0).to(tl.float16)
        q_r = tl.load(Q_ptr + q_base[:, None] + _HS_NOPE + rope_offs[None, :],
                       mask=h_valid[:, None], other=0.0)
        scores += tl.dot(q_r, tl.trans(kv_r))

        s = tl.where(idx_valid[None, :],
                     scores * (softmax_scale * LOG2E), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - m_safe))
        p = tl.where(idx_valid[None, :], tl.math.exp2(s - m_safe[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pf = p.to(tl.float16)
        acc_r = acc_r * alpha[:, None] + tl.dot(pf, kv_r)
        m_i = m_new

        if NCHUNK >= 1:
            a0 = a0 * alpha[:, None] + tl.dot(pf, kv0)
        if NCHUNK >= 2:
            a1 = a1 * alpha[:, None] + tl.dot(pf, kv1)
        if NCHUNK >= 3:
            a2 = a2 * alpha[:, None] + tl.dot(pf, kv2)
        if NCHUNK >= 4:
            a3 = a3 * alpha[:, None] + tl.dot(pf, kv3)
        if NCHUNK >= 5:
            a4 = a4 * alpha[:, None] + tl.dot(pf, kv4)
        if NCHUNK >= 6:
            a5 = a5 * alpha[:, None] + tl.dot(pf, kv5)
        if NCHUNK >= 7:
            a6 = a6 * alpha[:, None] + tl.dot(pf, kv6)
        if NCHUNK >= 8:
            a7 = a7 * alpha[:, None] + tl.dot(pf, kv7)
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    o_base = bid * stride_ob + offs_h * stride_oh
    if NCHUNK >= 1:
        tl.store(O_ptr + o_base[:, None] + c0[None, :],
                 (a0 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c0 < _HS_NOPE)[None, :])
    if NCHUNK >= 2:
        tl.store(O_ptr + o_base[:, None] + c1[None, :],
                 (a1 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c1 < _HS_NOPE)[None, :])
    if NCHUNK >= 3:
        tl.store(O_ptr + o_base[:, None] + c2[None, :],
                 (a2 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c2 < _HS_NOPE)[None, :])
    if NCHUNK >= 4:
        tl.store(O_ptr + o_base[:, None] + c3[None, :],
                 (a3 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c3 < _HS_NOPE)[None, :])
    if NCHUNK >= 5:
        tl.store(O_ptr + o_base[:, None] + c4[None, :],
                 (a4 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c4 < _HS_NOPE)[None, :])
    if NCHUNK >= 6:
        tl.store(O_ptr + o_base[:, None] + c5[None, :],
                 (a5 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c5 < _HS_NOPE)[None, :])
    if NCHUNK >= 7:
        tl.store(O_ptr + o_base[:, None] + c6[None, :],
                 (a6 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c6 < _HS_NOPE)[None, :])
    if NCHUNK >= 8:
        tl.store(O_ptr + o_base[:, None] + c7[None, :],
                 (a7 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (c7 < _HS_NOPE)[None, :])
    tl.store(O_ptr + o_base[:, None] + _HS_NOPE + rope_offs[None, :],
             (acc_r / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None])
    tl.store(LSE_ptr + bid * H + offs_h,
             tl.where(l_i > 0.0, m_i / tl.constexpr(1.4426950408889634) + tl.math.log(safe_l), float("-inf")),
             mask=h_valid)



def run_bh(q, kc, flat, scale, out, lse, bh, ncol, bt=16, warps=8, nohoist=0):
    """BLOCK_T must stay >=16: tl.dot lowers to mma, which needs M>=16."""
    B, _, H, D = q.shape
    page = kc.shape[1]
    page_bytes = kc.stride(0)
    total = kc.shape[0] * page_bytes
    raw_u8 = kc.as_strided((total,), (1,)).view(torch.uint8)
    raw_bf16 = raw_u8.view(torch.bfloat16)
    lut = M.fp8_payload_lut(q.device, torch.float32)
    q3 = q.squeeze(1).contiguous()
    grid = (B, triton.cdiv(H, bh))
    _k = _bh_kernel_nohoist if nohoist else _bh_kernel
    _k[grid](q3, raw_u8, raw_bf16, lut, flat,
                     torch.empty(0, device=q.device, dtype=torch.int32),
                     out, lse, scale, page, int(page_bytes), int(page * _HS_TOKEN_BYTES),
                     H, flat.shape[1], False,
                     q3.stride(0), q3.stride(1),
                     out.stride(0), out.stride(1),
                     BLOCK_H=bh, BLOCK_T=bt, NCOL=ncol,
                     **(dict() if not nohoist else {}),
                     num_warps=warps, num_stages=1)
    return out, lse


def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    B, H, D, topk = 173, 32, 512, 512
    q, kc, flat = A.build(B, H, D, topk, num_tokens=B * topk)
    scale = 1.0 / (D ** 0.5)
    out = torch.zeros(B, H, D, dtype=q.dtype, device=q.device)
    lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)

    prod, _ = M._run_headshared_sparse_decode(q, kc, flat, None, scale)
    torch.cuda.synchronize()
    ref = prod.squeeze(1).clone()

    # 2x2 factorial: the shipped kernel is (n256, warps=8), so attributing a win
    # to the narrower dot requires holding warps fixed, and vice versa.
    cfgs = [("n256 w8 (shipped)", dict(bh=16, ncol=256, warps=8)),
            ("n256 w4", dict(bh=16, ncol=256, warps=4)),
            ("n128 w8", dict(bh=16, ncol=128, warps=8)),
            ("n128 w4", dict(bh=16, ncol=128, warps=4)),
            ("n128 w2", dict(bh=16, ncol=128, warps=2)),
            ("n256 w2", dict(bh=16, ncol=256, warps=2))]

    def one(**kw):
        out.zero_()
        run_bh(q, kc, flat, scale, out, lse, **kw)
        torch.cuda.synchronize()

    print("shared-memory requirement per variant:")
    ok = []
    for name, kw in cfgs:
        try:
            one(**kw)
            rel = ((out.float() - ref.float()).abs().max().item()
                   / max(ref.abs().max().item(), 1e-6))
            print("  %-28s compiled   rel %.1e%s"
                  % (name, rel, "" if rel < 1e-2 else "  <-- MISMATCH"))
            if rel < 1e-2:
                ok.append((name, kw))
        except Exception as e:
            print("  %-28s %s" % (name, str(e).splitlines()[0][:64]))

    if not ok:
        print("\nnothing compiled")
        return
    ROUNDS = 7
    res = {n: [] for n, _ in ok}
    for r in range(ROUNDS):
        for name, kw in ok:
            res[name].append(timeit(lambda: one(**kw), iters=8, warmup=2))
    base = sorted(res[ok[0][0]])[ROUNDS // 2]
    print("\nmedian of %d interleaved rounds:" % ROUNDS)
    for name, _ in ok:
        v = sorted(res[name])
        med = v[ROUNDS // 2]
        print("  %-28s %7.3f ms  (%5.3fx)  spread %.1f%%"
              % (name, med, base / med, 100 * (v[-1] - v[0]) / med))


if __name__ == "__main__":
    main()
