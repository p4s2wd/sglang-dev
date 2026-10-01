"""Head-shared attention, orientation-flipped: coalesced gather, one transpose.

WHERE THIS CAME FROM. Attention is 31% of prefill device time. Three measurements
set up this design:

  1. A bare gather of the 151 MB a call touches runs in 0.24 ms = 614 GB/s, 100%
     of the card's 616 GB/s peak. The gather is free when it is coalesced.
  2. tl.trans is expensive: a QK-only probe measured 13.47 ms transposing eight
     KV chunks per tile against 3.45 ms gathering them already transposed.
  3. But the transposed gather is UNCOALESCED -- with the token axis last,
     consecutive lanes read bytes 576 apart -- so it costs 3.45 ms instead of the
     0.24 ms the natural layout achieves. The shipped commit (fcb22c9d9d) took
     that trade because it still beat tl.trans: 15.6 -> 10.8 ms.

The attribution of that kernel: QK+softmax 7.4 ms, PV dots 0.4 ms, second gather
3.4 ms. Both halves pay for the uncoalesced access.

THE FLIP. scores^T instead of scores. Gather KV in the natural, coalesced layout
and compute  scores_t[T,H] = dot(kv[T,NCOL], q_t[NCOL,H]). The softmax runs along
axis 0. PV then wants dot(p[H,T], kv[T,NCOL]), so p is transposed ONCE per tile --
a 16x16 fp16 tile, against the eight 16x64 KV transposes the old kernel paid --
and the natural KV tiles are reused from registers, so there is no second gather
at all. Every access is coalesced and the only shared-memory round-trip is the
small probability tile.

Correctness is checked against test_headshared_mla.py's fp32 reference, the same
one production scores 3.9e-04 against.

Run: python hs_flip.py [pool_tokens]
"""
import sys
import time

sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

import torch
import triton
import triton.language as tl

import ab_headshared_scale as A
import test_headshared_mla as T
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

LOG2E = tl.constexpr(1.4426950408889634)
TOKB = tl.constexpr(576)
SB = tl.constexpr(8)
GRP = tl.constexpr(64)
NOPE = tl.constexpr(448)
ROPE = tl.constexpr(64)


@triton.jit
def _kv_n(cu8, lut, tok_base, cols, soff, valid):
    """Gather [BLOCK_T, NCOL] nope: token axis first, so lanes walk contiguous
    bytes inside a token. This is the coalesced layout."""
    m = valid[:, None] & (cols < NOPE)[None, :]
    byte = tl.load(cu8 + tok_base[:, None] + cols[None, :], mask=m, other=0)
    sa = soff[:, None] + (cols // GRP)[None, :]
    scale = tl.math.exp2(tl.load(cu8 + sa, mask=m, other=127).to(tl.float32) - 127.0)
    return (tl.load(lut + byte) * scale).to(tl.float16)


@triton.jit
def _k(Q_ptr, cu8, cu16, lut, idx_ptr, O_ptr, LSE_ptr, softmax_scale,
       page_size, page_bytes, soff_base, H, topk,
       stride_qb, stride_qh, stride_ob, stride_oh,
       BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr, NCOL: tl.constexpr,
       REGATHER: tl.constexpr):
    bid = tl.program_id(0)
    hblk = tl.program_id(1)
    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_t = tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, NCOL)
    offs_r = tl.arange(0, ROPE)

    h_valid = offs_h < H
    q_base = bid * stride_qb + offs_h * stride_qh
    # Query transposed [NCOL, BLOCK_H]: the RIGHT operand of the flipped QK dot.
    # Loaded once per program, so its uncoalesced addressing is paid once.
    q0 = tl.load(Q_ptr + q_base[None, :] + (0 * NCOL + offs_c)[:, None], mask=h_valid[None, :], other=0.0)
    q1 = tl.load(Q_ptr + q_base[None, :] + (1 * NCOL + offs_c)[:, None], mask=h_valid[None, :], other=0.0)
    q2 = tl.load(Q_ptr + q_base[None, :] + (2 * NCOL + offs_c)[:, None], mask=h_valid[None, :], other=0.0)
    q3 = tl.load(Q_ptr + q_base[None, :] + (3 * NCOL + offs_c)[:, None], mask=h_valid[None, :], other=0.0)
    q4 = tl.load(Q_ptr + q_base[None, :] + (4 * NCOL + offs_c)[:, None], mask=h_valid[None, :], other=0.0)
    q5 = tl.load(Q_ptr + q_base[None, :] + (5 * NCOL + offs_c)[:, None], mask=h_valid[None, :], other=0.0)
    q6 = tl.load(Q_ptr + q_base[None, :] + (6 * NCOL + offs_c)[:, None], mask=h_valid[None, :], other=0.0)
    q7 = tl.load(Q_ptr + q_base[None, :] + (7 * NCOL + offs_c)[:, None], mask=h_valid[None, :], other=0.0)
    q_r = tl.load(Q_ptr + q_base[None, :] + NOPE + offs_r[:, None], mask=h_valid[None, :], other=0.0)

    # Online softmax state is per-head; the score tile is [BLOCK_T, BLOCK_H].
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    a0 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a1 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a2 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a3 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a4 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a5 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a6 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a7 = tl.zeros([BLOCK_H, NCOL], tl.float32)
    a_r = tl.zeros([BLOCK_H, ROPE], tl.float32)

    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + offs_t
        raw = tl.load(idx_ptr + bid * topk + t_idx, mask=t_idx < topk, other=-1)
        valid = (t_idx < topk) & (raw >= 0)
        safe = tl.where(valid, raw, 0).to(tl.int64)
        pg = safe // page_size
        po = safe % page_size
        tok_base = pg * page_bytes + po * TOKB
        soff_t = pg * page_bytes + soff_base + po * SB

        # ---- QK, flipped: scores_t[T, H] = dot(kv[T,C], q_t[C, H]) ----
        kv0 = _kv_n(cu8, lut, tok_base, 0 * NCOL + offs_c, soff_t, valid)
        st = tl.dot(kv0, q0)
        kv1 = _kv_n(cu8, lut, tok_base, 1 * NCOL + offs_c, soff_t, valid)
        st += tl.dot(kv1, q1)
        kv2 = _kv_n(cu8, lut, tok_base, 2 * NCOL + offs_c, soff_t, valid)
        st += tl.dot(kv2, q2)
        kv3 = _kv_n(cu8, lut, tok_base, 3 * NCOL + offs_c, soff_t, valid)
        st += tl.dot(kv3, q3)
        kv4 = _kv_n(cu8, lut, tok_base, 4 * NCOL + offs_c, soff_t, valid)
        st += tl.dot(kv4, q4)
        kv5 = _kv_n(cu8, lut, tok_base, 5 * NCOL + offs_c, soff_t, valid)
        st += tl.dot(kv5, q5)
        kv6 = _kv_n(cu8, lut, tok_base, 6 * NCOL + offs_c, soff_t, valid)
        st += tl.dot(kv6, q6)
        kv7 = _kv_n(cu8, lut, tok_base, 7 * NCOL + offs_c, soff_t, valid)
        st += tl.dot(kv7, q7)
        rb = ((tok_base + NOPE) // 2).to(tl.int64)
        kv_r = tl.load(cu16 + rb[:, None] + offs_r[None, :], mask=valid[:, None], other=0.0).to(tl.float16)
        st += tl.dot(kv_r, q_r)

        # Softmax over the token axis (axis 0), per head column.
        s = tl.where(valid[:, None], st * (softmax_scale * LOG2E), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - m_safe))
        p_t = tl.where(valid[:, None], tl.math.exp2(s - m_safe[None, :]), 0.0)
        l_i = l_i * alpha + tl.sum(p_t, axis=0)
        # The one transpose in the kernel: a [BLOCK_T, BLOCK_H] fp16 tile.
        pf = tl.trans(p_t.to(tl.float16))

        # ---- PV reuses the tiles already in registers: no second gather ----
        if REGATHER:
            a0 = a0 * alpha[:, None] + tl.dot(pf, _kv_n(cu8, lut, tok_base, 0 * NCOL + offs_c, soff_t, valid))
        else:
            a0 = a0 * alpha[:, None] + tl.dot(pf, kv0)
        if REGATHER:
            a1 = a1 * alpha[:, None] + tl.dot(pf, _kv_n(cu8, lut, tok_base, 1 * NCOL + offs_c, soff_t, valid))
        else:
            a1 = a1 * alpha[:, None] + tl.dot(pf, kv1)
        if REGATHER:
            a2 = a2 * alpha[:, None] + tl.dot(pf, _kv_n(cu8, lut, tok_base, 2 * NCOL + offs_c, soff_t, valid))
        else:
            a2 = a2 * alpha[:, None] + tl.dot(pf, kv2)
        if REGATHER:
            a3 = a3 * alpha[:, None] + tl.dot(pf, _kv_n(cu8, lut, tok_base, 3 * NCOL + offs_c, soff_t, valid))
        else:
            a3 = a3 * alpha[:, None] + tl.dot(pf, kv3)
        if REGATHER:
            a4 = a4 * alpha[:, None] + tl.dot(pf, _kv_n(cu8, lut, tok_base, 4 * NCOL + offs_c, soff_t, valid))
        else:
            a4 = a4 * alpha[:, None] + tl.dot(pf, kv4)
        if REGATHER:
            a5 = a5 * alpha[:, None] + tl.dot(pf, _kv_n(cu8, lut, tok_base, 5 * NCOL + offs_c, soff_t, valid))
        else:
            a5 = a5 * alpha[:, None] + tl.dot(pf, kv5)
        if REGATHER:
            a6 = a6 * alpha[:, None] + tl.dot(pf, _kv_n(cu8, lut, tok_base, 6 * NCOL + offs_c, soff_t, valid))
        else:
            a6 = a6 * alpha[:, None] + tl.dot(pf, kv6)
        if REGATHER:
            a7 = a7 * alpha[:, None] + tl.dot(pf, _kv_n(cu8, lut, tok_base, 7 * NCOL + offs_c, soff_t, valid))
        else:
            a7 = a7 * alpha[:, None] + tl.dot(pf, kv7)
        a_r = a_r * alpha[:, None] + tl.dot(pf, kv_r)
        m_i = m_new

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    o_base = bid * stride_ob + offs_h * stride_oh
    c = 0 * NCOL + offs_c
    tl.store(O_ptr + o_base[:, None] + c[None, :], (a0 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None] & (c < NOPE)[None, :])
    c = 1 * NCOL + offs_c
    tl.store(O_ptr + o_base[:, None] + c[None, :], (a1 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None] & (c < NOPE)[None, :])
    c = 2 * NCOL + offs_c
    tl.store(O_ptr + o_base[:, None] + c[None, :], (a2 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None] & (c < NOPE)[None, :])
    c = 3 * NCOL + offs_c
    tl.store(O_ptr + o_base[:, None] + c[None, :], (a3 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None] & (c < NOPE)[None, :])
    c = 4 * NCOL + offs_c
    tl.store(O_ptr + o_base[:, None] + c[None, :], (a4 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None] & (c < NOPE)[None, :])
    c = 5 * NCOL + offs_c
    tl.store(O_ptr + o_base[:, None] + c[None, :], (a5 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None] & (c < NOPE)[None, :])
    c = 6 * NCOL + offs_c
    tl.store(O_ptr + o_base[:, None] + c[None, :], (a6 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None] & (c < NOPE)[None, :])
    c = 7 * NCOL + offs_c
    tl.store(O_ptr + o_base[:, None] + c[None, :], (a7 / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None] & (c < NOPE)[None, :])
    tl.store(O_ptr + o_base[:, None] + NOPE + offs_r[None, :],
             (a_r / safe_l[:, None]).to(O_ptr.dtype.element_ty), mask=h_valid[:, None])
    tl.store(LSE_ptr + bid * H + offs_h,
             tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf")),
             mask=h_valid)


def run(q, kc, flat, softmax_scale, out, lse, bh=16, ncol=64, warps=4, stages=1, bt=16, regather=1):
    B, _, H, D = q.shape
    assert 8 * ncol >= 448, "NCOL=%d leaves %d of 448 nope columns unread" % (ncol, 448 - 8 * ncol)
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
             BLOCK_H=bh, BLOCK_T=bt, NCOL=ncol, REGATHER=regather,
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

    # ---- correctness against the test suite's fp32 reference ----
    NB, NT = 16, 4096
    kc2, truth = T.build_cache(NT)
    g = torch.Generator(device="cuda").manual_seed(3)
    q2 = (torch.randn(NB, 1, H, D, device="cuda", generator=g) * 0.5).half()
    idx2 = torch.randint(0, NT, (NB, 1, TOPK), device="cuda", generator=g, dtype=torch.int32)
    flat2 = idx2.reshape(NB, -1).contiguous()
    tlen2 = torch.full((NB,), TOPK, dtype=torch.int32, device="cuda")
    sc = D ** -0.5
    ref = T.reference(q2, truth, idx2, tlen2, None)
    o2 = torch.zeros(NB, H, D, dtype=torch.float16, device="cuda")
    l2 = torch.zeros(NB, H, dtype=torch.float32, device="cuda")
    # The n64 config stages 8 live KV tiles plus 8 query tiles and needs 74752 B
    # of shared memory against the 65536 B SM75 cap, so validate at a shape that
    # launches; the arithmetic is identical, only the tiling differs.
    ok_cfg = None
    for cfg in (dict(ncol=64, warps=4, stages=1, regather=1),
                dict(ncol=64, warps=4, stages=2, regather=1)):
        try:
            run(q2, kc2, flat2, sc, o2, l2, bh=16, **cfg)
            torch.cuda.synchronize()
            ok_cfg = cfg
            break
        except Exception as e:
            print("  cfg %s does not launch: %s" % (cfg, str(e).splitlines()[-1][:44]))
    if ok_cfg is None:
        print("no config launches"); return
    p_out, _ = M._run_headshared_sparse_decode(q2, kc2, flat2, tlen2, sc)
    rp, rf = T.rel(p_out.squeeze(1), ref), T.rel(o2, ref)
    print("vs fp32 reference (%s): production rel %.3e   flipped rel %.3e   %s"
          % (ok_cfg, rp, rf, "OK" if rf < 3 * rp else "FAIL"))
    if rf >= 3 * rp:
        return

    q, kc, idx = A.build(B, H, D, TOPK, POOL)
    out = torch.zeros(B, H, D, dtype=torch.float16, device="cuda")
    lse = torch.zeros(B, H, dtype=torch.float32, device="cuda")
    mb = B * TOPK * 576 / 1e6
    print("\npool %.0f MB; %.0f MB/call; gather floor 0.24 ms" % (POOL * 584 / 1e6, mb))

    t_prod = bench(lambda: M._run_headshared_sparse_decode(
        q, kc, idx, torch.full((B,), TOPK, dtype=torch.int32, device="cuda"), 0.088))
    print("production (transposed gather) : %7.2f ms" % t_prod)

    cands = [("flip+regather bt8 w4", dict(ncol=64, warps=4, stages=1, bt=8, regather=1)),
             ("flip+keep     bt8 w4", dict(ncol=64, warps=4, stages=1, bt=8, regather=0)),
             ("flip+keep     bt8 w8", dict(ncol=64, warps=8, stages=1, bt=8, regather=0)),
             ("flip+regather bt16 w4", dict(ncol=64, warps=4, stages=1, regather=1))]
    res = {n: [] for n, _ in cands}
    bad = set()
    for r in range(5):
        for name, kw in cands:
            if name in bad:
                continue
            try:
                res[name].append(bench(lambda k=kw: run(q, kc, idx, 0.088, out, lse, bh=16, **k)))
            except Exception as e:
                if r == 0:
                    print("  %-20s FAIL %s" % (name, str(e).splitlines()[-1][:50]))
                bad.add(name)
    for name, ts in res.items():
        if not ts:
            continue
        ts.sort()
        print("  %-20s %10.2f ms  %5.1f GB/s" % (name, ts[2], mb / ts[2]))


if __name__ == "__main__":
    main()
