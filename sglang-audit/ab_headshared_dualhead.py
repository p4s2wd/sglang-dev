"""Halve the gather work by giving one program both head blocks.

At the real prefill shape H=32 and BLOCK_H=16, so the grid is (B, 2): two
programs per request, each walking that request's entire selected-KV list. The
KV bytes and the gather instructions are therefore both paid twice for data that
only exists once.

Raising BLOCK_H to 32 would fix that but needs 90 KB of shared memory against
SM75's 64 KB cap. Instead: keep BLOCK_H=16 and loop the head blocks inside one
program, so each KV tile is gathered once and fed to both head groups' dots.

Why traffic is not the argument and issue is: the kernel runs at 1.6% of DRAM
bandwidth and ~2% of tensor-core peak, and is immune to grid order, num_stages,
scale handling and the LUT. It is waiting on scattered-gather issue and latency.
This variant cuts the number of gather instructions per KV token in half, which
is the only lever here that removes work rather than rearranging it.

The cost is registers: two sets of [16,576] fp32 accumulators plus two sets of q
tiles. If that spills, the variant loses and the answer is that the redundancy
is the price of fitting in 64 KB.

Run: python ab_headshared_dualhead.py
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
HALF = tl.constexpr(256)
NOPE = tl.constexpr(448)
ROPE = tl.constexpr(64)
TOKB = tl.constexpr(576)
SB = tl.constexpr(8)
GRP = tl.constexpr(64)


@triton.jit
def _load_kv(cache_u8_ptr, lut_ptr, tok_base, cols, pg, po, kv_valid,
             page_bytes, scale_off):
    mask = kv_valid[:, None] & (cols < NOPE)[None, :]
    byte = tl.load(cache_u8_ptr + tok_base[:, None] + cols[None, :], mask=mask, other=0)
    s_addr = (pg * page_bytes + scale_off + po * SB)[:, None] + (cols // GRP)[None, :]
    scale = tl.math.exp2(
        tl.load(cache_u8_ptr + s_addr, mask=mask, other=127).to(tl.float32) - 127.0)
    return (tl.load(lut_ptr + byte) * scale).to(tl.float16)


@triton.jit
def _dualhead_kernel(
    Q_ptr, cache_u8_ptr, cache_bf16_ptr, lut_ptr, indices_ptr,
    O_ptr, LSE_ptr, softmax_scale, page_size, page_bytes, scale_off,
    H, topk, stride_qb, stride_qh, stride_ob, stride_oh,
    BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr, NGROUP: tl.constexpr,
):
    """One program covers NGROUP head blocks and gathers each KV tile once."""
    bid = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    cols_a = tl.arange(0, HALF)
    cols_b = HALF + tl.arange(0, HALF)
    rope_offs = tl.arange(0, ROPE)

    # Per-group query tiles and online-softmax state. Triton rejects `other`
    # without a mask, so the head bound is spelled out even though H divides
    # BLOCK_H at the real shape.
    h0 = 0 * BLOCK_H + tl.arange(0, BLOCK_H)
    h1 = 1 * BLOCK_H + tl.arange(0, BLOCK_H)
    v0 = (h0 < H)[:, None]
    v1 = (h1 < H)[:, None]
    qb = Q_ptr + bid * stride_qb
    q_a0 = tl.load(qb + h0[:, None] * stride_qh + cols_a[None, :], mask=v0, other=0.0)
    q_b0 = tl.load(qb + h0[:, None] * stride_qh + cols_b[None, :], mask=v0, other=0.0)
    q_r0 = tl.load(qb + h0[:, None] * stride_qh + NOPE + rope_offs[None, :],
                   mask=v0, other=0.0)
    q_a1 = tl.load(qb + h1[:, None] * stride_qh + cols_a[None, :], mask=v1, other=0.0)
    q_b1 = tl.load(qb + h1[:, None] * stride_qh + cols_b[None, :], mask=v1, other=0.0)
    q_r1 = tl.load(qb + h1[:, None] * stride_qh + NOPE + rope_offs[None, :],
                   mask=v1, other=0.0)

    m0 = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l0 = tl.zeros([BLOCK_H], tl.float32)
    a0 = tl.zeros([BLOCK_H, HALF], tl.float32)
    b0 = tl.zeros([BLOCK_H, HALF], tl.float32)
    r0 = tl.zeros([BLOCK_H, ROPE], tl.float32)
    m1 = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l1 = tl.zeros([BLOCK_H], tl.float32)
    a1 = tl.zeros([BLOCK_H, HALF], tl.float32)
    b1 = tl.zeros([BLOCK_H, HALF], tl.float32)
    r1 = tl.zeros([BLOCK_H, ROPE], tl.float32)

    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + offs_t
        raw = tl.load(indices_ptr + bid * topk + t_idx, mask=t_idx < topk, other=-1)
        idx_valid = (t_idx < topk) & (raw >= 0)
        safe = tl.where(idx_valid, raw, 0).to(tl.int64)
        pg = safe // page_size
        po = safe % page_size
        tok_base = pg * page_bytes + po * TOKB

        kv_a = _load_kv(cache_u8_ptr, lut_ptr, tok_base, cols_a, pg, po,
                        idx_valid, page_bytes, scale_off)
        kv_b = _load_kv(cache_u8_ptr, lut_ptr, tok_base, cols_b, pg, po,
                        idx_valid, page_bytes, scale_off)
        rope_base = ((tok_base + NOPE) // 2).to(tl.int64)
        kv_r = tl.load(cache_bf16_ptr + rope_base[:, None] + rope_offs[None, :],
                       mask=idx_valid[:, None], other=0.0).to(tl.float16)

        kv_at, kv_bt, kv_rt = tl.trans(kv_a), tl.trans(kv_b), tl.trans(kv_r)

        for g in tl.static_range(NGROUP):
            if g == 0:
                s = tl.dot(q_a0, kv_at) + tl.dot(q_b0, kv_bt) + tl.dot(q_r0, kv_rt)
                s = tl.where(idx_valid[None, :], s * (softmax_scale * LOG2E), float("-inf"))
                mn = tl.maximum(m0, tl.max(s, axis=1))
                ms = tl.where(mn == float("-inf"), 0.0, mn)
                al = tl.where(m0 == float("-inf"), 0.0, tl.math.exp2(m0 - ms))
                p = tl.where(idx_valid[None, :], tl.math.exp2(s - ms[:, None]), 0.0)
                l0 = l0 * al + tl.sum(p, axis=1)
                pf = p.to(tl.float16)
                a0 = a0 * al[:, None] + tl.dot(pf, kv_a)
                b0 = b0 * al[:, None] + tl.dot(pf, kv_b)
                r0 = r0 * al[:, None] + tl.dot(pf, kv_r)
                m0 = mn
            else:
                s = tl.dot(q_a1, kv_at) + tl.dot(q_b1, kv_bt) + tl.dot(q_r1, kv_rt)
                s = tl.where(idx_valid[None, :], s * (softmax_scale * LOG2E), float("-inf"))
                mn = tl.maximum(m1, tl.max(s, axis=1))
                ms = tl.where(mn == float("-inf"), 0.0, mn)
                al = tl.where(m1 == float("-inf"), 0.0, tl.math.exp2(m1 - ms))
                p = tl.where(idx_valid[None, :], tl.math.exp2(s - ms[:, None]), 0.0)
                l1 = l1 * al + tl.sum(p, axis=1)
                pf = p.to(tl.float16)
                a1 = a1 * al[:, None] + tl.dot(pf, kv_a)
                b1 = b1 * al[:, None] + tl.dot(pf, kv_b)
                r1 = r1 * al[:, None] + tl.dot(pf, kv_r)
                m1 = mn

    for g in tl.static_range(NGROUP):
        offs_h = g * BLOCK_H + tl.arange(0, BLOCK_H)
        h_valid = offs_h < H
        if g == 0:
            sl, mi, aa, bb, rr = tl.where(l0 > 0.0, l0, 1.0), m0, a0, b0, r0
        else:
            sl, mi, aa, bb, rr = tl.where(l1 > 0.0, l1, 1.0), m1, a1, b1, r1
        o_base = bid * stride_ob + offs_h * stride_oh
        tl.store(O_ptr + o_base[:, None] + cols_a[None, :],
                 (aa / sl[:, None]).to(O_ptr.dtype.element_ty), mask=h_valid[:, None])
        tl.store(O_ptr + o_base[:, None] + cols_b[None, :],
                 (bb / sl[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (cols_b < NOPE)[None, :])
        tl.store(O_ptr + o_base[:, None] + NOPE + rope_offs[None, :],
                 (rr / sl[:, None]).to(O_ptr.dtype.element_ty), mask=h_valid[:, None])
        tl.store(LSE_ptr + bid * H + offs_h,
                 tl.where((l0 if g == 0 else l1) > 0.0,
                          mi / LOG2E + tl.math.log(sl), float("-inf")),
                 mask=h_valid)


def run_dual(q, kc, flat, scale, out, lse, num_warps=8):
    B, _, H, D = q.shape
    page = kc.shape[1]
    page_bytes = kc.stride(0)
    total = kc.shape[0] * page_bytes
    raw_u8 = kc.as_strided((total,), (1,)).view(torch.uint8)
    raw_bf16 = raw_u8.view(torch.bfloat16)
    lut = M.fp8_payload_lut(q.device, torch.float32)
    q3 = q.squeeze(1).contiguous()
    BH = 16
    ngroup = triton.cdiv(H, BH)
    # One program per request now: the head blocks live inside it.
    _dualhead_kernel[(B,)](
        q3, raw_u8, raw_bf16, lut, flat, out, lse,
        scale, page, int(page_bytes), int(page * TOKB),
        H, flat.shape[1], q3.stride(0), q3.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_H=BH, BLOCK_T=16, NGROUP=ngroup,
        num_warps=num_warps, num_stages=1)
    return out, lse


def timeit(fn, iters=15, warmup=3):
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

    cfgs = [("shipped grid=(B,hb)", "base", 8), ("dual-head grid=(B,)", "dual", 8),
            ("dual-head warps=4", "dual", 4), ("dual-head warps=16", "dual", 16)]
    ROUNDS = 6
    res = {n: [] for n, _, _ in cfgs}
    rels = {}
    for r in range(ROUNDS):
        for name, kind, nw in cfgs:
            out.zero_()
            try:
                if kind == "base":
                    A.run_variant(0, q, kc, flat, scale, out, lse)
                else:
                    run_dual(q, kc, flat, scale, out, lse, num_warps=nw)
                torch.cuda.synchronize()
            except Exception as e:
                if r == 0:
                    print("  %-22s %s" % (name, str(e).splitlines()[0][:60]))
                res[name].append(float("inf"))
                rels[name] = float("nan")
                continue
            if r == 0:
                rels[name] = ((out.float() - ref.float()).abs().max().item()
                              / max(ref.abs().max().item(), 1e-6))
            if kind == "base":
                res[name].append(timeit(
                    lambda: A.run_variant(0, q, kc, flat, scale, out, lse),
                    iters=15, warmup=3))
            else:
                res[name].append(timeit(
                    lambda: run_dual(q, kc, flat, scale, out, lse, num_warps=nw),
                    iters=6, warmup=2))

    base = sorted(res[cfgs[0][0]])[ROUNDS // 2]
    print("median of %d interleaved rounds:" % ROUNDS)
    for name, _, _ in cfgs:
        v = [x for x in sorted(res[name]) if x != float("inf")]
        if not v:
            continue
        med = v[len(v) // 2]
        bad = "" if rels[name] < 1e-2 else "  <-- MISMATCH"
        print("  %-22s %7.3f ms  (%5.3fx)  rel %.1e%s"
              % (name, med, base / med, rels[name], bad))


if __name__ == "__main__":
    main()
