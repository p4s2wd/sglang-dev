"""Is the per-lane scale gather what keeps the head-shared kernel off the roof?

The kernel is 27.9% of production EXTEND device time (3.0 ms/call) and, unlike
the all-reduce that sits next to it in the profile, it is balanced across TP
ranks to within 1% -- so it is real work on the critical path, not peer waiting.

Its bandwidth floor is not far away: 512 selected tokens x 576 B per request-head
group, amortised over BLOCK_H=16 heads. Yet it runs several times slower than
that floor. One candidate is the scale operand. Each token stores 8 UE8M0 group
scales, one per 64 columns, but the kernel loads them as a full [BLOCK_T, 256]
gather -- 256 lanes reaching for 4 distinct addresses per row. That is 64x waste
on the load-store unit for that one operand, plus the address arithmetic to build
it (a division by 64 per lane).

Variants, all numerically identical to the shipped kernel unless noted:
  shipped   -- as-is, control
  bitcast   -- UE8M0 byte -> fp32 via bitcast(b<<23) instead of exp2
  wide      -- load the 8 scales as one contiguous [T,8] row, broadcast in registers
  noscale   -- scale forced to 1.0: WRONG numerically, an upper bound on what any
               scale-handling change can buy

Run: python ab_headshared_scale.py
"""
import sys
import time

import torch
import triton
import triton.language as tl

DEV_G = "cuda"

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

from sglang.kernels.ops.attention import flash_mla_sm120_triton as M  # noqa: E402

LOG2E = tl.constexpr(1.4426950408889634)
_HS_NOPE_HALF = tl.constexpr(256)
_HS_NOPE = tl.constexpr(448)
_HS_ROPE = tl.constexpr(64)
_HS_TOKEN_BYTES = tl.constexpr(576)
_HS_SCALE_BYTES = tl.constexpr(8)
_HS_GROUP = tl.constexpr(64)


# --- variant loaders -------------------------------------------------------
@triton.jit
def _load_shipped(cache_u8_ptr, lut_ptr, tok_base, cols, pg, po, kv_valid,
                  page_bytes, scale_section_off):
    mask = kv_valid[:, None] & (cols < _HS_NOPE)[None, :]
    byte = tl.load(cache_u8_ptr + tok_base[:, None] + cols[None, :], mask=mask, other=0)
    scale_addr = (pg * page_bytes + scale_section_off + po * _HS_SCALE_BYTES)[:, None] \
        + (cols // _HS_GROUP)[None, :]
    scale = tl.math.exp2(
        tl.load(cache_u8_ptr + scale_addr, mask=mask, other=127).to(tl.float32) - 127.0)
    return (tl.load(lut_ptr + byte) * scale).to(tl.float16)


@triton.jit
def _load_bitcast(cache_u8_ptr, lut_ptr, tok_base, cols, pg, po, kv_valid,
                  page_bytes, scale_section_off):
    mask = kv_valid[:, None] & (cols < _HS_NOPE)[None, :]
    byte = tl.load(cache_u8_ptr + tok_base[:, None] + cols[None, :], mask=mask, other=0)
    scale_addr = (pg * page_bytes + scale_section_off + po * _HS_SCALE_BYTES)[:, None] \
        + (cols // _HS_GROUP)[None, :]
    scale = (
        tl.load(cache_u8_ptr + scale_addr, mask=mask, other=127).to(tl.int32) << 23
    ).to(tl.float32, bitcast=True)
    return (tl.load(lut_ptr + byte) * scale).to(tl.float16)


@triton.jit
def _load_noscale(cache_u8_ptr, lut_ptr, tok_base, cols, pg, po, kv_valid,
                  page_bytes, scale_section_off):
    mask = kv_valid[:, None] & (cols < _HS_NOPE)[None, :]
    byte = tl.load(cache_u8_ptr + tok_base[:, None] + cols[None, :], mask=mask, other=0)
    return tl.load(lut_ptr + byte).to(tl.float16)


@triton.jit
def _load_wide(cache_u8_ptr, lut_ptr, tok_base, cols, pg, po, kv_valid,
               page_bytes, scale_section_off, HALF: tl.constexpr):
    """Read all 8 group scales as one contiguous 8-byte row, then broadcast.

    HALF picks which four of the eight groups this nope half covers.
    """
    mask = kv_valid[:, None] & (cols < _HS_NOPE)[None, :]
    byte = tl.load(cache_u8_ptr + tok_base[:, None] + cols[None, :], mask=mask, other=0)
    g = tl.arange(0, _HS_SCALE_BYTES)
    s_addr = (pg * page_bytes + scale_section_off)[:, None] + po * _HS_SCALE_BYTES + g[None, :]
    s8 = tl.load(cache_u8_ptr + s_addr, mask=kv_valid[:, None], other=127).to(tl.int32)
    s8 = (s8 << 23).to(tl.float32, bitcast=True)
    # [T,8] -> [T,4,64] -> [T,256], each group scale repeated across its columns.
    s4 = tl.reshape(s8, (s8.shape[0], 2, _HS_SCALE_BYTES // 2))
    s4 = tl.where((tl.arange(0, 2)[None, :, None] == HALF)[None, :, :], s4, 0.0)
    s_sel = tl.sum(s4, axis=1)
    s = tl.reshape(tl.broadcast_to(s_sel[:, :, None],
                                   (s_sel.shape[0], 4, _HS_GROUP)),
                   (s_sel.shape[0], _HS_NOPE_HALF))
    return (tl.load(lut_ptr + byte) * s).to(tl.float16)


# --- kernel bodies (copy of the shipped one, loader swapped) ----------------
@triton.jit
def _kernel(Q_ptr, cache_u8_ptr, cache_bf16_ptr, lut_ptr, indices_ptr, topk_len_ptr,
            O_ptr, LSE_ptr, softmax_scale, page_size, page_bytes, scale_section_off,
            H, topk, HAS_TOPK_LEN, stride_qb, stride_qh, stride_ob, stride_oh,
            BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr, VARIANT: tl.constexpr,
            SWAP: tl.constexpr = 0):
    if SWAP:
        bid = tl.program_id(1)
        hblk = tl.program_id(0)
    else:
        bid = tl.program_id(0)
        hblk = tl.program_id(1)
    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    h_valid = offs_h < H
    offs_t = tl.arange(0, BLOCK_T)
    cols_a = tl.arange(0, _HS_NOPE_HALF)
    cols_b = _HS_NOPE_HALF + tl.arange(0, _HS_NOPE_HALF)
    rope_offs = tl.arange(0, _HS_ROPE)
    q_base = bid * stride_qb + offs_h * stride_qh
    q_a = tl.load(Q_ptr + q_base[:, None] + cols_a[None, :], mask=h_valid[:, None], other=0.0)
    q_b = tl.load(Q_ptr + q_base[:, None] + cols_b[None, :], mask=h_valid[:, None], other=0.0)
    q_r = tl.load(Q_ptr + q_base[:, None] + _HS_NOPE + rope_offs[None, :],
                  mask=h_valid[:, None], other=0.0)

    valid_len = topk
    if HAS_TOPK_LEN:
        valid_len = tl.load(topk_len_ptr + bid).to(tl.int32)

    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc_a = tl.zeros([BLOCK_H, _HS_NOPE_HALF], tl.float32)
    acc_b = tl.zeros([BLOCK_H, _HS_NOPE_HALF], tl.float32)
    acc_r = tl.zeros([BLOCK_H, _HS_ROPE], tl.float32)

    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + offs_t
        raw = tl.load(indices_ptr + bid * topk + t_idx, mask=t_idx < topk, other=-1)
        idx_valid = (t_idx < valid_len) & (raw >= 0)
        safe = tl.where(idx_valid, raw, 0).to(tl.int64)
        page_ids = safe // page_size
        page_offs = safe % page_size
        tok_base = page_ids * page_bytes + page_offs * _HS_TOKEN_BYTES

        if VARIANT == 0:
            kv_a = _load_shipped(cache_u8_ptr, lut_ptr, tok_base, cols_a, page_ids,
                                 page_offs, idx_valid, page_bytes, scale_section_off)
            kv_b = _load_shipped(cache_u8_ptr, lut_ptr, tok_base, cols_b, page_ids,
                                 page_offs, idx_valid, page_bytes, scale_section_off)
        elif VARIANT == 1:
            kv_a = _load_bitcast(cache_u8_ptr, lut_ptr, tok_base, cols_a, page_ids,
                                 page_offs, idx_valid, page_bytes, scale_section_off)
            kv_b = _load_bitcast(cache_u8_ptr, lut_ptr, tok_base, cols_b, page_ids,
                                 page_offs, idx_valid, page_bytes, scale_section_off)
        elif VARIANT == 3:
            kv_a = _load_wide(cache_u8_ptr, lut_ptr, tok_base, cols_a, page_ids,
                              page_offs, idx_valid, page_bytes, scale_section_off, 0)
            kv_b = _load_wide(cache_u8_ptr, lut_ptr, tok_base, cols_b, page_ids,
                              page_offs, idx_valid, page_bytes, scale_section_off, 1)
        else:
            kv_a = _load_noscale(cache_u8_ptr, lut_ptr, tok_base, cols_a, page_ids,
                                 page_offs, idx_valid, page_bytes, scale_section_off)
            kv_b = _load_noscale(cache_u8_ptr, lut_ptr, tok_base, cols_b, page_ids,
                                 page_offs, idx_valid, page_bytes, scale_section_off)

        rope_base = ((tok_base + _HS_NOPE) // 2).to(tl.int64)
        kv_r = tl.load(cache_bf16_ptr + rope_base[:, None] + rope_offs[None, :],
                       mask=idx_valid[:, None], other=0.0).to(tl.float16)

        scores = tl.dot(q_a, tl.trans(kv_a)) + tl.dot(q_b, tl.trans(kv_b)) \
            + tl.dot(q_r, tl.trans(kv_r))
        scores_log2 = scores * (softmax_scale * LOG2E)
        scores_log2 = tl.where(idx_valid[None, :], scores_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores_log2, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - m_safe))
        p = tl.math.exp2(scores_log2 - m_safe[:, None])
        p = tl.where(idx_valid[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pf = p.to(tl.float16)
        acc_a = acc_a * alpha[:, None] + tl.dot(pf, kv_a)
        acc_b = acc_b * alpha[:, None] + tl.dot(pf, kv_b)
        acc_r = acc_r * alpha[:, None] + tl.dot(pf, kv_r)
        m_i = m_new

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    o_base = bid * stride_ob + offs_h * stride_oh
    tl.store(O_ptr + o_base[:, None] + cols_a[None, :],
             (acc_a / safe_l[:, None]).to(O_ptr.dtype.element_ty), mask=h_valid[:, None])
    tl.store(O_ptr + o_base[:, None] + cols_b[None, :],
             (acc_b / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None] & (cols_b < _HS_NOPE)[None, :])
    tl.store(O_ptr + o_base[:, None] + _HS_NOPE + rope_offs[None, :],
             (acc_r / safe_l[:, None]).to(O_ptr.dtype.element_ty), mask=h_valid[:, None])
    tl.store(LSE_ptr + bid * H + offs_h,
             tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf")),
             mask=h_valid)


STRIDE = 576


def build(B, H, D, topk, num_tokens, page=64, seed=0):
    """Real DSv4 paged layout: [e4m3 nope 448 | bf16 rope 64] per 576-byte token
    slot, with the page's 8 UE8M0 group scales in a section at the page tail.

    Getting this right matters: the kernel addresses tokens by a 576-byte stride
    and reads scales at page*576, so a per-token interleaved 584-byte record --
    the obvious guess -- makes it read garbage and produce NaN in every variant,
    including a faithful copy of the shipped kernel.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_pages = -(-num_tokens // page)
    page_bytes = -(-584 * page // STRIDE) * STRIDE
    raw = torch.zeros(num_pages, page_bytes, dtype=torch.uint8, device="cuda")

    nope = (torch.randn(num_tokens, 448, generator=g, device="cuda") * 0.3)
    # Valid e4m3fn payloads only: random bytes hit the NaN encodings 0x7F/0xFF.
    q8 = nope.to(torch.float8_e4m3fn).view(torch.uint8)
    rope = (torch.randn(num_tokens, 64, generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    scale_exp = torch.randint(-6, 0, (num_tokens, 7), device=DEV_G, generator=g)
    sb = (127 + scale_exp).to(torch.uint8)

    pg = torch.arange(num_tokens, device="cuda") // page
    po = (torch.arange(num_tokens, device="cuda") % page)
    raw[pg[:, None], (po * STRIDE)[:, None] + torch.arange(448, device="cuda")[None, :]] = q8
    raw[pg[:, None], (po * STRIDE + 448)[:, None]
        + 2 * torch.arange(64, device="cuda")[None, :]] = rope.view(torch.uint8)[:, ::2]
    raw[pg[:, None], (page * STRIDE + po * 8)[:, None]
        + torch.arange(8, device="cuda")[None, :]] = torch.cat(
            [sb, torch.zeros(num_tokens, 1, dtype=torch.uint8, device=DEV_G)], 1)

    kc = raw.as_strided((num_pages, page, 1, STRIDE),
                        (page_bytes, STRIDE, STRIDE, 1)).view(torch.float8_e4m3fn)
    q = (torch.randn(B, 1, H, D, generator=g, device="cuda",
                     dtype=torch.float32) * 0.5).half()
    indices = torch.randint(0, num_tokens, (B, 1, topk),
                            generator=g, device="cuda", dtype=torch.int32)
    return q, kc, indices.reshape(B, -1).contiguous()


def run_variant(variant, q, kc, flat, scale, out, lse, swap=0, stages=1, bh=16):
    B, _, H, D = q.shape
    num_pages, page = kc.shape[0], kc.shape[1]
    page_bytes = kc.stride(0)
    total = num_pages * page_bytes
    raw_u8 = kc.as_strided((total,), (1,)).view(torch.uint8)
    raw_bf16 = raw_u8.view(torch.bfloat16)
    lut = M.fp8_payload_lut(q.device, torch.float32)
    q3 = q.squeeze(1).contiguous()
    nhb = triton.cdiv(H, bh)
    grid = (nhb, B) if swap else (B, nhb)
    _kernel[grid](q3, raw_u8, raw_bf16, lut, flat,
                  torch.empty(0, device=q.device, dtype=torch.int32),
                  out, lse, scale, page, int(page_bytes),
                  int(page * _HS_TOKEN_BYTES),
                  H, flat.shape[1], False,
                  q3.stride(0), q3.stride(1), out.stride(0), out.stride(1),
                  BLOCK_H=bh, BLOCK_T=16, VARIANT=variant, SWAP=swap,
                  num_warps=8, num_stages=stages)
    return out, lse


def timeit(fn, iters=20, warmup=5):
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
    q, kc, flat = build(B, H, D, topk, num_tokens=B * topk)
    scale = 1.0 / (D ** 0.5)
    out = torch.zeros(B, H, D, dtype=q.dtype, device=q.device)
    lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)

    prod_out, _ = M._run_headshared_sparse_decode(q, kc, flat, None, scale)
    torch.cuda.synchronize()
    run_variant(0, q, kc, flat, scale, out, lse)
    torch.cuda.synchronize()
    p3 = prod_out.squeeze(1)
    rel0 = ((out.float() - p3.float()).abs().max().item() / max(p3.abs().max().item(), 1e-6))
    print("control vs production: rel %.2e nan=%d %s"
          % (rel0, torch.isnan(out).any().item(),
             "ok" if rel0 < 1e-5 and not torch.isnan(out).any() else "BROKEN"))
    if rel0 >= 1e-5 or torch.isnan(out).any():
        raise SystemExit(1)
    ref = out.clone()

    cfgs = [
        ("shipped        grid=(B,hb) st=1", dict(variant=0, swap=0, stages=1)),
        ("grid=(hb,B)    L2 reuse of KV", dict(variant=0, swap=1, stages=1)),
        ("stages=2       prefetch gathers", dict(variant=0, swap=0, stages=2)),
        ("grid swap + stages=2", dict(variant=0, swap=1, stages=2)),
        # H=32 with BLOCK_H=16 puts 2 head-blocks per request on the grid, and
        # each one reads the request's whole KV set: 102 MB where 51 MB of data
        # exists. BLOCK_H=32 reads it once -- the only lever here that removes
        # work rather than reordering it. It costs accumulator registers
        # ([32,576] fp32 = 72 regs/thread at 8 warps), which may spill.
        ("BLOCK_H=32 (halves KV traffic)", dict(variant=0, swap=0, stages=1, bh=32)),
        ("BLOCK_H=32 warps=4", dict(variant=0, swap=0, stages=1, bh=32)),
        ("BLOCK_H=64", dict(variant=0, swap=0, stages=1, bh=64)),
        ("noscale (WRONG, upper bound)", dict(variant=2, swap=0, stages=1)),
    ]

    def timed(**kw):
        v = kw.pop("variant", 0)
        n = 15 if kw.get("bh", 16) <= 16 else 6
        return timeit(lambda: run_variant(v, q, kc, flat, scale, out, lse, **kw),
                      iters=n, warmup=3)

    ROUNDS = 6
    res = {n: [] for n, _ in cfgs}
    rels = {}
    for r in range(ROUNDS):
        for name, kw in cfgs:
            out.zero_()
            kw = dict(kw)
            vv = kw.pop("variant", 0)
            run_variant(vv, q, kc, flat, scale, out, lse, **kw)
            torch.cuda.synchronize()
            if r == 0:
                rels[name] = ((out.float() - ref.float()).abs().max().item()
                              / max(ref.abs().max().item(), 1e-6))
            res[name].append(timed(**kw))
    base = sorted(res[cfgs[0][0]])[ROUNDS // 2]
    print("\nmedian of %d interleaved rounds:" % ROUNDS)
    for name, _ in cfgs:
        v = sorted(res[name])
        med = v[ROUNDS // 2]
        bad = "" if (rels[name] < 1e-2 or "WRONG" in name) else "  <-- MISMATCH"
        print("  %-34s %7.3f ms  (%5.3fx)  spread %5.1f%%  rel %.1e%s"
              % (name, med, base / med, 100 * (v[-1] - v[0]) / med, rels[name], bad))


if __name__ == "__main__":
    main()
