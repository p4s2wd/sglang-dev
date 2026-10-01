"""Prototype: share the gathered KV across query heads in sparse MLA attention.

Why: DeepSeek-V4 MLA has num_key_value_heads=1, so all query heads of one token
attend to the SAME latent KV. The shipped kernel launches grid=(B, H), so every
head re-gathers and re-dequantizes the same bytes. Measured on this GPU
(B=512, topk=512): H=1 -> 2.34ms, H=64 -> 144.6ms -- linear in H (61.7x for 64
heads), so L2 absorbs none of the redundancy. In prefill that kernel is 51.5% of
all device time (923.8 of 1792.6 ms in the EXTEND trace).

The fix: one block owns BLOCK_H heads and BLOCK_T tokens. The KV tile is gathered
and dequantized ONCE and reused by all BLOCK_H heads, cutting KV traffic and
dequant ALU by BLOCK_Hx.

Layout constraints that shape the implementation:
  - tl.dot lowers to mma, so M >= 16 (BLOCK_H >= 16) and K >= 16 (BLOCK_T >= 16).
  - SM75 caps dynamic shared memory at 64KB, so the 448-wide nope half cannot be
    staged as one [BLOCK_T, 448] tile alongside q. It is split into two 256-wide
    halves with two explicit accumulators: 64 columns of the second half are
    padding (12.5% of the FLOPs), which is far cheaper than the SMEM a single
    512-wide tile needs.
  - Triton refuses non-constexpr module globals, hence tl.constexpr below.

Standalone: checks numerics against the shipped kernel and times both at the real
prefill shape before anything is wired into sglang.
"""
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

from sglang.kernels.ops.attention.flash_mla_sm120_triton import (  # noqa: E402
    flash_mla_sparse_decode_triton,
)
from sglang.kernels.ops.quantization.fp8_w8a16 import fp8_payload_lut  # noqa: E402

NOPE = tl.constexpr(448)
ROPE = tl.constexpr(64)
HALF = tl.constexpr(256)          # nope half width (256 + 256 >= 448)
D = 512                           # host-side: NOPE + ROPE
TOKEN_BYTES = tl.constexpr(576)   # 448 e4m3 + 64*2 bf16
TOKEN_BYTES_I = 576
SCALE_BYTES = tl.constexpr(8)     # 7 used, one per 64 nope dims
TILE = tl.constexpr(64)           # nope dims per scale group


@triton.jit
def _load_nope_tile(cache_u8_ptr, lut_ptr, tok_base, cols, scale_col, pg, po,
                    kv_valid, page_bytes, scale_section_off):
    """Gather + dequant one [BLOCK_T, ncols] slice of the nope half.

    Padded columns (>= 448) load byte 0 and LUT[0] == 0.0, so they contribute
    nothing to either dot product.
    """
    in_nope = cols < NOPE
    mask = kv_valid[:, None] & in_nope[None, :]
    byte = tl.load(cache_u8_ptr + tok_base[:, None] + cols[None, :],
                   mask=mask, other=0)
    s_addr = (pg * page_bytes + scale_section_off + po * SCALE_BYTES)[:, None] \
        + scale_col + (cols // TILE)[None, :]
    s_f = tl.math.exp2(
        tl.load(cache_u8_ptr + s_addr, mask=mask, other=127).to(tl.float32) - 127.0)
    return (tl.load(lut_ptr + byte) * s_f).to(tl.float16)


@triton.jit
def _headshared_kernel(
    Q_ptr,            # [B, H, D] fp16
    cache_u8_ptr,     # uint8 flat: e4m3 nope payload + ue8m0 scales
    cache_bf16_ptr,   # bfloat16 flat: rope
    lut_ptr,          # fp32 [256]: e4m3 byte -> value
    indices_ptr,      # [B, topk] int32
    O_ptr,            # [B, H, D]
    LSE_ptr,          # [B, H] fp32
    sm_scale,
    page_size,
    page_bytes,
    scale_section_off,
    H: tl.constexpr,
    topk,
    stride_qb: tl.int64,
    stride_qh: tl.int64,
    stride_ob: tl.int64,
    stride_oh: tl.int64,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    b = tl.program_id(0)
    hblk = tl.program_id(1)

    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    h_valid = offs_h < H
    offs_t = tl.arange(0, BLOCK_T)
    offs_a = tl.arange(0, HALF)                  # nope columns 0..255
    offs_b = HALF + tl.arange(0, HALF)           # nope columns 256..511 (>=448 pad)
    offs_r = tl.arange(0, ROPE)

    q_base = b * stride_qb + offs_h * stride_qh
    q_a = tl.load(Q_ptr + q_base[:, None] + offs_a[None, :],
                  mask=h_valid[:, None], other=0.0)
    q_b = tl.load(Q_ptr + q_base[:, None] + offs_b[None, :],
                  mask=h_valid[:, None] & (offs_b < NOPE)[None, :], other=0.0)
    q_r = tl.load(Q_ptr + q_base[:, None] + NOPE + offs_r[None, :],
                  mask=h_valid[:, None], other=0.0)

    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc_a = tl.zeros([BLOCK_H, HALF], tl.float32)
    acc_b = tl.zeros([BLOCK_H, HALF], tl.float32)
    acc_r = tl.zeros([BLOCK_H, ROPE], tl.float32)

    for t0 in range(0, topk, BLOCK_T):
        t_idx = t0 + offs_t
        raw = tl.load(indices_ptr + b * topk + t_idx,
                      mask=t_idx < topk, other=-1)
        kv_valid = raw >= 0
        safe = tl.where(kv_valid, raw, 0).to(tl.int64)
        pg = safe // page_size
        po = safe % page_size
        tok_base = pg * page_bytes + po * TOKEN_BYTES

        kv_a = _load_nope_tile(cache_u8_ptr, lut_ptr, tok_base, offs_a, 0, pg, po,
                               kv_valid, page_bytes, scale_section_off)
        # scale_col stays 0 for both halves: the helper derives the group index
        # from the absolute column via cols // TILE, so passing HALF // TILE here
        # as well would read scale groups 8..11, past the 7 that exist.
        kv_b = _load_nope_tile(cache_u8_ptr, lut_ptr, tok_base, offs_b,
                               0, pg, po, kv_valid, page_bytes,
                               scale_section_off)
        r_base = ((tok_base + NOPE) // 2).to(tl.int64)
        kv_r = tl.load(cache_bf16_ptr + r_base[:, None] + offs_r[None, :],
                       mask=kv_valid[:, None], other=0.0).to(tl.float16)

        scores = tl.dot(q_a, tl.trans(kv_a)) + tl.dot(q_b, tl.trans(kv_b)) \
            + tl.dot(q_r, tl.trans(kv_r))
        scores = scores * sm_scale
        scores = tl.where(kv_valid[None, :], scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_safe))
        w = tl.exp(scores - m_safe[:, None])
        w = tl.where(kv_valid[None, :], w, 0.0)
        l_i = l_i * alpha + tl.sum(w, axis=1)
        wf = w.to(tl.float16)

        acc_a = acc_a * alpha[:, None] + tl.dot(wf, kv_a)
        acc_b = acc_b * alpha[:, None] + tl.dot(wf, kv_b)
        acc_r = acc_r * alpha[:, None] + tl.dot(wf, kv_r)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    o_base = b * stride_ob + offs_h * stride_oh
    tl.store(O_ptr + o_base[:, None] + offs_a[None, :],
             (acc_a / l_safe[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None])
    tl.store(O_ptr + o_base[:, None] + offs_b[None, :],
             (acc_b / l_safe[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None] & (offs_b < NOPE)[None, :])
    tl.store(O_ptr + o_base[:, None] + NOPE + offs_r[None, :],
             (acc_r / l_safe[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None])
    tl.store(LSE_ptr + b * H + offs_h,
             tl.where(l_i == 0.0, float("-inf"), m_i + tl.log(l_i)),
             mask=h_valid)


def headshared_decode(q, k_cache, indices, lut, block_h, block_t, num_warps,
                      num_stages=1):
    B, _, H, Dq = q.shape
    num_pages, page_size = k_cache.shape[0], k_cache.shape[1]
    page_bytes = k_cache.stride(0)
    raw = k_cache.as_strided((num_pages * page_bytes,), (1,)).view(torch.uint8)
    q3 = q.squeeze(1).contiguous()
    out = torch.zeros(B, H, Dq, dtype=q.dtype, device=q.device)
    lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)
    _headshared_kernel[(B, triton.cdiv(H, block_h))](
        q3, raw, raw.view(torch.bfloat16), lut, indices.contiguous(), out, lse,
        Dq ** -0.5, page_size, page_bytes, page_size * TOKEN_BYTES_I,
        H=H, topk=indices.shape[-1],
        stride_qb=q3.stride(0), stride_qh=q3.stride(1),
        stride_ob=out.stride(0), stride_oh=out.stride(1),
        BLOCK_H=block_h, BLOCK_T=block_t,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out, lse


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    torch.cuda.set_device(0)
    dev = "cuda:0"
    PAGE, TOPK = 256, 512
    N_TOK = 8192
    page_bytes = -(-584 * PAGE // 576) * 576
    raw = torch.randint(0, 255, (N_TOK // PAGE, page_bytes), dtype=torch.uint8,
                        device=dev)
    kc = raw.as_strided((N_TOK // PAGE, PAGE, 1, TOKEN_BYTES_I),
                        (page_bytes, TOKEN_BYTES_I, TOKEN_BYTES_I, 1)).view(
        torch.float8_e4m3fn)
    lut = fp8_payload_lut(torch.device(dev), torch.float32)

    print("correctness vs the shipped kernel (B=4, H=16, topk=512):")
    q = (torch.randn(4, 1, 16, D, device=dev) * 0.3).half()
    idx = torch.randint(0, N_TOK, (4, TOPK), dtype=torch.int32, device=dev)
    ref_o, _ = flash_mla_sparse_decode_triton(q, kc, idx.unsqueeze(1), None, None,
                                              D, D ** -0.5)
    for bt in (16, 32):
        try:
            got_o, _ = headshared_decode(q, kc, idx, lut, 16, bt, 4)
            torch.cuda.synchronize()
            rel = ((got_o.squeeze(1).float() - ref_o.float()).abs().max().item()
                   / ref_o.float().abs().max().item())
            print(f"  BLOCK_T={bt}: rel {rel:.3e}  {'ok' if rel < 5e-3 else 'FAIL'}")
        except Exception as e:
            print(f"  BLOCK_T={bt}: {type(e).__name__}: {str(e)[:90]}")

    print("\ntiming at the real prefill shape (topk=512):")
    print(f"{'B':>5} {'H':>4} {'shipped':>9} {'shared':>9} {'speedup':>8}  cfg")
    for B, H in [(512, 32), (512, 64), (256, 32), (2, 64)]:
        q = (torch.randn(B, 1, H, D, device=dev) * 0.3).half()
        idx = torch.randint(0, N_TOK, (B, TOPK), dtype=torch.int32, device=dev)
        t_old = bench(lambda: flash_mla_sparse_decode_triton(
            q, kc, idx.unsqueeze(1), None, None, D, D ** -0.5))
        best = None
        for bh in (16, 32):
            for bt in (16, 32):
                for nw in (4, 8):
                    try:
                        t = bench(lambda: headshared_decode(q, kc, idx, lut,
                                                            bh, bt, nw))
                    except Exception:
                        continue
                    if best is None or t < best[0]:
                        best = (t, bh, bt, nw)
        if best is None:
            print(f"{B:5d} {H:4d} {t_old:9.2f}  all configs failed")
            continue
        print(f"{B:5d} {H:4d} {t_old:9.2f} {best[0]:9.2f} "
              f"{t_old / best[0]:7.1f}x  BH={best[1]} BT={best[2]} w={best[3]}")


if __name__ == "__main__":
    main()
