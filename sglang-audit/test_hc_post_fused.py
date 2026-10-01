"""Fused mhc_post for SM75, replacing the eager torch expression.

The torch fallback in deepseek_v4.py is

    (post.unsqueeze(-1) * x.unsqueeze(1)
     + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)).type_as(x)

sum(dim=1) reduces comb's *first* channel index, so the operation is

    out[t, j, h] = post[t, j] * x[t, h] + sum_m comb[t, m, j] * residual[t, m, h]

With hc_mult=4 and hidden=4096 the expression materialises [T,4,4,4096] fp32
twice (the comb*residual product and the reduction result), plus another fp32
buffer for the final add and a pass to cast back to fp16. At T=512 each of those
is 134 MB, so one call moves close to a gigabyte to produce 4 MB of output. It is
the #1 attributed line in the EXTEND profile (55.28 ms over 240 calls).

The operation is purely bandwidth-bound: per token read hc_mult*hidden of x,
hc_mult*hidden of residual and hc_mult^2 mixing weights, write hc_mult*hidden.
A kernel that keeps comb and post in registers and holds all hc_mult residual
tiles at once touches exactly that.

Run: python sglang-audit/test_hc_post_fused.py
"""
import sys
import time

import torch
import triton
import triton.language as tl

repo = "/data/nvme/sglang-codex/sglang"
sys.path.insert(0, repo + "/python")


@triton.jit
def _hc_post_kernel(
    x_ptr, res_ptr, post_ptr, comb_ptr, out_ptr,
    hidden,
    stride_xt, stride_rt, stride_pt, stride_ct, stride_ot,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_h = tl.program_id(0)
    t = tl.program_id(1)

    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < hidden

    x_row = tl.load(x_ptr + t * stride_xt + cols, mask=mask, other=0.0).to(tl.float32)

    # All hc_mult residual tiles at once: read once, reused for every output
    # channel. HC is 4 for this model, so this is 4 * BLOCK_H fp32 per program.
    r_base = res_ptr + t * stride_rt
    r0 = tl.load(r_base + 0 * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(r_base + 1 * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    r2 = tl.load(r_base + 2 * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    r3 = tl.load(r_base + 3 * hidden + cols, mask=mask, other=0.0).to(tl.float32)

    c_base = comb_ptr + t * stride_ct
    p_base = post_ptr + t * stride_pt

    o_base = out_ptr + t * stride_ot
    for j in tl.static_range(HC):
        p_j = tl.load(p_base + j).to(tl.float32)
        # sum_m comb[t, m, j] * residual[t, m]
        acc = p_j * x_row
        acc += tl.load(c_base + 0 * HC + j).to(tl.float32) * r0
        acc += tl.load(c_base + 1 * HC + j).to(tl.float32) * r1
        acc += tl.load(c_base + 2 * HC + j).to(tl.float32) * r2
        acc += tl.load(c_base + 3 * HC + j).to(tl.float32) * r3
        tl.store(o_base + j * hidden + cols, acc.to(out_ptr.dtype.element_ty),
                 mask=mask)


def hc_post_fused(x, residual, post, comb):
    """x [T, hidden], residual/out [T, HC, hidden], post [T, HC] f32,
    comb [T, HC, HC] f32."""
    T, hidden = x.shape
    HC = residual.shape[1]
    assert HC == 4, "this kernel hard-unrolls the 4 residual tiles"
    out = torch.empty_like(residual)
    BLOCK_H = 256
    grid = (triton.cdiv(hidden, BLOCK_H), T)
    _hc_post_kernel[grid](
        x, residual, post, comb, out,
        hidden,
        x.stride(0), residual.stride(0), post.stride(0), comb.stride(0),
        out.stride(0),
        HC=HC, BLOCK_H=BLOCK_H,
        num_warps=4, num_stages=1,
    )
    return out


def torch_ref(x, residual, post, comb):
    return (
        post.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    ).type_as(x)


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    torch.manual_seed(0)
    dev = "cuda:0"
    HC, HIDDEN = 4, 4096
    ok = True

    print("correctness vs the torch expression:")
    for T in (1, 8, 512, 1024):
        x = (torch.randn(T, HIDDEN, device=dev) * 0.5).half()
        res = (torch.randn(T, HC, HIDDEN, device=dev) * 0.5).half()
        post = torch.rand(T, HC, device=dev, dtype=torch.float32)
        comb = torch.rand(T, HC, HC, device=dev, dtype=torch.float32) * 0.5
        ref = torch_ref(x, res, post, comb)
        got = hc_post_fused(x, res, post, comb)
        rel = ((got.float() - ref.float()).abs().max().item()
               / max(ref.abs().max().item(), 1e-6))
        good = rel < 2e-3
        ok &= good
        print(f"  T={T:5d}: max rel {rel:.2e}  {'ok' if good else 'FAIL'}")

    print("\nperformance (chunk 512 is the production prefill chunk):")
    for T in (1, 2, 512):
        x = (torch.randn(T, HIDDEN, device=dev) * 0.5).half()
        res = (torch.randn(T, HC, HIDDEN, device=dev) * 0.5).half()
        post = torch.rand(T, HC, device=dev, dtype=torch.float32)
        comb = torch.rand(T, HC, HC, device=dev, dtype=torch.float32) * 0.5
        t_ref = bench(lambda: torch_ref(x, res, post, comb))
        t_new = bench(lambda: hc_post_fused(x, res, post, comb))
        mb = (3 * T * HC * HIDDEN * 2 + T * HC * HC * 4 + T * HC * 4) / 2**20
        floor = mb / 616e3 * 1e3
        print(f"  T={T:5d}: torch {t_ref:7.3f} ms   fused {t_new:7.3f} ms   "
              f"{t_ref / t_new:5.2f}x   (BW floor {floor:.4f} ms)")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
