"""Sweep BLOCK_T / num_warps / num_stages for the head-shared sparse kernel.

The shipped launcher hardcodes BLOCK_T=16, num_warps=8, num_stages=1. At the
production topk=512 that is 32 serial gather+mma iterations per program, and the
EXTEND trace shows grid=[173,2] -- only 346 programs on 43 SMs, so there may be
room to trade per-program width for fewer iterations. BLOCK_T=32 was rejected
earlier for a single 512-wide tile (70656 B of SMEM vs the 65536 B SM75 cap),
but the shipped kernel splits nope into two 256-wide halves, which changes the
SMEM math.

Mirrors _run_headshared_sparse_decode's exact call, parameterised.
"""
import sys
import time

import torch
import triton

repo = "/data/nvme/sglang-codex/sglang"
sys.path.insert(0, repo + "/python")

from sglang.kernels.ops.attention import flash_mla_sm120_triton as M  # noqa: E402

_HS_TOKEN_BYTES = M._HS_TOKEN_BYTES


def run_cfg(q, k_cache, flat_indices, softmax_scale, page_size, page_bytes,
            block_t, num_warps, num_stages, bufs=None):
    B, _, H, D = q.shape
    topk = flat_indices.shape[1]
    total_elems = k_cache.shape[0] * page_bytes
    raw_uint8 = k_cache.as_strided((total_elems,), (1,)).view(torch.uint8)
    raw_bf16 = raw_uint8.view(torch.bfloat16)
    lut = M.fp8_payload_lut(q.device, torch.float32)
    q3 = q.squeeze(1).contiguous()
    if bufs is None:
        out = torch.zeros(B, H, D, dtype=q.dtype, device=q.device)
        lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)
    else:
        out, lse = bufs
    grid = (B, triton.cdiv(H, 16))
    M._headshared_sparse_kernel[grid](
        q3, raw_uint8, raw_bf16, lut, flat_indices,
        torch.empty(0, device=q.device, dtype=torch.int32),
        out, lse,
        softmax_scale, page_size, int(page_bytes),
        int(page_size * _HS_TOKEN_BYTES),
        H, topk, False,
        q3.stride(0), q3.stride(1), out.stride(0), out.stride(1),
        BLOCK_H=16, BLOCK_T=block_t,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out, lse


def build(B, H, D, topk, num_pages, page_size, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(B, 1, H, D, generator=g, device="cuda",
                     dtype=torch.float32) * 0.5).half()
    vals = (torch.randn(num_pages * page_size * 448, generator=g, device="cuda")
            * 0.3).to(torch.float8_e4m3fn).view(torch.uint8)
    cache = torch.zeros(num_pages, page_size, 576 + 8, dtype=torch.uint8,
                        device="cuda")
    cache[:, :, :448] = vals.view(num_pages, page_size, 448)
    cache[:, :, 576:] = 120
    indices = torch.randint(0, num_pages * page_size, (B, 1, topk),
                            generator=g, device="cuda", dtype=torch.int32)
    return q, cache, indices.reshape(B, -1).contiguous()


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
    page_size = 64
    num_pages = (B * topk) // page_size + 8
    q, cache, flat = build(B, H, D, topk, num_pages, page_size)
    scale = 1.0 / (D ** 0.5)
    page_bytes = cache.stride(0)

    out_ref, lse_ref = run_cfg(q, cache, flat, scale, page_size, page_bytes,
                               16, 8, 1)
    torch.cuda.synchronize()
    t_ship = timeit(lambda: run_cfg(q, cache, flat, scale, page_size, page_bytes,
                                    16, 8, 1))
    print(f"shipped (BLOCK_T=16, warps=8, stages=1): {t_ship:7.3f} ms")

    bufs = (torch.zeros(B, H, D, dtype=q.dtype, device=q.device),
            torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device))
    for bt in (16, 32, 64):
        for nw in (4, 8):
            for ns in (1, 2):
                try:
                    out, lse = run_cfg(q, cache, flat, scale, page_size,
                                       page_bytes, bt, nw, ns, bufs)
                    torch.cuda.synchronize()
                    t = timeit(lambda: run_cfg(q, cache, flat, scale, page_size,
                                               page_bytes, bt, nw, ns, bufs))
                    rel = ((out.float() - out_ref.float()).abs().max().item()
                           / max(out_ref.abs().max().item(), 1e-6))
                    flag = "" if rel < 1e-2 else "  <-- MISMATCH"
                    print(f"  BLOCK_T={bt:3d} warps={nw} stages={ns}: "
                          f"{t:7.3f} ms ({t_ship / t:5.2f}x)  rel {rel:.1e}{flag}")
                except Exception as e:
                    msg = str(e).splitlines()[0][:70]
                    print(f"  BLOCK_T={bt:3d} warps={nw} stages={ns}: {msg}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
