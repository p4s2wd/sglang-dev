"""Decode GEMV must equal the dequantized reference, and beat the mma kernel.

The kernel assumes the decode layout moe_align_block_size produces: one expert
per 16-row block, one real row per block, the other 15 slots carrying the
sentinel. It also must ignore slots past num_valid, whose buffers are
capacity-sized torch.empty memory.

Correctness is checked against dequant_mxfp4_reference + a plain matmul, which
is independent of both kernels. Then the same shapes go through
mxfp4_w4a16_gemm_ptx_direct so the speedup is measured, not assumed.
"""
import os as _os
import pathlib as _pb
import time

import torch


def _find_repo():
    r = _os.environ.get("SGLANG_REPO")
    if r:
        return r
    d = _pb.Path(__file__).resolve().parent
    for _ in range(8):
        for cand in (d, d / "sglang"):
            if (cand / "python" / "sglang").is_dir():
                return str(cand)
        d = d.parent
    raise RuntimeError("set SGLANG_REPO")


_PY = _find_repo() + "/python"

from sglang.kernels.ops.moe.mxfp4_w4a16_kernels import (  # noqa: E402
    dequant_mxfp4_reference,
    get_gemv_module,
    mxfp4_w4a16_gemm_ptx_direct,
    mxfp4_w4a16_gemv,
)

torch.cuda.set_device(0)
dev = "cuda:0"
BLOCK_M = 16


def build(n_act, n, k, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    m_total = n_act  # one real row per expert
    num_slots = n_act * BLOCK_M
    sorted_ids = torch.full((num_slots,), m_total, dtype=torch.int32, device=dev)
    for e in range(n_act):
        sorted_ids[e * BLOCK_M] = e
    eids = torch.arange(n_act, dtype=torch.int32, device=dev)
    num_valid = torch.full((1,), num_slots, dtype=torch.int32, device=dev)

    a = (torch.randn(m_total, k, generator=g, device=dev) * 0.3).half()
    w = torch.randint(-128, 127, (n_act, n, k // 2), dtype=torch.int8, device=dev)
    s = torch.randint(118, 127, (n_act, n, k // 32), dtype=torch.uint8, device=dev)
    return a, w, s, sorted_ids, eids, num_valid, m_total, num_slots


def reference(a, w, s, sorted_ids, eids, num_slots, n):
    """out[slot] = a[id] @ dequant(w[expert]).T, computed in fp32."""
    wdeq = dequant_mxfp4_reference(w, s).float()   # [E, N, K]
    out = torch.zeros((num_slots, n), dtype=torch.float32, device=dev)
    for slot in range(num_slots):
        sid = int(sorted_ids[slot])
        if sid >= a.shape[0]:
            continue
        e = int(eids[slot // BLOCK_M])
        out[slot] = a[sid].float() @ wdeq[e].t()
    return out


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


ok = True
for (n_act, n, k) in ((6, 4096, 4096), (6, 2048, 4096), (48, 2048, 4096),
                      (6, 2048, 1024)):
    a, w, s, sorted_ids, eids, num_valid, m_total, num_slots = build(n_act, n, k)
    ref = reference(a, w, s, sorted_ids, eids, num_slots, n)

    out = torch.zeros((num_slots, n), dtype=torch.float16, device=dev)
    try:
        mxfp4_w4a16_gemv(a, w, s, sorted_ids, eids, out,
                         sentinel=m_total, num_valid=num_valid)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"n_act={n_act} n={n} k={k}: GEMV FAILED "
              f"{type(e).__name__}: {str(e)[:150]}")
        ok = False
        continue

    # Compare only the real rows; padding rows are intentionally untouched.
    real = torch.tensor([slot for slot in range(num_slots)
                         if int(sorted_ids[slot]) < m_total], device=dev)
    got = out[real].float()
    exp = ref[real]
    rel = (got - exp).abs().max().item() / max(exp.abs().max().item(), 1e-9)
    good = rel < 2e-3
    ok = ok and good

    t_g = bench(lambda: mxfp4_w4a16_gemv(
        a, w, s, sorted_ids, eids, out, sentinel=m_total, num_valid=num_valid))
    out2 = torch.zeros((num_slots, n), dtype=torch.float16, device=dev)
    t_m = bench(lambda: mxfp4_w4a16_gemm_ptx_direct(
        a, w, s, sorted_ids, eids, out2, num_valid=num_valid, sentinel=m_total))
    wbytes = w.numel()
    print(f"n_act={n_act:3d} n={n:5d} k={k:5d}  rel={rel:.1e} "
          f"{'ok' if good else 'FAIL'}  gemv {t_g:6.3f}ms ({wbytes/t_g/1e6:4.0f} GB/s) "
          f"mma {t_m:6.3f}ms ({wbytes/t_m/1e6:4.0f} GB/s)  = {t_m/t_g:.2f}x")

print("\nW4A16 DECODE GEMV:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
