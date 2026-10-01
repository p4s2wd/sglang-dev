"""Gate test for the W8A16 kernel before it touches the loader.

The loader change is only worth making if this kernel (a) matches an explicit
dequant + fp32 matmul and (b) actually streams at a useful fraction of the card's
bandwidth on the shapes DeepSeek-V4-Flash runs at TP2. Correctness failures here
would silently corrupt every dense projection in the model, so the reference is
computed independently of the kernel's own scale arithmetic.
"""
import os
import sys
import time

import torch


def _find_repo():
    import pathlib

    r = os.environ.get("SGLANG_REPO")
    if r:
        return r
    d = pathlib.Path(__file__).resolve().parent
    for _ in range(8):
        for cand in (d, d / "sglang"):
            if (cand / "python" / "sglang").is_dir():
                return str(cand)
        d = d.parent
    raise RuntimeError("set SGLANG_REPO to the sglang checkout")


sys.path.insert(0, _find_repo() + "/python")

from sglang.kernels.ops.quantization.fp8_w8a16 import (
    _should_use_w8a16_wide,
    w8a16_linear,
)

torch.cuda.set_device(0)
dev = "cuda:0"
BS = 128
PEAK_GBPS = 616.0


def make(n, k, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device=dev) * 0.05).to(torch.float8_e4m3fn)
    # ceil: a ragged trailing block still gets a scale, as in the checkpoint.
    exp = torch.randint(
        -8, -1, ((n + BS - 1) // BS, (k + BS - 1) // BS), generator=g, device=dev
    )
    scale = torch.pow(2.0, exp.float())
    return w, scale


def reference(x, w, scale):
    """Explicit dequant + fp32 matmul, independent of the kernel's arithmetic.

    repeat_interleave then slice rather than view: the last block row/column can
    be short, and a view would reject a ragged shape.
    """
    n, k = w.shape
    nb, kb = scale.shape
    full = (
        scale.repeat_interleave(BS, dim=0)[:n]
        .repeat_interleave(BS, dim=1)[:, :k]
    )
    return x.float() @ (w.float() * full).t()


def bench(fn, iters=80, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


SHAPES = [
    ("q_a  4096x1536", 1536, 4096),
    ("kv_a 4096x640", 640, 4096),
    ("q_b  4096x16384", 16384, 4096),
    ("o    16384x4096", 4096, 16384),
    ("idx  4096x4096", 4096, 4096),
]

ok = True
print(f"peak {PEAK_GBPS:.0f} GB/s (2080 Ti datasheet)\n")
print(f"{'shape':>18} {'rel':>9} {'ms':>7} {'GB/s':>7} {'%pk':>6} {'vs fp16':>8}")
total_fp8 = total_fp16 = 0.0
for label, n, k in SHAPES:
    w, scale = make(n, k)
    x = (torch.randn(1, k, device=dev) * 0.5).half()
    ref = reference(x, w, scale)

    got = w8a16_linear(x, w, scale)
    torch.cuda.synchronize()
    rel = (got.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    correct = rel < 5e-3
    ok = ok and correct

    dt = bench(lambda: w8a16_linear(x, w, scale))
    gbps = w.numel() / dt / 1e9

    w16 = w.float().half()
    dt16 = bench(lambda: torch.mm(x, w16.t()))
    total_fp8 += dt
    total_fp16 += dt16

    # Per-shape speedup is reported but not required: a 2.6 MB projection is
    # launch bound, where no kernel beats cuBLAS and the absolute cost is
    # microseconds. What matters is the sum over the projections a token runs.
    print(f"{label:>18} {rel:>9.1e} {dt*1e3:>7.3f} {gbps:>7.1f} "
          f"{100*gbps/PEAK_GBPS:>5.1f}% {dt16/dt:>7.2f}x"
          f"{'' if correct else '  <-- INCORRECT'}")

print(f"\nper-layer dense GEMV total: fp8 {total_fp8*1e3:.3f} ms vs "
      f"fp16 {total_fp16*1e3:.3f} ms -> {total_fp16/total_fp8:.2f}x, "
      f"saves {(total_fp16-total_fp8)*1e3*43:.1f} ms over 43 layers")
ok = ok and total_fp8 < total_fp16

assert not _should_use_w8a16_wide(1, 7168, 1024)
assert _should_use_w8a16_wide(1, 7168, 1024, allow_k1024=True)

# Ragged K: the model's projections are all 128-divisible, but a masked path
# that never runs is not a tested path.
w, scale = make(640, 1000)
x = (torch.randn(1, 1000, device=dev) * 0.5).half()
ref = reference(x, w, scale)
got = w8a16_linear(x, w, scale)
rel = (got.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
print(f"\nragged K=1000 (masked path): rel {rel:.1e} "
      f"{'ok' if rel < 5e-3 else 'FAIL'}")
ok = ok and rel < 5e-3

# Batch > 1 exercises the tl.dot path used by prefill and by decode at bs>1.
w, scale = make(4096, 4096)
x = (torch.randn(8, 4096, device=dev) * 0.5).half()
ref = reference(x, w, scale)
got = w8a16_linear(x, w, scale)
rel = (got.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
print(f"batch=8 GEMM path: rel {rel:.1e} {'ok' if rel < 5e-3 else 'FAIL'}")
ok = ok and rel < 5e-3

print("\nW8A16 KERNEL:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
