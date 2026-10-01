"""Can a Triton GEMV beat cuBLAS on the bf16 linears that are 14% of decode?

The true DECODE ranking puts cuBLAS internal::gemvx::kernel at 14% -- 184 calls per
step at 58 us each. The stack join attributes this family to
kernels/ops/attention/dsv4/gemm.py:_linear_bf16_fp32_cublas, used by the compressor
(compressor.py:492). Those weights are BF16 in the checkpoint, so they have no block
scale and cannot use the W8A16 GEMV, which measures 483-495 GB/s on this card.

At 58 us a call moves only a few megabytes, so it is running an order of magnitude
below the bandwidth the same card sustains elsewhere. The module comment records that
this path was already improved once (61.5 -> 30.5 us at N=384,K=4096,M=1) by switching
to torch.mm(out_dtype=fp32); the remaining cost is cuBLAS being poor at tiny-M GEMVs.

Measure cuBLAS against the existing W8A16-style Triton GEMV shape at the real
compressor dimensions. If a Triton GEMV wins, routing these few call sites is a
bounded change worth ~10% of decode.
"""
import json, sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
from safetensors import safe_open

D = "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731"
idx = json.load(open(D + "/model.safetensors.index.json"))
dev = torch.device("cuda")

shapes = {}
for k, f in idx["weight_map"].items():
    if k.startswith("layers.1.") and k.endswith(".weight"):
        with safe_open(D + "/" + f, framework="pt") as sf:
            t = sf.get_slice(k)
            sh = tuple(t.get_shape())
            if len(sh) == 2 and t.get_dtype() in ("BF16", "F32") and ".experts." not in k:
                shapes[k[len("layers.1."):-len(".weight")]] = (sh, t.get_dtype())
print("non-FP8 2-D per-layer weights:")
for n, (sh, dt) in shapes.items():
    print("  %-28s %-14s %s" % (n[:28], "x".join(map(str, sh)), dt))


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


print("\nM=1 linear, cuBLAS out_dtype=fp32 vs plain fp16 mm vs Triton GEMV")
print("%-26s %12s %12s %12s %10s" % ("shape", "cublas fp32", "mm fp16", "w8a16-style", "GB/s best"))
from sglang.kernels.ops.quantization.fp8_w8a16 import w8a16_linear
for n, (sh, dt) in shapes.items():
    N, K = sh
    if N * K > 64e6:
        continue
    x = torch.randn(1, K, dtype=torch.float16, device=dev)
    w = torch.randn(N, K, dtype=torch.float16, device=dev)
    t_cublas = bench(lambda: torch.mm(x, w.t(), out_dtype=torch.float32))
    t_mm = bench(lambda: torch.mm(x, w.t()))
    wq = torch.randint(0, 255, (N, K), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    sc = torch.ones(max(1, N // 128), max(1, K // 128), dtype=torch.float32, device=dev)
    try:
        t_tri = bench(lambda: w8a16_linear(x, wq, sc))
    except Exception:
        t_tri = float("nan")
    gb = N * K * 2 / 1e9
    best = min(v for v in (t_cublas, t_mm, t_tri) if v == v)
    print("%-26s %12.1f %12.1f %12.1f %10.1f"
          % ("%s %dx%d" % (n[:18], N, K), t_cublas, t_mm, t_tri, gb / (best / 1e6) * 1e-3 * 1e3))
