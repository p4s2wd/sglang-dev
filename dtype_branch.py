"""Is linear_bf16_fp32 taking the fp32-copy fallback because x is fp16 and y is bf16?

_linear_bf16_fp32_cublas (dsv4/gemm.py:115) fast path requires
    x.dtype == y.dtype and y.dtype in (bf16, fp16)
The checkpoint stores ffn.gate.weight as BF16. On SM75 the model runs its activations in
float16 (those cards have no bfloat16 math), so if the weight is left bf16 the equality
test fails and the fallback runs torch.mm(x.float(), y.float().t()) -- which allocates a
fresh fp32 copy of the weight on EVERY call. The module comment records this exact
mistake costing 2x at router shapes, and the trace shows an aten::copy_ at that line
costing 4.0 ms/token, 7.4% of decode compute.

Reproduce both branches at the real router shape (N=256, K=4096, M=1) with the real
dtype combinations and measure. If fp16-activation vs bf16-weight really lands in the
fallback, casting the weight once at load is a one-line fix worth ~4 ms/token.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch

dev = torch.device("cuda")
N, K, M = 256, 4096, 1


def bench(fn, iters=300, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


print("branch condition: x.dtype == y.dtype and y.dtype in (bf16, fp16)")
for xd, yd in ((torch.float16, torch.float16), (torch.float16, torch.bfloat16),
               (torch.bfloat16, torch.bfloat16)):
    fast = xd == yd and yd in (torch.bfloat16, torch.float16)
    x = torch.randn(M, K, dtype=xd, device=dev)
    y = torch.randn(N, K, dtype=yd, device=dev)
    if fast:
        t = bench(lambda: torch.mm(x, y.t(), out_dtype=torch.float32))
        path = "FAST  torch.mm(out_dtype=fp32)"
    else:
        t = bench(lambda: torch.mm(x.float(), y.float().t()))
        path = "SLOW  x.float(), y.float().t()"
    print("  x=%-9s y=%-9s -> %s  %6.1f us" % (str(xd).split(".")[-1],
                                               str(yd).split(".")[-1], path, t))

print("\nwhat the fix would cost: cast weight to activation dtype once, then fast path")
x = torch.randn(M, K, dtype=torch.float16, device=dev)
y = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
t_slow = bench(lambda: torch.mm(x.float(), y.float().t()))
t_fast = bench(lambda: torch.mm(x, y.to(torch.float16).t(), out_dtype=torch.float32))
print("  SLOW (per-call fp32 copy) : %6.1f us" % t_slow)
print("  cast-then-fast per call   : %6.1f us  (includes the .to() each call)" % t_fast)
w = y.to(torch.float16)
t_pre = bench(lambda: torch.mm(x, w.t(), out_dtype=torch.float32))
print("  weight pre-cast at load   : %6.1f us  -> %.2fx vs SLOW" % (t_pre, t_slow / t_pre))
print("\nper token over 43 layers: SLOW %.2f ms, pre-cast %.2f ms, saving %.2f ms"
      % (t_slow * 43 / 1e3, t_pre * 43 / 1e3, (t_slow - t_pre) * 43 / 1e3))
