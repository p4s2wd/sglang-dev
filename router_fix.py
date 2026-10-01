"""Fix for the bf16-weight linear on SM75: the fast path is unreachable, not just slow.

Measured at the real router shape (N=256, K=4096, M=1) on a 2080 Ti:
  x=fp16 y=fp16 -> torch.mm(x, y.t(), out_dtype=fp32)   33.5 us
  x=fp16 y=bf16 -> torch.mm(x.float(), y.float().t())   58.0 us   (the fallback)
and the fast path is not merely skipped by the dtype equality test -- it cannot be
used at all, because torch.mm with bf16 inputs and fp32 output raises
  "gemm input type BFloat16 and output type float is only supported for CUDA devices
   with compute capability 8.0 or higher"
So on SM75 a bf16 weight is permanently stuck on the fallback that copies the whole
weight to fp32 on every call. The fix is to hold the weight in fp16 (the dtype the
model's activations already use on sub-90) and get fp32 by converting the small
output, not the large weight.

Also check the other bf16 weights that reach a linear, and whether the router weight is
actually bf16 at runtime or already cast by the loader.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch

dev = torch.device("cuda")


def bench(fn, iters=400, warmup=60):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


print("router shape N=256 K=4096 M=1 (and M=2 for bs=2)")
for M in (1, 2):
    N, K = 256, 4096
    x16 = torch.randn(M, K, dtype=torch.float16, device=dev)
    w16 = torch.randn(N, K, dtype=torch.float16, device=dev)
    w32 = w16.float()
    cur = bench(lambda: torch.mm(x16.float(), w32.t()))
    # candidate: fp16 GEMM then widen the small output
    cand = bench(lambda: torch.mm(x16, w16.t()).float())
    # candidate: fp16 GEMM with fp32 out_dtype (works on SM75 for fp16 inputs)
    try:
        cand2 = bench(lambda: torch.mm(x16, w16.t(), out_dtype=torch.float32))
    except Exception:
        cand2 = float("nan")
    print("  M=%d  current fallback %6.1f us | fp16 mm then .float() %6.1f us (%.2fx)"
          " | fp16 mm out_dtype=fp32 %6.1f us (%.2fx)"
          % (M, cur, cand, cur / cand, cand2, cur / cand2))

# Numerical check: does widening the output change the router's decisions?
print("\nnumerical agreement, fp16-mm-then-float vs the fp32 fallback")
torch.manual_seed(0)
worst = 0.0
for trial in range(20):
    N, K, M = 256, 4096, 1
    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    w = (torch.randn(N, K, device=dev) * 0.02).half()
    ref = torch.mm(x.float(), w.float().t())
    got = torch.mm(x, w.t()).float()
    rel = ((got - ref).abs() / ref.abs().clamp(min=1e-6)).max().item()
    # top-6 selection is what actually matters for a router
    top_ref = ref.topk(6, dim=-1).indices
    top_got = got.topk(6, dim=-1).indices
    same = (top_ref == top_got).float().mean().item()
    worst = max(worst, rel)
    if trial < 3:
        print("  trial %d: max rel %.3e  top-6 identical %.0f%%" % (trial, rel, same * 100))
print("  worst rel over 20 trials: %.3e" % worst)
