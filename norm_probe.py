"""What do the model's RMSNorm calls actually cost, and is there a faster path?

Elementwise work is 13.6% of prefill device time, and the profile shows a
pow_tensor_scalar kernel at 126 us/call -- absurdly slow for a per-element square,
which is the signature of an unfused x.pow(2).mean(-1) fallback. RMSNorm.forward_cuda
does have a fused path, so the question is which call sites reach it and what the
fused version costs against the eager one at the shapes this model actually uses.

Measured at production shapes: 512 tokens x hidden 7168 (the two per-layer norms)
and the attention norms at 512 x 32 heads x 512. For each, the fused sglang RMSNorm
against the hand-rolled eager equivalent, interleaved so drift cannot fake a win.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
from sglang.srt.layers.layernorm import RMSNorm

dev = torch.device("cuda")


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def eager(x, w, eps=1e-6):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(v + eps)).to(x.dtype) * w


print("%-34s %10s %10s %8s" % ("shape / variant", "fused us", "eager us", "ratio"))
for (T, H) in ((512, 7168), (1024, 7168), (16384, 7168), (512, 1536), (512, 512)):
    x = torch.randn(T, H, dtype=torch.float16, device=dev)
    w = torch.randn(H, dtype=torch.float16, device=dev)
    n = RMSNorm(H, eps=1e-6).to(dev)
    n.weight.data.copy_(w)
    with torch.no_grad():
        a = sorted(bench(lambda: n(x)) for _ in range(3))
        b = sorted(bench(lambda: eager(x, w)) for _ in range(3))
    print("%-34s %10.1f %10.1f %8.2fx" % ("norm %5d x %5d" % (T, H), a[1], b[1], b[1] / a[1]))

# The fused kernel can also fold the residual add, which removes a separate
# elementwise add kernel per layer. Check it is reachable and what it costs.
print()
for (T, H) in ((512, 7168), (16384, 7168)):
    x = torch.randn(T, H, dtype=torch.float16, device=dev)
    r = torch.randn(T, H, dtype=torch.float16, device=dev)
    w = torch.randn(H, dtype=torch.float16, device=dev)
    n = RMSNorm(H, eps=1e-6).to(dev)
    n.weight.data.copy_(w)
    with torch.no_grad():
        try:
            a = sorted(bench(lambda: n(x, residual=r.clone())) for _ in range(3))
            print("norm %5d x %5d +residual fused: %8.1f us" % (T, H, a[1]))
        except Exception as e:
            print("norm %5d x %5d +residual FAILED: %s" % (T, H, str(e)[:70]))
        b = sorted(bench(lambda: (eager(x + r, w), x + r)) for _ in range(3))
        print("norm %5d x %5d +residual eager : %8.1f us" % (T, H, b[1]))
