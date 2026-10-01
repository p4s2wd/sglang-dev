"""True memory bandwidth of GPU 2 (2080 Ti) — clean copy + read-only tests."""
import time
import torch

dev = "cuda:0"
N = 1 << 28  # 256 MiB
x = torch.empty(N, dtype=torch.uint8, device=dev)
y = torch.empty(N, dtype=torch.uint8, device=dev)

def timeit(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

# DtoD copy: 2x bytes moved
ms = timeit(lambda: y.copy_(x)) * 1000
print(f"DtoD copy:   {ms:7.3f} ms  {2*N/ms/1e6:6.0f} GB/s")

# read-only reduce (uint8 -> sum in fp32 via view trick to avoid materializing)
ms = timeit(lambda: x.view(torch.int32).sum()) * 1000
print(f"read sum:    {ms:7.3f} ms  {N/ms/1e6:6.0f} GB/s")

# fp16 GEMM flops (Turing TC)
a = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
b = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
ms = timeit(lambda: a @ b) * 1000
print(f"fp16 GEMM:   {ms:7.3f} ms  {2*4096**3/ms/1e9:6.0f} GFLOP/s")

props = torch.cuda.get_device_properties(0)
print(f"clock check: name={props.name}")
