"""SM75 Triton smoke test: verify Triton compiles/runs on Turing (SM 7.5).

Run: CUDA_VISIBLE_DEVICES=<n> python sm75_triton_smoke.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x + 1.0, mask=mask)


@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, K: tl.constexpr):
    offs = tl.arange(0, 16)
    a = tl.load(a_ptr + offs[:, None] * K + tl.arange(0, K)[None, :])
    b = tl.load(b_ptr + tl.arange(0, K)[:, None] * 16 + offs[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs[:, None] * 16 + offs[None, :], c)


def main():
    dev = "cuda:0"
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    x = torch.randn(1024, device=dev, dtype=torch.float16)
    y = torch.empty_like(x)
    add_kernel[(4,)](x, y, 1024, BLOCK=256)
    torch.cuda.synchronize()
    print("triton elementwise on sm75 OK:", torch.allclose(y, x + 1.0))

    a = torch.randn(16, 64, device=dev, dtype=torch.float16)
    b = torch.randn(64, 16, device=dev, dtype=torch.float16)
    c = torch.empty(16, 16, device=dev, dtype=torch.float32)
    dot_kernel[(1,)](a, b, c, K=64)
    torch.cuda.synchronize()
    print(
        "tl.dot fp16 on sm75 OK:",
        torch.allclose(c, (a.float() @ b.float()), atol=1e-1),
    )


if __name__ == "__main__":
    main()
