#!/usr/bin/env python
"""Does tl.sort behave at the magnitude the packed keys actually use?

The exact-merge prototype lost every winner at the final merge level: it read
1024 real values plus 1024 zero padding, sorted, and wrote 512 zeros. The same
kernel on synthetic values under 2^30 was correct in all six shapes tried.

The real difference is magnitude. A packed key is (float_bits << 32) | column,
so its int64 value sits just under 2^63 -- the top of the int64 range, not the
bottom. The earlier probe only generated small values and so never touched the
regime where the kernel actually runs.

This sweeps value magnitude against dtype and width, and separately checks
tl.maximum, since the bitonic network is built from it.

Run: tl_sort_magnitude.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sort_full(in_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs,
             tl.sort(tl.load(in_ptr + offs), descending=True))


@triton.jit
def _sort_head(in_ptr, out_ptr, N: tl.constexpr, K: tl.constexpr):
    offs = tl.arange(0, N)
    s = tl.sort(tl.load(in_ptr + offs), descending=True)
    tl.store(out_ptr + offs, s, mask=offs < K)


@triton.jit
def _max_reduce(in_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    v = tl.load(in_ptr + offs)
    tl.store(out_ptr, tl.max(v, axis=0))


def band(name, hi_bits):
    def f(n, dev, dt):
        # values whose top 32 bits equal hi_bits: exactly the packed-key regime
        cols = torch.arange(n, device=dev, dtype=torch.int64)
        return (torch.full((n,), hi_bits, device=dev, dtype=torch.int64) << 32) | cols
    f.__name__ = name
    return f


def main():
    dev = "cuda"
    torch.manual_seed(0)
    bands = [
        ("small  <2^30", 0x00000012),
        ("mid    <2^62", 0x1FFFFFFF),
        ("packed ~2^63", 0xFF800000),
        ("near-max  ", 0xFFFFFFFF),
    ]
    print(f"{'band':>13} {'N':>6} {'full sort':>10} {'head(512)':>11} "
          f"{'tl.max':>8}  got[:3]")
    for name, hb in bands:
        for N in (1024, 2048):
            src = band(name, hb)(N, dev, torch.int64)
            exp = torch.sort(src, descending=True).values
            dst = torch.zeros(N, device=dev, dtype=torch.int64)
            _sort_full[(1,)](src, dst, N=N, num_warps=4, num_stages=1)
            torch.cuda.synchronize()
            full = bool((exp == dst).all())
            dst2 = torch.zeros(N, device=dev, dtype=torch.int64)
            _sort_head[(1,)](src, dst2, N=N, K=512, num_warps=4, num_stages=1)
            torch.cuda.synchronize()
            head = bool((exp[:512] == dst2[:512]).all())
            out = torch.zeros(1, device=dev, dtype=torch.int64)
            _max_reduce[(1,)](src, out, N=N, num_warps=4, num_stages=1)
            torch.cuda.synchronize()
            mx = int(out.item()) == int(exp[0].item())
            print(f"{name:>13} {N:>6} {str(full):>10} {str(head):>11} "
                  f"{str(mx):>8}  {dst2[:3].tolist()}")


if __name__ == "__main__":
    main()