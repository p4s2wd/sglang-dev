#!/usr/bin/env python
"""Is tl.sort itself correct on this GPU, at the widths the merge needs?

The exact-merge prototype produced plausible timings but its output was wrong:
after the final merge level, 512 of 512 true winners were missing. The per-slab
stage was exact, which isolates the fault to tl.sort (or to the masked store
that reads its result).

This tests the primitive on its own -- plain sort, and the masked head-store the
merge kernel actually uses -- so a wrong answer here is a Triton/SM75 problem
rather than anything about top-K.

Run: tl_sort_probe.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sort_full(in_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    s = tl.sort(tl.load(in_ptr + offs), descending=True)
    tl.store(out_ptr + offs, s)


@triton.jit
def _sort_head(in_ptr, out_ptr, N: tl.constexpr, K: tl.constexpr):
    """The merge kernel's shape: sort N, store only the leading K lanes."""
    offs = tl.arange(0, N)
    s = tl.sort(tl.load(in_ptr + offs), descending=True)
    tl.store(out_ptr + offs, s, mask=offs < K)


def main():
    dev = "cuda"
    torch.manual_seed(0)
    print(f"{'N':>6} {'dtype':>8} {'warps':>5}  {'full':>7}  {'head(K=512)':>11}")
    for dt in (torch.int64, torch.int32):
        for N in (512, 1024, 2048, 4096):
            for warps in (4, 8):
                src = torch.randint(0, 1 << 30, (N,), device=dev, dtype=dt)
                dst = torch.zeros(N, device=dev, dtype=dt)
                exp = torch.sort(src, descending=True).values
                try:
                    _sort_full[(1,)](src, dst, N=N, num_warps=warps,
                                     num_stages=1)
                    torch.cuda.synchronize()
                    full = bool((exp == dst).all())
                except Exception as e:
                    full = f"FAIL {type(e).__name__}"
                try:
                    dst2 = torch.zeros(N, device=dev, dtype=dt)
                    _sort_head[(1,)](src, dst2, N=N, K=512, num_warps=warps,
                                     num_stages=1)
                    torch.cuda.synchronize()
                    head = bool((exp[:512] == dst2[:512]).all())
                except Exception as e:
                    head = f"FAIL {type(e).__name__}"
                name = str(dt).split(".")[-1]
                print(f"{N:>6} {name:>8} {warps:>5}  {str(full):>7}  "
                      f"{str(head):>11}")


if __name__ == "__main__":
    main()