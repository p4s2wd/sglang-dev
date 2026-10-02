#!/usr/bin/env python
"""Packed top-K keys are negative int64. Confirm, then confirm uint64 fixes it.

A packed key is (float_bits << 32) | column. For a positive float the key is
float_bits ^ 0x80000000, which reaches 0xFFFFFFFF -- above 0x7FFFFFFF -- so the
shift puts a 1 in bit 63 and the packed value is negative when read as int64.
About half of all real scores land there.

The production kernel accumulates in tl.uint64 and is unaffected. The prototype
merge loaded the same values into int64 and sorted them signed, which orders the
largest scores last, selects the 512 smallest instead of the 512 largest, and
lets zero padding -- larger than any negative value -- take every output slot.
That is the whole cause of the lost winners, and it also invalidates the earlier
conclusion that the production merge is inexact on pre-sorted partials.

Every reference here is computed on the 32-bit key alone, which is non-negative
and so sorts identically signed or unsigned. torch.sort on the packed int64
cannot be used as a reference for the same bug it is meant to detect.

Run: tl_sort_sign.py
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")


@triton.jit
def _sort_head(in_ptr, out_ptr, N: tl.constexpr, K: tl.constexpr,
               UNSIGNED: tl.constexpr):
    offs = tl.arange(0, N)
    v = tl.load(in_ptr + offs)
    if UNSIGNED:
        v = v.to(tl.uint64, bitcast=True)
    s = tl.sort(v, descending=True)
    if UNSIGNED:
        s = s.to(tl.int64, bitcast=True)
    tl.store(out_ptr + offs, s, mask=offs < K)


def pack(scores):
    """The kernel's encoding, done exactly: (key << 32) | column."""
    bits = scores.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign = 1 << 31
    key = torch.where((bits & sign) != 0, (~bits) & 0xFFFFFFFF, bits ^ sign)
    cols = torch.arange(scores.numel(), device=scores.device, dtype=torch.int64)
    return (key << 32) | cols, key


def main():
    dev = "cuda"
    torch.manual_seed(0)
    K = 512
    print(f"{'cap':>8} {'neg frac':>9} {'min packed':>22} {'N':>6}  "
          f"{'int64 sort':>12}  {'uint64 sort':>12}")
    for cap in (16384, 150000):
        scores = torch.randn(cap, device=dev)
        p, key = pack(scores)
        neg = (p < 0).float().mean().item()
        # reference from the non-negative 32-bit key: signed/unsigned agree there
        order = torch.argsort(key, descending=True)[:K]
        ref = (key[order] << 32) | order.to(torch.int64)
        for N in (2048, 4096):
            src = p[:N].contiguous()
            k32 = key[:N]
            o = torch.argsort(k32, descending=True)[:K]
            ref_n = (k32[o] << 32) | o.to(torch.int64)
            res = []
            for un in (0, 1):
                dst = torch.zeros(N, device=dev, dtype=torch.int64)
                _sort_head[(1,)](src, dst, N=N, K=K, UNSIGNED=un, num_warps=4,
                                num_stages=1)
                torch.cuda.synchronize()
                bad = int((dst[:K] != ref_n).sum())
                res.append("exact" if bad == 0 else f"wrong {bad}")
            print(f"{cap:>8} {neg:>9.3f} {p.min().item():>22} {N:>6}  "
                  f"{res[0]:>12}  {res[1]:>12}")
    # and the whole point: does unsigned order recover torch.topk?
    scores = torch.randn(150000, device=dev)
    p, key = pack(scores)
    ref_idx = set(torch.topk(scores, K).indices.tolist())
    # descending by key, computed on the non-negative 32-bit half
    got = set(torch.sort(key, descending=True).values[:K].new_tensor([]).tolist()) \
        if False else set(torch.argsort(key, descending=True)[:K].tolist())
    print(f"\nsanity: unsigned key order == torch.topk indices? "
          f"{ref_idx == got}")
    print(f"         torch.sort on packed int64 == torch.topk indices? "
          f"{ref_idx == set((torch.sort(p, descending=True).values[:K] & 0xFFFFFFFF).tolist())}")


if __name__ == "__main__":
    main()