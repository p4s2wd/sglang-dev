#!/usr/bin/env python
"""SUPERSEDED -- kept only as a record of the investigation.

This script compares merge variants on pre-sorted partials, and it stores the
packed keys in an int64 buffer. That is the bug: a packed key is
(float_bits << 32) | column, and for a positive float the key reaches
0xFFFFFFFF, so the shift sets bit 63 and roughly half of all real keys are
negative as int64. Compared signed, every variant here "loses" 5 to 251
selections and the production merge looks inexact when it is exact.

The conclusion this script was used to draw is wrong. See
NOTES-2026-10-02-decode-topk.md section 7.1, and use topk_merge_search.py,
which does the same sweep on uint64 and reports 64/64 exact.

Which merge formulation reproduces torch.topk on pre-sorted partials?

The production kernel merges with

    acc = tl.bitonic_merge(acc)
    acc = tl.maximum(acc, tl.topk(candidate, K_POW2, dim=0))

and matches torch.topk exactly when `candidate` is a fresh block of raw scores.
Feeding it an already-sorted 512-wide partial instead does not reproduce it --
497 of 512 selections differ -- so the formulation's correctness depends on the
input, and guessing which variant is safe is cheaper measured than reasoned.

Each variant is checked against torch.topk on correct partials, so a pass here
means the merge is exact, not merely self-consistent.
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")


@triton.jit
def _merge_variant(
    partials_ptr, out_raw_ptr, seq_lens_ptr,
    K: tl.constexpr, K_POW2: tl.constexpr, N_PARTIALS: tl.constexpr,
    VARIANT: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, K_POW2)
    acc = tl.zeros((K_POW2,), dtype=tl.uint64)
    for s in range(0, N_PARTIALS):
        cand = tl.load(partials_ptr + (row * N_PARTIALS + s) * K_POW2 + offs)
        if VARIANT == 0:      # production shape
            acc = tl.bitonic_merge(acc)
            acc = tl.maximum(acc, tl.topk(cand, K_POW2, dim=0))
        elif VARIANT == 1:    # keep acc genuinely sorted before merging
            acc = tl.sort(acc, descending=True)
            acc = tl.bitonic_merge(acc)
            acc = tl.maximum(acc, tl.topk(cand, K_POW2, dim=0))
        elif VARIANT == 2:    # sort both sides, no bitonic_merge
            acc = tl.sort(acc, descending=True)
            acc = tl.maximum(acc, tl.sort(cand, descending=True))
        elif VARIANT == 3:    # sort acc, then re-sort the max (cheap at 25 steps)
            acc = tl.sort(acc, descending=True)
            m = tl.maximum(acc, cand)
            acc = tl.sort(m, descending=True)
    acc = tl.sort(acc, descending=True)
    raw = (acc & 0xFFFFFFFF).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + row)
    tl.store(out_raw_ptr + row * K + offs, tl.where(offs < K, raw, -1), mask=offs < K)


def pack_desc(vals):
    sb = vals.to(torch.float32).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign = torch.tensor(0x80000000, dtype=torch.int64, device=vals.device)
    key = torch.where((sb & sign) != 0, (~sb) & 0xFFFFFFFF, sb ^ sign)
    return key << 32


def main():
    dev = "cuda"
    torch.manual_seed(0)
    K, KP = 512, 512
    names = {0: "production shape", 1: "sort acc first",
             2: "sort both, no bitonic_merge", 3: "sort + sort(max)"}
    print(f"{'cap':>8} {'npart':>6} " + " ".join(f"{names[v]:>24}" for v in names))
    for cap in (4096, 37500, 150000):
        for npart_req in (8, 32):
            slab = ((triton.cdiv(cap, npart_req) + 511) // 512) * 512
            npart = triton.cdiv(cap, slab)
            scores = torch.randn(1, cap, device=dev)
            part = torch.zeros(npart * KP, dtype=torch.int64, device=dev)
            for s in range(npart):
                lo, hi = s * slab, min((s + 1) * slab, cap)
                v, idx = torch.topk(scores[0, lo:hi], min(K, hi - lo))
                p = torch.sort(pack_desc(v) | (idx.to(torch.int64) + lo),
                               descending=True).values
                part[s * KP: s * KP + len(p)] = p
            sl_t = torch.full((1,), cap, dtype=torch.int32, device=dev)
            ref = set(torch.topk(scores[0], K).indices.tolist())
            line = f"{cap:>8} {npart:>6} "
            for v in names:
                o = torch.empty(1, K, dtype=torch.int32, device=dev)
                _merge_variant[(1,)](part, o, sl_t, K=K, K_POW2=KP,
                                     N_PARTIALS=npart, VARIANT=v,
                                     num_warps=4, num_stages=1)
                torch.cuda.synchronize()
                got = set(o[0].tolist())
                line += f"{'OK' if got == ref else f'{len(got^ref)//2} wrong':>24} "
            print(line, flush=True)


if __name__ == "__main__":
    main()
