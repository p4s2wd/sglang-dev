#!/usr/bin/env python
"""Re-test the production merge recurrence on pre-sorted partials, in uint64.

This recurrence -- bitonic_merge then elementwise maximum -- was previously
judged inexact on pre-sorted partials, losing 5 to 251 selections depending on
shape, and was replaced by a full sort. That verdict was an artifact: the packed
keys were compared as signed int64, so the sort took the 512 smallest and zero
padding took every slot. The production kernel itself accumulates in tl.uint64
and is exact.

So the cheap recurrence deserves a second hearing. If it is exact, it avoids the
2048-wide sorts that dominate the full-sort merge, and the whole kernel gets
faster. Compared against torch.topk on real scores, so a pass means exact rather
than self-consistent.

Sweeps the two choices that were confounded before: whether the slab kernel
sorts its accumulator before writing it, and whether the merge applies tl.topk
to the incoming partial.

Run: topk_merge_search.py
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")


@triton.jit
def _slab(
    scores_ptr, seq_lens_ptr, part_ptr,
    stride_scores, n_part, K: tl.constexpr, K_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr, SLAB: tl.constexpr, DO_SORT: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // n_part
    s = pid % n_part
    seq_len = tl.load(seq_lens_ptr + row)
    start0 = s * SLAB
    offs_k = tl.arange(0, K_POW2)
    if start0 >= seq_len:
        tl.store(part_ptr + pid * K_POW2 + offs_k, tl.zeros((K_POW2,), tl.uint64)
                 .to(tl.int64, bitcast=True))
        return
    end = tl.minimum(start0 + SLAB, seq_len)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((K_POW2,), dtype=tl.uint64)
    for start in range(start0, end, BLOCK_N):
        cols = start + offs_n
        score = tl.load(scores_ptr + row * stride_scores + cols,
                        mask=cols < end, other=float("-inf"))
        sb = score.to(tl.uint32, bitcast=True)
        sign = tl.full(sb.shape, 0x80000000, tl.uint32)
        key = tl.where((sb & sign) != 0, ~sb, sb ^ sign)
        packed = (key.to(tl.uint64) << 32) | cols.to(tl.uint64)
        cand = tl.topk(packed, K_POW2, dim=0)
        acc = tl.bitonic_merge(acc)
        acc = tl.maximum(acc, tl.topk(cand, K_POW2, dim=0))
    if DO_SORT:
        acc = tl.sort(acc, descending=True)
    tl.store(part_ptr + pid * K_POW2 + offs_k, acc.to(tl.int64, bitcast=True))


@triton.jit
def _merge(
    part_ptr, out_raw_ptr, seq_lens_ptr,
    K: tl.constexpr, K_POW2: tl.constexpr, N_PARTIALS: tl.constexpr,
    USE_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, K_POW2)
    acc = tl.zeros((K_POW2,), dtype=tl.uint64)
    for s in range(0, N_PARTIALS):
        cand = tl.load(part_ptr + (row * N_PARTIALS + s) * K_POW2 + offs)
        # Partials are packed keys, and half of them are negative as int64.
        # The recurrence compares with unsigned maximum, so both the load and
        # the store have to be reinterpreted, not converted.
        cand = cand.to(tl.uint64, bitcast=True)
        if USE_TOPK:
            cand = tl.topk(cand, K_POW2, dim=0)
        acc = tl.bitonic_merge(acc)
        acc = tl.maximum(acc, tl.topk(cand, K_POW2, dim=0))
    acc = tl.sort(acc, descending=True)
    raw = (acc & 0xFFFFFFFF).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + row)
    tl.store(out_raw_ptr + row * K + offs, tl.where(offs < K, raw, -1),
             mask=offs < K)


def main():
    dev = "cuda"
    K = KP = 512
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    print(f"{'cap':>8} {'npart':>6} " +
          " ".join(f"{'sortA=%d,topkB=%d' % c:>16}" for c in combos))
    worst = 0
    for cap in (4096, 16384, 37500, 150000):
        for n_req in (8, 25, 32, 64):
            slab = ((triton.cdiv(cap, n_req) + 511) // 512) * 512
            npart = triton.cdiv(cap, slab)
            torch.manual_seed(cap + n_req)
            scores = torch.randn(1, cap, device=dev)
            sl_t = torch.full((1,), cap, dtype=torch.int32, device=dev)
            ref = set(torch.topk(scores[0], K).indices.tolist())
            line = f"{cap:>8} {npart:>6} "
            for a_sort, b_topk in combos:
                part = torch.zeros(npart * KP, dtype=torch.int64, device=dev)
                _slab[(npart,)](scores, sl_t, part, scores.stride(0), npart,
                                K=K, K_POW2=KP, BLOCK_N=512, SLAB=slab,
                                DO_SORT=bool(a_sort), num_warps=4,
                                num_stages=1)
                o = torch.empty(1, K, dtype=torch.int32, device=dev)
                _merge[(1,)](part, o, sl_t, K=K, K_POW2=KP,
                             N_PARTIALS=npart, USE_TOPK=bool(b_topk),
                             num_warps=4, num_stages=1)
                torch.cuda.synchronize()
                wrong = len(set(o[0].tolist()) ^ ref) // 2
                worst = max(worst, wrong)
                line += f"{('OK' if wrong == 0 else str(wrong)):>16} "
            print(line, flush=True)
    print(f"\n最大错误数: {worst}")


if __name__ == "__main__":
    main()