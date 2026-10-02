#!/usr/bin/env python
"""Exact merge by full sort, replacing the production recurrence.

The production merge -- bitonic_merge then elementwise maximum -- is only
*approximately* a top-k merge; it is exact for the production input shape (raw
score blocks, hundreds of iterations) and measurably wrong for pre-sorted
slabs, by 5 to 251 selections depending on shape. Since the reason is not
understood, relying on a shape where it happens to be exact is not something to
ship.

This drops the recurrence entirely: a merge of F sorted lists is a single
tl.sort over their concatenation, truncated to K. That is exact by definition,
and the F partials per program are loaded as one [F*K_POW2] block so the sort is
a genuine whole-array sort rather than a per-row one.

Structure is fixed (N_PARTIAL and FANIN are compile-time) so it can be captured
in the decode CUDA graph: N_PARTIAL partials -> N_PARTIAL/FANIN -> 1.

Run: topk_fullsort_merge.py
"""
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
from sglang.kernels.ops.attention.dsv4.topk import (  # noqa: E402
    topk_transform_paged_triton,
)


@triton.jit
def _slab_kernel(
    scores_ptr, seq_lens_ptr, part_ptr,
    stride_scores, K: tl.constexpr, K_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr, SLAB: tl.constexpr, N_PARTIAL: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // N_PARTIAL
    s = pid % N_PARTIAL
    seq_len = tl.load(seq_lens_ptr + row)
    start0 = s * SLAB
    offs_k = tl.arange(0, K_POW2)
    if start0 >= seq_len:
        tl.store(part_ptr + pid * K_POW2 + offs_k, tl.zeros((K_POW2,), tl.uint64))
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
    acc = tl.sort(acc, descending=True)
    tl.store(part_ptr + pid * K_POW2 + offs_k, acc)


@triton.jit
def _merge_kernel(
    in_ptr, out_ptr, K: tl.constexpr, K_POW2: tl.constexpr,
    FANIN: tl.constexpr, N_OUT: tl.constexpr, N_IN: tl.constexpr,
):
    """out[g] = top-K of in[g*FANIN : (g+1)*FANIN], as one full sort.

    The sort runs on uint64, not int64. A packed key is (float_bits << 32) |
    column, and for a positive float the key is float_bits ^ 0x80000000, which
    reaches 0xFFFFFFFF -- so the shift sets bit 63 and the packed value is
    negative as int64. About half of all real scores land there. Sorting signed
    puts the largest scores last, selects the 512 smallest, and lets the zero
    padding (larger than any negative value) take every slot. The buffers are
    int64 for torch's sake, hence the bitcasts in both directions.
    """
    pid = tl.program_id(0)
    row = pid // N_OUT
    g = pid % N_OUT
    base = (row * N_IN + g * FANIN) * K_POW2
    offs = tl.arange(0, FANIN * K_POW2)
    vals = tl.load(in_ptr + base + offs).to(tl.uint64, bitcast=True)
    srt = tl.sort(vals, descending=True)
    # The K largest of the concatenation are its leading K_POW2 lanes, so store
    # the flat sorted block with a mask instead of slicing (Triton has no
    # general tensor slicing, and tl.sort returns a tensor, not a pointer).
    tl.store(out_ptr + (row * N_OUT + g) * K_POW2 + offs,
             srt.to(tl.int64, bitcast=True), mask=offs < K_POW2)


def run_fullsort(scores, seq_lens, page_tables, out_page, out_raw, page_size,
                 n_partial, block_n=512, fanin=4, bufs=None):
    """Exact merge by full sort, in levels small enough for 64 KB of shared
    memory.

    Two constraints shaped this. Triton lowers tl.sort into a bitonic network
    whose shared-memory use grows with the sort width, and SM75 has 64 KB:
    FANIN=16 (8192 int64) asks for 256 KB and fails to launch. So levels are
    FANIN=4. And every level's buffer is padded to FANIN*K_POW2 per output so a
    level whose partial count is not a multiple of FANIN reads zeros rather
    than past the end -- an earlier fixed-FANIN version overran and only looked
    correct because the overrun landed on zero padding, which sorts last.
    """
    rows, cap = scores.shape
    K = out_page.shape[1]
    K_POW2 = triton.next_power_of_2(K)
    assert fanin <= 4, "FANIN>4 exceeds SM75 shared memory for tl.sort"
    n_blocks = triton.cdiv(cap, block_n)
    slab = triton.cdiv(n_blocks, n_partial) * block_n
    n_partial = triton.cdiv(n_blocks, triton.cdiv(n_blocks, n_partial)) \
        if False else n_partial
    if bufs is None:
        p = torch.zeros(rows * n_partial * K_POW2, dtype=torch.int64,
                        device=scores.device)
        bufs = [torch.zeros(rows * n_partial * fanin * K_POW2, dtype=torch.int64,
                            device=scores.device) for _ in range(4)]
        bufs.append(torch.zeros(rows * fanin * K_POW2, dtype=torch.int64,
                                device=scores.device))
    _slab_kernel[(rows * n_partial,)](
        scores, seq_lens, p, scores.stride(0), K=K, K_POW2=K_POW2,
        BLOCK_N=block_n, SLAB=slab, N_PARTIAL=n_partial,
        num_warps=4, num_stages=1)
    cur, cur_n = p, n_partial
    for lvl, dst in enumerate(bufs):
        if cur_n <= 1:
            break
        nxt_n = triton.cdiv(cur_n, fanin)
        _merge_kernel[(rows * nxt_n,)](
            cur, dst, K=K, K_POW2=K_POW2, FANIN=fanin, N_OUT=nxt_n, N_IN=cur_n,
            num_warps=4, num_stages=1)
        cur, cur_n = dst, nxt_n
    assert cur_n == 1, f"{n_partial} partials did not reduce in {len(bufs)} levels"
    # final: the K winners -> page indices (identical tail to production)
    best = cur.reshape(-1)[:K_POW2]
    raw = (best & 0xFFFFFFFF).to(torch.int32)
    offs = torch.arange(K, device=scores.device, dtype=torch.int32)
    valid = offs.unsqueeze(0) < seq_lens.to(torch.int32).unsqueeze(1)
    page_ids = torch.where(valid, raw // page_size, torch.zeros_like(raw))
    in_table = valid & (page_ids >= 0) & (page_ids < page_tables.shape[1])
    pages = torch.gather(page_tables, 1, torch.where(in_table, page_ids,
                                                    torch.zeros_like(page_ids)))
    page_indices = torch.where(in_table, pages * page_size + raw % page_size,
                               torch.full_like(raw, -1))
    out_page.copy_(page_indices)
    if out_raw is not None:
        out_raw.copy_(torch.where(valid, raw, torch.full_like(raw, -1)))
    return cur


def main():
    dev = "cuda"
    torch.manual_seed(0)
    K = KP = 512
    print(f"{'cap':>8} {'npart':>6} {'base us':>10} {'fullsort us':>12} {'speedup':>8}  exact")
    for cap in (4096, 16384, 37500, 150000):
        for npart in (8, 16, 32, 64):
            scores = torch.randn(1, cap, device=dev, dtype=torch.float32)
            seq_lens = torch.full((1,), cap, dtype=torch.int32, device=dev)
            pt = torch.arange(16384, device=dev, dtype=torch.int32).repeat(1, 1) % 4096
            o1 = torch.empty(1, K, dtype=torch.int32, device=dev)
            r1 = torch.empty(1, K, dtype=torch.int32, device=dev)
            o2 = torch.empty(1, K, dtype=torch.int32, device=dev)
            r2 = torch.empty(1, K, dtype=torch.int32, device=dev)

            def timeit(fn, n=20):
                fn(o1, r1); torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n):
                    fn(o2, r2)
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / n * 1e6

            base = timeit(lambda a, b: topk_transform_paged_triton(
                scores, seq_lens, pt, a, 16, b))
            fs = timeit(lambda a, b: run_fullsort(
                scores, seq_lens, pt, a, b, 16, npart))
            exact = torch.equal(r1, r2) and torch.equal(o1, o2)
            print(f"{cap:>8} {npart:>6} {base:>10.1f} {fs:>12.1f} "
                  f"{base/fs:>7.1f}x  {'OK' if exact else 'MISMATCH'}", flush=True)


if __name__ == "__main__":
    main()
