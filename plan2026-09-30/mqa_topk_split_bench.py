#!/usr/bin/env python
"""Microbenchmark: does splitting the top-k over the sequence pay for itself?

The production kernel launches grid=(rows,), so at decode bs=1 -- which is what
the whole long-context problem looks like -- a single 128-thread block on a
68-SM card walks the entire row: seq_len/512 iterations of a 512-wide bitonic
top-k merged into a 512-element accumulator. The trace agrees: grid [1,1,1] at
both 440 and 150K tokens, 25 us -> 912 us per launch.

Top-k is associative over a partition, so splitting the row into contiguous
slabs, taking the top-K of each, and taking the top-K of the union is exact --
not an approximation. This measures what that buys before touching the
production path, and checks the selection is bit-identical.

Run: mqa_topk_split_bench.py [--seq-lens 4096,37500,150000]
"""
import argparse
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
def _slab_topk(
    scores_ptr, seq_lens_ptr, partials_ptr,
    stride_scores, n_partials, seq_cap,
    K: tl.constexpr, K_POW2: tl.constexpr, BLOCK_N: tl.constexpr,
    SLAB: tl.constexpr,
):
    """Top-K of one slab of one row -> partials[row, slab]."""
    pid = tl.program_id(0)
    row = pid // n_partials
    slab = pid % n_partials
    seq_len = tl.load(seq_lens_ptr + row)
    start0 = slab * SLAB
    if start0 >= seq_len:
        # Empty slab: emit sentinels so the merge still sees K_POW2 slots.
        offs = tl.arange(0, K_POW2)
        tl.store(partials_ptr + pid * K_POW2 + offs, tl.zeros((K_POW2,), tl.uint64))
        return
    end = tl.minimum(start0 + SLAB, seq_len)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((K_POW2,), dtype=tl.uint64)
    for start in range(start0, end, BLOCK_N):
        cols = start + offs_n
        valid = cols < end
        score = tl.load(scores_ptr + row * stride_scores + cols, mask=valid,
                        other=float("-inf"))
        sb = score.to(tl.uint32, bitcast=True)
        sign = tl.full(sb.shape, 0x80000000, tl.uint32)
        key = tl.where((sb & sign) != 0, ~sb, sb ^ sign)
        packed = (key.to(tl.uint64) << 32) | cols.to(tl.uint64)
        cand = tl.topk(packed, K_POW2, dim=0)
        acc = tl.bitonic_merge(acc)
        acc = tl.maximum(acc, tl.topk(cand, K_POW2, dim=0))
    acc = tl.sort(acc, descending=True)
    offs = tl.arange(0, K_POW2)
    tl.store(partials_ptr + pid * K_POW2 + offs, acc)


@triton.jit
def _merge(
    scores_ptr, seq_lens_ptr, page_tables_ptr, out_page_indices_ptr,
    out_raw_indices_ptr, partials_ptr,
    stride_scores, stride_page_tables, seq_cap,
    K: tl.constexpr, K_POW2: tl.constexpr, WRITE_RAW: tl.constexpr,
    PAGE_SIZE: tl.constexpr, N_PARTIALS: tl.constexpr, SLAB: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
):
    """Merge the per-slab top-K lists, then the page-table lookup the
    production kernel ends with (kept identical so parity is meaningful)."""
    row = tl.program_id(0)
    offs_k = tl.arange(0, K_POW2)
    acc = tl.zeros((K_POW2,), dtype=tl.uint64)
    for s in range(0, N_PARTIALS):
        cand = tl.load(partials_ptr + (row * N_PARTIALS + s) * K_POW2 + offs_k)
        acc = tl.bitonic_merge(acc)
        acc = tl.maximum(acc, tl.topk(cand, K_POW2, dim=0))
    acc = tl.sort(acc, descending=True)
    raw = (acc & 0xFFFFFFFF).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + row)
    valid = (offs_k < K) & (offs_k < seq_len)
    page_ids = raw // PAGE_SIZE
    page_ids = tl.where(valid, page_ids, 0)
    in_table = valid & (page_ids >= 0) & (page_ids < PAGE_TABLE_WIDTH)
    pages = tl.load(page_tables_ptr + row * stride_page_tables + page_ids,
                    mask=in_table, other=0)
    page_indices = pages * PAGE_SIZE + raw % PAGE_SIZE
    page_indices = tl.where(in_table, page_indices, -1).to(tl.int32)
    tl.store(out_page_indices_ptr + row * K + offs_k, page_indices,
             mask=offs_k < K)
    if WRITE_RAW:
        r = tl.where(valid, raw, -1)
        tl.store(out_raw_indices_ptr + row * K + offs_k, r, mask=offs_k < K)


def run_split(scores, seq_lens, page_tables, out_page, out_raw, page_size,
              n_partials, block_n=512, partials=None):
    rows, cap = scores.shape
    K = out_page.shape[1]
    K_POW2 = triton.next_power_of_2(K)
    # slab and cap are both in elements, so the block count is cdiv(cap, slab).
    # (Mixing in cdiv(cap, block_n) here silently yields 1 and the merge then
    # reads only the first slab -- which is exactly what it did once.)
    n_blocks = triton.cdiv(cap, block_n)
    slab = triton.cdiv(n_blocks, n_partials) * block_n
    n_partials = triton.cdiv(cap, slab)
    if partials is None or partials.numel() < rows * n_partials * K_POW2:
        partials = torch.zeros(rows * n_partials * K_POW2, dtype=torch.int64,
                               device=scores.device)
    _slab_topk[(rows * n_partials,)](
        scores, seq_lens, partials, scores.stride(0), n_partials, cap,
        K=K, K_POW2=K_POW2, BLOCK_N=block_n, SLAB=slab, num_warps=4, num_stages=1)
    _merge[(rows,)](
        scores, seq_lens, page_tables, out_page, out_raw, partials,
        scores.stride(0), page_tables.stride(1), cap,
        K=K, K_POW2=K_POW2, WRITE_RAW=out_raw is not None, PAGE_SIZE=page_size,
        N_PARTIALS=n_partials, SLAB=slab, PAGE_TABLE_WIDTH=page_tables.shape[1],
        num_warps=4, num_stages=1)
    return partials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", default="4096,37500,150000")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--pages-per-row", type=int, default=16384)
    ap.add_argument("--parts", default="32,64,128")
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    K = args.topk

    print(f"rows={args.rows} K={K} page_size={args.page_size}")
    print(f"{'seq_len':>9} {'baseline us':>12} " +
          " ".join(f"{'split'+p:>11}" for p in args.parts.split(",")) +
          f" {'best x':>8}  identical")
    for sl in (int(x) for x in args.seq_lens.split(",")):
        scores = torch.randn(args.rows, sl, device=dev, dtype=torch.float32)
        seq_lens = torch.full((args.rows,), sl, dtype=torch.int32, device=dev)
        pt = torch.arange(args.pages_per_row, device=dev,
                          dtype=torch.int32).repeat(args.rows, 1) % 4096

        def once(fn):
            o = torch.empty(args.rows, K, dtype=torch.int32, device=dev)
            r = torch.empty(args.rows, K, dtype=torch.int32, device=dev)
            fn(o, r)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                fn(o, r)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / 20 * 1e6, o.clone(), r.clone()

        base_us, base_o, base_r = once(
            lambda o, r: topk_transform_paged_triton(
                scores, seq_lens, pt, o, args.page_size, r))

        line = f"{sl:>9} {base_us:>12.1f} "
        best, best_tag = base_us, "-"
        for p in (int(x) for x in args.parts.split(",")):
            try:
                us, o, r = once(lambda o, r, p=p: run_split(
                    scores, seq_lens, pt, o, r, args.page_size, p))
            except Exception as e:
                line += f"{type(e).__name__:>11} "
                continue
            same = torch.equal(base_o, o) and torch.equal(base_r, r)
            line += f"{us:>11.1f} "
            if us < best:
                best, best_tag = us, f"split{p}" + ("" if same else "(MISMATCH)")
        line += f" {base_us/best:>7.1f}x  {best_tag}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
