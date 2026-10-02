#!/usr/bin/env python
"""Time the exact top-K merge the way decode actually runs it: inside a CUDA
graph.

The first wall-clock measurement of the full-sort merge showed a flat ~750 us
floor at every context length, which is not a property of the kernels. Each
call allocated six tensors and issued three Triton launches, and that Python
side cost dominated. Production decode captures this sequence in a CUDA graph,
where the launches are free, so the graph is the only measurement that answers
the question worth asking -- does this make the decode step faster.

Both arms are captured and replayed the same way, so neither pays a launch cost
the other avoids.

Run: topk_graph_bench.py
"""
import sys
import time

import torch
import triton

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
sys.path.insert(0, "/data/nvme/sglang-codex/plan2026-09-30")

from sglang.kernels.ops.attention.dsv4.topk import (  # noqa: E402
    topk_transform_paged_triton,
)
from topk_fullsort_merge import _merge_kernel, _slab_kernel  # noqa: E402


def make_graph(fn):
    """Capture fn into a CUDA graph after a side-stream warmup."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def time_graph(g, n=200):
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def fullsort(scores, seq_lens, page_tables, out_page, out_raw, page_size,
             n_partial, bufs, block_n=512, fanin=4):
    """Same body as run_fullsort but on preallocated buffers, so the graph can
    bake the pointers in and nothing is allocated per call."""
    rows, cap = scores.shape
    K = out_page.shape[1]
    K_POW2 = triton.next_power_of_2(K)
    n_blocks = triton.cdiv(cap, block_n)
    slab = triton.cdiv(n_blocks, n_partial) * block_n
    p = bufs["part"]
    _slab_kernel[(rows * n_partial,)](
        scores, seq_lens, p, scores.stride(0), K=K, K_POW2=K_POW2,
        BLOCK_N=block_n, SLAB=slab, N_PARTIAL=n_partial,
        num_warps=4, num_stages=1)
    cur, cur_n, lvl = p, n_partial, 0
    while cur_n > 1:
        dst = bufs[f"m{lvl}"]
        nxt_n = triton.cdiv(cur_n, fanin)
        _merge_kernel[(rows * nxt_n,)](
            cur, dst, K=K, K_POW2=K_POW2, FANIN=fanin, N_OUT=nxt_n, N_IN=cur_n,
            num_warps=4, num_stages=1)
        cur, cur_n, lvl = dst, nxt_n, lvl + 1
    best = cur.reshape(-1)[:K_POW2]
    raw = (best & 0xFFFFFFFF).to(torch.int32)
    offs = torch.arange(K, device=scores.device, dtype=torch.int32)
    valid = offs.unsqueeze(0) < seq_lens.to(torch.int32).unsqueeze(1)
    page_ids = torch.where(valid, raw // page_size, torch.zeros_like(raw))
    in_table = valid & (page_ids >= 0) & (page_ids < page_tables.shape[1])
    pages = torch.gather(page_tables, 1,
                         torch.where(in_table, page_ids,
                                     torch.zeros_like(page_ids)))
    page_indices = torch.where(in_table, pages * page_size + raw % page_size,
                               torch.full_like(raw, -1))
    out_page.copy_(page_indices)
    if out_raw is not None:
        out_raw.copy_(torch.where(valid, raw, torch.full_like(raw, -1)))


def alloc_bufs(rows, cap, K, n_partial, fanin, dev, page_size):
    K_POW2 = triton.next_power_of_2(K)
    # fullsort() always launches exactly n_partial slab programs -- the ones past
    # a short row write zeros and sort last -- so the merge must see exactly
    # n_partial partials. Sizing the buffers off a reduced count is how the
    # earlier out-of-bounds read happened.
    bufs = {"part": torch.zeros(rows * n_partial * K_POW2, dtype=torch.int64,
                                device=dev)}
    cur_n = n_partial
    lvl = 0
    while cur_n > 1:
        nxt = triton.cdiv(cur_n, fanin)
        bufs[f"m{lvl}"] = torch.zeros(rows * nxt * fanin * K_POW2,
                                      dtype=torch.int64, device=dev)
        cur_n, lvl = nxt, lvl + 1
    return bufs


def main():
    dev = "cuda"
    torch.manual_seed(0)
    K = KP = 512
    page_size = 16
    print(f"{'cap':>8} {'npart':>6} {'fanin':>6} {'base us':>10} "
          f"{'exact us':>10} {'speedup':>8}  match")
    for cap in (4096, 16384, 37500, 150000):
        scores = torch.randn(1, cap, device=dev, dtype=torch.float32)
        seq_lens = torch.full((1,), cap, dtype=torch.int32, device=dev)
        pt = (torch.arange(8192, device=dev, dtype=torch.int32)
              .repeat(1, 1) % 4096)
        o1 = torch.empty(1, K, dtype=torch.int32, device=dev)
        r1 = torch.empty(1, K, dtype=torch.int32, device=dev)
        o2 = torch.empty(1, K, dtype=torch.int32, device=dev)
        r2 = torch.empty(1, K, dtype=torch.int32, device=dev)

        g_base = make_graph(lambda: topk_transform_paged_triton(
            scores, seq_lens, pt, o1, page_size, r1))
        base = time_graph(g_base)

        row = []
        for npart, fanin in ((8, 4), (16, 4), (32, 4), (32, 2), (64, 4),
                             (64, 2), (128, 4)):
            try:
                bufs = alloc_bufs(1, cap, K, npart, fanin, dev, page_size)
                g_ex = make_graph(lambda: fullsort(
                    scores, seq_lens, pt, o2, r2, page_size, npart, bufs,
                    fanin=fanin))
                t = time_graph(g_ex)
            except triton.runtime.errors.OutOfResources as e:
                row.append(f"{npart}/{fanin}: OOM({e})")
                continue
            o2.zero_(); r2.zero_()
            g_ex.replay(); torch.cuda.synchronize()
            ok = torch.equal(r1, r2) and torch.equal(o1, o2)
            row.append((npart, fanin, t, ok))
        best = min((r for r in row if len(r) == 4), key=lambda r: r[2],
                   default=None)
        for r in row:
            if len(r) != 4:
                print(f"{cap:>8} {r}")
                continue
            npart, fanin, t, ok = r
            mark = "  <-- best" if r is best else ""
            print(f"{cap:>8} {npart:>6} {fanin:>6} {base:>10.1f} {t:>10.1f} "
                  f"{base/t:>7.1f}x  {'exact' if ok else 'MISMATCH'}{mark}",
                  flush=True)
        print()


if __name__ == "__main__":
    main()