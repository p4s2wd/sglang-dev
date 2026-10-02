#!/usr/bin/env python
"""Three-way comparison, all exact, all timed the way decode runs them.

Three arms for the top-K step, which the trace showed occupying a single SM and
costing about 16% of long-context decode:

  base       the production kernel: one program, grid [1,1,1]
  sort       parallel slabs + a merge that fully sorts each fan-in group
  recur      parallel slabs + the production merge recurrence (cheapest merge)

All three are captured in a CUDA graph and replayed, so none pays a launch cost
the others avoid. Every arm is compared byte-for-byte against base, and base is
itself checked against torch.topk, so an "exact" verdict cannot come from a
shared mistake.

Both parallel arms pay a tail of small torch ops that base fuses into its kernel.
That makes their measured times pessimistic by roughly 15 us per token, which is
noted rather than corrected -- fusing the tail is a separate change.

Run: topk_three_way.py
"""
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
sys.path.insert(0, "/data/nvme/sglang-codex/plan2026-09-30")

from sglang.kernels.ops.attention.dsv4.topk import (  # noqa: E402
    topk_transform_paged_triton,
)
from topk_merge_search import _merge as _recur_merge  # noqa: E402
from topk_merge_search import _slab  # noqa: E402


@triton.jit
def _sort_merge(in_ptr, out_ptr, K: tl.constexpr, K_POW2: tl.constexpr,
                FANIN: tl.constexpr, N_OUT: tl.constexpr,
                N_IN: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // N_OUT
    g = pid % N_OUT
    base = (row * N_IN + g * FANIN) * K_POW2
    offs = tl.arange(0, FANIN * K_POW2)
    vals = tl.load(in_ptr + base + offs).to(tl.uint64, bitcast=True)
    srt = tl.sort(vals, descending=True)
    tl.store(out_ptr + (row * N_OUT + g) * K_POW2 + offs,
             srt.to(tl.int64, bitcast=True), mask=offs < K_POW2)


def make_graph(fn):
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


def drop_graph(g):
    """A captured graph owns a private memory pool that ordinary caching
    allocator calls do not reclaim. Several arms times several context sizes
    exhausts the device otherwise."""
    g.reset()
    del g
    torch.cuda.empty_cache()


def tail(raw, seq_lens, page_tables, out_page, out_raw, page_size):
    """The page-table transform, unfused -- base does this inside its kernel.

    Takes raw column indices, not packed keys, so both parallel arms share it:
    the sort arm unpacks first, the recurrence arm already wrote them out.
    """
    K = out_page.shape[1]
    offs = torch.arange(K, device=out_page.device, dtype=torch.int32)
    valid = offs.unsqueeze(0) < seq_lens.to(torch.int32).unsqueeze(1)
    page_ids = torch.where(valid, raw // page_size, torch.zeros_like(raw))
    in_table = valid & (page_ids >= 0) & (page_ids < page_tables.shape[1])
    pages = torch.gather(page_tables, 1,
                         torch.where(in_table, page_ids,
                                     torch.zeros_like(page_ids)))
    page_indices = torch.where(in_table, pages * page_size + raw % page_size,
                               torch.full_like(raw, -1))
    out_page.copy_(page_indices)
    out_raw.copy_(torch.where(valid, raw, torch.full_like(raw, -1)))


def arm_sort(scores, seq_lens, pt, o, r, ps, npart, fanin, bufs):
    rows, cap = scores.shape
    K = o.shape[1]
    KP = triton.next_power_of_2(K)
    slab = triton.cdiv(triton.cdiv(cap, 512), npart) * 512
    _slab[(rows * npart,)](scores, seq_lens, bufs["part"], scores.stride(0),
                           npart, K=K, K_POW2=KP, BLOCK_N=512, SLAB=slab,
                           DO_SORT=True, num_warps=4, num_stages=1)
    cur, cur_n, lvl = bufs["part"], npart, 0
    while cur_n > 1:
        nxt = triton.cdiv(cur_n, fanin)
        _sort_merge[(rows * nxt,)](cur, bufs[f"m{lvl}"], K=K, K_POW2=KP,
                                   FANIN=fanin, N_OUT=nxt, N_IN=cur_n,
                                   num_warps=4, num_stages=1)
        cur, cur_n, lvl = bufs[f"m{lvl}"], nxt, lvl + 1
    # Every row has its own K winners, laid out consecutively. Slicing the first
    # K elements instead silently gives every row row 0's answer.
    best = cur[: rows * KP].view(rows, KP)
    tail((best & 0xFFFFFFFF).to(torch.int32), seq_lens, pt, o, r, ps)


def arm_recur(scores, seq_lens, pt, o, r, ps, npart, fanin, bufs):
    """fanin is accepted only so both arms share one call signature; the
    recurrence merges in registers and never reads a level buffer."""
    rows, cap = scores.shape
    K = o.shape[1]
    KP = triton.next_power_of_2(K)
    slab = triton.cdiv(triton.cdiv(cap, 512), npart) * 512
    _slab[(rows * npart,)](scores, seq_lens, bufs["part"], scores.stride(0),
                           npart, K=K, K_POW2=KP, BLOCK_N=512, SLAB=slab,
                           DO_SORT=True, num_warps=4, num_stages=1)
    _recur_merge[(rows,)](bufs["part"], r, seq_lens, K=K, K_POW2=KP,
                          N_PARTIALS=npart, USE_TOPK=True, num_warps=4,
                          num_stages=1)
    tail(r, seq_lens, pt, o, r, ps)


def alloc(rows, npart, fanin, K, dev):
    """Partial buffer, plus one merge buffer per level.

    The recurrence arm needs only the partials -- it merges in registers -- so it
    passes fanin < 2 and gets no level buffers. Passing fanin 1 would make
    cdiv(cur_n, 1) == cur_n and never terminate.
    """
    KP = triton.next_power_of_2(K)
    b = {"part": torch.zeros(rows * npart * KP, dtype=torch.int64, device=dev)}
    if fanin < 2:
        return b
    cur_n, lvl = npart, 0
    while cur_n > 1:
        nxt = triton.cdiv(cur_n, fanin)
        b[f"m{lvl}"] = torch.zeros(rows * nxt * fanin * KP, dtype=torch.int64,
                                   device=dev)
        cur_n, lvl = nxt, lvl + 1
    return b


def main():
    dev = "cuda"
    K = 512
    ps = 16
    caps = (4096, 16384, 37500, 150000, 262144)
    print(f"{'cap':>8} {'arm':>22} {'us':>9} {'vs base':>8}  match")
    for cap in caps:
        torch.manual_seed(cap)
        scores = torch.randn(1, cap, device=dev)
        seq_lens = torch.full((1,), cap, dtype=torch.int32, device=dev)
        pt = (torch.arange(1 << 17, device=dev, dtype=torch.int32)
              .repeat(1, 1) % 4096)
        o1 = torch.empty(1, K, dtype=torch.int32, device=dev)
        r1 = torch.empty(1, K, dtype=torch.int32, device=dev)
        o2 = torch.empty(1, K, dtype=torch.int32, device=dev)
        r2 = torch.empty(1, K, dtype=torch.int32, device=dev)
        g_base = make_graph(lambda: topk_transform_paged_triton(
            scores, seq_lens, pt, o1, ps, r1))
        base = time_graph(g_base)
        g_base.replay()
        torch.cuda.synchronize()
        truth = set(torch.topk(scores[0], K).indices.tolist())
        base_ok = set(r1[0].tolist()) == truth
        print(f"{cap:>8} {'base (production)':>22} {base:>9.1f} {'1.00x':>8}  "
              f"{'exact' if base_ok else 'MISMATCH vs torch'}", flush=True)
        for label, fn, npart, fanin in (
            ("sort", arm_sort, 64, 2),
            ("sort", arm_sort, 32, 2),
            ("recur", arm_recur, 64, 0),
            ("recur", arm_recur, 128, 0),
            ("recur", arm_recur, 256, 0),
        ):
            bufs = alloc(1, npart, max(fanin, 1), K, dev)
            try:
                g = make_graph(lambda: fn(scores, seq_lens, pt, o2, r2, ps,
                                          npart, fanin or 1, bufs))
            except triton.runtime.errors.OutOfResources as e:
                print(f"{cap:>8} {label + f' n={npart}':>22} {'-':>9} {'-':>8}  "
                      f"OOM {str(e)[:40]}")
                continue
            t = time_graph(g)
            o2.zero_()
            r2.zero_()
            g.replay()
            torch.cuda.synchronize()
            ok = torch.equal(r1, r2) and torch.equal(o1, o2)
            print(f"{cap:>8} {label + f' n={npart}':>22} {t:>9.1f} "
                  f"{base / t:>7.2f}x  {'exact' if ok else 'MISMATCH'}",
                  flush=True)
            drop_graph(g)
        drop_graph(g_base)
        del scores, pt, o1, r1, o2, r2
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()