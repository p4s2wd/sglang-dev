#!/usr/bin/env python
"""Two things the winner still needs: where to switch arms, and whether the
parallel path is exact outside the single-full-length-row case.

Dispatch. The full-sort merge is 15.6x the production kernel at 262144 but 0.62x
at 4096 -- a single program over a short row is already cheap, and splitting it
adds kernels. So production will need a length threshold, and that threshold has
to be measured, not guessed.

Batch. Every measurement so far used one row of length exactly cap. Production
runs ragged batches: many rows, each with its own seq_len, most shorter than the
allocated width. The slab kernel must leave the tail of a short row as zero
padding that sorts last, and rows must not read each other's partials.

Run: topk_dispatch_batch.py
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
from topk_three_way import (  # noqa: E402
    alloc,
    arm_sort,
    drop_graph,
    make_graph,
    time_graph,
)

K = 512
NPART = 64
FANIN = 2
PS = 16


def bench(caps):
    dev = "cuda"
    print(f"{'cap':>8} {'base us':>9} {'sort us':>9} {'ratio':>7}  exact")
    out = {}
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
        gb = make_graph(lambda: topk_transform_paged_triton(
            scores, seq_lens, pt, o1, PS, r1))
        base = time_graph(gb)
        bufs = alloc(1, NPART, FANIN, K, dev)
        gs = make_graph(lambda: arm_sort(scores, seq_lens, pt, o2, r2, PS,
                                        NPART, FANIN, bufs))
        t = time_graph(gs)
        o2.zero_()
        r2.zero_()
        gs.replay()
        gb.replay()
        torch.cuda.synchronize()
        ok = torch.equal(r1, r2) and torch.equal(o1, o2)
        out[cap] = base / t
        print(f"{cap:>8} {base:>9.1f} {t:>9.1f} {base / t:>6.2f}x  "
              f"{'exact' if ok else 'MISMATCH'}", flush=True)
        drop_graph(gs)
        drop_graph(gb)
        del scores, pt, o1, r1, o2, r2, bufs
        torch.cuda.empty_cache()
    return out


def batch_exactness():
    dev = "cuda"
    print(f"\n{'rows':>5} {'cap':>8} {'lens':>28}  match")
    cases = [
        (1, 4096, [4096]),
        (4, 8192, [8192, 4096, 1, 300]),
        (8, 37500, [37500, 20000, 37500, 1, 511, 512, 513, 37500]),
        (8, 262144, [262144, 150000, 65536, 1024, 262144, 3, 77777, 262144]),
        (16, 65536, [65536 if i % 2 else 5000 + i for i in range(16)]),
    ]
    allok = True
    for rows, cap, lens in cases:
        torch.manual_seed(cap + rows)
        scores = torch.randn(rows, cap, device=dev)
        seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
        pt = (torch.arange(1 << 17, device=dev, dtype=torch.int32)
              .repeat(rows, 1) % 4096)
        o1 = torch.empty(rows, K, dtype=torch.int32, device=dev)
        r1 = torch.empty(rows, K, dtype=torch.int32, device=dev)
        o2 = torch.empty(rows, K, dtype=torch.int32, device=dev)
        r2 = torch.empty(rows, K, dtype=torch.int32, device=dev)
        topk_transform_paged_triton(scores, seq_lens, pt, o1, PS, r1)
        bufs = alloc(rows, NPART, FANIN, K, dev)
        arm_sort(scores, seq_lens, pt, o2, r2, PS, NPART, FANIN, bufs)
        torch.cuda.synchronize()
        ok = torch.equal(r1, r2) and torch.equal(o1, o2)
        # and against torch.topk directly, since base is only a peer. Rows
        # shorter than K carry -1 padding past seq_len in both kernels, so only
        # the real slots take part.
        truth = all(set(v for v in r2[b].tolist() if v >= 0) ==
                    set(torch.topk(scores[b, :lens[b]],
                                   min(K, lens[b])).indices.tolist())
                    for b in range(rows))
        allok = allok and ok and truth
        print(f"{rows:>5} {cap:>8} {str(lens[:6]):>28}  "
              f"{'exact' if ok else 'MISMATCH vs base'}"
              f"{'' if truth else ' / MISMATCH vs torch'}", flush=True)
        del scores, pt, o1, r1, o2, r2, bufs
        torch.cuda.empty_cache()
    print(f"\n批量/不等长: {'全部 exact' if allok else '存在 MISMATCH'}")


def main():
    bench((2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768))
    batch_exactness()


if __name__ == "__main__":
    main()