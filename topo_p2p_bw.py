#!/usr/bin/env python
"""Measure P2P bandwidth between every GPU pair on this box.

The topology question ("is TP2xPP4 landing each PP stage inside one NVLink pair,
and is NCCL actually using the NVLink?") is answerable directly instead of by
reading sglang's rank-to-device mapping.

Two things are measured per pair:

  cudaMemcpyPeerAsync  - the raw P2P path PyTorch will use. If P2P is not
                         available for a pair this falls back to a staged copy
                         through host memory and lands near PCIe speed instead
                         of NVLink speed.
  ncclAllReduce on 2    - what TP2 actually runs per layer. A 2-rank all-reduce
                         moves (n-1)/n of the buffer each way, so it is not
                         directly comparable to the memcpy number; the ratio
                         between pairs is what matters.

Everything is timed with CUDA events, median of N, and repeated interleaved so a
clock-drift difference between pairs cannot be mistaken for a topology effect.

Usage: topo_p2p_bw.py [--size-mb 64] [--rounds 10]
"""
import argparse
import itertools
import statistics
import sys

import torch


def timeit(fn, n=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a = torch.cuda.Event(True)
        b = torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-mb", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=8)
    a = ap.parse_args()

    n = torch.cuda.device_count()
    print(f"{n} GPUs, buffer {a.size_mb} MiB\n")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  SMs={p.multi_processor_count}  "
              f"mem={p.total_memory/2**30:.1f} GiB")
    print()

    mb = a.size_mb * 2**20
    bufs = [torch.empty(mb, dtype=torch.uint8, device=f"cuda:{i}") for i in range(n)]
    for b in bufs:
        b.fill_(1)

    # ---- peer copy bandwidth ----
    print("=" * 78)
    print("1. cudaMemcpyPeerAsync  (GB/s, higher = direct P2P path)")
    print("=" * 78)
    print("%-12s %12s %12s" % ("pair", "GB/s", "class"))
    results = {}
    for i, j in itertools.combinations(range(n), 2):
        # interleave rounds across pairs
        samples = []
        for _ in range(2):
            for _ in range(a.rounds):
                def go():
                    torch.cuda.set_device(i)
                    torch.cuda.copy_(bufs[j], bufs[i], non_blocking=True)
                samples.append(timeit(go, n=3, warmup=1))
        ms = statistics.median(samples)
        gbs = mb / (ms * 1e-3) / 1e9
        results[(i, j)] = gbs
        cls = "NVLink" if gbs > 20 else ("PCIe" if gbs > 5 else "host-staged")
        print("%-12s %12.1f %12s" % (f"{i}<->{j}", gbs, cls))

    # ---- group the pairs ----
    print()
    print("=" * 78)
    print("2. pair classes")
    print("=" * 78)
    fast = [k for k, v in results.items() if v > 20]
    slow = [k for k, v in results.items() if v <= 20]
    print("  fast (>20 GB/s, consistent with NVLink):")
    for k in fast:
        print(f"    {k[0]} <-> {k[1]}   {results[k]:.1f} GB/s")
    print("  slow:")
    for k in slow:
        print(f"    {k[0]} <-> {k[1]}   {results[k]:.1f} GB/s")

    # ---- 2-rank NCCL all-reduce, which is what TP2 does per layer ----
    print()
    print("=" * 78)
    print("3. 2-rank NCCL all-reduce (what TP2 runs per layer)")
    print("=" * 78)
    try:
        print(f"{'pair':<12}{'GB/s':>10}  note")
        for i, j in fast + slow:
            # nccl needs one process per GPU; do it in two child processes
            import subprocess
            code = r'''
import os, sys, statistics, torch, torch.distributed as dist
rank = int(sys.argv[1]); peer = int(sys.argv[2]); mb = int(sys.argv[3])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=2,
                        init_method=f"tcp://127.0.0.1:{29500+rank}")
t = torch.empty(mb*2**20, dtype=torch.uint8, device=f"cuda:{rank}"); t.fill_(1)
for _ in range(5): dist.all_reduce(t)
torch.cuda.synchronize()
ts=[]
for _ in range(20):
    a0=torch.cuda.Event(True); b0=torch.cuda.Event(True)
    a0.record(); dist.all_reduce(t); b0.record(); torch.cuda.synchronize()
    ts.append(a0.elapsed_time(b0))
print(statistics.median(ts))
dist.destroy_process_group()
'''
            outs = []
            for r in (i, j):
                pass
            # launch two procs
            procs = []
            for rank, peer_ in ((i, j), (j, i)):
                p = subprocess.Popen([sys.executable, "-c", code, str(rank),
                                      str(peer_), str(a.size_mb)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env={**__import__("os").environ})
                procs.append(p)
            res = []
            for p in procs:
                o, e = p.communicate(timeout=180)
                if p.returncode != 0:
                    res = None
                    break
                res.append(float(o.decode().strip().splitlines()[-1]))
            if res is None or len(res) != 2:
                print(f"{i}<->{j:<8}  all-reduce FAILED")
                continue
            ms = max(res)
            # all-reduce of N bytes moves 2(N-1)/N ~= 2N over the wire
            gbs = 2 * mb / (ms * 1e-3) / 1e9
            print(f"{i}<->{j:<8}{gbs:>10.1f}")
    except Exception as exc:
        print("  nccl all-reduce probe failed:", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
