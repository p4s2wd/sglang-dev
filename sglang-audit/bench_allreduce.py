"""Is TP2's all-reduce cost real transfer time, or peer waiting?

The production EXTEND profile puts all_reduce_1shot_push at 27.7% (506.8 ms over
174 launches = 2.91 ms per call). A 512-token x 4096-hidden fp16 tensor is 4 MB;
over PCIe that should be a fraction of a millisecond, so most of 2.91 ms is
probably the kernel spinning for the lagging rank (a pipeline-parallel bubble),
not bytes on the wire. Which of the two it is decides whether there is anything
to optimise: a bandwidth problem is a kernel problem, a wait is a scheduling
problem and no kernel change will help.

Times the same collective two ways: back-to-back (both ranks arrive together, so
the wait is minimal and the number is close to pure transfer) and with one rank
held back by a dummy kernel (the wait becomes explicit).

Run under torchrun: torchrun --nproc-per-node 2 sglang-audit/bench_allreduce.py
"""
import os
import sys
import time

import torch
import torch.distributed as dist

repo = "/data/nvme/sglang-codex/sglang"
sys.path.insert(0, repo + "/python")


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    dist.barrier()
    return dt * 1e3


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    dev = f"cuda:{os.environ['LOCAL_RANK']}"

    for tokens, hidden in ((1, 4096), (512, 4096), (2048, 4096)):
        x = torch.randn(tokens, hidden, dtype=torch.float16, device=dev)
        mb = tokens * hidden * 2 / 2**20
        t = bench(lambda: dist.all_reduce(x))
        # NCCL ring over 2 ranks sends the buffer once each way.
        gbps = mb / 2**10 / (t / 1e3) / 1e3 if t > 0 else 0
        if rank == 0:
            print(f"  nccl   {tokens:5d}x{hidden}: {t:7.3f} ms  "
                  f"({mb:6.2f} MiB, {gbps:5.2f} GB/s effective)")

    # The custom kernel sglang actually uses, driven the same way.
    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        ca = CustomAllreduce(group=dist.group.WORLD, device=dev)
        if ca.disabled:
            if rank == 0:
                print("  custom allreduce: disabled (no P2P?)")
        else:
            for tokens, hidden in ((1, 4096), (512, 4096)):
                x = torch.randn(tokens, hidden, dtype=torch.float16, device=dev)
                mb = tokens * hidden * 2 / 2**20
                t = bench(lambda: ca.all_reduce(x))
                if rank == 0:
                    print(f"  custom {tokens:5d}x{hidden}: {t:7.3f} ms  "
                          f"({mb:6.2f} MiB)")
    except Exception as e:
        if rank == 0:
            print(f"  custom allreduce: {type(e).__name__}: {str(e)[:70]}")

    # Now the skewed case: rank 1 burns GPU time before joining, so the kernel
    # duration measured on rank 0 includes the wait.
    burn = torch.randn(4096, 4096, device=dev)
    for tokens, hidden in ((512, 4096),):
        x = torch.randn(tokens, hidden, dtype=torch.float16, device=dev)

        def skewed():
            if rank == 1:
                for _ in range(20):
                    burn @ burn
            dist.all_reduce(x)

        t = bench(skewed, iters=15, warmup=4)
        if rank == 0:
            print(f"  skewed {tokens:5d}x{hidden}: {t:7.3f} ms  "
                  f"(rank 1 burns 20 matmuls first)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
