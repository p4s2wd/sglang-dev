"""Is the 2.32 ms allreduce real cost, or a spin-wait absorbing pipeline skew?

The prefill profile ranks all_reduce_2shot_pull at 14.8% (204 ms, 88 calls, 2.32 ms
each). For a 512-token x hidden-size fp16 message that is implausible: two ranks
exchange a few megabytes over NVLink (TP2 always lands inside an NVLink pair),
which at even 20 GB/s is tens of microseconds. A pull-style allreduce spends most
of its time spinning until the peer publishes, so its measured duration is
max(work, skew) -- if the two ranks are not in lockstep, the kernel "runs" for the
whole wait and the profile blames the wrong thing.

Measured standalone with both ranks arriving together, this says what the kernel
costs on its own. If it is ~0.1 ms, the 2.32 ms is skew and allreduce is a symptom
of the serialized pipeline, not a target. If it is ~2 ms, the kernel itself is the
problem and there is 15% of prefill to win by fixing it.
"""
import os, sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import torch.distributed as dist

def main():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    r = dist.get_rank()
    dev = torch.device("cuda")

    from sglang.srt.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    ar = CustomAllreduce(group=dist.group.WORLD, device=dev)

    for hidden in (4096, 7168):
        for tokens in (512, 1024):
            x = torch.randn(tokens, hidden, dtype=torch.float16, device=dev)
            if not ar.may_use_custom_all_reduce(x):
                if r == 0: print("hidden=%d tokens=%d: custom AR unavailable" % (hidden, tokens))
                continue
            for _ in range(20):
                ar.all_reduce(x)
            torch.cuda.synchronize(); dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            N = 200
            for _ in range(N):
                ar.all_reduce(x)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / N * 1e3
            if r == 0:
                mb = tokens * hidden * 2 / 1e6
                # 2-shot: each rank reads the peer's half once.
                eff = mb * (2 - 1) / (dt / 1e3) / 1e3
                print("hidden=%5d tokens=%4d  msg %6.2f MB  %6.3f ms  ~%5.1f GB/s peer read"
                      % (hidden, tokens, mb, dt, eff))
    dist.destroy_process_group()

main()
