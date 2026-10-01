"""Floor for a TP2 allreduce at prefill message size, measured two ways.

The prefill profile ranks all_reduce_2shot_pull at 14.8% -- 204 ms over 88 calls,
2.32 ms each. A pull-style allreduce spends its time spinning until the peer
publishes, so its measured duration is max(work, skew): if the two ranks drift
apart, the kernel "runs" for the whole wait and the profile blames the kernel for
a scheduling problem. To tell those apart I need the floor.

Two ranks, the real message size, measured with both ranks arriving together:
NCCL's own all_reduce (a lower bound any implementation approaches) and a plain
peer-to-peer copy (the raw NVLink limit). If both are far under 2.32 ms, the
allreduce line in the profile is skew, and chasing it would waste the round.
"""
import os, sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group("nccl")
r = dist.get_rank()
dev = torch.device("cuda")
HID = int(sys.argv[1]) if len(sys.argv) > 1 else 7168

def bench(fn, n=100, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

if r == 0:
    print("hidden=%d; 2 ranks; both arrive together" % HID)
    print("%10s %10s %12s %12s" % ("tokens", "msg MB", "nccl ms", "p2p copy ms"))
for tokens in (512, 1024, 2048):
    x = torch.randn(tokens, HID, dtype=torch.float16, device=dev)
    t_nccl = bench(lambda: dist.all_reduce(x))
    # Raw link floor: push the whole message to the peer and read it back.
    peer = torch.empty_like(x)
    grp = dist.group.WORLD
    def p2p():
        dist.broadcast(x, src=0, group=grp)
    t_p2p = bench(p2p)
    if r == 0:
        mb = tokens * HID * 2 / 1e6
        print("%10d %10.2f %12.3f %12.3f" % (tokens, mb, t_nccl, t_p2p))
dist.destroy_process_group()
