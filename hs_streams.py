"""Does running the head-shared kernel's topk chunks CONCURRENTLY beat one call?

Correction to the previous probe: it split topk across n SEQUENTIAL launches and
concluded splitting was linear-slower. That was an artifact -- sequential launches
serialize and pay per-call cost n times, whereas real flash-decoding splits inside
one launch so the blocks are resident together.

The single-call topk sweep settles what the kernel actually costs:
  topk 512 -> 0.715 ms (32 tiles)   topk 128 -> 0.162 ms (8 tiles)
  topk 256 -> 0.363 ms (16 tiles)   topk  16 -> 0.156 ms (1 tile)
so there is a ~0.155 ms floor and the tile loop adds ~22 us per tile above ~8 tiles.
The tile loop is serial per block, and 22 us for 16 tokens x 576 B is exposed
latency, not bandwidth. That is exactly the condition an in-kernel topk split fixes.

Test the prediction without writing a kernel: issue the n_splits calls on n_splits
CUDA streams so their blocks are resident at the same time. 4 blocks per call means
8 streams put 32 blocks on 68 SMs, all co-resident. If wall time approaches the
0.155 ms floor, an in-kernel split (one launch, grid B x H/16 x n_splits, plus the
existing LSE merge) is worth implementing; if it stays near 0.7 ms, the kernel does
not overlap and the split is dead for real.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import ab_headshared_scale as A
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

H, D, TOPK = 64, 512, 512
q, kc, idx = A.build(1, H, D, TOPK, 236800)
qb = q.expand(1, 1, H, D).contiguous()
full = idx.reshape(1, -1).contiguous()
streams = [torch.cuda.Stream() for _ in range(16)]


def bench(fn, iters=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


base = sorted(bench(lambda: M._run_headshared_sparse_decode(
    qb, kc, full, None, 0.088)) for _ in range(3))[1]
ph = sorted(bench(lambda: M._run_triton_sparse_decode(
    qb, kc, full, None, 0.088)) for _ in range(3))[1]
print("one head-shared call (32 tiles): %.4f ms   per-head (production): %.4f ms"
      % (base, ph))
print("\nchunks issued on separate streams (blocks co-resident):")
print("%6s %8s %10s %10s %10s" % ("splits", "tiles/blk", "ms", "vs 1 call", "vs per-head"))
for ns in (2, 4, 8, 16):
    chunk = TOPK // ns
    slices = [(s * chunk, (s + 1) * chunk) for s in range(ns)]
    tlens = [torch.full((1,), chunk, dtype=torch.int32, device="cuda") for _ in slices]

    def run():
        outs = []
        ev = torch.cuda.Event()
        ev.record()
        for (lo, hi), tl_, st in zip(slices, tlens, streams):
            st.wait_event(ev)
            with torch.cuda.stream(st):
                outs.append(M._run_headshared_sparse_decode(
                    qb, kc, full[:, lo:hi].contiguous(), tl_, 0.088))
        for st in streams[:ns]:
            torch.cuda.current_stream().wait_stream(st)
        return outs

    t = sorted(bench(run) for _ in range(3))[1]
    print("%6d %8d %10.4f %9.2fx %10.2fx" % (ns, chunk // 16, t, base / t, ph / t))
