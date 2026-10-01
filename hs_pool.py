"""Production attention kernel time vs KV pool size, at the real call shape.

Two diagnoses were on the table for why prefill runs at 890 tok/s when one
pipeline stage appeared to do only 172 ms of device work per 512-token chunk. One
said the four PP stages serialize; the other said the trace was truncated by
profile_by_stage and the stages really do overlap. The PP=4 vs PP=2 comparison
settles that: serialized stages predict equal throughput (both move 43 layers of
work per chunk), perfect pipelining predicts PP4 = 2x PP2, and measured is 890 vs
626 = 1.42x, so pipelining works partially and the trace was indeed undercounting.

That leaves the second-order effect as the real lever: attention gathers topk=512
entries per query token from the KV pool, so as a long prompt grows the pool grows
and the same gather walks from L2-resident to DRAM-resident scattered pages. This
measures the shipped kernel at a fixed call shape across pool sizes spanning that
transition, interleaved so thermal drift cannot masquerade as a pool effect. If
attention time tracks pool size, the fix is to make the gather cheaper at large
pools; if it is flat, the length effect lives elsewhere.
"""
import os, sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import ab_headshared_scale as A
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

B, H, D, TOPK = 512, 32, 512, 512
POOLS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else
                          ["16384", "32768", "65536", "131072", "236800"])]
ROUNDS = 3

def bench(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

sets = {}
for p in POOLS:
    q, kc, idx = A.build(B, H, D, TOPK, p)
    tlen = torch.full((B,), TOPK, dtype=torch.int32, device="cuda")
    sets[p] = (q, kc, idx, tlen)
    print("pool %7d tokens = %6.0f MB" % (p, p * 584 / 1e6))

res = {p: [] for p in POOLS}
for r in range(ROUNDS):
    for p in POOLS:
        q, kc, idx, tlen = sets[p]
        res[p].append(bench(lambda a=(q, kc, idx, tlen): M._run_headshared_sparse_decode(
            a[0], a[1], a[2], a[3], 0.088)))

mb = B * TOPK * 576 / 1e6
print("\nshipped kernel, %.0f MB gathered per call (fixed regardless of pool)" % mb)
print("%12s %10s %9s %9s %9s" % ("pool tokens", "ms med", "best", "worst", "GB/s"))
base = None
for p in POOLS:
    v = sorted(res[p]); med = v[len(v) // 2]
    if base is None: base = med
    print("%12d %10.2f %9.2f %9.2f %9.1f  %.2fx" % (p, med, v[0], v[-1], mb / med, med / base))
