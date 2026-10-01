"""Does removing tl.trans from the QK dot speed the attention kernel up?

The gather is free: a bare gather of the same 151 MB runs in 0.24 ms, which is
614 GB/s -- 100% of the card's 616 GB/s peak. The production kernel takes 19.3 ms
for the same bytes, so ~18 ms is the mma path, running at 1.5% of the 58 TFLOP/s
measured tensor-core peak. The one structurally expensive thing in that path is
tl.trans: every one of the 8 nope chunks is gathered as [BLOCK_T, NCOL] and then
transposed, which in Triton means a shared-memory round-trip per chunk. Gathering
straight into [NCOL, BLOCK_T] should remove it.

Correctness matters as much as speed here, so the transposed variant is compared
against the non-transposed one on the same inputs, not just timed.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import ab_headshared_scale as A
import bh32_nohoist as B
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

POOL = 236800
Bt, H, D, TOPK = 512, 32, 512, 512
q, kc, idx = A.build(Bt, H, D, TOPK, POOL)
out = torch.zeros(Bt, H, D, dtype=torch.float16, device="cuda")
lse = torch.zeros(Bt, H, dtype=torch.float32, device="cuda")
mb = Bt * TOPK * 576 / 1e6

def bench(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

def go(ncol, warps, trans, bt=16, stages=2):
    o = torch.zeros(Bt, H, D, dtype=torch.float16, device="cuda")
    l = torch.zeros(Bt, H, dtype=torch.float32, device="cuda")
    B.run(q, kc, idx, 0.088, o, l, bh=16, ncol=ncol, warps=warps, hoist=1,
          stages=stages, bt=bt, deq=2, trans=trans)
    return o, l

print("correctness: transposed gather vs tl.trans, same inputs")
ref_o, ref_l = go(64, 4, 0)
for ncol, warps in ((64, 4), (128, 4), (64, 8)):
    try:
        o, l = go(ncol, warps, 1)
        torch.cuda.synchronize()
        do = (o.float() - ref_o.float()).abs().max().item()
        dl = (l - ref_l).abs().max().item()
        print("  ncol=%3d w%d  max|dO|=%.3e max|dLSE|=%.3e  %s"
              % (ncol, warps, do, dl, "OK" if do < 2e-2 else "MISMATCH"))
    except Exception as e:
        print("  ncol=%3d w%d  FAIL %s" % (ncol, warps, str(e).splitlines()[-1][:56]))

print("\nspeed (median of 5 interleaved rounds), %.0f MB useful/call" % mb)
cands = [("production", None),
         ("tl.trans n64 w4", (64, 4, 0)), ("TRANS  n64 w4", (64, 4, 1)),
         ("tl.trans n128 w4", (128, 4, 0)), ("TRANS  n128 w4", (128, 4, 1)),
         ("TRANS  n64 w4 bt32", (64, 4, 1)), ("TRANS  n128 w8", (128, 8, 1))]
res = {n: [] for n, _ in cands}
t_prod = bench(lambda: M._run_headshared_sparse_decode(
    q, kc, idx, torch.full((Bt,), TOPK, dtype=torch.int32, device="cuda"), 0.088))
for r in range(5):
    for name, cfg in cands:
        if cfg is None: continue
        ncol, warps, trans = cfg
        bt = 32 if "bt32" in name else 16
        res[name].append(bench(lambda c=(ncol, warps, trans, bt): B.run(
            q, kc, idx, 0.088, out, lse, bh=16, ncol=c[0], warps=c[1], hoist=1,
            stages=2, bt=c[3], deq=2, trans=c[2])))
print("  %-22s %8s %9s" % ("config", "ms", "GB/s"))
print("  %-22s %8.2f %9.1f" % ("production", t_prod, mb / t_prod))
for name, ts in res.items():
    if not ts: continue
    ts.sort(); med = ts[2]
    print("  %-22s %8.2f %9.1f" % (name, med, mb / med))
