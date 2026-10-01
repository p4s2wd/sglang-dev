"""Can BLOCK_H=32 fit SMEM if NCOL shrinks, and does it pay?

The grid is (B, cdiv(H, BLOCK_H)) = (512, 2), so every token's 576-byte KV entry
is gathered AND staged through shared memory twice per call, once per head block.
The kernel is not mma-bound -- it runs at 1.5% of the 58 TFLOP/s measured peak --
it is bound by Triton staging each tl.dot operand through SMEM, so halving the
number of times the same KV tile is staged should be worth up to ~1.7x.

BLOCK_H=32 was rejected earlier at 77824 B against the 65536 B SM75 cap. But SMEM
scales with NCOL as well as BLOCK_H: the eight KV tiles are [NCOL, BLOCK_T] each.
Halving NCOL to 32 doubles the chunk count (16 instead of 8) but halves every
staged tile, so the combination may fit. This measures feasibility and speed for
the whole (BLOCK_H, NCOL) grid rather than assuming either axis is fixed.

Correctness is checked against the fp32 reference for any config that launches,
because a kernel that reads only 8*NCOL of the 448 nope columns would look fast
and be wrong -- the trap an earlier version of this probe fell into.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import ab_headshared_scale as A
import test_headshared_mla as T
import hs_transposed as H
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

POOL = 236800
B, H_, D, TOPK = 512, 32, 512, 512

# --- correctness for each config that launches ---
NB, NT = 16, 4096
kc2, truth = T.build_cache(NT)
g = torch.Generator(device="cuda").manual_seed(3)
q2 = (torch.randn(NB, 1, H_, D, device="cuda", generator=g) * 0.5).half()
idx2 = torch.randint(0, NT, (NB, 1, TOPK), device="cuda", generator=g, dtype=torch.int32)
flat2 = idx2.reshape(NB, -1).contiguous()
tlen2 = torch.full((NB,), TOPK, dtype=torch.int32, device="cuda")
sc = D ** -0.5
ref = T.reference(q2, truth, idx2, tlen2, None)
p_out, _ = M._run_headshared_sparse_decode(q2, kc2, flat2, tlen2, sc)
rp = T.rel(p_out.squeeze(1), ref)
print("production rel vs fp32 reference: %.3e" % rp)

CFGS = [(16, 64), (32, 32), (32, 64), (16, 32)]
for bh, ncol in CFGS:
    o = torch.zeros(NB, H_, D, dtype=torch.float16, device="cuda")
    l = torch.zeros(NB, H_, dtype=torch.float32, device="cuda")
    try:
        H.run(q2, kc2, flat2, sc, o, l, bh=bh, ncol=ncol, warps=4, stages=1, pv_trans=0)
        torch.cuda.synchronize()
        print("  bh=%2d ncol=%3d  launches, rel %.3e  %s"
              % (bh, ncol, T.rel(o, ref), "OK" if T.rel(o, ref) < 3 * rp else "FAIL"))
    except Exception as e:
        print("  bh=%2d ncol=%3d  DOES NOT LAUNCH: %s"
              % (bh, ncol, str(e).splitlines()[-1][:60]))

q, kc, idx = A.build(B, H_, D, TOPK, POOL)
out = torch.zeros(B, H_, D, dtype=torch.float16, device="cuda")
lse = torch.zeros(B, H_, dtype=torch.float32, device="cuda")
mb = B * TOPK * 576 / 1e6

def bench(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

t_prod = bench(lambda: M._run_headshared_sparse_decode(
    q, kc, idx, torch.full((B,), TOPK, dtype=torch.int32, device="cuda"), 0.088))
print("\nproduction kernel: %.2f ms (%.0f MB/call)" % (t_prod, mb))

res = {}
bad = set()
for r in range(5):
    for bh, ncol in CFGS:
        name = "bh=%d ncol=%d" % (bh, ncol)
        if name in bad: continue
        try:
            res.setdefault(name, []).append(bench(
                lambda b=bh, n=ncol: H.run(q, kc, idx, 0.088, out, lse,
                                           bh=b, ncol=n, warps=4, stages=1, pv_trans=0)))
        except Exception:
            bad.add(name)
for name, ts in sorted(res.items()):
    ts.sort()
    print("  %-16s %7.2f ms  %.2fx vs production" % (name, ts[2], t_prod / ts[2]))
