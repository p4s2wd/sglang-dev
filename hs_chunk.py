"""Is the 17 ms the 64-byte gather granularity or the dots? Chunk a bare gather.

hs_floor.py's hand-written kernel read the same 151 MB at 641 GB/s in 0.24 ms,
but it loaded all 512 columns of a token in ONE tl.load (512 contiguous bytes per
token row). The production kernel loads NCOL=64 columns at a time, 8 chunks, so
each load touches a 64-byte slice of a 576-byte-strided token. The standalone
kernel's DEQ=0 knob still runs the dots and the trans, so it cannot separate the
two. This kernel has no dots at all -- just the gather summed into an
accumulator -- and a CHUNKS knob: CHUNKS=1 is the fast wide load, CHUNKS=8 is
production's 64-byte granularity. If CHUNKS=8 collapses to ~17 ms the wall is the
gather shape and the fix is wide loads. If it stays ~0.24 ms the wall is the dots
and trans, and the gather was never the problem.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch, triton
import triton.language as tl
import ab_headshared_scale as A

TOKEN_BYTES = tl.constexpr(576)

@triton.jit
def g(idx_ptr, cache_ptr, out_ptr, topk, page_size, page_bytes,
      BLOCK_T: tl.constexpr, NCOL: tl.constexpr, NCHUNK: tl.constexpr):
    bid = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, NCOL)
    acc = tl.zeros([BLOCK_T, NCOL], dtype=tl.float32)
    for ts in range(0, topk, BLOCK_T):
        t_idx = ts + offs_t
        raw = tl.load(idx_ptr + bid * topk + t_idx, mask=t_idx < topk, other=0).to(tl.int64)
        base = (raw // page_size) * page_bytes + (raw % page_size) * TOKEN_BYTES
        for c in tl.static_range(NCHUNK):
            b = tl.load(cache_ptr + base[:, None] + (c * NCOL + offs_c)[None, :],
                        mask=(t_idx < topk)[:, None], other=0)
            acc += b.to(tl.float32)
    if tl.sum(acc) == 12345.678:
        tl.store(out_ptr + bid, 1.0)

POOL = 236800
Bt, H, D, TOPK = 512, 32, 512, 512
q, kc, idx = A.build(Bt, H, D, TOPK, POOL)
flat = kc.as_strided((kc.numel(),), (1,)).view(torch.uint8)
out = torch.zeros(Bt, dtype=torch.float32, device="cuda")
ps, pb = kc.shape[1], kc.stride(0)
mb = Bt * TOPK * 576 / 1e6

def bench(fn, iters=10, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

print("bare gather, no dots. pool %.0f MB, %.0f MB useful/call" % (POOL*584/1e6, mb))
print("%-26s %8s %9s" % ("load shape", "ms", "GB/s"))
for ncol, nchunk, warps in ((512, 1, 4), (256, 2, 4), (128, 4, 4), (64, 8, 4),
                            (64, 8, 8), (128, 4, 8)):
    try:
        ms = bench(lambda n=ncol, c=nchunk, w=warps: g[(Bt,)](
            idx, flat, out, TOPK, ps, int(pb), BLOCK_T=16, NCOL=n, NCHUNK=c,
            num_warps=w))
        print("%-26s %8.3f %9.1f" % ("%d B x %d chunks w%d" % (ncol, nchunk, warps),
                                     ms, 2*mb/ms))
    except Exception as e:
        print("%-26s FAIL %s" % ("%d B x %d" % (ncol, nchunk), str(e).splitlines()[-1][:44]))
