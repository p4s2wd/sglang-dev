"""The attention gather floor at PRODUCTION pool size, not a toy cache.

The carried-over verdict "attention is at its floor: 2.524 ms vs 2.368 ms for a
bare gather" was measured with a small KV cache that fits in L2 (5.4 MB). In
production the pool is 236800 tokens x 584 B = 138 MB, and the prefill trace
shows the same kernel launch taking 2 ms early in a prompt (most topk entries
are padding, so masked loads all hit token 0, L2-resident) and 23-25 ms late in
one (all 512 entries valid, scattered across the whole pool). 302 MB of useful
reads in 23 ms is 13 GB/s -- 2.5% of the card's streaming rate. Either the
random 576-byte gather pattern really costs that much, or the kernel is leaving
10x on the table. This measures both at the real shape: the production kernel,
and a gather-only kernel with identical addressing that just sums the bytes.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch, triton
import triton.language as tl
import ab_headshared_scale as A
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

TOKEN_BYTES = tl.constexpr(576)
SCALE_BYTES = tl.constexpr(8)

@triton.jit
def gather_only(idx_ptr, cache_ptr, out_ptr, topk, page_size, page_bytes,
                scale_off, BLOCK_T: tl.constexpr):
    bid = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, 512)
    acc = tl.zeros([BLOCK_T, 512], dtype=tl.float32)
    for ts in range(0, topk, BLOCK_T):
        t_idx = ts + offs_t
        raw = tl.load(idx_ptr + bid * topk + t_idx, mask=t_idx < topk, other=0).to(tl.int64)
        page_ids = raw // page_size
        page_offs = raw % page_size
        base = page_ids * page_bytes + page_offs * TOKEN_BYTES
        b = tl.load(cache_ptr + base[:, None] + offs_c[None, :],
                    mask=(t_idx < topk)[:, None], other=0)
        acc += b.to(tl.float32)
    if tl.sum(acc) == 12345.678:
        tl.store(out_ptr + bid, 1.0)

def bench(fn, iters=10, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 236800
B, H, D, TOPK = 512, 32, 512, 512
q, kc, indices = A.build(B, H, D, TOPK, POOL)
flat = kc.as_strided((kc.numel(),), (1,)).view(torch.uint8)
scale = flat.view(torch.bfloat16)
out = torch.zeros(B, H, D, dtype=torch.float16, device="cuda")
lse = torch.zeros(B, H, dtype=torch.float32, device="cuda")
outsum = torch.zeros(B, dtype=torch.float32, device="cuda")
page_size = kc.shape[1]
page_bytes = kc.stride(0)
print("pool %d tokens = %.0f MB; B=%d H=%d topk=%d" % (POOL, POOL*584/1e6, B, H, TOPK))

# production kernel, all-valid (late-chunk steady state)
tlen = torch.full((B,), TOPK, dtype=torch.int32, device="cuda")
t_full = bench(lambda: M._run_headshared_sparse_decode(
    q.view(B, 1, H, D), kc, indices.view(B, 1, TOPK), tlen, 0.088))

# gather-only floor, same addressing, x2 head-block redundancy included
idx = indices.view(B, -1).contiguous()
t_gather = bench(lambda: gather_only[(B,)](idx, flat, outsum, TOPK, page_size,
                                           int(page_bytes), page_size * 576,
                                           BLOCK_T=16, num_warps=4))
useful = B * TOPK * 584 / 1e6
print("\nuseful bytes per call: %.0f MB (x2 with head-block redundancy)" % useful)
print("production kernel : %7.2f ms  %5.1f GB/s useful" % (t_full, useful / t_full))
print("gather-only floor : %7.2f ms  %5.1f GB/s useful" % (t_gather, useful / t_gather))
print("ratio kernel/floor: %.2fx" % (t_full / t_gather))
