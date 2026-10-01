"""Ablate the W4A16 kernel to find the real bottleneck on SM75."""
import time
import torch
import triton
import triton.language as tl

@triton.jit
def _decode(code):
    c = code.to(tl.int32)
    sign = tl.where((c & 0x08) != 0, -1.0, 1.0).to(tl.float32)
    e = (c >> 1) & 0x03
    m = c & 0x01
    val = tl.where(e == 0, 0.5 * m.to(tl.float32),
                   tl.exp2((e - 1).to(tl.float32)) * (1.0 + 0.5 * m.to(tl.float32)))
    return (sign * val).to(tl.float16)

@triton.jit
def k_variant(a_ptr, w_ptr, s_ptr, c_ptr, EIDS, N, K,
              sam, swe, swn, swk, sse, ssn, ssk, scm,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              MODE: tl.constexpr):
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BLOCK_N)
    pid_m = pid // npn
    pid_n = pid % npn
    expert = tl.load(EIDS + pid_m).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    BK2: tl.constexpr = BLOCK_K // 2
    offs_kh = tl.arange(0, BK2)
    a_ptrs = a_ptr + offs_m[:, None] * sam + tl.arange(0, BLOCK_K)[None, :]
    w_ptrs = w_ptr + expert * swe + offs_n[None, :] * swn + offs_kh[:, None] * swk
    s_ptrs = s_ptr + expert * sse + offs_n[None, :] * ssn
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b_u8 = tl.load(w_ptrs)
        lo = _decode(b_u8 & 0x0F)
        hi = _decode((b_u8 >> 4) & 0x0F)
        if MODE == 0:      # gather scale [BK2, BN] (current)
            s_raw = tl.load(s_ptrs + (kt * (BLOCK_K // 32) + offs_kh[:, None] // 16) * ssk)
            s = s_raw.to(tl.float16)
            b = tl.trans(tl.interleave(tl.trans(lo * s), tl.trans(hi * s)))
        elif MODE == 1:    # no scale at all
            b = tl.trans(tl.interleave(tl.trans(lo), tl.trans(hi)))
        elif MODE == 2:    # scalar scale per tile (coalesced [BN])
            s_raw = tl.load(s_ptrs + (kt * (BLOCK_K // 32)) * ssk)
            s = s_raw.to(tl.float16)
            b = tl.trans(tl.interleave(tl.trans(lo * s), tl.trans(hi * s)))
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        w_ptrs += BK2 * swk
    tl.store(c_ptr + offs_m[:, None].to(tl.int64) * scm + offs_n[None, :], acc.to(tl.float16))

def bench(mode, iters=10):
    dev = "cuda:0"
    E, K, N = 6, 4096, 4096
    nblocks = 6
    BM, BN, BK = 16, 64, 128
    a = torch.randn(nblocks * BM, K, device=dev, dtype=torch.float16)
    w = torch.randint(-128, 127, (E, N, K // 2), dtype=torch.int8, device=dev)
    s = torch.rand(E, N, K // 32, device=dev)
    c = torch.zeros(nblocks * BM, N, device=dev, dtype=torch.float16)
    eids = torch.arange(nblocks, dtype=torch.int32, device=dev)
    wu = w.view(torch.uint8)
    grid = (nblocks * triton.cdiv(N, BN),)
    def run():
        k_variant[grid](a, wu, s, c, eids, N, K, a.stride(0), wu.stride(0), wu.stride(1),
                        wu.stride(2), s.stride(0), s.stride(1), s.stride(2), c.stride(0),
                        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, MODE=mode,
                        num_warps=4, num_stages=2)
    for _ in range(2): run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    wbytes = E * N * (K // 2)
    print(f"MODE={mode}: {ms:7.2f} ms   weight-bw {wbytes/ms/1e6:6.0f} GB/s")

for mode in (0, 1, 2):
    bench(mode)
