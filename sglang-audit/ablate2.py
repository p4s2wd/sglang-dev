"""Decisive ablation: is tl.dot on register-computed operands the bottleneck?"""
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
def k(a_ptr, w_ptr, wf_ptr, c_ptr, N, K,
      sam, swe, swn, swk, scm,
      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pid_m, pid_n = pid // npn, pid % npn
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    BK2: tl.constexpr = BK // 2
    offs_kh = tl.arange(0, BK2)
    a_ptrs = a_ptr + offs_m[:, None] * sam + tl.arange(0, BK)[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * swn + offs_kh[None, :] * swk
    acc = tl.zeros((BM, BN), tl.float32)
    for kt in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        if MODE == 0:      # plain fp16 GEMM: load b directly, dot
            b = tl.load(wf_ptr + offs_n[:, None] * BK + tl.arange(0, BK)[None, :])
            acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
            wf_ptr += 0  # placeholder
        elif MODE == 1:    # load packed bytes only, no decode, no dot
            b_u8 = tl.load(w_ptrs)
            acc += tl.sum(b_u8.to(tl.float32), axis=1)[:, None] * 0.0 + b_u8[:, :BN].to(tl.float32)
        elif MODE == 2:    # decode + FMA (no tl.dot, no trans)
            b_u8 = tl.load(w_ptrs)
            lo = _decode(b_u8 & 0x0F); hi = _decode((b_u8 >> 4) & 0x0F)
            b = tl.interleave(lo, hi)          # [BN, BK]
            acc += tl.sum(a[:, :, None] * b[None, :, :], axis=2)  # [BM,BN] heavy
        elif MODE == 3:    # decode + dot (current design)
            b_u8 = tl.load(w_ptrs)
            lo = _decode(b_u8 & 0x0F); hi = _decode((b_u8 >> 4) & 0x0F)
            b = tl.trans(tl.interleave(lo, hi))
            acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BK
        w_ptrs += BK2 * swk
    tl.store(c_ptr + offs_m[:, None].to(tl.int64) * scm + offs_n[None, :], acc.to(tl.float16))

E, K, N = 6, 4096, 4096

def bench(mode, iters=10):
    dev = "cuda:0"
    BM, BN, BK = 16, 64, 128
    nblocks = E
    a = torch.randn(nblocks * BM, K, device=dev, dtype=torch.float16)
    w = torch.randint(-128, 127, (E, N, K // 2), dtype=torch.int8, device=dev).view(torch.uint8)
    wf = torch.randn(E, N, K, device=dev, dtype=torch.float16)
    c = torch.zeros(nblocks * BM, N, device=dev, dtype=torch.float16)
    grid = (nblocks * triton.cdiv(N, BN),)
    def run():
        k[grid](a, w, wf, c, N, K, a.stride(0), w.stride(0), w.stride(1), w.stride(2),
                c.stride(0), BM=BM, BN=BN, BK=BK, MODE=mode, num_warps=4, num_stages=2)
    try:
        for _ in range(2): run()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"MODE={mode}: ERROR {type(e).__name__}: {str(e)[:120]}")
        return
    t0 = time.perf_counter()
    for _ in range(iters): run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    wbytes = E * N * (K // 2)
    print(f"MODE={mode}: {ms:8.2f} ms   {wbytes/ms/1e6:6.0f} GB/s")

# raw achievable read bandwidth for reference
x = torch.randint(0, 255, (E * N * (K // 2),), dtype=torch.uint8, device="cuda:0")
for _ in range(3): s = x.to(torch.float32).sum()
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(20): s = x.to(torch.float32).sum()
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / 20 * 1000
print(f"torch uint8->f32 sum: {ms:8.2f} ms   {E*N*(K//2)/ms/1e6:6.0f} GB/s (reference)")

for mode in (0, 1, 3):
    bench(mode)
