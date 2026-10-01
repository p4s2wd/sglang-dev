"""gather-LUT vs bit-decode at nw=8, plus BM sweep (more rows amortize dequant)."""
import time
import torch
import triton
import triton.language as tl

E, K, N = 6, 4096, 4096
LUT = torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float16, device="cuda:0")

@triton.jit
def _decode_bits(code):
    c = code.to(tl.int32)
    e = (c >> 1) & 0x03
    m = c & 0x01
    expf = e + 14
    mant = tl.where(e == 0, 0, m << 9)
    bits = tl.where((e == 0) & (m == 0), 0, (expf << 10) | mant)
    bits = tl.where((c & 0x08) != 0, bits | 0x8000, bits)
    return bits.to(tl.int16).to(tl.float16, bitcast=True)

@triton.jit
def k(a_ptr, w_ptr, lut_ptr, c_ptr, N, K, sam, swe, swn, swk, scm,
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
        b_u8 = tl.load(w_ptrs)
        if MODE == 0:  # bit decode
            lo = _decode_bits(b_u8 & 0x0F); hi = _decode_bits((b_u8 >> 4) & 0x0F)
        else:          # gather LUT
            lo = tl.gather(lut_ptr, (b_u8 & 0x0F).to(tl.int32), axis=0)
            hi = tl.gather(lut_ptr, ((b_u8 >> 4) & 0x0F).to(tl.int32), axis=0)
        b = tl.trans(tl.interleave(lo, hi))
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BK
        w_ptrs += BK2 * swk
    tl.store(c_ptr + offs_m[:, None].to(tl.int64) * scm + offs_n[None, :], acc.to(tl.float16))

def bench(mode, BM, BN, BK, nw, ns, iters=15):
    dev = "cuda:0"
    nblocks = E
    a = torch.randn(nblocks * BM, K, device=dev, dtype=torch.float16)
    w = torch.randint(-128, 127, (E, N, K // 2), dtype=torch.int8, device=dev).view(torch.uint8)
    c = torch.zeros(nblocks * BM, N, device=dev, dtype=torch.float16)
    grid = (nblocks * triton.cdiv(N, BN),)
    def run():
        k[grid](a, w, LUT, c, N, K, a.stride(0), w.stride(0), w.stride(1), w.stride(2),
                c.stride(0), BM=BM, BN=BN, BK=BK, MODE=mode, num_warps=nw, num_stages=ns)
    try:
        for _ in range(3): run()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"mode={mode} BM={BM} BN={BN} BK={BK} nw={nw}: ERR {str(e)[:70]}"); return
    t0 = time.perf_counter()
    for _ in range(iters): run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"mode={mode} BM={BM:3d} BN={BN} BK={BK} nw={nw} ns={ns}: {ms:6.2f} ms  {E*N*(K//2)/ms/1e6:5.0f} GB/s")

for mode in (0,):
    for BM in (16, 32, 64):
        bench(mode, BM, 64, 128, 8, 2)
