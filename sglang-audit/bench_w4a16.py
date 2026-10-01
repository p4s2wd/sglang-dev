"""W4A16 kernel perf at real expert shapes on the 2080 Ti (SM75).

Real shapes: w13 [E, 4096, 2048-packed] (gate|up, K=4096), w2 [E, 4096, 1024-packed].
Decode-like: topk=6, E=256 experts active -> slots = B*6, padded per expert.
"""
import os as _os, pathlib as _pb
def _find_repo():
    r = _os.environ.get("SGLANG_REPO")
    if r:
        return r
    d = _pb.Path(__file__).resolve().parent
    for _ in range(8):
        for cand in (d, d / "sglang"):
            if (cand / "python" / "sglang").is_dir():
                return str(cand)
        d = d.parent
    raise RuntimeError("set SGLANG_REPO to the sglang checkout")
_PY = _find_repo() + "/python"

import importlib.util
import time
import torch

spec = importlib.util.spec_from_file_location(
    "m", _PY + "/sglang/kernels/ops/moe/mxfp4_w4a16_kernels.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def bench(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

def main():
    dev = "cuda:0"
    E, K, inter = 256, 4096, 2048
    for B in (1, 8, 32):
        topk = 6
        block_m = 16
        # spread B*topk slots over ~min(E, B*topk) experts (decode: few experts hit)
        n_act = min(E, B * topk)
        slots = B * topk
        # simple layout: one expert per slot group of block_m
        per_e = (slots + n_act - 1) // n_act
        nblocks_per_e = (per_e + block_m - 1) // block_m
        num_slots = n_act * nblocks_per_e * block_m
        ids = torch.full((num_slots,), B * topk, dtype=torch.int32, device=dev)
        cnt = 0
        for e in range(n_act):
            take = min(per_e, slots - cnt)
            ids[cnt:cnt+take] = torch.arange(cnt, cnt+take, dtype=torch.int32, device=dev)
            cnt += take
        eids = torch.repeat_interleave(
            torch.arange(n_act, dtype=torch.int32, device=dev),
            torch.full((n_act,), nblocks_per_e, dtype=torch.int32, device=dev),
        )
        a = (torch.randn(B * topk, K, device=dev) * 0.05).half()
        w13 = torch.randint(-128, 127, (n_act, 2 * inter, K // 2), dtype=torch.int8, device=dev)
        s13 = torch.pow(2.0, torch.randint(-9, 0, (n_act, 2 * inter, K // 32)).float().to(dev))
        w2 = torch.randint(-128, 127, (n_act, K, inter // 2), dtype=torch.int8, device=dev)
        s2 = torch.pow(2.0, torch.randint(-9, 0, (n_act, K, inter // 32)).float().to(dev))
        out1 = torch.zeros(num_slots, 2 * inter, dtype=torch.float16, device=dev)
        act = torch.zeros(num_slots, inter, dtype=torch.float16, device=dev)
        out2 = torch.zeros(num_slots, K, dtype=torch.float16, device=dev)
        slot_ids = torch.arange(num_slots, dtype=torch.int32, device=dev)

        t1 = bench(lambda: m.mxfp4_w4a16_gemm(a, w13, s13, ids, eids, out1, sentinel=B*topk))
        t2 = bench(lambda: m.mxfp4_w4a16_gemm(act, w2, s2, slot_ids, eids, out2, sentinel=num_slots))
        # bytes actually read (weights of active experts, both GEMMs)
        wbytes = n_act * (2 * inter * (K // 2) + K * (inter // 2))  # int8 bytes
        print(f"B={B:3d} active_e={n_act:3d} slots={num_slots:5d}  gemm1 {t1:6.2f}ms gemm2 {t2:6.2f}ms "
              f"total {t1+t2:6.2f}ms  weight-bw {wbytes/(t1+t2)/1e6:5.0f} GB/s")

if __name__ == "__main__":
    main()
