"""Correctness + speed for block_m > 16 in the repacked W4A16 grouped GEMM.

Why: the grid is (m_block, n_tile) and each m_block is 16 rows, so an expert that
needs 32 rows pays two full walks of its weight slice. Measured at the prefill
shape (chunk 512 x topk 6 over 128 experts per TP2 rank): 1 block/expert 10.08 ms
at 80 GB/s of weight traffic, 2 blocks/expert 19.85 ms at 41 GB/s -- linear in the
block count. block_m=32 makes one warp walk the weight for two 16-row tiles.

Layout note: moe_align_block_size lays out sorted_token_ids as
[expert0: ceil(cnt0/bm)*bm slots][expert1: ...], so a row's slot is
expert_block_start + offset_within_expert, and expert_ids has one entry per bm
rows. This test builds that layout by hand for a chosen block_m.
"""
import sys
import time

import torch

repo = "/data/nvme/sglang-codex/sglang"
sys.path.insert(0, repo + "/python")

from sglang.kernels.ops.moe.mxfp4_w4a16_kernels import (  # noqa: E402
    mxfp4_w4a16_gemm_ptx,
    repack_mxfp4_for_ptx,
)

torch.manual_seed(0)
dev = "cuda:0"
E_SMALL, HIDDEN_S, INTER_S, TOPK = 4, 256, 128, 6


def e2m1_lut():
    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return torch.tensor(vals + [-v for v in vals], device=dev)


LUT = e2m1_lut()


def dequant_ref(w, s):
    """[E, N, K//2] int8-packed + [E, N, K//32] UE8M0 bytes -> [E, N, K] fp32."""
    E_, N, Kh = w.shape
    K = Kh * 2
    b = w.view(torch.uint8)
    vals = torch.empty(E_, N, K, device=dev, dtype=torch.float32)
    vals[:, :, 0::2] = LUT[(b & 0xF).long()]
    vals[:, :, 1::2] = LUT[(b >> 4).long()]
    scale = torch.pow(2.0, s.float() - 127.0)
    return (vals.view(E_, N, K // 32, 32) * scale.unsqueeze(-1)).view(E_, N, K)


def build_layout(rows_per_expert, n_act, block_m):
    """Hand-build moe_align_block_size's output for a uniform expert load.

    sorted_token_ids entries index the topk-flattened activation buffer, whose
    length is num_tokens*topk. With a uniform load that is exactly
    n_act*rows_per_expert rows, so row r of expert e points at activation row
    e*rows_per_expert + r and lands at slot e*slots_per_expert + r.
    """
    blocks_per_expert = -(-rows_per_expert // block_m)
    slots_per_expert = blocks_per_expert * block_m
    num_slots = n_act * slots_per_expert
    ids = torch.full((num_slots,), n_act * rows_per_expert, dtype=torch.int32,
                     device=dev)
    for e in range(n_act):
        for r in range(rows_per_expert):
            ids[e * slots_per_expert + r] = e * rows_per_expert + r
    eids = torch.arange(n_act, dtype=torch.int32, device=dev).repeat_interleave(
        blocks_per_expert)
    # every block is live, so the padded count is the whole buffer
    num_valid = torch.full((1,), num_slots, dtype=torch.int32, device=dev)
    return ids, eids, num_valid, num_slots, blocks_per_expert


def run_case(n_act, rows_per_expert, block_m):
    N, K = 2 * INTER_S, HIDDEN_S
    ids, eids, num_valid, num_slots, _ = build_layout(rows_per_expert, n_act, block_m)
    a = (torch.randn(n_act * rows_per_expert, K, device=dev) * 0.3).half()
    w = torch.randint(-128, 127, (n_act, N, K // 2), dtype=torch.int8, device=dev)
    s = torch.randint(118, 127, (n_act, N, K // 32), dtype=torch.uint8, device=dev)
    out = torch.zeros(num_slots, N, dtype=torch.float16, device=dev)
    w_rep = repack_mxfp4_for_ptx(w.view(torch.uint8))

    # sentinel = number of activation rows; ids are e*rows_per_expert + r
    mxfp4_w4a16_gemm_ptx(a, w_rep, s, ids, eids, out, n_act * rows_per_expert,
                         num_valid=num_valid)
    torch.cuda.synchronize()

    wq = dequant_ref(w, s)
    bad, worst = 0, 0.0
    slots_per_expert = -(-rows_per_expert // block_m) * block_m
    for e in range(n_act):
        for r in range(rows_per_expert):
            slot = e * slots_per_expert + r
            got = out[slot].float()
            ref = a[e * rows_per_expert + r].float() @ wq[e].t()
            rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
            worst = max(worst, rel)
            if rel > 2e-2:
                bad += 1
    print(f"  rows/expert={rows_per_expert:3d} block_m={block_m:2d}: "
          f"{bad:3d} bad rows, worst rel {worst:.2e}  {'ok' if bad == 0 else 'FAIL'}")
    return bad == 0


def bench(fn, iters=15, warmup=4):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def perf_case(label, n_act, rows_per_expert, block_m):
    """TP2 shards the intermediate for gemm1 and hidden for gemm2."""
    for which, N, K in (("gemm1", 2 * 2048 // 2, 4096), ("gemm2", 4096 // 2, 2048)):
        ids, eids, num_valid, num_slots, _ = build_layout(
            rows_per_expert, n_act, block_m)
        a = (torch.randn(n_act * rows_per_expert, K, device=dev) * 0.05).half()
        w = torch.randint(-128, 127, (n_act, N, K // 2), dtype=torch.int8, device=dev)
        s = torch.randint(118, 127, (n_act, N, K // 32), dtype=torch.uint8, device=dev)
        out = torch.zeros(num_slots, N, dtype=torch.float16, device=dev)
        w_rep = repack_mxfp4_for_ptx(w.view(torch.uint8))
        t = bench(lambda: mxfp4_w4a16_gemm_ptx(a, w_rep, s, ids, eids, out,
                                               n_act * rows_per_expert,
                                               num_valid=num_valid))
        print(f"  {label} {which} experts={n_act:4d} rows/exp={rows_per_expert:3d} "
              f"bm={block_m:2d}: {t:8.3f} ms  weight-bw {w.numel()/t/1e6:5.0f} GB/s")
        del w_rep, w, out


def main():
    print("correctness (fp32 dequantized reference):")
    ok = True
    for rows in (1, 16, 17, 32, 33, 64, 65):
        ok &= run_case(E_SMALL, rows, 16)

    print("\nperformance, decode (6 experts, 1 row each):")
    perf_case("decode ", 6, 1, 16)
    print("\nperformance, prefill (chunk 512 x topk 6 over 128 experts = 24 rows):")
    perf_case("prefill", 128, 24, 16)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
