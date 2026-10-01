"""Sweep the per-head decode kernel's launch space on DEVICE time.

This is the kernel production decode actually runs at bs=1 (the head-shared one is
1.81x slower there, measured device-only). It costs 0.279 ms for topk=512, and the
time scales with tile count (0.039 ms at 4 tiles, 0.144 at 16, 0.279 at 32), so it is
latency-bound at ~8.7 us per 16-token tile rather than bandwidth-bound.

The @triton.autotune space is only three configs (BLOCK_T 16/32, num_warps 4/8,
num_stages 2). For a latency-bound gather loop, num_stages is the knob that matters
most -- it controls how many tiles' loads are prefetched ahead -- and it is pinned at
2. Widen the sweep and measure kernel duration with the profiler, since host wall
time around the wrapper has a ~0.155 ms floor that hides everything.
"""
import sys
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch, triton
from torch.profiler import ProfilerActivity, profile
import ab_headshared_scale as A
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

H, D, TOPK = 64, 512, 512
q, kc, idx = A.build(1, H, D, TOPK, 236800)
qb = q.expand(1, 1, H, D).contiguous()
full = idx.reshape(1, -1).contiguous()

B = 1
q3 = qb.squeeze(1).contiguous()
flat = full.contiguous()
total_elems = kc.shape[0] * kc.stride(0)
raw_u8 = kc.as_strided((total_elems,), (1,)).view(torch.uint8)
raw_bf16 = raw_u8.view(torch.bfloat16)
lut = M.fp8_payload_lut(q.device, torch.float32)
out = torch.zeros(B, H, D, dtype=torch.float16, device="cuda")
lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device="cuda")
tk = torch.empty(0, device="cuda", dtype=torch.int32)
KERN = M._tiled_sparse_decode_kernel.fn  # unwrap autotune


def launch(bt, warps, stages):
    KERN[(B, H)](
        q3, raw_u8, lut, raw_u8, raw_bf16, flat, tk, out, lse,
        0.088, kc.shape[1], int(kc.stride(0)),
        int(kc.shape[1] * M._TOKEN_DATA_STRIDE),
        H, TOPK, triton.next_power_of_2(TOPK), False,
        q3.stride(0), q3.stride(1), out.stride(0), out.stride(1), flat.stride(0),
        NOPE_PAD=512, ROPE_DIM=M._ROPE_DIM, NOPE_DIM_RT=M._NOPE_DIM,
        BLOCK_T=bt, num_warps=warps, num_stages=stages,
    )


def device_ms(cfg, iters=30):
    for _ in range(8):
        launch(*cfg)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(iters):
            launch(*cfg)
        torch.cuda.synchronize()
    tot = sum(e.device_time_total for e in pr.key_averages()
              if e.device_time_total and "tiled_sparse" in e.key)
    return tot / 1e3 / iters


print("per-head kernel, device time at production shape (bs=1, topk=512, 236800-token pool)")
print("%6s %6s %7s %10s" % ("BT", "warps", "stages", "ms"))
res = []
for bt in (16, 32, 64):
    for warps in (4, 8, 16):
        for stages in (1, 2, 3, 4):
            try:
                t = device_ms((bt, warps, stages))
                res.append((t, bt, warps, stages))
                print("%6d %6d %7d %10.4f" % (bt, warps, stages, t))
            except Exception as e:
                print("%6d %6d %7d   FAIL %s" % (bt, warps, stages, str(e)[:44]))
res.sort()
print("\nbest 5:")
for t, bt, w, s in res[:5]:
    print("  BT=%d warps=%d stages=%d  %.4f ms" % (bt, w, s, t))
print("current autotune best of its 3 configs is what production uses; baseline 0.2794 ms")
