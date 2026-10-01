"""Run the V4 JIT store kernel (KV -> FP8 paged cache) on SM75 for real.

Real DSv4 SWA pool layout: input [T, 512] bf16 (448 nope + 64 rope),
page stride = ceil(page_size*584/576)*576 bytes, page_size=128 -> 74880.

Scale granularity (from csrc/deepseek_v4/store.cuh): the block is 256 threads
= 8 warps, each warp covers 64 elements and writes ONE UE8M0 scale byte to
scale_ptr[wid], i.e. 8 scale bytes per token at PAGE*576 + token*8 + wid.
Warps 0..6 quantize the 448 nope dims (7*64=448); warp 7 copies the 64 rope
dims through as bf16. Verifying with a single per-token scale is therefore
wrong -- it only passes when all seven warps happen to share an exponent.

Verifies the cuda_fp8.h software e4m3 conversion path works on Turing.
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

import sys
sys.path.insert(0, _PY)
import math
import torch

torch.cuda.set_device(0)
from sglang.kernels.ops.attention.dsv4.attn import _jit_fused_store_module

NOPE, ROPE = 448, 64
WARPS = 7          # warps that quantize nope; each owns 64 elements
PER_WARP = NOPE // WARPS  # 64
PAGE = 128
STRIDE = math.ceil(PAGE * 584 / 576) * 576  # 74880
T = 8
TRIALS = 20
# e4m3 has 3 mantissa bits -> worst-case relative rounding error is 2**-4.
E4M3_MAX_REL = 2 ** -4

fails = 0
for trial in range(TRIALS):
    torch.manual_seed(1000 + trial)
    inp = torch.randn(T, NOPE + ROPE, device="cuda:0", dtype=torch.bfloat16)
    cache = torch.zeros(4, STRIDE, device="cuda:0", dtype=torch.uint8)
    idx = torch.tensor([0, 1, 2, 3, 129, 130, 255, 256], device="cuda:0", dtype=torch.int32)

    mod = _jit_fused_store_module(name="flashmla", input_dtype=torch.bfloat16,
                                  index_dtype=torch.int32, page_size=PAGE)
    mod.run(inp, cache, idx)
    torch.cuda.synchronize()

    # verify every token against its own page/offset, per-warp scale
    worst = 0.0
    worst_desc = ""
    rope_ok = True
    for t in range(T):
        page = int(idx[t]) >> 7
        off = int(idx[t]) & 127
        tok = cache[page, off * 576: off * 576 + 576]
        scale_base = page * STRIDE + PAGE * 576 + off * 8
        for w in range(WARPS):
            lo, hi = w * PER_WARP, (w + 1) * PER_WARP
            payload = tok[lo:hi].view(torch.float8_e4m3fn).float()
            sb = int(cache.view(-1)[scale_base + w])
            scale = math.ldexp(1.0, sb - 127)
            want = inp[t, lo:hi].float()
            rel = ((payload * scale - want).abs().max() / want.abs().max()).item()
            if rel > worst:
                worst, worst_desc = rel, f"token{t} warp{w} scale_byte={sb}"
        rope_ok &= torch.equal(
            tok[NOPE:NOPE + ROPE * 2].view(torch.bfloat16), inp[t, NOPE:]
        )

    ok = worst <= E4M3_MAX_REL and rope_ok
    print(f"trial {trial}: worst per-warp rel err {worst:.4f} ({worst_desc}) "
          f"rope_ok={rope_ok} -> {'ok' if ok else 'BAD'}")
    if not ok:
        fails += 1

print(f"e4m3 worst-case bound = 2**-4 = {E4M3_MAX_REL:.4f}")
print("JIT STORE REAL RUN:", "PASS" if fails == 0 else f"FAIL ({fails}/{TRIALS})")
raise SystemExit(0 if fails == 0 else 1)
