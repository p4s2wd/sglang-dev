"""Measure the actual rel-err distribution of the V4 FP8 store kernel.

The store path quantizes each token's 448 nope dims to e4m3 with a per-token
UE8M0 scale. e4m3 carries 3 mantissa bits, so the worst-case relative
rounding error is 2**-4 = 6.25% -- a 5% pass threshold is below the format's
own bound and will fail on unlucky draws. This script measures the empirical
distribution so the threshold can be set from the format, not from luck.
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
PAGE = 128
STRIDE = math.ceil(PAGE * 584 / 576) * 576  # 74880
T = 8
N = 40

rels = []
for trial in range(N):
    torch.manual_seed(1000 + trial)
    inp = torch.randn(T, NOPE + ROPE, device="cuda:0", dtype=torch.bfloat16)
    cache = torch.zeros(4, STRIDE, device="cuda:0", dtype=torch.uint8)
    idx = torch.tensor([0, 1, 2, 3, 129, 130, 255, 256], device="cuda:0", dtype=torch.int32)
    mod = _jit_fused_store_module(name="flashmla", input_dtype=torch.bfloat16,
                                  index_dtype=torch.int32, page_size=PAGE)
    mod.run(inp, cache, idx)
    torch.cuda.synchronize()
    tok0 = cache[0, : 576]
    nope_fp8 = tok0[:NOPE].view(torch.float8_e4m3fn).float()
    scale_byte = cache[0, PAGE * 576 + 0]
    scale = math.ldexp(1.0, int(scale_byte) - 127)
    recon = nope_fp8 * scale
    err = (recon - inp[0, :NOPE].float()).abs().max().item()
    rel = err / inp[0, :NOPE].float().abs().max().item()
    rels.append(rel)

rels.sort()
print(f"trials={N} min={rels[0]:.4f} median={rels[N//2]:.4f} max={rels[-1]:.4f}")
print(f"e4m3 theoretical worst-case rel err = 2**-4 = {2**-4:.4f}")
print(f"count over 0.05: {sum(1 for r in rels if r > 0.05)} / {N}")
print(f"count over 0.0625: {sum(1 for r in rels if r > 0.0625)} / {N}")
