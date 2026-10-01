"""Diagnose the FP8 store kernel's occasional large reconstruction error.

Baseline: with the assumed scale offset (PAGE*576 + token), most trials
reconstruct to within 1 ULP of the e4m3 payload, which is correct. A minority
of trials show ~0.86 relative error, which is ~2x and therefore not a rounding
artefact. This script runs many trials, and for each FAILING trial dumps the
per-element worst offender plus a brute-force search for the byte that best
explains the stored payload, so we can tell a wrong scale offset apart from a
genuine kernel defect.
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

assumed = PAGE * 576
nbad = 0
for trial in range(N):
    torch.manual_seed(1000 + trial)
    inp = torch.randn(T, NOPE + ROPE, device="cuda:0", dtype=torch.bfloat16)
    cache = torch.zeros(4, STRIDE, device="cuda:0", dtype=torch.uint8)
    idx = torch.tensor([0, 1, 2, 3, 129, 130, 255, 256], device="cuda:0", dtype=torch.int32)
    mod = _jit_fused_store_module(name="flashmla", input_dtype=torch.bfloat16,
                                  index_dtype=torch.int32, page_size=PAGE)
    mod.run(inp, cache, idx)
    torch.cuda.synchronize()

    c0 = cache[0]
    payload = c0[:NOPE].view(torch.float8_e4m3fn).float()
    want = inp[0, :NOPE].float()
    sb = int(c0[assumed])
    scale = math.ldexp(1.0, sb - 127)
    diff = (payload * scale - want).abs()
    wi = int(diff.argmax())
    rel = float(diff.max() / want.abs().max())
    if rel <= 0.0625:
        continue
    nbad += 1
    print(f"--- trial {trial} rel={rel:.4f} scale_byte={sb} scale={scale:.3e}")
    print(f"    worst elem {wi}: want={float(want[wi]):.4f} "
          f"payload={float(payload[wi]):.2f} recon={float(payload[wi]*scale):.4f}")
    print(f"    want amax={float(want.abs().max()):.4f} "
          f"payload amax={float(payload.abs().max()):.2f}")
    # implied scale if the payload were correct for the worst element
    if float(payload[wi]) != 0:
        print(f"    implied scale at worst elem = {float(want[wi]/payload[wi]):.3e} "
              f"(ratio to assumed {float(want[wi]/payload[wi]/scale):.3f})")
    # brute force best byte, vectorized
    bytes_f = c0.float()
    errs = torch.empty(STRIDE, device="cuda:0")
    for lo in range(0, STRIDE, 4096):
        hi = min(lo + 4096, STRIDE)
        s = torch.ldexp(torch.ones(hi - lo, device="cuda:0"), bytes_f[lo:hi] - 127)
        errs[lo:hi] = (s[:, None] * payload[None, :] - want[None, :]).abs().amax(dim=1)
    bi = int(errs.argmin())
    print(f"    brute-force best byte={bi} val={int(c0[bi])} err={float(errs[bi]):.4f} "
          f"| assumed err={float(diff.max()):.4f}")
    # is the payload itself consistent (single scale across the token)?
    ratio = want / torch.where(payload == 0, torch.ones_like(payload), payload)
    r_ok = ratio[payload != 0]
    if r_ok.numel():
        print(f"    payload-implied scale spread: min={float(r_ok.min()):.3e} "
              f"max={float(r_ok.max()):.3e}")
print(f"BAD {nbad} / {N}")
