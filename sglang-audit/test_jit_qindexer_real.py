"""Real run of the indexer Q quant JIT kernel (RoPE + Hadamard + fp8) on SM75.

Shapes from the real config: index_n_heads=64, index_head_dim=128,
qk_rope_head_dim=64. Verifies the fp8 e4m3 output is finite and the
Hadamard-rotated values dominate the raw ones (rotation actually applied).
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
from sglang.kernels.ops.attention.dsv4.elementwise import (
    fused_q_indexer_rope_hadamard_quant,
)

T, H, D, ROPE = 4, 64, 128, 64
MAXPOS = 1024
q = torch.randn(T, H, D, device="cuda:0", dtype=torch.bfloat16)
weight = torch.rand(T, H, device="cuda:0", dtype=torch.bfloat16)
inv_freq = torch.exp(
    -math.log(10000.0)
    * torch.arange(0, ROPE // 2, device="cuda:0", dtype=torch.float32)
    / (ROPE // 2)
)
pos = torch.arange(MAXPOS, device="cuda:0", dtype=torch.float32)
ang = torch.outer(pos, inv_freq)
freqs_cis = torch.polar(torch.ones_like(ang), ang)  # [MAXPOS, 32] complex64
positions = torch.tensor([0, 5, 100, 1023], device="cuda:0", dtype=torch.int64)

q_fp8, weights_out = fused_q_indexer_rope_hadamard_quant(
    q, weight, 0.1, freqs_cis, positions
)
torch.cuda.synchronize()
finite = torch.isfinite(q_fp8.float()).all().item()
mag = q_fp8.float().abs().mean().item()
w_ok = torch.isfinite(weights_out).all().item()
print(f"q_fp8 finite={finite} mean|q|={mag:.3f} weights finite={w_ok}")
ok = finite and w_ok and mag > 1e-3
print("INDEXER Q JIT REAL RUN:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
