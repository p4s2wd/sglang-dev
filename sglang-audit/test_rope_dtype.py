"""fused_rope_inplace on SM75: the kernel used to be bf16-only, but sub-90
runs the model in fp16. Checks both dtypes against a complex-multiply reference.
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
import torch

torch.cuda.set_device(0)
from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace

ROPE_DIM = 64
MAXPOS = 512
dev = "cuda:0"
torch.manual_seed(0)

half = ROPE_DIM // 2
inv = torch.pow(10000.0, -(torch.arange(half, device=dev, dtype=torch.float32) / half))
t = torch.arange(MAXPOS, device=dev, dtype=torch.float32)
ang = torch.outer(t, inv)
freqs_cis = torch.polar(torch.ones_like(ang), ang)  # [MAXPOS, half] complex64

pos = torch.randint(0, MAXPOS, (37,), dtype=torch.int32, device=dev)
ok = True
for dt in (torch.bfloat16, torch.float16):
    q = torch.randn(37, 64, ROPE_DIM, device=dev).to(dt)
    k = torch.randn(37, 1, ROPE_DIM, device=dev).to(dt)
    q_ref = q.clone().float()
    k_ref = k.clone().float()

    fused_rope_inplace(q, k, freqs_cis, pos)
    torch.cuda.synchronize()

    # reference: rotate consecutive pairs as complex numbers
    for x, xr in ((q, q_ref), (k, k_ref)):
        v = xr.view(*xr.shape[:-1], half, 2)
        z = torch.view_as_complex(v)
        f = freqs_cis[pos.to(torch.long)].to(torch.complex64)  # [B, half]
        zrot = z * f.unsqueeze(-2)
        want = torch.view_as_real(zrot).flatten(-2)
        got = x.float().view(*x.shape[:-1], half, 2).flatten(-2)
        err = (got - want).abs().max().item()
        rel = err / max(want.abs().max().item(), 1e-9)
        tol = 1e-2 if dt is torch.bfloat16 else 1e-3
        good = rel < tol
        ok = ok and good
        print(f"{str(dt):16s} shape={tuple(x.shape)}: rel {rel:.2e} -> "
              f"{'PASS' if good else 'FAIL'}")

# q/k dtype mismatch must be rejected, not silently mis-rotated
try:
    fused_rope_inplace(
        torch.randn(4, 2, ROPE_DIM, device=dev, dtype=torch.float16),
        torch.randn(4, 1, ROPE_DIM, device=dev, dtype=torch.bfloat16),
        freqs_cis, torch.arange(4, device=dev, dtype=torch.int32))
    print("dtype mismatch rejected: False -> FAIL")
    ok = False
except ValueError:
    print("dtype mismatch rejected: True")

print("FUSED ROPE DTYPE:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
