"""Minimal repro: E=1, N=8, K=32, 16 slots. Inspect the error pattern."""
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
sys.path.insert(0, str(_pb.Path(__file__).resolve().parent))
import torch
from test_w4a16_ptx import load_module, repack_mxfp4

dev = "cuda:0"
mod = load_module()

E, N, K = 1, 8, 32
# W all = nibble 2 (=1.0): packed byte 0x22 (both nibbles = 2)
w = torch.full((E, N, K // 2), 0x22, dtype=torch.int8, device=dev)
s = torch.ones((E, N, K // 32), device=dev)
wr = repack_mxfp4(w)
print("repacked u32 (expect all 0x22222222):", wr.flatten()[:4].tolist(),
      "hex:", [hex(x & 0xFFFFFFFF) for x in wr.flatten()[:4].tolist()])

M = 16
a = torch.ones(M, K, dtype=torch.float16, device=dev)
sorted_ids = torch.arange(M, dtype=torch.int32, device=dev)
eids = torch.zeros(1, dtype=torch.int32, device=dev)
out = torch.zeros(M, N, dtype=torch.float16, device=dev)
mod.run(a, wr, s, sorted_ids, eids, out, float(M))
torch.cuda.synchronize()
print("out[0] (expect all 32.0):", out[0].tolist())
print("out[1] :", out[1].tolist())

# now vary A along k: a[m][k] = k  -> C[m][n] = sum_k k = 496
a2 = torch.arange(K, dtype=torch.float16, device=dev)[None].repeat(M, 1)
out2 = torch.zeros(M, N, dtype=torch.float16, device=dev)
mod.run(a2, wr, s, sorted_ids, eids, out2, float(M))
torch.cuda.synchronize()
print("out2[0] (expect all 496):", out2[0].tolist())

# vary A along m: a[m][k] = m -> C[m][n] = 32*m
a3 = torch.arange(M, dtype=torch.float16, device=dev)[:, None].repeat(1, K)
out3 = torch.zeros(M, N, dtype=torch.float16, device=dev)
mod.run(a3, wr, s, sorted_ids, eids, out3, float(M))
torch.cuda.synchronize()
print("out3 diag (expect 32*m):", [out3[m, 0].item() for m in range(0, 16, 3)])
print("out3 row0 all cols (expect 0):", out3[0].tolist())
print("out3 row1 all cols (expect 32):", out3[1].tolist())

# vary W along n: nibble pattern per row: row n has nibble (n%8 in {0..7} value)
# use distinct scale per n instead: s[0][n][0] = n+1
s2 = torch.arange(1, N + 1, dtype=torch.float32, device=dev)[None, :, None].expand(E, N, 1).contiguous()
out4 = torch.zeros(M, N, dtype=torch.float16, device=dev)
mod.run(a, wr, s2, sorted_ids, eids, out4, float(M))
torch.cuda.synchronize()
print("out4[0] (expect 32*[1..8]):", out4[0].tolist())
