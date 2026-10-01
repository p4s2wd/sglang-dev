"""Compile-test the V4 JIT store kernel on SM75 (no sgl_kernel needed)."""
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
print("capability:", torch.cuda.get_device_capability(0))

from sglang.kernels.ops.attention.dsv4.attn import _jit_fused_store_module

# bf16 input, int32 indices, page_size 128 (SWA pool page size)
inp = torch.randn(4, 512, device="cuda:0", dtype=torch.bfloat16)
cache = torch.zeros(2, 128 * 576, device="cuda:0", dtype=torch.uint8)
idx = torch.tensor([0, 1, 2, 3], device="cuda:0", dtype=torch.int32)
mod = _jit_fused_store_module(name="flashmla", input_dtype=torch.bfloat16,
                              index_dtype=torch.int32, page_size=128)
mod.run(inp, cache.view(torch.float8_e4m3fn), idx)
torch.cuda.synchronize()
print("JIT store kernel compiled + ran on SM75: OK")
