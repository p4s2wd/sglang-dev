"""Compile-test the indexer Q quant JIT kernel on SM75."""
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
from sglang.kernels.ops.attention.dsv4.elementwise import (
    _jit_main_q_indexer_rope_hadamard_quant_module,
)
mod = _jit_main_q_indexer_rope_hadamard_quant_module(torch.bfloat16)
print("indexer Q quant JIT kernel compiled on SM75: OK", mod is not None)
