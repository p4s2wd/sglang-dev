"""Find which linear hits the fp32-copy fallback in linear_bf16_fp32.

_linear_bf16_fp32_cublas (kernels/ops/attention/dsv4/gemm.py:115) has two branches:
  if x.dtype == y.dtype and y.dtype in (bf16, fp16): torch.mm(x, y.t(), out_dtype=fp32)
  else:                                              torch.mm(x.float(), y.float().t())
The fallback makes a fresh fp32 copy of the weight on EVERY call, and the module's own
comment records that this exact mistake cost 2x at router shapes until the dtype test
was widened to accept float16. The attribution shows an aten::copy_ at that line
costing 4.0 ms/token across the 4 stages -- 7.4% of decode compute -- which is the
signature of the fallback branch being taken.

Instrument the function itself, run one decode step, and print the dtypes and shapes
that reach each branch. That names the call site instead of guessing.
"""
import sys, collections
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import sglang.kernels.ops.attention.dsv4.gemm as G

hits = collections.Counter()
slow = collections.Counter()
orig = G._linear_bf16_fp32_cublas


def traced(x, y):
    fast = x.is_cuda and x.dtype == y.dtype and y.dtype in (torch.bfloat16, torch.float16)
    key = ("FAST" if fast else "SLOW") + " x=%s y=%s %s x %s y %s" % (
        x.dtype, y.dtype, tuple(x.shape), tuple(y.shape))
    hits[key] += 1
    if not fast:
        import traceback
        st = [l.strip() for l in traceback.format_stack()
              if "/sglang/" in l and "gemm.py" not in l]
        slow[" <- ".join(st[-3:])] += 1
    return orig(x, y)


G._linear_bf16_fp32_cublas = traced
# the public entry resolves the name at call time in this module
G.linear_bf16_fp32.__globals__["_linear_bf16_fp32_cublas"] = traced

import urllib.request, json, time
def post(p, pl=None):
    r = urllib.request.Request("http://127.0.0.1:30000" + p,
                               data=json.dumps(pl or {}).encode(),
                               headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=600).read().decode()

# The server is a separate process, so instrumenting here proves nothing about it.
# Instead: reproduce the model's own dtypes locally from the loaded config.
print("instrumentation only works in-process; the server is a separate process.")
print("Reproducing the branch decision from the checkpoint dtypes instead.\n")
