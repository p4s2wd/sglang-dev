"""Validate the transposed attention kernel with the test suite's own reference.

hs_ref.py's hand-rolled dequant produced NaN, so rather than debug it this uses
test_headshared_mla.py's build_cache/reference pair, which is already trusted
(the production kernel scores rel 4.136e-04 against it). Both kernels are then
measured against the same fp32 reference. The transposed kernel puts kv in the
mma's A operand where production puts it in B, so a few ulps of extra rounding
difference is expected; a systematic error would be a bug.
"""
import sys
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import test_headshared_mla as T
import hs_full as F
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

B, H, D = 16, 32, 512
NT, TOPK = 4096, 512
kc, truth = T.build_cache(NT)
g = torch.Generator(device="cuda").manual_seed(3)
q = (torch.randn(B, 1, H, D, device="cuda", generator=g) * 0.5).half()
idx = torch.randint(0, NT, (B, 1, TOPK), device="cuda", generator=g, dtype=torch.int32)
flat = idx.reshape(B, -1).contiguous()
tlen = torch.full((B,), TOPK, dtype=torch.int32, device="cuda")
scale = D ** -0.5
ref = T.reference(q, truth, idx, tlen, None)

p_out, _ = M._run_headshared_sparse_decode(q, kc, flat, tlen, scale)
t_out = torch.zeros(B, H, D, dtype=torch.float16, device="cuda")
t_lse = torch.zeros(B, H, dtype=torch.float32, device="cuda")
F.run(q, kc, flat, scale, t_out, t_lse, bh=16, ncol=16, warps=4, stages=2)
torch.cuda.synchronize()

print("against the fp32 reference (rel = max abs err / max |ref|):")
for name, o in (("production", p_out.squeeze(1)), ("transposed", t_out)):
    print("  %-12s rel %.3e" % (name, T.rel(o, ref)))
print("  transposed vs production: max abs %.3e"
      % (t_out.float() - p_out.squeeze(1).float()).abs().max().item())
rp, rt = T.rel(p_out.squeeze(1), ref), T.rel(t_out, ref)
print("\n%s" % ("PASS: transposed is within 3x production's own error"
                if rt < 3 * rp else "FAIL: transposed error is systematic"))
