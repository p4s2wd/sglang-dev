"""Three-way check of the transposed attention kernel against a torch reference.

max|dO| = 1.578e-02 against the production kernel is not by itself a pass or a
fail: two kernels can each be within 4e-4 of the true answer and still be 1.5e-2
apart only if the outputs are large, so the absolute number is meaningless
without the output scale. This computes a float32 reference from the same cache
and indices and reports each kernel's error against it. The transposed kernel
puts kv in the mma's A operand where production puts it in B, so the fp16
accumulation splits differently -- a few ulps of extra error is expected and
fine, a systematic error is a bug.
"""
import sys
sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import ab_headshared_scale as A
import hs_full as F
from sglang.kernels.ops.attention import flash_mla_sm120_triton as M

B, H, D, TOPK = 64, 32, 512, 512
POOL = 20000
q, kc, idx = A.build(B, H, D, TOPK, POOL)
scale = 0.088

# --- reference: dequant the cache in fp32 and do attention directly ---
page = kc.shape[1]
pb = kc.stride(0)
flat = kc.as_strided((kc.shape[0] * pb,), (1,)).view(torch.uint8)
lut = M.fp8_payload_lut(torch.device("cuda"), torch.float32)
soff = page * 576
i = idx.long()
pg = i // page
po = i % page
base = pg * page + po * 576
cols = torch.arange(448, device="cuda")
nope = lut[flat[(base[:, :, None] + cols[None, None, :]).reshape(-1)].long()].reshape(B, TOPK, 448)
gcols = torch.arange(7, device="cuda")
sbyte = flat[(pg * page + soff + po * 8)[:, :, None] + gcols].reshape(B, TOPK, 7).float()
sgn = torch.exp2(sbyte - 127.0)
col_scale = sgn.repeat_interleave(64, dim=-1)[:, :, :448]
nope = nope * col_scale
rbase = (base + 448) // 2
rope = flat.view(torch.bfloat16)[(rbase[:, :, None] + torch.arange(64, device="cuda")).reshape(-1).long()].reshape(B, TOPK, 64).float()
kv = torch.cat([nope, rope], -1)                          # [B, TOPK, 512]
qf = q.squeeze(1).float()                                  # [B, H, 512]
sc = torch.einsum("bhd,btd->bht", qf, kv) * scale
mask = torch.ones(B, TOPK, dtype=torch.bool, device="cuda")
sc = sc.masked_fill(~mask[:, None, :], float("-inf"))
p = torch.softmax(sc, dim=-1)
ref = torch.einsum("bht,btd->bhd", p, kv)

p_out, p_lse = M._run_headshared_sparse_decode(
    q, kc, idx, torch.full((B,), TOPK, dtype=torch.int32, device="cuda"), scale)
p_out = p_out.squeeze(1)
t_out = torch.zeros(B, H, D, dtype=torch.float16, device="cuda")
t_lse = torch.zeros(B, H, dtype=torch.float32, device="cuda")
F.run(q, kc, idx, scale, t_out, t_lse, bh=16, ncol=16, warps=4, stages=2)
torch.cuda.synchronize()

r = ref.abs().max().item()
print("reference output max magnitude: %.3f" % r)
for name, o in (("production", p_out), ("transposed", t_out)):
    e = (o.float() - ref).abs()
    rel = (e / (ref.abs() + 1e-3)).max().item()
    print("  %-12s vs ref: max abs %.3e   max rel %.3e" % (name, e.max().item(), rel))
d = (t_out.float() - p_out.float()).abs()
print("  transposed vs production: max abs %.3e" % d.max().item())
print("\nverdict: transposed is a rounding-level variant of production"
      if (t_out.float() - ref).abs().max().item() < 3 * (p_out.float() - ref).abs().max().item()
      else "\nverdict: transposed error is NOT rounding-level -- investigate")
