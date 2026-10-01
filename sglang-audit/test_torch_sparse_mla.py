"""Verify the pure-PyTorch sparse-MLA decode fallback on SM75 with the real
DSv4-Flash FP8 paged KV layout.

This is the path sub-90 now takes after FlashInfer's sm_120-only CUTLASS
module was ruled out; it had never executed, because the first request used to
die inside the FlashInfer branch before reaching it.

Layout (page_size=256, from csrc/deepseek_v4/store.cuh):
  page_bytes = ceil(584*256/576)*576 = 149760
  NOPE   fp8 e4m3 : page[off*576 + 0   : 448]
  ROPE   bf16     : page[off*576 + 448 : 576]
  SCALES ue8m0    : page[256*576 + off*8 + 0:7]   (one per 64 nope dims)
A UE8M0 byte b encodes the value 2**(b - 127).
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

from sglang.kernels.ops.attention.flash_mla_sm120 import (
    _NOPE_DIM,
    _NOPE_ROPE_STRIDE,
    _ROPE_DIM,
    _SCALE_STRIDE,
    _default_backend,
    flash_mla_with_kvcache_sm120,
)

PAGE = 256
PAGE_BYTES = -(-584 * PAGE // 576) * 576
NUM_PAGES = 4
B, S_Q, H_Q, D_QK = 2, 1, 8, 512
HEAD_DIM_V = 512
TOPK = 64

assert PAGE_BYTES == 149760, f"page_bytes {PAGE_BYTES}"
assert _NOPE_ROPE_STRIDE == 576 and _SCALE_STRIDE == 8
print(f"page_bytes={PAGE_BYTES} stride={_NOPE_ROPE_STRIDE} scale_stride={_SCALE_STRIDE}")
backend = _default_backend()
print(f"resolved backend = {backend}")
assert backend == "torch", f"expected the torch fallback, got {backend}"

torch.manual_seed(0)
dev = "cuda:0"

raw = torch.zeros(NUM_PAGES, PAGE_BYTES, dtype=torch.uint8, device=dev)
N_TOK = NUM_PAGES * PAGE
kv_true = torch.zeros(N_TOK, D_QK, device=dev)

# ue8m0 exponent per 64 nope dims; byte = 127 + exp, value = 2**exp
scale_exp = torch.randint(-6, 0, (N_TOK, 7), device=dev)
scale = torch.pow(2.0, scale_exp.float())

nope_f32 = torch.randn(N_TOK, _NOPE_DIM, device=dev)
nope_blocks = nope_f32.view(N_TOK, 7, 64)
q8 = (nope_blocks / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
nope_deq = q8.float().view(N_TOK, _NOPE_DIM) * scale.repeat_interleave(64, dim=-1)
kv_true[:, :_NOPE_DIM] = nope_deq

rope = (torch.randn(N_TOK, _ROPE_DIM, device=dev) * 0.5).to(torch.bfloat16)
kv_true[:, _NOPE_DIM:] = rope.float()

q_bytes = q8.view(torch.uint8).reshape(N_TOK, _NOPE_DIM)
s_bytes = (127 + scale_exp).to(torch.uint8)
r_bytes = rope.view(torch.uint8).reshape(N_TOK, _ROPE_DIM * 2)
# Page addressing is NOT a flat byte stream: the page stride (149760) is not a
# multiple of the per-token stride (576), so address (page, offset-in-page).
tok = torch.arange(N_TOK, device=dev)
pg = tok // PAGE
in_page = (tok % PAGE) * _NOPE_ROPE_STRIDE
scale_region = PAGE * _NOPE_ROPE_STRIDE
for c in range(_NOPE_DIM):
    raw[pg, in_page + c] = q_bytes[:, c]
for c in range(_ROPE_DIM * 2):
    raw[pg, in_page + _NOPE_DIM + c] = r_bytes[:, c]
for c in range(7):
    raw[pg, scale_region + (tok % PAGE) * _SCALE_STRIDE + c] = s_bytes[:, c]

k_cache = raw.as_strided(
    (NUM_PAGES, PAGE, 1, _NOPE_ROPE_STRIDE),
    (PAGE_BYTES, _NOPE_ROPE_STRIDE, _NOPE_ROPE_STRIDE, 1),
).view(torch.float8_e4m3fn)

q_t = (torch.randn(B, S_Q, H_Q, D_QK, device=dev) * 0.3).to(torch.float16)
indices = torch.randint(0, N_TOK, (B, S_Q, TOPK), dtype=torch.int32, device=dev)
indices[0, :, -8:] = -1
topk_length = torch.tensor([TOPK, TOPK - 8], dtype=torch.int32, device=dev)
attn_sink = (torch.randn(H_Q, device=dev) * 0.1).to(torch.float16)
softmax_scale = D_QK ** -0.5

out, lse = flash_mla_with_kvcache_sm120(
    q=q_t, k_cache=k_cache, indices=indices, topk_length=topk_length,
    attn_sink=attn_sink, head_dim_v=HEAD_DIM_V, softmax_scale=softmax_scale,
)
torch.cuda.synchronize()
print(f"out {tuple(out.shape)} {out.dtype}; lse "
      f"{'None' if lse is None else tuple(lse.shape)}")

# Reference: identical math in fp32, including the attn_sink denominator term.
kv_all = kv_true.to(torch.float32)
ref = torch.zeros(B, S_Q, H_Q, HEAD_DIM_V, device=dev)
for b in range(B):
    rng = torch.arange(TOPK, device=dev)
    valid = (indices[b, 0] >= 0) & (rng < topk_length[b])
    idx = indices[b, 0][valid].long()
    kv = kv_all[idx]
    for h in range(H_Q):
        s = (q_t[b, 0, h].float() @ kv.t()) * softmax_scale
        lse_h = torch.logsumexp(s, dim=0)
        lse_out = torch.logsumexp(
            torch.stack([lse_h, attn_sink[h].float()]), dim=0)
        w = torch.exp(s - lse_out)
        ref[b, 0, h] = w @ kv[:, :HEAD_DIM_V]

err = (out.float() - ref).abs().max().item()
rel = err / ref.abs().max().item()
print(f"vs reference (with sink): abs {err:.5f} rel {rel:.3e}")

finite = bool(torch.isfinite(out).all())
shape_ok = out.shape == (B, S_Q, H_Q, HEAD_DIM_V) and out.dtype == q_t.dtype
ok = finite and shape_ok and rel < 5e-3
print(f"finite={finite} shape_ok={shape_ok} rel_ok={rel < 5e-3}")

# Chunking equivalence: the dispatcher splits the batch to bound the gather
# peak, and attention is independent per query position, so a forced-small
# chunk must reproduce the unchunked result exactly. Compare against the chunk
# function directly so the check is about the split, not the math.
from sglang.kernels.ops.attention import flash_mla_sm120 as m

ref_out, ref_lse = m._sm120_sparse_decode_fwd_chunk(
    q_t, k_cache, indices, topk_length, attn_sink, HEAD_DIM_V, softmax_scale)

saved = m._SPARSE_MLA_TORCH_CHUNK_BYTES
worst = 0.0
try:
    for budget in (256 * 1024 * 1024, 1, 3):
        m._SPARSE_MLA_TORCH_CHUNK_BYTES = budget
        c_out, c_lse = m._sm120_sparse_decode_fwd(
            q_t, k_cache, indices, topk_length, attn_sink, HEAD_DIM_V,
            softmax_scale)
        torch.cuda.synchronize()
        d = (c_out.float() - ref_out.float()).abs().max().item()
        dl = (c_lse.float() - ref_lse.float()).abs().max().item()
        worst = max(worst, d, dl)
        print(f"  budget={budget:>12}  max|chunked-unchunked| out {d:.2e} lse {dl:.2e}")
finally:
    m._SPARSE_MLA_TORCH_CHUNK_BYTES = saved
chunk_ok = worst <= 2 * float(torch.finfo(torch.float16).eps) * max(
    ref_out.abs().max().item(), 1.0)
print(f"chunking within fp16 rounding={chunk_ok} (worst {worst:.2e}, "
      f"tol {2 * float(torch.finfo(torch.float16).eps) * max(ref_out.abs().max().item(), 1.0):.2e})")
ok = ok and chunk_ok
ok = ok and chunk_ok

print("TORCH SPARSE MLA FALLBACK:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
