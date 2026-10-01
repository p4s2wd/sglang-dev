"""Triton sparse-MLA decode must match the PyTorch fallback on SM75.

The Triton kernel was unreachable here until the e4m3 payload stopped being
loaded as an fp8 dtype: Triton on SM75 refuses fp8 loads outright
("type fp8e4nv not supported in this architecture. The supported fp8 dtypes
are ('fp8e4b15', 'fp8e5')"). The fix routes the payload as uint8 through a
256-entry fp32 LUT, the same trick as the dense W8A16 GEMV.

Both backends run on the identical paged cache built with the real DSv4-Flash
layout (page_size 256, page_bytes 149760, NOPE fp8 [0:448], ROPE bf16
[448:576], UE8M0 scales at page*576 + off*8). The fp32 reference is computed
independently, so a disagreement between the two backends is not excused by a
shared bug in either.

Why this matters: the Triton path replaces ~81 small kernels per layer
(_gather_and_dequant + _sm120_sparse_decode_fwd_chunk, 10.8 ms/token of the
73 ms decode step), so it is the largest single decode lever left -- but only
if it is correct.
"""
import os as _os
import pathlib as _pb


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
import torch  # noqa: E402

torch.cuda.set_device(0)

from sglang.kernels.ops.attention import flash_mla_sm120 as m  # noqa: E402
from sglang.kernels.ops.attention.flash_mla_sm120 import (  # noqa: E402
    _NOPE_DIM,
    _NOPE_ROPE_STRIDE,
    _ROPE_DIM,
    _SCALE_STRIDE,
)
from sglang.kernels.ops.attention.flash_mla_sm120_triton import (  # noqa: E402
    flash_mla_sparse_decode_triton,
)

PAGE = 256
PAGE_BYTES = -(-584 * PAGE // 576) * 576
NUM_PAGES = 4
B, S_Q, H_Q, D_QK = 2, 1, 8, 512
HEAD_DIM_V = 512
TOPK = 64

dev = "cuda:0"
torch.manual_seed(0)

# ---- Build the paged cache exactly as the store kernel writes it ----
raw = torch.zeros(NUM_PAGES, PAGE_BYTES, dtype=torch.uint8, device=dev)
N_TOK = NUM_PAGES * PAGE
kv_true = torch.zeros(N_TOK, D_QK, device=dev)

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

# The kernels take a float8_e4m3fn view of the raw pages; the Triton path now
# re-views it as uint8 internally, so both backends see the same bytes.
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


def reference():
    """fp32 attention with the attn_sink denominator term, independent of both."""
    kv_all = kv_true.to(torch.float32)
    ref = torch.zeros(B, S_Q, H_Q, HEAD_DIM_V, device=dev)
    ref_lse = torch.zeros(B, H_Q, S_Q, device=dev)
    for b in range(B):
        rng = torch.arange(TOPK, device=dev)
        valid = (indices[b, 0] >= 0) & (rng < topk_length[b])
        idx = indices[b, 0][valid].long()
        kv = kv_all[idx]
        for h in range(H_Q):
            s = (q_t[b, 0, h].float() @ kv.t()) * softmax_scale
            lse_h = torch.logsumexp(s, dim=0)
            ref_lse[b, h, 0] = lse_h
            lse_out = torch.logsumexp(
                torch.stack([lse_h, attn_sink[h].float()]), dim=0)
            w = torch.exp(s - lse_out)
            ref[b, 0, h] = w @ kv[:, :HEAD_DIM_V]
    return ref, ref_lse


ref, ref_lse = reference()
results = {}

# ---- PyTorch fallback (the current production path) ----
t_out, t_lse = m._sm120_sparse_decode_fwd_chunk(
    q_t, k_cache, indices, topk_length, attn_sink, HEAD_DIM_V, softmax_scale)
torch.cuda.synchronize()
results["torch"] = t_out

# ---- Triton path (uint8 payload + fp32 LUT) ----
try:
    tr_out, tr_lse = flash_mla_sparse_decode_triton(
        q_t, k_cache, indices, topk_length, attn_sink, HEAD_DIM_V, softmax_scale)
    torch.cuda.synchronize()
    results["triton"] = tr_out
except Exception as e:
    print(f"TRITON FAILED: {type(e).__name__}: {str(e)[:220]}")

ok = True
for name, out in results.items():
    err = (out.float() - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    finite = bool(torch.isfinite(out).all())
    good = finite and rel < 5e-3
    ok = ok and good
    print(f"{name:7s} vs fp32 reference: abs {err:.5f} rel {rel:.3e} "
          f"finite={finite} {'ok' if good else 'FAIL'}")

if len(results) == 2:
    d = (results["triton"].float() - results["torch"].float()).abs().max().item()
    rel = d / results["torch"].float().abs().max().item()
    print(f"triton vs torch: abs {d:.5f} rel {rel:.3e}")

# ---- extra_k_cache (c4 / c128) path ----
# The Triton wrapper runs the kernel a second time on a separate cache and merges
# by LSE, so this branch is a distinct code path and was not covered above.
# Production reaches it whenever the SWA window and the compressed caches are both
# live, which is the normal long-context case.
def build_cache(page, num_pages, n_tok, seed):
    """Same real layout as above, its own pages and its own ground truth."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    page_bytes = -(-584 * page // 576) * 576
    raw = torch.zeros(num_pages, page_bytes, dtype=torch.uint8, device=dev)
    s_exp = torch.randint(-6, 0, (n_tok, 7), device=dev, generator=g)
    sc = torch.pow(2.0, s_exp.float())
    nf = torch.randn(n_tok, _NOPE_DIM, device=dev, generator=g)
    q8x = (nf.view(n_tok, 7, 64) / sc.unsqueeze(-1)).to(torch.float8_e4m3fn)
    kvx = torch.zeros(n_tok, D_QK, device=dev)
    kvx[:, :_NOPE_DIM] = (q8x.float().view(n_tok, _NOPE_DIM)
                          * sc.repeat_interleave(64, dim=-1))
    rx = (torch.randn(n_tok, _ROPE_DIM, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    kvx[:, _NOPE_DIM:] = rx.float()
    qb = q8x.view(torch.uint8).reshape(n_tok, _NOPE_DIM)
    sb = (127 + s_exp).to(torch.uint8)
    rb = rx.view(torch.uint8).reshape(n_tok, _ROPE_DIM * 2)
    tk = torch.arange(n_tok, device=dev)
    pg, ip = tk // page, (tk % page) * _NOPE_ROPE_STRIDE
    sreg = page * _NOPE_ROPE_STRIDE
    for c in range(_NOPE_DIM):
        raw[pg, ip + c] = qb[:, c]
    for c in range(_ROPE_DIM * 2):
        raw[pg, ip + _NOPE_DIM + c] = rb[:, c]
    for c in range(7):
        raw[pg, sreg + (tk % page) * _SCALE_STRIDE + c] = sb[:, c]
    kc = raw.as_strided(
        (num_pages, page, 1, _NOPE_ROPE_STRIDE),
        (page_bytes, _NOPE_ROPE_STRIDE, _NOPE_ROPE_STRIDE, 1),
    ).view(torch.float8_e4m3fn)
    return kc, kvx


# A smaller page size here as well: the c4/c128 caches use different page sizes
# than the SWA cache, and the kernel derives addressing from page_size.
XPAGE, XPAGES, XTOK, XTOPK = 64, 3, 192, 32
x_kc, x_kv = build_cache(XPAGE, XPAGES, XTOK, seed=7)
x_idx = torch.randint(0, XTOK, (B, S_Q, XTOPK), dtype=torch.int32, device=dev)
x_idx[1, :, -5:] = -1
x_len = torch.tensor([XTOPK, XTOPK - 5], dtype=torch.int32, device=dev)

# reference over BOTH caches: one softmax across the concatenated token set
kv_all = torch.cat([kv_true, x_kv], dim=0).to(torch.float32)
ref_x = torch.zeros(B, S_Q, H_Q, HEAD_DIM_V, device=dev)
for b in range(B):
    rng = torch.arange(TOPK, device=dev)
    v1 = (indices[b, 0] >= 0) & (rng < topk_length[b])
    idx1 = indices[b, 0][v1].long()
    xr = torch.arange(XTOPK, device=dev)
    v2 = (x_idx[b, 0] >= 0) & (xr < x_len[b])
    idx2 = x_idx[b, 0][v2].long() + N_TOK
    kv = kv_all[torch.cat([idx1, idx2])]
    for h in range(H_Q):
        s = (q_t[b, 0, h].float() @ kv.t()) * softmax_scale
        lse_h = torch.logsumexp(s, dim=0)
        lse_o = torch.logsumexp(torch.stack([lse_h, attn_sink[h].float()]), dim=0)
        ref_x[b, 0, h] = torch.exp(s - lse_o) @ kv[:, :HEAD_DIM_V]

t_out2, _ = m._sm120_sparse_decode_fwd_chunk(
    q_t, k_cache, indices, topk_length, attn_sink, HEAD_DIM_V, softmax_scale,
    extra_k_cache=x_kc, extra_indices=x_idx, extra_topk_length=x_len)
tr_out2, _ = flash_mla_sparse_decode_triton(
    q_t, k_cache, indices, topk_length, attn_sink, HEAD_DIM_V, softmax_scale,
    extra_k_cache=x_kc, extra_indices=x_idx, extra_topk_length=x_len)
torch.cuda.synchronize()

print("\nwith extra_k_cache (c4/c128 merge path):")
for name, o in (("torch", t_out2), ("triton", tr_out2)):
    e = (o.float() - ref_x).abs().max().item()
    r = e / ref_x.abs().max().item()
    fin = bool(torch.isfinite(o).all())
    good = fin and r < 5e-3
    ok = ok and good
    print(f"  {name:7s} vs fp32 reference: abs {e:.5f} rel {r:.3e} "
          f"finite={fin} {'ok' if good else 'FAIL'}")

print("TRITON SPARSE MLA (SM75 LUT PATH):", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
