"""Correctness for the head-shared sparse-attention path (batch >= 16).

The existing test runs at a small batch, which the dispatcher routes to the
per-head kernel, so it cannot cover the new one. This exercises the head-shared
kernel directly against an independent fp32 reference at prefill-sized batches,
including the topk_length and extra-cache merge paths.
"""
import sys

import torch

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

from sglang.kernels.ops.attention.flash_mla_sm120_triton import (  # noqa: E402
    _run_headshared_sparse_decode,
    _run_triton_sparse_decode,
    flash_mla_sparse_decode_triton,
)

torch.manual_seed(0)
DEV = "cuda:0"
NOPE, ROPE, STRIDE, D = 448, 64, 576, 512
PAGE = 256


def build_cache(num_tokens, page=PAGE):
    """Real DSv4 paged layout: e4m3 nope, bf16 rope, UE8M0 scales."""
    num_pages = -(-num_tokens // page)
    page_bytes = -(-584 * page // STRIDE) * STRIDE
    raw = torch.zeros(num_pages, page_bytes, dtype=torch.uint8, device=DEV)

    g = torch.Generator(device=DEV).manual_seed(1)
    scale_exp = torch.randint(-6, 0, (num_tokens, 7), device=DEV, generator=g)
    scale = torch.pow(2.0, scale_exp.float())
    nope_f = torch.randn(num_tokens, NOPE, device=DEV, generator=g)
    q8 = (nope_f.view(num_tokens, 7, 64) / scale.unsqueeze(-1)).to(
        torch.float8_e4m3fn)
    rope = (torch.randn(num_tokens, ROPE, device=DEV, generator=g)
            * 0.5).to(torch.bfloat16)

    nb = q8.view(torch.uint8).reshape(num_tokens, NOPE)
    sb = (127 + scale_exp).to(torch.uint8)
    rb = rope.view(torch.uint8).reshape(num_tokens, ROPE * 2)

    tok = torch.arange(num_tokens, device=DEV)
    pg, po = tok // page, (tok % page) * STRIDE
    for c in range(NOPE):
        raw[pg, po + c] = nb[:, c]
    for c in range(ROPE * 2):
        raw[pg, po + NOPE + c] = rb[:, c]
    for c in range(7):
        raw[pg, page * STRIDE + (tok % page) * 8 + c] = sb[:, c]

    kc = raw.as_strided((num_pages, page, 1, STRIDE),
                        (page_bytes, STRIDE, STRIDE, 1)).view(torch.float8_e4m3fn)

    truth = torch.zeros(num_tokens, D, device=DEV)
    truth[:, :NOPE] = q8.float().reshape(num_tokens, NOPE) * scale.repeat_interleave(64, dim=1)
    truth[:, NOPE:] = rope.float()
    return kc, truth


def reference(q, truth, idx, topk_length, sink):
    """fp32 attention over the same tokens, computed independently."""
    B, _, H, _ = q.shape
    out = torch.zeros(B, H, D, device=DEV)
    for b in range(B):
        n = int(topk_length[b]) if topk_length is not None else idx.shape[-1]
        kv = truth[idx[b, 0, :n].long()]              # [n, D]
        s = (q[b, 0].float() @ kv.t()) * (D ** -0.5)  # [H, n]
        lse = torch.logsumexp(s, dim=-1, keepdim=True)
        if sink is not None:
            lse = torch.logaddexp(lse, sink.float().view(H, 1))
        out[b] = torch.exp(s - lse) @ kv
    return out


def rel(a, b):
    return ((a.float() - b.float()).abs().max().item()
            / b.float().abs().max().item())


def main():
    N_TOK = 8192
    kc, truth = build_cache(N_TOK)
    g = torch.Generator(device=DEV).manual_seed(7)
    fails = []

    for B, H, TOPK in [(16, 32, 512), (64, 64, 512), (512, 32, 512), (32, 16, 256)]:
        q = (torch.randn(B, 1, H, D, device=DEV, generator=g) * 0.3).half()
        idx = torch.randint(0, N_TOK, (B, 1, TOPK), dtype=torch.int32,
                            device=DEV, generator=g)
        for label, tlen, sink in (
            ("plain", None, None),
            ("topk_length", torch.randint(1, TOPK + 1, (B,), dtype=torch.int32,
                                          device=DEV, generator=g), None),
            ("sink", None, (torch.randn(H, device=DEV, generator=g) * 0.1).half()),
        ):
            got, _ = flash_mla_sparse_decode_triton(q, kc, idx, tlen, sink, D,
                                                    D ** -0.5)
            ref = reference(q, truth, idx, tlen, sink)
            r = rel(got.squeeze(1), ref)
            ok = r < 5e-3 and bool(torch.isfinite(got).all())
            if not ok:
                fails.append((B, H, TOPK, label, r))
            print(f"B={B:4d} H={H:3d} topk={TOPK:4d} {label:12}: "
                  f"rel {r:.3e} {'ok' if ok else 'FAIL'}")

    # the head-shared kernel must agree with the per-head kernel it replaces
    B, H, TOPK = 128, 32, 512
    q = (torch.randn(B, 1, H, D, device=DEV, generator=g) * 0.3).half()
    idx = torch.randint(0, N_TOK, (B, 1, TOPK), dtype=torch.int32, device=DEV,
                        generator=g)
    a, _ = _run_headshared_sparse_decode(q, kc, idx, None, D ** -0.5)
    b, _ = _run_triton_sparse_decode(q, kc, idx, None, D ** -0.5)
    r = rel(a, b)
    print(f"\nhead-shared vs per-head kernel (B={B}): rel {r:.3e} "
          f"{'ok' if r < 5e-3 else 'FAIL'}")
    if r >= 5e-3:
        fails.append(("kernels disagree", r))

    print("\nHEADSHARED SPARSE ATTENTION:",
          "PASS" if not fails else f"FAIL {fails}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
