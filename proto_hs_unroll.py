#!/usr/bin/env python
"""Prototype: collapse the 8 chunk accumulators into one looped accumulator.

Why
---
The shipped `_headshared_sparse_kernel` holds EIGHT separate [BLOCK_H, NCOL] fp32
accumulators (acc0..acc7) plus acc_r, because Triton allows neither a list
comprehension nor the tuple builtin inside @jit. Measured `metadata.shared` for
the shipped config: **40 960 B**, against SM75's 64 KB -- so exactly ONE block
fits per SM, i.e. 4 warps, i.e. **6% occupancy**. Registers are irrelevant: even
at 32 regs/thread the smem cap still admits one block.

At 6% occupancy the FFMA pipeline cannot fill. `tl.dot` lowers to scalar FFMA on
sm_75 (PTX-verified: 0 mma.sync, 2048 fma.rn), so each dot issues a long dependent
FMA chain that needs many warps to hide latency.

Hypothesis under test
---------------------
Replacing the 8 unrolled accumulators with ONE accumulator updated inside
`tl.static_range` cuts the accumulator state from 32 KB to 4 KB, which should let
4-8 blocks fit per SM and lift occupancy from 6% to 25-50%.

The loop is over NCHUNK = 512 // NCOL iterations of a *static* range, so it
unrolls at compile time and each iteration still gets its own SSA value -- but
Triton can reuse one destination buffer instead of needing them all live.

This prototype deliberately does NOT change the math. It is a measurement of
whether the smem/occupancy hypothesis is the real lever, before touching the
shipped kernel.

Run:  proto_hs_unroll.py [--B 512] [--ncol 64] [--warps 4]
"""
import argparse
import os
import statistics
import subprocess
import sys
import tempfile

REPO = "/data/nvme/sglang-codex/sglang/python"

WORKER = r'''
import sys, statistics, torch, triton, triton.language as tl
sys.path.insert(0, REPO_PATH)
d = torch.load(sys.argv[1], weights_only=False)
B = d["q"].shape[0]; H = d["q"].shape[2]
TOPK = d["indices"].shape[1]
import sglang.kernels.ops.attention.flash_mla_sm120_triton as M
from sglang.kernels.ops.attention.flash_mla_sm120_triton import (
    _hs_load_nope, _hs_load_nope_t, fp8_payload_lut)

NOPE = tl.constexpr(448)
ROPE = tl.constexpr(64)
PAD  = tl.constexpr(512)
G    = tl.constexpr(64)
SB   = tl.constexpr(8)
DS   = tl.constexpr(576)
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _hs_looped(
    Q_ptr, cache_u8_ptr, cache_bf16_ptr, lut_ptr, indices_ptr,
    topk_len_ptr, O_ptr, LSE_ptr,
    softmax_scale, page_size, page_bytes, scale_section_off,
    H: tl.constexpr, topk, HAS_TOPK_LEN: tl.constexpr,
    stride_qb, stride_qh, stride_os, stride_ls, stride_ob, stride_oh,
    BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr, NCOL: tl.constexpr,
):
    """Same math as the shipped kernel, one accumulator instead of NCHUNK."""
    bid = tl.program_id(0)
    hblk = tl.program_id(1)
    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    h_valid = offs_h < H
    offs_t = tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, NCOL)
    rope_offs = tl.arange(0, ROPE)
    NCHUNK: tl.constexpr = PAD // NCOL

    q_base = bid * stride_qb + offs_h * stride_qh
    acc = tl.zeros([BLOCK_H, NCOL], tl.float32)
    acc_r = tl.zeros([BLOCK_H, ROPE], tl.float32)
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)

    q0 = tl.load(Q_ptr + q_base[:, None] + offs_c[None, :],
                 mask=h_valid[:, None] & (offs_c < NOPE)[None, :], other=0.0)
    q_r = tl.load(Q_ptr + q_base[:, None] + NOPE + rope_offs[None, :],
                  mask=h_valid[:, None], other=0.0)

    valid_len = topk
    if HAS_TOPK_LEN:
        valid_len = tl.load(topk_len_ptr + bid).to(tl.int32)

    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + offs_t
        raw = tl.load(indices_ptr + bid * topk + t_idx,
                      mask=t_idx < topk, other=-1)
        idx_valid = (t_idx < valid_len) & (raw >= 0)
        safe = tl.where(idx_valid, raw, 0).to(tl.int64)
        page_ids = safe // page_size
        page_offs = safe % page_size
        tok_base = page_ids * page_bytes + page_offs * DS

        scores = tl.zeros([BLOCK_H, BLOCK_T], tl.float32)
        # one destination, reused across chunks
        for c in tl.static_range(NCHUNK):
            kv = _hs_load_nope_t(cache_u8_ptr, lut_ptr, tok_base,
                                 c * NCOL + offs_c, page_ids, page_offs,
                                 idx_valid, page_bytes, scale_section_off)
            qc = tl.load(Q_ptr + q_base[:, None] + (c * NCOL + offs_c)[None, :],
                         mask=h_valid[:, None] & ((c * NCOL + offs_c) < NOPE)[None, :],
                         other=0.0)
            scores += tl.dot(qc, kv)

        rope_base = ((tok_base + NOPE) // 2).to(tl.int64)
        kv_r_t = tl.load(cache_bf16_ptr + rope_base[None, :] + rope_offs[:, None],
                         mask=idx_valid[None, :], other=0.0).to(tl.float16)
        scores += tl.dot(q_r, kv_r_t)

        s = tl.where(idx_valid[None, :], scores * (softmax_scale * LOG2E),
                     float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - m_safe))
        p = tl.where(idx_valid[None, :], tl.math.exp2(s - m_safe[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pf = p.to(tl.float16)
        kv_r_n = tl.load(cache_bf16_ptr + rope_base[:, None] + rope_offs[None, :],
                         mask=idx_valid[:, None], other=0.0).to(tl.float16)
        acc_r = acc_r * alpha[:, None] + tl.dot(pf, kv_r_n)
        for c in tl.static_range(NCHUNK):
            kv = _hs_load_nope(cache_u8_ptr, lut_ptr, tok_base,
                               c * NCOL + offs_c, page_ids, page_offs,
                               idx_valid, page_bytes, scale_section_off)
            acc = acc * alpha[:, None] + tl.dot(pf, kv)
        m_i = m_new

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    o_base = bid * stride_ob + offs_h * stride_oh
    for c in tl.static_range(NCHUNK):
        ci = c * NCOL + offs_c
        tl.store(O_ptr + o_base[:, None] + ci[None, :],
                 (acc / safe_l[:, None]).to(O_ptr.dtype.element_ty),
                 mask=h_valid[:, None] & (ci < NOPE)[None, :])
    tl.store(O_ptr + o_base[:, None] + NOPE + rope_offs[None, :],
             (acc_r / safe_l[:, None]).to(O_ptr.dtype.element_ty),
             mask=h_valid[:, None])
    tl.store(LSE_ptr + bid * stride_ls + offs_h,
             tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf")),
             mask=h_valid)


raw_u8 = d["k_cache"].as_strided((d["n_pages"] * d["page_bytes"],), (1,)).view(torch.uint8)
raw_bf16 = raw_u8.view(torch.bfloat16)
lut = fp8_payload_lut(d["q"].device, torch.float32)
flat = d["indices"].reshape(B, -1).contiguous()
q3 = d["q"].squeeze(1).contiguous()
out = torch.zeros(B, H, 512, dtype=torch.float16, device="cuda")
lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device="cuda")
empty = torch.empty(0, dtype=torch.int32, device="cuda")
NCOL = int(sys.argv[2]); NW = int(sys.argv[3])


def looped():
    _hs_looped[(B, triton.cdiv(H, 16))](
        q3, raw_u8, raw_bf16, lut, flat, empty, out, lse,
        d["softmax_scale"], 16, d["page_bytes"], 16 * 576,
        H, TOPK, False,
        q3.stride(0), q3.stride(1), 0, 0, out.stride(0), out.stride(1),
        BLOCK_H=16, BLOCK_T=16, NCOL=NCOL, num_warps=NW, num_stages=2)


def shipped():
    M._run_headshared_sparse_decode(d["q"], d["k_cache"], d["indices"], None,
                                   d["softmax_scale"])


def timeit(f, n=8):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)


# report the compiled resource usage, which is the whole point of the prototype
import re
def usage():
    for dev, cache in _hs_looped.device_caches.items():
        for key, k in cache[0].items():
            md = k.metadata
            return getattr(md, "shared", None), getattr(md, "num_regs", None)
    return None, None

tl_ms = timeit(looped)
sh, nreg = usage()
print("looped\t%.4f\tshared=%s\tnreg=%s" % (tl_ms, sh, nreg))
sh_ms = timeit(shipped)
for dev, cache in M._headshared_sparse_kernel.device_caches.items():
    for key, k in cache[0].items():
        print("shipped\t%.4f\tshared=%s\tnreg=%s" % (sh_ms, getattr(k.metadata,"shared",None),
                                                    getattr(k.metadata,"num_regs",None)))
        break
    break
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=512)
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--cache-tokens", type=int, default=16384)
    ap.add_argument("--ncols", default="64,128")
    ap.add_argument("--warps", default="4")
    a = ap.parse_args()

    import torch
    import pf_hs_sweep as S

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    for ncol in a.ncols.split(","):
        for nw in a.warps.split(","):
            d = S.build(a.B, 64, a.topk, a.cache_tokens)
            with tempfile.TemporaryDirectory() as td:
                ip = os.path.join(td, "i.pt")
                wp = os.path.join(td, "w.py")
                torch.save(d, ip)
                with open(wp, "w") as f:
                    f.write("REPO_PATH = %r\n" % REPO + WORKER)
                r = subprocess.run([sys.executable, wp, ip, ncol, nw],
                                   env=env, capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"NCOL={ncol} warps={nw} FAILED")
                    print(r.stderr[-1200:])
                    continue
                for line in r.stdout.splitlines():
                    print(f"NCOL={ncol:>4} warps={nw}  {line}")


if __name__ == "__main__":
    main()
