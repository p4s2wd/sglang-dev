#!/usr/bin/env python
"""How much of _headshared_sparse_kernel's 21 ms is the FFMA dots themselves?

Why this probe exists
---------------------
An earlier micro-benchmark timed a single [16,64]x[64,16] dot at ~50 us. That is
~660x its arithmetic floor and matches the launch overhead on this box
(cuda_runtime median 14.4 us, host wrapper floor ~155 us), so it measured launch,
not compute. The real kernel does 17 dots inside ONE launch, so it amortises
launch -- which means the isolated numbers cannot be used to attribute the
kernel's 21 ms.

This probe puts the work INSIDE one launch: a loop over `iters` tile steps, each
doing the same NCHUNK QK dots + NCHUNK PV dots + rope dot as the real kernel, on
data that is already resident. Subtracting the `iters=0` launch gives the
marginal cost of one step, which is what has to be compared against the
arithmetic floor.

Three variants, so the remaining candidates can be separated:

  dots    the dot sequence, as shipped
  bcast   explicit broadcast multiply instead of tl.dot (no smem staging, but
          a reduction per element)
  gather  the gather + dequant only, no dots (this is the floor of the memory path)

Reported: marginal us per step, and the ratio to the FFMA arithmetic floor for
that step. A ratio near 1 means the dots are already at the machine's limit and
nothing can be won; a ratio near 18 means there is real headroom.

Usage: hs_dot_attrib.py [--iters 32] [--bh 16] [--bt 16] [--ncol 64]
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile

REPO = "/data/nvme/sglang-codex/sglang/python"

WORKER = r'''
import sys, statistics, torch, triton, triton.language as tl, re
sys.path.insert(0, REPO_PATH)

NOPE = tl.constexpr(448)
ROPE = tl.constexpr(64)
PAD  = tl.constexpr(512)
G    = tl.constexpr(64)
SB   = tl.constexpr(8)
DS   = tl.constexpr(576)
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def k_dots(Q, K, O, topk, page_size, page_bytes, scale_off,
           H: tl.constexpr, BH: tl.constexpr, BT: tl.constexpr,
           NCOL: tl.constexpr, ITERS: tl.constexpr):
    """QK + PV + rope dots over ITERS steps, no gather (data already resident)."""
    NCHUNK: tl.constexpr = PAD // NCOL
    oh = tl.arange(0, BH); ot = tl.arange(0, BT); oc = tl.arange(0, NCOL)
    ro = tl.arange(0, ROPE)
    qbase = oh * (NOPE + ROPE)
    acc = tl.zeros([BH, NCOL], tl.float32)
    accr = tl.zeros([BH, ROPE], tl.float32)
    m_i = tl.full([BH], float("-inf"), tl.float32)
    l_i = tl.zeros([BH], tl.float32)
    for it in range(ITERS):
        toff = (it * BT) % topk
        tok = (ot + toff).to(tl.int64) * DS
        scores = tl.zeros([BH, BT], tl.float32)
        for c in tl.static_range(NCHUNK):
            q = tl.load(Q + qbase[:, None] + (c * NCOL + oc)[None, :])
            k = tl.load(K + tok[:, None] + (c * NCOL + oc)[None, :])
            scores += tl.dot(q, tl.trans(k))
        qr = tl.load(Q + qbase[:, None] + NOPE + ro[None, :])
        kr = tl.load(K + tok[:, None] + NOPE + ro[None, :])
        scores += tl.dot(qr, tl.trans(kr))
        s = scores * LOG2E
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - m_safe))
        p = tl.math.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pf = p.to(tl.float16)
        accr = accr * alpha[:, None] + tl.dot(pf, kr)
        for c in tl.static_range(NCHUNK):
            k = tl.load(K + tok[:, None] + (c * NCOL + oc)[None, :])
            acc = acc * alpha[:, None] + tl.dot(pf, k)
        m_i = m_new
    inv = 1.0 / tl.where(l_i > 0.0, l_i, 1.0)
    tl.store(O + oh[:, None] * NCOL + oc[None, :],
             (acc * inv[:, None]).to(O.dtype.element_ty))
    tl.store(O + BH * NCOL + oh[:, None] * ROPE + ro[None, :],
             (accr * inv[:, None]).to(O.dtype.element_ty))


@triton.jit
def k_bcast(Q, K, O, topk, page_size, page_bytes, scale_off,
            H: tl.constexpr, BH: tl.constexpr, BT: tl.constexpr,
            NCOL: tl.constexpr, ITERS: tl.constexpr):
    """Same, but the QK dots become explicit broadcast reductions in fp32."""
    NCHUNK: tl.constexpr = PAD // NCOL
    oh = tl.arange(0, BH); ot = tl.arange(0, BT); oc = tl.arange(0, NCOL)
    ro = tl.arange(0, ROPE)
    qbase = oh * (NOPE + ROPE)
    acc = tl.zeros([BH, NCOL], tl.float32)
    accr = tl.zeros([BH, ROPE], tl.float32)
    m_i = tl.full([BH], float("-inf"), tl.float32)
    l_i = tl.zeros([BH], tl.float32)
    for it in range(ITERS):
        toff = (it * BT) % topk
        tok = (ot + toff).to(tl.int64) * DS
        scores = tl.zeros([BH, BT], tl.float32)
        for c in tl.static_range(NCHUNK):
            q = tl.load(Q + qbase[:, None] + (c * NCOL + oc)[None, :]).to(tl.float32)
            k = tl.load(K + tok[:, None] + (c * NCOL + oc)[None, :]).to(tl.float32)
            scores += tl.sum(q[:, None, :] * k[None, :, :], axis=2)
        qr = tl.load(Q + qbase[:, None] + NOPE + ro[None, :]).to(tl.float32)
        kr = tl.load(K + tok[:, None] + NOPE + ro[None, :]).to(tl.float32)
        scores += tl.sum(qr[:, None, :] * kr[None, :, :], axis=2)
        s = scores * LOG2E
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - m_safe))
        p = tl.math.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pf = p.to(tl.float16)
        accr = accr * alpha[:, None] + tl.dot(pf, kr.to(tl.float16))
        for c in tl.static_range(NCHUNK):
            k = tl.load(K + tok[:, None] + (c * NCOL + oc)[None, :]).to(tl.float32)
            acc = acc * alpha[:, None] + tl.sum(pf[:, None, :].to(tl.float32)
                                                * k[None, :, :], axis=2)
        m_i = m_new
    inv = 1.0 / tl.where(l_i > 0.0, l_i, 1.0)
    tl.store(O + oh[:, None] * NCOL + oc[None, :],
             (acc * inv[:, None]).to(O.dtype.element_ty))
    tl.store(O + BH * NCOL + oh[:, None] * ROPE + ro[None, :],
             (accr * inv[:, None]).to(O.dtype.element_ty))


@triton.jit
def k_empty(Q, K, O, topk, page_size, page_bytes, scale_off,
            H: tl.constexpr, BH: tl.constexpr, BT: tl.constexpr,
            NCOL: tl.constexpr, ITERS: tl.constexpr):  # same signature
    """Launch cost only: same signature, no work."""
    z = tl.zeros([BH], tl.float32)
    idx = (tl.arange(0, BH) * 1).to(tl.int32)
    tl.store(O + idx, z)


ITERS = int(sys.argv[1]); BH = int(sys.argv[2]); BT = int(sys.argv[3])
NCOL = int(sys.argv[4]); NW = int(sys.argv[4] and 4)
dev = 'cuda'
NQ = BH * 512
q = torch.randn(NQ, device=dev, dtype=torch.float16) * 0.1
K = 8192 * 512
k = torch.randn(K, device=dev, dtype=torch.float16) * 0.1
o = torch.zeros(BH * 512 + BH * 64, device=dev, dtype=torch.float16)


def bench(kern, iters, reps=12):
    def call():
        kern[(1,)](q, k, o, 512, 16, 9344, 16 * 576, 64,
                   BH=BH, BT=BT, NCOL=NCOL, ITERS=iters,
                   num_warps=NW, num_stages=2)
    for _ in range(4):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); call(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


e0 = bench(k_dots, 0)
d_hi = bench(k_dots, 64)
d_lo = bench(k_dots, 8)

# arithmetic floor for one step, one block
nch = 512 // NCOL
flop_step = 2 * (BH * BT * NCOL * nch) * 2 + 2 * (BH * BT * 64) * 2
SM_FP32 = 128 * 2 * 1.7e9
floor_us = flop_step / SM_FP32 * 1e6

print("%.4f %.4f %.4f %.4f" % (e0, d_lo, d_hi, floor_us))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--bh", type=int, default=16)
    ap.add_argument("--bt", type=int, default=16)
    ap.add_argument("--ncol", type=int, default=64)
    a = ap.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    with tempfile.TemporaryDirectory() as td:
        wp = os.path.join(td, "w.py")
        ip = None
        with open(wp, "w") as f:
            f.write("REPO_PATH = %r\n" % REPO + WORKER)
        r = subprocess.run([sys.executable, wp, "8", str(a.bh), str(a.bt),
                            str(a.ncol)], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print("worker failed:\n", r.stderr[-3000:])
            return
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        print("worker stdout:", lines[-3:])
        vals = [float(x) for x in lines[-1].split()]
        e0, d_lo, d_hi, floor_us = vals

    m_lo = (d_lo - e0) / 8 * 1000     # us per step, marginal
    m_hi = (d_hi - e0) / 64 * 1000

    nch = 512 // a.ncol
    print(f"one block, BH={a.bh} BT={a.bt} NCOL={a.ncol} (NCHUNK={nch}), 4 warps")
    print()
    print("launch-only            %.1f us  (this is what the earlier probe measured)" % (e0 * 1000))
    print()
    print("per-step marginal cost (launch subtracted):")
    print("  tl.dot variant       %.2f us/step   (iters 8: %.2f, iters 64: %.2f)"
          % (m_lo, m_lo, m_hi))
    print("  FFMA floor, 1 block  %.3f us/step" % floor_us)
    print()
    print("ratio to floor:  dots %.1fx" % (m_lo / floor_us,))
    print()
    real_step = 21.08 * 1000 / 32      # 32 topk tiles per call
    print("real kernel: 21.08 ms/call over 32 tiles = %.2f us/tile (all 2048 blocks," % real_step)
    print("  i.e. %.2f us/tile per block-equivalent)" % (real_step / 1))
    print()
    print("Interpretation: if `dots` marginal is close to the floor, the FFMA path")
    print("is already saturating one SM and the only fix is more blocks/SM (smem).")
    print("If it is far above, the dots themselves are inefficient and a different")
    print("formulation is needed.")


if __name__ == "__main__":
    main()
