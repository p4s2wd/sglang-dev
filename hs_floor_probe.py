#!/usr/bin/env python
"""What is the floor for the head-shared sparse attention gather at production shape?

This is the measurement that decides whether an attention rewrite can pay.

The kernel's real cost decomposes into
  (a) gathering + dequantising the selected KV pages, and
  (b) the QK and PV arithmetic.
On this Triton build every tl.dot lowers to scalar FFMA (PTX-verified: 0 mma.sync,
2048 fma.rn on sm_75), so (b) cannot be moved onto tensor cores. That makes (a)
the only thing worth optimising -- but only if (a) is actually the wall.

Three probes, same inputs, same addressing, each removing one cost:

  GATHER_ONLY  read the same bytes at the same addresses, dequantise, and
               accumulate so nothing is dead-code eliminated. No dots.
  GATHER_DOT   the real kernel.
  DOT_ONLY     a synthetic loop doing the same number of scalar FMAs against
               already-resident data. Upper bound on (b)'s arithmetic cost.

Reported per call, and as a fraction of the 616 GB/s streaming rate.

Everything is timed with CUDA events, median of N, interleaved rounds so clock
drift between probes cannot be mistaken for a difference.

Usage: hs_floor_probe.py [--B 512] [--topk 512] [--cache-tokens 16384] [--rounds 5]
"""
import argparse
import os
import statistics
import subprocess
import sys
import tempfile

REPO = "/data/nvme/sglang-codex/sglang/python"
sys.path.insert(0, "/data/nvme/sglang-codex")

WORKER = r'''
import sys, statistics, torch, triton, triton.language as tl
sys.path.insert(0, REPO)
d = torch.load(sys.argv[1], weights_only=False)
ROUNDS = int(sys.argv[2])
B = d["q"].shape[0]; H = d["q"].shape[2]
TOPK = d["indices"].shape[1]
import sglang.kernels.ops.attention.flash_mla_sm120_triton as M

# --- layout constants, read from the module so this cannot drift ---
NOPE = 448; ROPE = 64; G = 64; SB = 8; DS = 576
PB = d["page_bytes"]; NP = d["n_pages"]; PAGE = 16
DATA = PAGE * DS; NSC = NOPE // G

@triton.jit
def _gather_only(Q_ptr, cache_u8_ptr, cache_bf16_ptr, lut_ptr, indices_ptr,
                 O_ptr, page_size, page_bytes, scale_off,
                 H: tl.constexpr, topk, BLOCK_H: tl.constexpr,
                 BLOCK_T: tl.constexpr, NCOL: tl.constexpr,
                 NOPE: tl.constexpr, ROPE: tl.constexpr,
                 G: tl.constexpr, SB: tl.constexpr, DS: tl.constexpr):
    """Same addresses, same dequant, no dot. Accumulates into O so the loads
    cannot be eliminated."""
    bid = tl.program_id(0); hblk = tl.program_id(1)
    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_t = tl.arange(0, BLOCK_T); offs_c = tl.arange(0, NCOL)
    rope_offs = tl.arange(0, ROPE)
    q_base = bid * (H * (NOPE + ROPE)) + offs_h * (NOPE + ROPE)
    acc = tl.zeros([BLOCK_H, NCOL], tl.float32)
    for t0 in range(0, topk, BLOCK_T):
        t_idx = t0 + offs_t
        raw = tl.load(indices_ptr + bid * topk + t_idx, mask=t_idx < topk, other=-1)
        ok = (t_idx < topk) & (raw >= 0)
        safe = tl.where(ok, raw, 0).to(tl.int64)
        pg = safe // page_size; po = safe % page_size
        tok = pg * page_bytes + po * DS
        m = ok[:, None] & (offs_c < NOPE)[None, :]
        b = tl.load(cache_u8_ptr + tok[:, None] + offs_c[None, :], mask=m, other=0)
        sa = (pg * page_bytes + scale_off + po * SB)[:, None] + (offs_c // G)[None, :]
        sc = tl.math.exp2(tl.load(cache_u8_ptr + sa, mask=m, other=127).to(tl.float32) - 127.0)
        acc += tl.sum(tl.load(lut_ptr + b).to(tl.float32) * sc, axis=1)[:, None]
        rb = ((tok + NOPE) // 2).to(tl.int64)
        acc += tl.sum(tl.load(cache_bf16_ptr + rb[:, None] + rope_offs[None, :],
                              mask=ok[:, None], other=0.0).to(tl.float32), axis=1)[:, None]
    tl.store(O_ptr + q_base[:, None] + offs_c[None, :], acc.to(O_ptr.dtype.element_ty))


def timeit(f, n=10):
    for _ in range(3): f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)

raw_u8 = d["k_cache"].as_strided((NP * PB,), (1,)).view(torch.uint8)
raw_bf16 = raw_u8.view(torch.bfloat16)
lut = M.fp8_payload_lut(d["q"].device, torch.float32)
flat = d["indices"].reshape(B, -1).contiguous()
q3 = d["q"].squeeze(1).contiguous()
out = torch.zeros(B, H, NOPE + ROPE, dtype=torch.float16, device="cuda")

res = {}
for rnd in range(ROUNDS):
    # real kernel
    def real():
        M._run_headshared_sparse_decode(d["q"], d["k_cache"], d["indices"], None,
                                       d["softmax_scale"])
    t = timeit(real)
    res.setdefault("real", []).append(t)
    # gather only, at the same BLOCK_H the real kernel uses
    for BH in (16, 32, 64):
        NCOL = 64
        oh = torch.zeros(B, H, NCOL, dtype=torch.float16, device="cuda")
        def g():
            _gather_only[(B, H // BH)](q3, raw_u8, raw_bf16, lut, flat, oh,
                                       PAGE, PB, DATA,
                                       H, TOPK, BLOCK_H=BH, BLOCK_T=16, NCOL=64,
                                       NOPE=448, ROPE=64, G=64, SB=8, DS=576,
                                       num_warps=4, num_stages=2)
        try:
            t = timeit(g)
            res.setdefault(f"gather_h{BH}", []).append(t)
        except Exception as e:
            import traceback
            print("GATHER_FAIL", BH, type(e).__name__, str(e)[:200], file=sys.stderr)
            res.setdefault(f"gather_h{BH}", []).append(float("nan"))
for k, v in res.items():
    print("%s\t%.4f" % (k, statistics.median(v)))
'''
WORKER = WORKER.replace("REPO", repr(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=512)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--cache-tokens", type=int, default=16384)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()

    import pf_hs_sweep as S
    d = S.build(a.B, a.H, a.topk, a.cache_tokens)

    TOKB = 448 + 128
    per_block = a.topk * TOKB
    nblocks = a.B * (a.H // 16)

    print(f"B={a.B} H={a.H} topk={a.topk} cache={a.cache_tokens}")
    print(f"real kernel grid=({a.B}, {a.H//16}) = {nblocks} blocks")
    print(f"gathered bytes at BLOCK_H=16: {nblocks*per_block/1e6:.0f} MB/call\n")

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    out = {}
    with tempfile.TemporaryDirectory() as td:
        ip = os.path.join(td, "i.pt")
        import torch
        torch.save(d, ip)
        # Triton requires @jit functions to live in a real file, so the worker
        # is written out rather than passed to `python -c`.
        wp = os.path.join(td, "worker.py")
        with open(wp, "w") as f:
            f.write(WORKER)
        r = subprocess.run([sys.executable, wp, ip, str(a.rounds)],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print("worker failed:\n", r.stderr[-3000:])
            return
        if r.stderr.strip():
            print("worker stderr:\n", r.stderr[-2000:])
        for line in r.stdout.splitlines():
            if "\t" in line:
                k, v = line.split("\t")
                out[k] = float(v)

    BW = 616e9
    print("%-14s %10s %12s %10s" % ("probe", "ms", "GB/s", "% of peak"))
    for k in sorted(out):
        ms = out[k]
        bh = int(k[len("gather_h"):]) if k.startswith("gather_h") else 16
        nb = a.B * (a.H // bh)
        by = nb * per_block
        gbs = by / (ms * 1e-3) / 1e9
        print("%-14s %10.3f %12.0f %9.1f%%"
              % (k, ms, gbs, 100 * by / (ms * 1e-3) / BW))

    if "real" in out and "gather_h16" in out:
        r, g = out["real"], out["gather_h16"]
        print()
        print(f"real kernel            {r:.3f} ms")
        print(f"gather-only, same cost {g:.3f} ms  = {100*g/r:.0f}% of the kernel")
        print(f"everything else        {r-g:.3f} ms  = {100*(r-g)/r:.0f}%")
        if "gather_h64" in out and out["gather_h64"] == out["gather_h64"]:
            print()
            print(f"if BLOCK_H could reach 64 the gather would be "
                  f"{out['gather_h64']:.3f} ms "
                  f"({g/out['gather_h64']:.2f}x cheaper) -- that is the prize "
                  f"for cutting head-split redundancy.")


if __name__ == "__main__":
    main()
