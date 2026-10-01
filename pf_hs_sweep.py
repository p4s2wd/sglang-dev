#!/usr/bin/env python
"""Sweep the head-shared attention launch knobs at the PREFILL shape.

The comment above the knobs in flash_mla_sm120_triton.py records a sweep, but it
was measured at B=1 (decode): "4/8/16/32 blocks all cost 0.710 ms". Prefill runs
B=512, where the grid is 1024 programs and the kernel's cost structure is
completely different -- neither block-starved nor split-limited. The shipped
constants (NCOL=64, warps=4, stages=2) are decode-optimal and unvalidated here.

Every config is checked against an fp32 reference over the same gather, so a fast
but wrong config cannot win. Timing is CUDA events around the kernel only: the
host wrapper has a ~0.155 ms floor that would mask everything.

Cache layout replicated here (from _hs_load_nope / _hs_load_nope_t):
  k_cache is [n_pages, page_bytes] uint8-viewed; per token
    [0:448]   e4m3 payload, 32 columns share one ue8m0 scale byte
    [448:576] bf16 rope
  the scale section starts at page_size * 448.

Usage: pf_hs_sweep.py [--quick] [--B 512] [--topk 512] [--cache-tokens 16384]
"""
import argparse
import itertools
import os
import statistics
import subprocess
import sys
import tempfile

import torch

REPO = "/data/nvme/sglang-codex/sglang/python"
H_DEFAULT = 64
NOPE = 448
ROPE = 64
GROUP = 64          # nope columns per ue8m0 scale byte (_HS_GROUP)
SCALE_BYTES = 8     # bytes of scale section per token (_HS_SCALE_BYTES)
DATA_STRIDE = 576   # _TOKEN_DATA_STRIDE: cache data bytes per token
D_HEADS = 512       # _D: q/o width = 448 nope + 64 rope
PAGE = 16

RUNNER = r'''
import sys, statistics, torch
sys.path.insert(0, %(repo)r)
import sglang.kernels.ops.attention.flash_mla_sm120_triton as M
d = torch.load(sys.argv[1], weights_only=False)
H = d["q"].shape[2]
def call():
    return M._run_headshared_sparse_decode(d["q"], d["k_cache"], d["indices"], None,
                                           d["softmax_scale"])
for _ in range(3): call()
torch.cuda.synchronize()
ts = []
for _ in range(%(n)d):
    a = torch.cuda.Event(True); b = torch.cuda.Event(True)
    a.record(); call(); b.record(); torch.cuda.synchronize()
    ts.append(a.elapsed_time(b))
o, l = call()
torch.save({"o": o.cpu(), "lse": l.cpu(), "ms": statistics.median(ts)},
           sys.argv[2])
'''


def build(B, H, topk, cache_tokens, seed=0):
    """Build the cache exactly as the kernel addresses it.

    Layout, from _TOKEN_DATA_STRIDE=576 and scale_section_off = page_size*576:
      page = [ data section: PAGE*576 ][ scale section: PAGE*8 ]  = PAGE*584
      token t in a page: data at t*576, scale bytes at PAGE*576 + t*8
      per token data: [0:448] e4m3 nope (one ue8m0 per 64 cols) + [448:576] bf16 rope
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    data = PAGE * DATA_STRIDE
    page_bytes = data + PAGE * SCALE_BYTES
    n_pages = (cache_tokens + PAGE - 1) // PAGE
    nscale = NOPE // GROUP

    buf = torch.zeros((n_pages, page_bytes), device="cuda", dtype=torch.uint8)
    # e4m3 payload. Restrict the exponent band: a uniform byte draw routinely
    # yields 2^8-scale values that overflow the fp16 the kernel dequantises into.
    buf[:, :PAGE * NOPE] = torch.randint(0, 0x70, (n_pages, PAGE * NOPE),
                                         device="cuda", dtype=torch.uint8,
                                         generator=g)
    # Rope is interleaved per token: token t's 128 bf16 bytes sit at
    # t*576+448, not in one contiguous block per page. Writing it as a single
    # [n_pages, PAGE*128] block silently puts tokens 4..7 over token 0 and
    # leaves the real slots as uninitialised garbage (which then decodes to
    # ~1e28 and makes every downstream comparison NaN).
    rope = torch.randn(n_pages, PAGE, ROPE, device="cuda",
                       dtype=torch.bfloat16, generator=g) * 0.5
    rope_bytes = rope.view(torch.uint8).reshape(n_pages, PAGE, ROPE * 2)
    for t in range(PAGE):
        buf[:, t * DATA_STRIDE + NOPE:t * DATA_STRIDE + DATA_STRIDE] = \
            rope_bytes[:, t]
    # ue8m0 scale byte b means 2^(b-127); stay near 127 so dequant is O(1).
    sc = torch.randint(124, 131, (n_pages, PAGE, nscale), device="cuda",
                       dtype=torch.uint8, generator=g)
    buf[:, data:data + PAGE * nscale] = sc.reshape(n_pages, PAGE * nscale)

    # The runner wants [n_pages, page_size] with stride(0)=page_bytes.
    k_cache = buf.reshape(-1).as_strided((n_pages, PAGE), (page_bytes, DATA_STRIDE))

    # q is [B, 1, H, _D=512]: 448 nope then 64 rope, matching the kernel's
    # q_r load at offset _HS_NOPE=448.
    q = (torch.randn(B, 1, H, D_HEADS, device="cuda", dtype=torch.float16,
                     generator=g) * 0.5)
    idx = torch.stack([
        torch.randperm(cache_tokens, device="cuda", generator=g)[:topk]
        for _ in range(B)
    ]).to(torch.int32)
    return dict(q=q, k_cache=k_cache, indices=idx,
                softmax_scale=NOPE ** -0.5, page_bytes=page_bytes,
                n_pages=n_pages, tok_bytes=DATA_STRIDE)


def reference(d, H, topk):
    """fp32 reference over the same gather, decoding the cache the way the kernel does.

    page = [data: PAGE*576][scales: PAGE*8]; token t has data at t*576 and its
    7 ue8m0 scale bytes at PAGE*576 + t*8, one per 64 nope columns.
    """
    kc = d["k_cache"]                     # [n_pages, PAGE], stride (page_bytes, 576)
    n_pages, PAGE = kc.shape
    page_bytes = d["page_bytes"]
    T = n_pages * PAGE
    data = PAGE * DATA_STRIDE
    nscale = NOPE // GROUP
    # buf[p] is one page; offsets WITHIN it are t*576 (data) and
    # data + t*8 (scales). Do not add p*page_bytes here -- that is already the
    # row stride, and doing it twice walks off the end of the tensor.
    buf = kc.as_strided((n_pages, page_bytes), (page_bytes, 1))

    # Gather the per-token rows. A python loop over pages would be far too slow
    # at production cache size, so do it with strided views:
    #   data[t]   for t in page p, token s  ->  buf[p, s*576 : s*576+576]
    #   scales[t]                            ->  buf[p, data + s*8 : +8]
    tok = torch.arange(T, device="cuda")
    pg, sl = tok // PAGE, tok % PAGE
    pay = buf[pg[:, None], sl[:, None] * DATA_STRIDE + torch.arange(DATA_STRIDE,
                                                                    device="cuda")]
    scb = buf[pg[:, None], data + sl[:, None] * SCALE_BYTES
              + torch.arange(nscale, device="cuda")]

    e4 = torch.arange(256, device="cuda", dtype=torch.uint8)
    sign = torch.where((e4 & 0x80) != 0, -1.0, 1.0)
    exp = ((e4 >> 3) & 0x0F).float()
    man = (e4 & 0x07).float()
    lut = sign * (1.0 + man / 8.0) * torch.pow(2.0, exp - 7.0)
    lut[(e4 & 0x7F) == 0x7F] = 0.0

    nope = lut[pay[:, :NOPE].long()]
    sc = torch.exp2(scb.float() - 127.0).repeat_interleave(GROUP, dim=1)[:, :NOPE]
    nope = nope * sc
    rope = pay[:, NOPE:NOPE + ROPE * 2].contiguous().view(torch.bfloat16).float()
    kv = torch.cat([nope, rope], dim=1)                     # [T, 512] fp32

    idx = d["indices"].long()
    gath = kv[idx]
    q = d["q"].float().squeeze(1)
    s = torch.einsum("bhd,btd->bht", q, gath) * d["softmax_scale"]
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bht,btd->bhd", p, gath)


def run_cfg(cfg, d, n=12):
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    env.update({k: str(v) for k, v in cfg.items()})
    code = RUNNER % dict(repo=REPO, n=n)
    with tempfile.TemporaryDirectory() as td:
        ip, op = os.path.join(td, "i.pt"), os.path.join(td, "o.pt")
        torch.save(d, ip)
        r = subprocess.run([sys.executable, "-c", code, ip, op],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr.strip()[-600:]
        return torch.load(op, weights_only=False), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=512)
    ap.add_argument("--H", type=int, default=H_DEFAULT)
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--cache-tokens", type=int, default=16384)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    print(f"shape: B={a.B} H={a.H} topk={a.topk} cache={a.cache_tokens} "
          f"grid=({a.B},{a.H//16},1)")
    print("shipped default = NCOL 64, warps 4, stages 2\n")

    d = build(a.B, a.H, a.topk, a.cache_tokens)
    ref = reference(d, a.H, a.topk)

    if a.quick:
        combos = list(itertools.product((64,), (4,), (1, 2, 3, 4)))
    else:
        combos = list(itertools.product((64, 128), (2, 4, 8), (1, 2, 3)))

    rows = []
    print("%-6s %-6s %-7s %10s %11s" % ("NCOL", "warps", "stages", "ms", "max|err|"))
    for ncol, w, st in combos:
        cfg = {"SGLANG_SM75_HS_NCOL": ncol, "SGLANG_SM75_HS_WARPS": w,
               "SGLANG_SM75_HS_STAGES": st}
        got, err = run_cfg(cfg, d)
        if got is None:
            print("%-6d %-6d %-7d  FAIL %s" % (ncol, w, st, err[:60]))
            continue
        o = got["o"].float().to(ref.device).reshape(ref.shape)
        e = (o - ref).abs().max().item()
        rows.append((got["ms"], e, (ncol, w, st)))
        print("%-6d %-6d %-7d %10.3f %11.2e" % (ncol, w, st, got["ms"], e))

    base = next((r for r in rows if r[2] == (64, 4, 2)), None)
    rows.sort()
    print()
    if base:
        print(f"shipped (64/4/2) = {base[0]:.3f} ms, err {base[1]:.2e}")
    good = [r for r in rows if r[1] < 5e-2]
    if good:
        errs = [r[1] for r in good]
        print(f"correctness gate: {len(good)}/{len(rows)} configs within 5e-2 "
              f"(errors {min(errs):.1e}..{max(errs):.1e})")
    print("\nranked:")
    for ms, e, key in rows:
        sp = (base[0] / ms) if base else 0
        print("  NCOL=%-4d warps=%-2d stages=%-2d  %7.3f ms  %5.2fx  err %.2e"
              % (key[0], key[1], key[2], ms, sp, e))


if __name__ == "__main__":
    main()
