"""Is NCOL=128 / num_warps=4 right for every shape the server actually sees?

The kernel change was tuned at one point -- B=173, H=32, topk=512 -- taken from a
single EXTEND profile. But the kernel runs across a range: B is the number of
query tokens in a prefill chunk (16 to chunk_size, and the dispatch floor is
SGLANG_SM75_HEADSHARED_MIN_BATCH=16), H is 32 or 64 depending on which cache
group is being read, and topk is index_topk=512.

If the optimum moves with shape, fixed constants are wrong somewhere. This sweeps
the shipped pair against the original across that whole grid, interleaved, and
reports where each wins.

Run: python sweep_hs_shapes.py
"""
import sys
import time

import torch
import triton

sys.path.insert(0, "/data/nvme/sglang-codex/sglang-audit")
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

import ab_headshared_bh32 as BH  # noqa: E402
import ab_headshared_scale as A  # noqa: E402
import sglang.kernels.ops.attention.flash_mla_sm120_triton as M  # noqa: E402


def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


# (B, H, topk) spanning what the server sees: the dispatch floor, a mid chunk, a
# full 512-token chunk, and both head counts.
SHAPES = [
    (16, 32, 512), (16, 64, 512),
    (64, 32, 512), (64, 64, 512),
    (173, 32, 512), (173, 64, 512),
    (512, 32, 512), (512, 64, 512),
]
CFGS = [
    ("n256 w8 (original)", dict(ncol=256, warps=8)),
    ("n128 w8", dict(ncol=128, warps=8)),
    ("n128 w4 (shipped)", dict(ncol=128, warps=4)),
    ("n64  w4", dict(ncol=64, warps=4)),
]
ROUNDS = 5


def main():
    print("%-16s %s" % ("shape", "  ".join("%-19s" % c[0] for c in CFGS)))
    wins = {c[0]: 0 for c in CFGS}
    for (B, H, topk) in SHAPES:
        D = 512
        q, kc, flat = A.build(B, H, D, topk, num_tokens=min(B * topk, 64 * 1024))
        scale = 1.0 / (D ** 0.5)
        out = torch.zeros(B, H, D, dtype=q.dtype, device=q.device)
        lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)

        res = {n: [] for n, _ in CFGS}
        rels = {}
        ok = []
        for name, kw in CFGS:
            try:
                out.zero_()
                BH.run_bh(q, kc, flat, scale, out, lse, bh=16, **kw)
                torch.cuda.synchronize()
                ok.append((name, kw))
            except Exception as e:
                print("  %-14s %-19s %s" % (B, H, name, str(e).splitlines()[0][:40]))
        if not ok:
            continue

        # Reference from the shipped entry point, which is the thing that must
        # stay correct.
        ref_out, _ = M._run_headshared_sparse_decode(q, kc, flat, None, scale)
        torch.cuda.synchronize()
        ref = ref_out.squeeze(1)

        for r in range(ROUNDS):
            for name, kw in ok:
                out.zero_()
                BH.run_bh(q, kc, flat, scale, out, lse, bh=16, **kw)
                torch.cuda.synchronize()
                if r == 0:
                    rels[name] = ((out.float() - ref.float()).abs().max().item()
                                  / max(ref.abs().max().item(), 1e-6))
                res[name].append(timeit(
                    lambda: BH.run_bh(q, kc, flat, scale, out, lse, bh=16, **kw),
                    iters=8 if B >= 173 else 14, warmup=2))

        cells = []
        best = min(sorted(res[n])[len(res[n]) // 2] for n, _ in ok)
        for name, _ in CFGS:
            if name not in [n for n, _ in ok]:
                cells.append("%-21s" % "--")
                continue
            v = sorted(res[name])
            med = v[len(v) // 2]
            star = "*" if med <= best * 1.02 else " "
            if med <= best * 1.02:
                wins[name] += 1
            cells.append("%7.3f %5.3fx%s  " % (med, best / med, star))
        print("%-16s %s" % ("B=%d H=%d" % (B, H), "  ".join(cells)))

    print("\nwithin 2% of best on %d shapes:" % len(SHAPES))
    for name, _ in CFGS:
        print("  %-21s %d" % (name, wins[name]))
    bad = [(n, r) for n, r in rels.items() if r > 1e-2]
    print("\nnumerics vs shipped entry point: %s"
          % ("all within 1e-2" if not bad else "OUT OF TOLERANCE %s" % bad))


if __name__ == "__main__":
    main()
