"""JIT moe_align_block_size on SM75.

The AOT (sgl_kernel) build ships no sm_75 cubin, so the sub-90 MoE path needs
the JIT variant, which compiles for the local architecture. This checks it
against a reference alignment for uneven expert loads, including the cases that
matter: an expert with zero tokens, and per-expert counts that are not multiples
of BLOCK.
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
import torch

torch.cuda.set_device(0)
from sglang.kernels.ops.moe.moe_align import moe_align_block_size as jit_align

dev = "cuda:0"
BLOCK = 16
ok = True

for counts in ([37, 5, 3, 1, 1, 0], [1, 1, 1], [64] * 8, [0, 0, 5], [16, 17, 15]):
    E = len(counts)
    # topk_ids grouped by expert, so expert e owns flattened positions
    # [sum(counts[:e]), sum(counts[:e]) + counts[e])
    topk_ids = torch.cat([
        torch.full((c, 1), e, dtype=torch.int64, device=dev)
        for e, c in enumerate(counts)]).to(torch.int32)
    numel = topk_ids.numel()
    capacity = numel + (E + 1) * (BLOCK - 1)
    sorted_ids = torch.full((capacity,), -7, dtype=torch.int32, device=dev)
    expert_ids = torch.full((-(-capacity // BLOCK),), -7, dtype=torch.int32, device=dev)
    post_pad = torch.empty((1,), dtype=torch.int32, device=dev)
    cumsum = torch.empty((E + 2,), dtype=torch.int32, device=dev)

    jit_align(topk_ids, E + 1, BLOCK, sorted_ids, expert_ids, post_pad, cumsum, True)
    torch.cuda.synchronize()
    nvalid = int(post_pad.item())

    want_valid = sum(-(-c // BLOCK) for c in counts) * BLOCK
    want_eids = [e for e, c in enumerate(counts) for _ in range(-(-c // BLOCK))]
    got_ids = sorted_ids[:nvalid].tolist()
    got_eids = expert_ids[: len(want_eids)].tolist()

    exact = nvalid == want_valid and got_eids == want_eids
    off = 0
    base = 0
    for e, c in enumerate(counts):
        nblk = -(-c // BLOCK)
        seg = got_ids[off : off + nblk * BLOCK]
        live = sorted(v for v in seg if v != numel)
        exact = exact and live == list(range(base, base + c))
        exact = exact and seg.count(numel) == nblk * BLOCK - c
        off += nblk * BLOCK
        base += c

    ok = ok and exact
    print(f"counts={counts}: num_valid={nvalid} (want {want_valid}) "
          f"eids_ok={got_eids == want_eids} -> {'PASS' if exact else 'FAIL'}")

print("JIT MOE ALIGN:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
