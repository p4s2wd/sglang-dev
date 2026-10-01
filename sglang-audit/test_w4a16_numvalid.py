"""Regression: W4A16 GEMM must not touch entries past num_tokens_post_padded.

moe_align_block_size returns capacity-sized torch.empty buffers; only the first
num_valid entries are written. The production path used to launch over the whole
capacity, so a block read a garbage expert id (out-of-bounds weight address) and
wrote past `out` -- an illegal memory access that surfaced much later, at an
unrelated call. Here we poison the tail so the bug is visible as wrong numbers
instead of a crash.
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

import importlib.util
import sys
sys.path.insert(0, _PY)
import torch

torch.cuda.set_device(0)
spec = importlib.util.spec_from_file_location(
    "m", _PY + "/sglang/kernels/ops/moe/mxfp4_w4a16_kernels.py")
tri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tri)

dev = "cuda:0"
E, N, K = 6, 256, 128
BLOCK_M = 16
# Real distribution: expert 0 gets most tokens, the rest get a few, so the
# per-expert block counts leave a long unwritten tail in the capacity buffer.
counts = [37, 5, 3, 1, 1, 0]
T = sum(counts)
topk_ids = torch.cat([
    torch.full((c, 1), e, dtype=torch.int64, device=dev)
    for e, c in enumerate(counts)]).to(torch.int32)
max_slot = topk_ids.shape[0]

w = torch.randint(-128, 127, (E, N, K // 2), dtype=torch.int8, device=dev)
s_bytes = (127 + torch.randint(-6, 0, (E, N, K // 32), device=dev)).to(torch.uint8)
a_g = (torch.randn(max_slot, K, device=dev) * 0.05).half()

# Build the align output directly, matching moe_align_block_size's contract:
# capacity-sized buffers whose first num_valid entries are written and whose
# tail is left as uninitialized torch.empty memory.
capacity = max_slot + (E + 1) * (BLOCK_M - 1)
nvalid = sum(-(-c // BLOCK_M) for c in counts) * BLOCK_M
sorted_ids = torch.empty(capacity, dtype=torch.int32, device=dev)
expert_ids = torch.empty(-(-capacity // BLOCK_M), dtype=torch.int32, device=dev)
num_valid = torch.tensor([nvalid], dtype=torch.int32, device=dev)
pos = 0
for e, c in enumerate(counts):
    nblk = -(-c // BLOCK_M)
    ids = torch.arange(pos, pos + c, dtype=torch.int32, device=dev)
    pad = torch.full((nblk * BLOCK_M - c,), max_slot, dtype=torch.int32, device=dev)
    sorted_ids[pos : pos + nblk * BLOCK_M] = torch.cat([ids, pad])
    expert_ids[pos // BLOCK_M : pos // BLOCK_M + nblk] = e
    pos += nblk * BLOCK_M
capacity = sorted_ids.shape[0]
print(f"capacity={capacity} num_valid={nvalid} (unwritten tail: {capacity - nvalid})")
assert capacity > nvalid, "fixture must leave an unwritten tail to be meaningful"

# Poison ONLY the unwritten tail, with values that would index far out of range.
sorted_ids[nvalid:] = 10 ** 6
expert_ids[nvalid // BLOCK_M:] = 10 ** 6

R = tri.dequant_mxfp4_reference(w, s_bytes)
ok = True
for name, fn in (
    ("PTX-direct", tri.mxfp4_w4a16_gemm_ptx_direct),
    ("Triton", tri.mxfp4_w4a16_gemm),
):
    out = torch.zeros(capacity, N, dtype=torch.float16, device=dev)
    try:
        # keyword on purpose: the two wrappers differ in positional order
        # (PTX takes sentinel 7th, Triton takes block_m there).
        fn(a_g, w, s_bytes, sorted_ids, expert_ids, out, sentinel=max_slot,
           num_valid=num_valid)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"{name}: RAISED {type(e).__name__}: {str(e)[:90]} -> FAIL")
        ok = False
        continue
    # reference over the valid region only
    sid = sorted_ids[:nvalid].long()
    slot_expert = torch.repeat_interleave(
        expert_ids[: capacity // BLOCK_M].long(),
        torch.full((capacity // BLOCK_M,), BLOCK_M, dtype=torch.int64, device=dev))[:nvalid]
    ref = torch.zeros(nvalid, N, device=dev)
    live = sid < max_slot
    A = a_g[sid.clamp(max=max_slot - 1)].float()
    for e in range(E):
        sel = (slot_expert == e) & live
        if sel.any():
            ref[sel] = A[sel] @ R[e].t()
    err = (out[:nvalid].float() - ref).abs().max().item()
    rel = err / max(ref.abs().max().item(), 1e-9)
    good = rel < 2e-2
    ok = ok and good
    print(f"{name}: rel {rel:.2e} -> {'PASS' if good else 'FAIL'}")

print("NUM_VALID REGRESSION:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
