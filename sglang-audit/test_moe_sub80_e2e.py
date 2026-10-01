"""End-to-end test of the sub-80 MoE path logic (B2 wiring).

Replicates _apply_sub80_mxfp4_w4a16 (fp8.py) against a naive reference MoE
layer on the 2080 Ti. moe_align_block_size is reproduced locally (the real one
is an sgl_kernel AOT op unavailable on this analysis box): flattened slot
index i = token*topk + k, per-expert padding to block_m, sentinel = numel.
Run: CUDA_VISIBLE_DEVICES=2 python test_moe_sub80_e2e.py
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
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    "m", _PY + "/sglang/kernels/ops/moe/mxfp4_w4a16_kernels.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def moe_align_ref(topk_ids, block_m, E):
    """Pure-torch replica of moe_align_block_size (CUDA kernel semantics)."""
    dev = topk_ids.device
    numel = topk_ids.numel()
    flat = topk_ids.flatten().to(torch.int64)
    counts = torch.bincount(flat, minlength=E)
    blocks = (counts + block_m - 1) // block_m
    expert_ids = torch.repeat_interleave(
        torch.arange(E, dtype=torch.int32, device=dev), blocks.to(torch.int32)
    )
    order = torch.argsort(flat, stable=True)  # slot indices grouped by expert
    parts, cur = [], 0
    for e in range(E):
        idx = order[cur : cur + int(counts[e])].to(torch.int32)
        cur += int(counts[e])
        pad = int(blocks[e]) * block_m - idx.numel()
        parts.append(torch.cat([idx, torch.full((pad,), numel, dtype=torch.int32, device=dev)]))
    return torch.cat(parts), expert_ids


def main():
    torch.manual_seed(3)
    dev = "cuda:0"
    E, hidden, inter, topk, T = 8, 64, 128, 2, 5
    block_m = 16

    codes = torch.randint(0, 16, (E, 2 * inter, hidden), dtype=torch.int64)
    w13 = (codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)).to(torch.uint8).view(torch.int8).to(dev)
    s13 = torch.pow(2.0, torch.randint(-9, 0, (E, 2 * inter, hidden // 32)).float()).to(dev)
    codes2 = torch.randint(0, 16, (E, hidden, inter), dtype=torch.int64)
    w2 = (codes2[:, :, 0::2] | (codes2[:, :, 1::2] << 4)).to(torch.uint8).view(torch.int8).to(dev)
    s2 = torch.pow(2.0, torch.randint(-9, 0, (E, hidden, inter // 32)).float()).to(dev)

    x = (torch.randn(T, hidden, device=dev) * 0.2).half()
    topk_weights, topk_ids = torch.topk(torch.softmax(torch.randn(T, E, device=dev), -1), topk, dim=-1)
    topk_weights = topk_weights.float()
    topk_ids = topk_ids.int()
    limit = 7.0

    # --- reference ---
    w13_ref = m.dequant_mxfp4_reference(w13, s13)
    w2_ref = m.dequant_mxfp4_reference(w2, s2)
    ref = torch.zeros(T, hidden, device=dev)
    for t in range(T):
        for k in range(topk):
            e = int(topk_ids[t, k])
            y = x[t].float() @ w13_ref[e].t()
            gate, up = y.chunk(2, -1)
            act = F.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
            ref[t] += (act @ w2_ref[e].t()) * topk_weights[t, k]

    # --- kernel path (mirrors _apply_sub80_mxfp4_w4a16) ---
    sorted_ids, expert_ids = moe_align_ref(topk_ids, block_m, E)
    num_slots = sorted_ids.shape[0]
    max_slot = T * topk
    # kernel gathers rows of `a` by the flattened slot id (value in sorted_ids),
    # so `a` must be x repeated topk times: row j = x[j // topk].
    a_g = x.repeat_interleave(topk, dim=0).to(torch.float16).contiguous()

    inter_slots = torch.zeros((num_slots, 2 * inter), dtype=torch.float16, device=dev)
    m.mxfp4_w4a16_gemm(a_g, w13, s13, sorted_ids, expert_ids, inter_slots,
                       sentinel=max_slot)
    nan_rows = torch.nonzero(torch.isnan(inter_slots).any(dim=-1)).flatten().tolist()
    print("inter nan rows:", nan_rows[:20], "of", num_slots)
    print("sorted_ids[nan rows]:", sorted_ids[nan_rows].tolist()[:20])
    print("expert per nan block:", [int(expert_ids[r // block_m]) for r in nan_rows[:20]])
    gate, up = inter_slots.chunk(2, dim=-1)
    act = (F.silu(gate.float()).clamp(max=limit) * up.float().clamp(-limit, limit)).to(torch.float16)

    # GEMM2 input `act` is indexed by PADDED SLOT POSITION (row i = slot i),
    # so the gather ids are the identity; expert block structure is unchanged.
    slot_ids = torch.arange(num_slots, dtype=torch.int32, device=dev)
    down_slots = torch.zeros((num_slots, hidden), dtype=torch.float16, device=dev)
    m.mxfp4_w4a16_gemm(act, w2, s2, slot_ids, expert_ids, down_slots,
                       sentinel=num_slots)

    out_ts = torch.zeros((max_slot, hidden), dtype=torch.float16, device=dev)
    valid = sorted_ids < max_slot
    out_ts.index_copy_(0, sorted_ids[valid].long(), down_slots[valid])
    out = (out_ts.view(T, topk, hidden).float() * topk_weights.unsqueeze(-1)).sum(dim=1)

    print("out nan:", torch.isnan(out).any().item(), "ref nan:", torch.isnan(ref).any().item())
    print("inter nan:", torch.isnan(inter_slots).any().item(), "act nan:", torch.isnan(act).any().item(), "down nan:", torch.isnan(down_slots).any().item())
    per_tok = (out - ref).abs().max(dim=-1)
    print("per-token err:", [f"{v:.3f}" for v in per_tok.values.tolist()])
    err = (out - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    print(f"max abs err {err:.5f}  max rel {rel:.2e}")
    ok = rel < 5e-2
    print("sub80 MoE e2e vs reference:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
