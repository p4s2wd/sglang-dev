"""E2E: full _apply_sub80_mxfp4_w4a16 sequence via the PTX-direct kernel."""
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
import torch.nn.functional as F

from sglang.kernels.jit.utils import load_jit

CUH = _PY + "/sglang/kernels/jit/csrc/moe/mxfp4_w4a16_ptx_direct.cuh"
spec = importlib.util.spec_from_file_location(
    "m", _PY + "/sglang/kernels/ops/moe/mxfp4_w4a16_kernels.py"
)
tri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tri)

mod = load_jit("w4a16_ptx_direct", cuda_files=[CUH],
               cuda_wrappers=[("run", "W4A16PtxDirectKernel::run")])


def moe_align_ref(topk_ids, block_m, E):
    dev = topk_ids.device
    numel = topk_ids.numel()
    flat = topk_ids.flatten().to(torch.int64)
    counts = torch.bincount(flat, minlength=E)
    blocks = (counts + block_m - 1) // block_m
    expert_ids = torch.repeat_interleave(
        torch.arange(E, dtype=torch.int32, device=dev), blocks.to(torch.int32))
    order = torch.argsort(flat, stable=True)
    parts, cur = [], 0
    for e in range(E):
        idx = order[cur:cur + int(counts[e])].to(torch.int32)
        cur += int(counts[e])
        pad = int(blocks[e]) * block_m - idx.numel()
        parts.append(torch.cat([idx, torch.full((pad,), numel, dtype=torch.int32, device=dev)]))
    return torch.cat(parts), expert_ids


def main():
    torch.manual_seed(5)
    dev = "cuda:0"
    # real V4 shapes: hidden 4096, inter 2048, E=8 (subset), topk 3
    E, hidden, inter, topk, T = 8, 4096, 2048, 3, 24
    limit = 7.0
    w13 = torch.randint(-128, 127, (E, 2 * inter, hidden // 2), dtype=torch.int8, device=dev)
    # UE8M0 bytes, exactly as the sub-80 loader keeps them in HBM.
    s13 = (127 + torch.randint(-9, 0, (E, 2 * inter, hidden // 32), device=dev)).to(torch.uint8)
    w2 = torch.randint(-128, 127, (E, hidden, inter // 2), dtype=torch.int8, device=dev)
    s2 = (127 + torch.randint(-9, 0, (E, hidden, inter // 32), device=dev)).to(torch.uint8)

    x = (torch.randn(T, hidden, device=dev) * 0.05).half()
    tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device=dev), -1), topk, dim=-1)
    tw = tw.float(); ti = ti.int()

    # ---- wiring (mirrors _apply_sub80_mxfp4_w4a16 with PTX-direct) ----
    block_m = 16
    sorted_ids, expert_ids = moe_align_ref(ti, block_m, E)
    num_slots = sorted_ids.shape[0]
    max_slot = T * topk
    a_g = x.repeat_interleave(topk, dim=0).to(torch.float16).contiguous()
    inter_slots = torch.zeros((num_slots, 2 * inter), dtype=torch.float16, device=dev)
    tri.mxfp4_w4a16_gemm_ptx_direct(a_g, w13, s13, sorted_ids, expert_ids, inter_slots, max_slot)
    gate, up = inter_slots.chunk(2, dim=-1)
    act = (F.silu(gate.float()).clamp(max=limit) * up.float().clamp(-limit, limit)).to(torch.float16)
    slot_ids = torch.arange(num_slots, dtype=torch.int32, device=dev)
    down_slots = torch.zeros((num_slots, hidden), dtype=torch.float16, device=dev)
    tri.mxfp4_w4a16_gemm_ptx_direct(act, w2, s2, slot_ids, expert_ids, down_slots, num_slots)
    out_ts = torch.zeros((max_slot, hidden), dtype=torch.float16, device=dev)
    valid = sorted_ids < max_slot
    out_ts.index_copy_(0, sorted_ids[valid].long(), down_slots[valid])
    out = (out_ts.view(T, topk, hidden).float() * tw.unsqueeze(-1)).sum(dim=1)

    # ---- fp32 reference ----
    R13 = tri.dequant_mxfp4_reference(w13, s13)  # [E, 2*inter, hidden]
    R2 = tri.dequant_mxfp4_reference(w2, s2)     # [E, hidden, inter]
    ref = torch.zeros(T, hidden, device=dev)
    for t in range(T):
        for k in range(topk):
            e = int(ti[t, k])
            y = x[t].float() @ R13[e].t()
            g, u = y.chunk(2, -1)
            a = F.silu(g).clamp(max=limit) * u.clamp(-limit, limit)
            ref[t] += (a @ R2[e].t()) * tw[t, k]

    err = (out - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    ok = rel < 2e-2
    print(f"PTX-direct MoE e2e: abs {err:.5f} rel {rel:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
