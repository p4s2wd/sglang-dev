"""B2 verification against REAL checkpoint weights on the 2080 Ti (SM75).

Loads layer-2 expert-0 MXFP4 tensors straight from the safetensors shard
(mmap, no full-shard load), then:
  1. dequant_mxfp4_reference sanity (value histogram matches the e2m1 grid)
  2. mxfp4_w4a16_gemm vs reference matmul on real w1/w2 shapes
  3. full MoE layer replay: 6 routed experts of layer 2 through the exact
     _apply_sub80_mxfp4_w4a16 sequence (align -> GEMM1 -> silu -> GEMM2 ->
     combine) vs a naive fp32 reference.
Run: CUDA_VISIBLE_DEVICES=2 python test_real_weights.py
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
import json
import torch
import triton
import torch.nn.functional as F
from safetensors import safe_open

CKPT = _os.environ.get("DSV4_CKPT", "/data/nvme/models/DeepSeek/V4/DeepSeek-V4-Flash-0731")
SHARD = f"{CKPT}/model-00004-of-00048.safetensors"

spec = importlib.util.spec_from_file_location(
    "m", _PY + "/sglang/kernels/ops/moe/mxfp4_w4a16_kernels.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def moe_align_ref(topk_ids, block_m, E):
    dev = topk_ids.device
    numel = topk_ids.numel()
    flat = topk_ids.flatten().to(torch.int64)
    counts = torch.bincount(flat, minlength=E)
    blocks = (counts + block_m - 1) // block_m
    expert_ids = torch.repeat_interleave(
        torch.arange(E, dtype=torch.int32, device=dev), blocks.to(torch.int32)
    )
    order = torch.argsort(flat, stable=True)
    parts, cur = [], 0
    for e in range(E):
        idx = order[cur : cur + int(counts[e])].to(torch.int32)
        cur += int(counts[e])
        pad = int(blocks[e]) * block_m - idx.numel()
        parts.append(torch.cat([idx, torch.full((pad,), numel, dtype=torch.int32, device=dev)]))
    return torch.cat(parts), expert_ids


def main():
    torch.manual_seed(11)
    dev = "cuda:0"
    ok_all = True

    f = safe_open(SHARD, framework="pt")
    # expert 0 of layer 2: w1 [2048, 2048] I8 + scale [2048, 128] E8M0
    w1 = f.get_tensor("layers.2.ffn.experts.0.w1.weight")
    s1 = f.get_tensor("layers.2.ffn.experts.0.w1.scale")
    w2 = f.get_tensor("layers.2.ffn.experts.0.w2.weight")
    s2 = f.get_tensor("layers.2.ffn.experts.0.w2.scale")
    print(f"w1 {tuple(w1.shape)} {w1.dtype}  s1 {tuple(s1.shape)} {s1.dtype}")
    print(f"w2 {tuple(w2.shape)} {w2.dtype}  s2 {tuple(s2.shape)} {s2.dtype}")

    # scale: e8m0 bytes -> float32 true values (what the loader copy_ produces)
    s1f = s1.view(torch.float8_e8m0fnu).float().to(dev)
    s2f = s2.view(torch.float8_e8m0fnu).float().to(dev)
    w1d = w1.to(dev)
    w2d = w2.to(dev)

    # ---- 1. reference dequant sanity: values must live on the e2m1 grid ----
    ref1 = m.dequant_mxfp4_reference(w1d[None], s1f[None])[0]  # [2048, 4096]
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
    # every |value| / scale must be on the grid
    sc = s1f.repeat_interleave(32, dim=-1)
    ratio = (ref1.abs() / sc).unique()
    on_grid = torch.isclose(ratio[:, None], grid[None, :]).any(dim=1).all().item()
    print(f"dequant grid check: |v|/scale in e2m1 set = {on_grid}")
    ok_all = ok_all and on_grid

    # ---- 2. kernel vs reference on real shapes (single expert) ----
    N1, K1 = w1.shape[0], w1.shape[1] * 2  # gate: N=2048, K=4096 elements
    M = 32
    a = (torch.randn(M, K1, device=dev) * 0.05).half()
    ids = torch.arange(M, dtype=torch.int32, device=dev)
    # one expert_id per BLOCK_M-row block (M=32, block_m=16 -> 2 blocks)
    eids = torch.zeros(triton.cdiv(M, 16), dtype=torch.int32, device=dev)
    out = torch.zeros(M, N1, device=dev, dtype=torch.float16)
    m.mxfp4_w4a16_gemm(a, w1d[None], s1f[None], ids, eids, out)
    torch.cuda.synchronize()
    ref = a.float() @ ref1.t()
    err = (out.float() - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    print(f"real w1 GEMM: max abs {err:.5f} rel {rel:.2e} -> {'PASS' if rel < 2e-2 else 'FAIL'}")
    ok_all = ok_all and rel < 2e-2

    # ---- 3. MoE layer replay: 6 experts x (w1,w2) through the wiring ----
    E = 6
    topk = 3
    T = 24
    limit = 7.0
    # checkpoint layout: w1=gate [inter, K/2], w3=up [inter, K/2], w2=down [K, inter/2]
    # sglang fuses w13 = cat(gate, up) along dim 0.
    w13s, s13s, w2s, s2s = [], [], [], []
    for e in range(E):
        g = f.get_tensor(f"layers.2.ffn.experts.{e}.w1.weight")
        u = f.get_tensor(f"layers.2.ffn.experts.{e}.w3.weight")
        gs = f.get_tensor(f"layers.2.ffn.experts.{e}.w1.scale")
        us = f.get_tensor(f"layers.2.ffn.experts.{e}.w3.scale")
        w13s.append(torch.cat([g, u], dim=0).to(dev))
        s13s.append(torch.cat([gs, us], dim=0).view(torch.float8_e8m0fnu).float().to(dev))
        w2s.append(f.get_tensor(f"layers.2.ffn.experts.{e}.w2.weight").to(dev))
        s2s.append(f.get_tensor(f"layers.2.ffn.experts.{e}.w2.scale").view(torch.float8_e8m0fnu).float().to(dev))
    W1 = torch.stack(w13s)   # [E, 2*inter, K/2]
    S1 = torch.stack(s13s)   # [E, 2*inter, K/32]
    W2 = torch.stack(w2s)    # [E, K, inter/2]
    S2 = torch.stack(s2s)    # [E, K, inter/32]
    R1 = m.dequant_mxfp4_reference(W1, S1)  # [E, 2*inter, K]
    R2 = m.dequant_mxfp4_reference(W2, S2)  # [E, K, inter]

    hidden = W2.shape[1]  # 4096
    x = (torch.randn(T, hidden, device=dev) * 0.05).half()
    topk_weights, topk_ids = torch.topk(torch.softmax(torch.randn(T, E, device=dev), -1), topk, dim=-1)
    topk_weights = topk_weights.float()
    topk_ids = topk_ids.int()

    # reference (fp32, naive)
    ref_out = torch.zeros(T, hidden, device=dev)
    for t in range(T):
        for k in range(topk):
            e = int(topk_ids[t, k])
            y = x[t].float() @ R1[e].t()
            g, u = y.chunk(2, -1)
            act = F.silu(g).clamp(max=limit) * u.clamp(-limit, limit)
            ref_out[t] += (act @ R2[e].t()) * topk_weights[t, k]

    # wiring (mirrors _apply_sub80_mxfp4_w4a16 exactly)
    block_m = 16
    sorted_ids, expert_ids = moe_align_ref(topk_ids, block_m, E)
    num_slots = sorted_ids.shape[0]
    max_slot = T * topk
    a_g = x.repeat_interleave(topk, dim=0).to(torch.float16).contiguous()
    inter_slots = torch.zeros((num_slots, W1.shape[1]), dtype=torch.float16, device=dev)
    m.mxfp4_w4a16_gemm(a_g, W1, S1, sorted_ids, expert_ids, inter_slots, sentinel=max_slot)
    gate, up = inter_slots.chunk(2, dim=-1)
    act = (F.silu(gate.float()).clamp(max=limit) * up.float().clamp(-limit, limit)).to(torch.float16)
    slot_ids = torch.arange(num_slots, dtype=torch.int32, device=dev)
    down_slots = torch.zeros((num_slots, hidden), dtype=torch.float16, device=dev)
    m.mxfp4_w4a16_gemm(act, W2, S2, slot_ids, expert_ids, down_slots, sentinel=num_slots)
    out_ts = torch.zeros((max_slot, hidden), dtype=torch.float16, device=dev)
    valid = sorted_ids < max_slot
    out_ts.index_copy_(0, sorted_ids[valid].long(), down_slots[valid])
    out = (out_ts.view(T, topk, hidden).float() * topk_weights.unsqueeze(-1)).sum(dim=1)

    err = (out - ref_out).abs().max().item()
    rel = err / ref_out.abs().max().item()
    print(f"real MoE replay (6 experts, topk=3, T=24): max abs {err:.5f} rel {rel:.2e} -> {'PASS' if rel < 2e-2 else 'FAIL'}")
    ok_all = ok_all and rel < 2e-2

    print("REAL-WEIGHT VALIDATION:", "PASS" if ok_all else "FAIL")
    return ok_all


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
