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

torch.manual_seed(3)
dev = "cuda:0"
E, hidden, inter, topk, T = 8, 64, 128, 2, 5
block_m = 16
codes = torch.randint(0, 16, (E, 2 * inter, hidden), dtype=torch.int64)
w13 = (codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)).to(torch.uint8).view(torch.int8).to(dev)
s13 = torch.pow(2.0, torch.randint(-9, 0, (E, 2 * inter, hidden // 32)).float()).to(dev)
x = (torch.randn(T, hidden, device=dev) * 0.2).half()
topk_weights, topk_ids = torch.topk(torch.softmax(torch.randn(T, E, device=dev), -1), topk, dim=-1)
topk_ids = topk_ids.int()

sorted_ids, expert_ids = moe_align_ref(topk_ids, block_m, E)
max_slot = T * topk
a_g = x.repeat_interleave(topk, dim=0).to(torch.float16).contiguous()
inter_slots = torch.zeros((sorted_ids.numel(), 2 * inter), dtype=torch.float16, device=dev)
m.mxfp4_w4a16_gemm(a_g, w13, s13, sorted_ids, expert_ids, inter_slots, sentinel=max_slot)
torch.cuda.synchronize()

w13_ref = m.dequant_mxfp4_reference(w13, s13)
bad = 0
for i in range(sorted_ids.numel()):
    sid = int(sorted_ids[i])
    if sid >= max_slot:
        if inter_slots[i].abs().max().item() > 1e-3:
            print(f"slot {i}: PAD row nonzero {inter_slots[i].abs().max().item()}")
            bad += 1
        continue
    t, k = sid // topk, sid % topk
    e = int(topk_ids[t, k])
    ref = a_g[sid].float() @ w13_ref[e].t()
    err = (inter_slots[i].float() - ref).abs().max().item()
    if err > 0.05:
        eb = int(expert_ids[i // block_m])
        print(f"slot {i} (t={t},k={k}) e={e} expert_block={eb} err {err:.4f}")
        bad += 1
print("bad slots:", bad, "of", sorted_ids.numel())
print("sorted_ids:", sorted_ids.tolist())
print("expert_ids:", expert_ids.tolist())
print("topk_ids:", topk_ids.tolist())

# --- continue to GEMM2 ---
codes2 = torch.randint(0, 16, (E, hidden, inter), dtype=torch.int64)
w2 = (codes2[:, :, 0::2] | (codes2[:, :, 1::2] << 4)).to(torch.uint8).view(torch.int8).to(dev)
s2 = torch.pow(2.0, torch.randint(-9, 0, (E, hidden, inter // 32)).float()).to(dev)
limit = 7.0
gate, up = inter_slots.chunk(2, dim=-1)
act = (F.silu(gate.float()).clamp(max=limit) * up.float().clamp(-limit, limit)).to(torch.float16)
print("act nan:", torch.isnan(act).any().item(), "act max:", act.abs().max().item())
num_slots = sorted_ids.numel()
slot_ids = torch.arange(num_slots, dtype=torch.int32, device=dev)
down_slots = torch.zeros((num_slots, hidden), dtype=torch.float16, device=dev)
m.mxfp4_w4a16_gemm(act, w2, s2, slot_ids, expert_ids, down_slots, sentinel=num_slots)
torch.cuda.synchronize()
print("down nan:", torch.isnan(down_slots).any().item())
w2_ref = m.dequant_mxfp4_reference(w2, s2)
bad = 0
for i in range(num_slots):
    sid = int(sorted_ids[i])
    if sid >= max_slot: continue
    t, k = sid // topk, sid % topk
    e = int(topk_ids[t, k])
    ref = act[i].float() @ w2_ref[e].t()
    err = (down_slots[i].float() - ref).abs().max().item()
    if err > 0.05:
        print(f"slot {i} e={e} err {err:.4f}"); bad += 1
print("gemm2 bad:", bad)
