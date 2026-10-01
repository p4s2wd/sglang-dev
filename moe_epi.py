"""Measure the routed-expert epilogue at production decode shapes, device time.

The eager decode trace attributes ~2.0 ms/token/stage to unfused torch glue inside
_apply_sub80_mxfp4_w4a16 (fp8.py:2759-2808): chunk -> F.silu(gate.float()) -> clamp ->
act*up -> index_copy_ -> weighted sum over topk. sglang already ships fused kernels for
both halves (silu_and_mul, moe_sum), so this is objective item (3) with a ready-made
fix -- but eager device times for tiny tensors are not trustworthy, so measure the
epilogue alone at the real shapes.

Production decode at bs=1: topk=6, block_m=16, so num_slots = 6*16 = 96;
moe_intermediate_size 2048 sharded over TP2 gives 1024 per rank; hidden 4096 sharded
gives 2048 per rank. Report each variant's kernel duration and the numerics.
"""
import sys, time
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

dev = torch.device("cuda")
NUM_SLOTS, INTER, HIDDEN, TOPK, NTOK = 96, 1024, 2048, 6, 1

inter_slots = torch.randn(NUM_SLOTS, 2 * INTER, dtype=torch.float16, device=dev)
down_slots = torch.randn(NUM_SLOTS, HIDDEN, dtype=torch.float16, device=dev)
sorted_ids = torch.randint(0, NTOK * TOPK, (NUM_SLOTS,), dtype=torch.int32, device=dev)
num_valid = torch.tensor([NTOK * TOPK], dtype=torch.int32, device=dev)
topk_weights = torch.rand(NTOK, TOPK, dtype=torch.float16, device=dev)
max_slot = NTOK * TOPK


def dev_ms(fn, iters=50, label=""):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = sum(e.device_time_total for e in pr.key_averages() if e.device_time_total)
    n = sum(e.count for e in pr.key_averages() if e.device_time_total)
    return tot / 1e3 / iters, n / iters


def current():
    gate, up = inter_slots.chunk(2, dim=-1)
    act = F.silu(gate.float())
    act = (act * up).to(torch.float16)
    slot_ids = torch.arange(NUM_SLOTS, dtype=torch.int32, device=dev)
    out_ts = torch.zeros((max_slot + 1, HIDDEN), dtype=torch.float16, device=dev)
    valid = (sorted_ids < max_slot) & (slot_ids < num_valid)
    dst = torch.where(valid, sorted_ids, torch.full_like(sorted_ids, max_slot))
    out_ts.index_copy_(0, dst.long(), down_slots)
    return (out_ts[:max_slot].view(NTOK, TOPK, HIDDEN).float()
            * topk_weights.float().unsqueeze(-1)).sum(dim=1)


def fused_act():
    from sglang.kernels.ops.activation import silu_and_mul
    return silu_and_mul(inter_slots)


t_cur, n_cur = dev_ms(current)
print("current epilogue: %.4f ms/token in %.0f kernels" % (t_cur, n_cur))

a = current()
b = fused_act()
print("silu_and_mul output vs current act: shape %s dtype %s" % (tuple(b.shape), b.dtype))
gate, up = inter_slots.chunk(2, dim=-1)
ref = (F.silu(gate.float()) * up).to(torch.float16)
rel = ((b.float() - ref.float()).abs() / ref.float().abs().clamp(min=1e-3)).max().item()
print("  max rel err vs current silu path: %.3e" % rel)
t_f, n_f = dev_ms(fused_act)
print("silu_and_mul alone: %.4f ms in %.0f kernels" % (t_f, n_f))

# what the act half costs inside current()
def act_only():
    gate, up = inter_slots.chunk(2, dim=-1)
    act = F.silu(gate.float())
    return (act * up).to(torch.float16)


t_act, n_act = dev_ms(act_only)
print("current act half (chunk+silu+mul+cast): %.4f ms in %.0f kernels -> fused saves %.4f ms"
      % (t_act, n_act, t_act - t_f))

# combine half
def combine_only():
    slot_ids = torch.arange(NUM_SLOTS, dtype=torch.int32, device=dev)
    out_ts = torch.zeros((max_slot + 1, HIDDEN), dtype=torch.float16, device=dev)
    valid = (sorted_ids < max_slot) & (slot_ids < num_valid)
    dst = torch.where(valid, sorted_ids, torch.full_like(sorted_ids, max_slot))
    out_ts.index_copy_(0, dst.long(), down_slots)
    return (out_ts[:max_slot].view(NTOK, TOPK, HIDDEN).float()
            * topk_weights.float().unsqueeze(-1)).sum(dim=1)


t_c, n_c = dev_ms(combine_only)
print("combine half (zeros+where+index_copy_+mul+sum): %.4f ms in %.0f kernels" % (t_c, n_c))
print("\nper stage per token; x4 stages for the decode total")
