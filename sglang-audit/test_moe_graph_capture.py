"""CUDA-graph capture of the sub-90 W4A16 MoE sequence.

Decode CUDA graph is the difference between ~1.6 tok/s and usable throughput on
this box, so every op on the MoE path must be capturable. The two things that
break capture are a host synchronization (nonzero via boolean-mask indexing, or
a .item()) and an allocation the graph pool cannot serve. This replays the exact
op sequence _apply_sub80_mxfp4_w4a16 runs and asserts it captures and replays.
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
E, N, K = 8, 256, 128
INTER2, HIDDEN = 2 * 128, K
BLOCK_M, TOPK, T = 16, 3, 4

torch.manual_seed(0)
w13 = torch.randint(-128, 127, (E, INTER2, K // 2), dtype=torch.int8, device=dev)
s13 = (127 + torch.randint(-6, 0, (E, INTER2, K // 32), device=dev)).to(torch.uint8)
w2 = torch.randint(-128, 127, (E, HIDDEN, INTER2 // 2), dtype=torch.int8, device=dev)
s2 = (127 + torch.randint(-6, 0, (E, HIDDEN, INTER2 // 32), device=dev)).to(torch.uint8)

# Static buffers, the way a graph runner holds them: allocated once, refilled
# in place, never reallocated inside the capture.
x = (torch.randn(T, K, device=dev) * 0.05).half()
topk_ids = torch.randint(0, E, (T, TOPK), dtype=torch.int32, device=dev)
topk_w = torch.rand(T, TOPK, device=dev)
a_g = torch.empty(T * TOPK, K, dtype=torch.float16, device=dev)

max_slot = T * TOPK
capacity = max_slot + (E + 1) * (BLOCK_M - 1)
nblk_cap = -(-capacity // BLOCK_M)
sorted_ids = torch.empty(capacity, dtype=torch.int32, device=dev)
expert_ids = torch.empty(nblk_cap, dtype=torch.int32, device=dev)
num_valid = torch.empty(1, dtype=torch.int32, device=dev)
inter_slots = torch.zeros(capacity, INTER2, dtype=torch.float16, device=dev)
act = torch.empty(capacity, INTER2 // 2, dtype=torch.float16, device=dev)
down_slots = torch.zeros(capacity, HIDDEN, dtype=torch.float16, device=dev)
out_ts = torch.zeros(max_slot + 1, HIDDEN, dtype=torch.float16, device=dev)
slot_range = torch.arange(capacity, dtype=torch.int32, device=dev)
slot_ids = torch.arange(capacity, dtype=torch.int32, device=dev)
discard_fill = torch.full((capacity,), max_slot, dtype=torch.int32, device=dev)
out = torch.empty(T, HIDDEN, dtype=torch.float16, device=dev)

from sglang.kernels.ops.moe.moe_align import moe_align_block_size as jit_align


def run_body(gemm):
    """The op sequence of _apply_sub80_mxfp4_w4a16, onto preallocated buffers."""
    jit_align(topk_ids, E + 1, BLOCK_M, sorted_ids, expert_ids, num_valid,
              torch.empty(E + 2, dtype=torch.int32, device=dev), True)
    a_g.copy_(x.repeat_interleave(TOPK, dim=0))
    gemm(a_g, w13, s13, sorted_ids, expert_ids, inter_slots,
         sentinel=max_slot, num_valid=num_valid)
    gate, up = inter_slots.chunk(2, dim=-1)
    a = torch.nn.functional.silu(gate.float()).clamp(max=10.0) * up.float().clamp(-10.0, 10.0)
    act.copy_(a.to(torch.float16))
    gemm(act, w2, s2, slot_ids, expert_ids, down_slots,
         sentinel=capacity, num_valid=num_valid)
    out_ts.zero_()
    valid = (sorted_ids < max_slot) & (slot_range < num_valid)
    dst = torch.where(valid, sorted_ids, discard_fill)
    out_ts.index_copy_(0, dst.long(), down_slots)
    out.copy_((out_ts[:max_slot].view(T, TOPK, HIDDEN).float()
               * topk_w.float().unsqueeze(-1)).sum(dim=1).half())


ok = True
for name, gemm in (("PTX-direct", tri.mxfp4_w4a16_gemm_ptx_direct),
                   ("Triton", tri.mxfp4_w4a16_gemm)):
    # eager once so Triton/JIT compilation happens outside capture
    run_body(gemm)
    torch.cuda.synchronize()
    eager = out.clone()

    try:
        # Warm up on a side stream, then capture -- the order torch requires.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                run_body(gemm)
        stream.synchronize()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run_body(gemm)
        torch.cuda.synchronize()

        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        run_body(gemm)
        torch.cuda.synchronize()
        err = (replayed.float() - out.float()).abs().max().item()
        rel = err / max(out.float().abs().max().item(), 1e-9)
        good = rel < 1e-3
        print(f"{name}: capture OK, replay vs eager rel {rel:.2e} -> "
              f"{'PASS' if good else 'FAIL'}")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:160]
        print(f"{name}: capture FAILED: {type(e).__name__}: {msg} -> FAIL")
        good = False
    ok = ok and good

print("MOE CUDA GRAPH CAPTURE:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
