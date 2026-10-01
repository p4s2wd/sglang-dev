"""B8: hand-written PTX mma.sync W4A16 on SM75 — correctness + perf vs Triton."""
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
import time

sys.path.insert(0, _PY)
import torch
import torch.nn.functional as F

from sglang.kernels.jit.utils import load_jit

CUH = str(_pb.Path(__file__).resolve().parent / "w4a16_ptx.cuh")

spec = importlib.util.spec_from_file_location(
    "m", _PY + "/sglang/kernels/ops/moe/mxfp4_w4a16_kernels.py"
)
tri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tri)


def load_module():
    return load_jit(
        "w4a16_ptx_b8",
        cuda_files=[CUH],
        cuda_wrappers=[("run", "W4A16PtxKernel::run")],
    )


def repack_mxfp4(w):
    """[E, N, K/2] int8 -> uint32 [E, N/8, K/32, 32], lane = gid*4 + c4.

    byte j (j=0..3) of the lane's u32 = packed byte
      W[e][nt*8 + gid][ks*16 + j*4 + c4]
    = the B-fragment byte for mma k-step j. Pure reshape/permute/view (no
    advanced indexing) so it stays memory-efficient on multi-GB weights.
    """
    E, N, Kh = w.shape
    nt, ks = N // 8, Kh // 16
    # [E, nt, gid, ks, j, c4]  (kh = j*4 + c4 -> j slow, c4 fast)
    wv = w.view(E, nt, 8, ks, 4, 4)
    # want [E, nt, ks, gid, c4, j] with j innermost-contiguous
    wv = wv.permute(0, 1, 3, 2, 5, 4).contiguous()
    # pack the 4 j-bytes into one little-endian uint32
    u32 = wv.view(torch.uint8).view(torch.uint32)  # [..., 4]u8 -> [...]u32
    return u32.reshape(E, nt, ks, 32).contiguous()


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
    torch.manual_seed(3)
    dev = "cuda:0"
    mod = load_module()
    print("PTX module loaded")

    ok_all = True
    for (E, N, K) in ((4, 64, 128), (6, 4096, 4096)):
        topk, T = 3, 12
        w = torch.randint(-128, 127, (E, N, K // 2), dtype=torch.int8, device=dev)
        s = torch.pow(2.0, torch.randint(-9, 0, (E, N, K // 32), device=dev).float())
        wr = repack_mxfp4(w)
        x = (torch.randn(T, K, device=dev) * 0.05).half()
        topk_weights, topk_ids = torch.topk(torch.softmax(torch.randn(T, E, device=dev), -1), topk, dim=-1)
        topk_weights = topk_weights.float()
        topk_ids = topk_ids.int()

        sorted_ids, eids = moe_align_ref(topk_ids, 16, E)
        num_slots = sorted_ids.shape[0]
        max_slot = T * topk
        a_g = x.repeat_interleave(topk, dim=0).to(torch.float16).contiguous()

        out = torch.zeros(num_slots, N, dtype=torch.float16, device=dev)
        mod.run(a_g, wr, s, sorted_ids, eids, out, float(max_slot))
        torch.cuda.synchronize()

        # reference: dequant + gather matmul (fp32), per-slot expert
        R = tri.dequant_mxfp4_reference(w, s)  # [E, N, K]
        valid = sorted_ids < max_slot
        sid = sorted_ids.long()
        rows = torch.where(valid, sid.clamp(max=max_slot - 1), torch.zeros_like(sid))
        A = a_g[rows].float()  # [num_slots, K]
        # expert of each slot = eids[slot // 16]
        slot_expert = torch.repeat_interleave(
            eids.long(), torch.full_like(eids, 16, dtype=torch.int64)
        )
        ref = torch.zeros(num_slots, N, device=dev)
        for e in range(E):
            sel = slot_expert == e
            if sel.any():
                ref[sel] = A[sel] @ R[e].t()
        ref[~valid] = 0.0

        err = (out.float() - ref).abs().max().item()
        rel = err / max(ref.abs().max().item(), 1e-9)
        status = "PASS" if rel < 2e-2 else "FAIL"
        ok_all = ok_all and rel < 2e-2
        print(f"E={E} N={N} K={K}: max abs {err:.5f} rel {rel:.2e} -> {status}")

    # perf: real w13 shape, decode-like (few experts)
    E, N, K = 6, 4096, 4096
    w = torch.randint(-128, 127, (E, N, K // 2), dtype=torch.int8, device=dev)
    s = torch.pow(2.0, torch.randint(-9, 0, (E, N, K // 32), device=dev).float())
    wr = repack_mxfp4(w)
    slots = E * 16
    a_g = (torch.randn(slots, K, device=dev) * 0.05).half()
    sorted_ids = torch.arange(slots, dtype=torch.int32, device=dev) % slots
    eids = torch.repeat_interleave(
        torch.arange(E, dtype=torch.int32, device=dev),
        torch.full((E,), N // 8 // (N // 8), dtype=torch.int32, device=dev) * 0 + 1,
    )
    # one m-block per expert (16 rows each)
    out = torch.zeros(slots, N, dtype=torch.float16, device=dev)

    def run_ptx():
        mod.run(a_g, wr, s, sorted_ids, eids, out, float(slots))

    def run_tri():
        tri.mxfp4_w4a16_gemm(a_g, w, s, sorted_ids, eids, out, sentinel=slots)

    for name, fn in (("PTX", run_ptx), ("Triton", run_tri)):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20): fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 20 * 1000
        wbytes = E * N * (K // 2)
        print(f"{name}: {ms:7.2f} ms   weight-bw {wbytes/ms/1e6:6.0f} GB/s")

    print("B8 PTX VALIDATION:", "PASS" if ok_all else "FAIL")
    return ok_all


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
