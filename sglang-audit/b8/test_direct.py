"""Direct-layout PTX W4A16 (no repack) vs Triton on SM75.

Goes through the production wrappers (mxfp4_w4a16_gemm_ptx_direct /
mxfp4_w4a16_gemm) rather than calling the JIT module directly: an earlier
version of this test called mod.run() itself and so silently kept passing after
the kernel's scale contract changed under it. The scale is built as raw UE8M0
bytes, which is what the sub-80 loader now keeps in HBM.
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
import time

sys.path.insert(0, _PY)
import torch

CUH = _PY + "/sglang/kernels/jit/csrc/moe/mxfp4_w4a16_ptx_direct.cuh"
spec = importlib.util.spec_from_file_location(
    "m", _PY + "/sglang/kernels/ops/moe/mxfp4_w4a16_kernels.py"
)
tri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tri)

assert tri.get_ptx_direct_module() is not None, "PTX direct module failed to build"
print("direct module loaded")


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


def byte_scale(b_u8, as_e8m0):
    """The same UE8M0 bytes, viewed the two ways the loader may hand them over."""
    return b_u8.view(torch.float8_e8m0fnu) if as_e8m0 else b_u8


def main():
    torch.manual_seed(3)
    dev = "cuda:0"
    ok = True
    for (E, N, K) in ((4, 64, 128), (6, 4096, 4096)):
        topk, T = 3, 12
        w = torch.randint(-128, 127, (E, N, K // 2), dtype=torch.int8, device=dev)
        x = (torch.randn(T, K, device=dev) * 0.05).half()
        tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device=dev), -1), topk, dim=-1)
        tw = tw.float(); ti = ti.int()
        sorted_ids, eids = moe_align_ref(ti, 16, E)
        num_slots = sorted_ids.shape[0]; max_slot = T * topk
        a_g = x.repeat_interleave(topk, dim=0).to(torch.float16).contiguous()
        out = torch.zeros(num_slots, N, dtype=torch.float16, device=dev)

        R_ref = None
        b_u8 = (127 + torch.randint(-9, 0, (E, N, K // 32), device=dev)).to(torch.uint8)
        for as_e8m0 in (True, False):
            s = byte_scale(b_u8, as_e8m0)
            tri.mxfp4_w4a16_gemm_ptx_direct(a_g, w, s, sorted_ids, eids, out, max_slot)
            torch.cuda.synchronize()
            R = tri.dequant_mxfp4_reference(w, s)
            if R_ref is None:
                R_ref = R
            else:
                # uint8 and float8_e8m0fnu must reach the kernel identically
                same = torch.equal(R, R_ref)
                ok = ok and same
                print(f"  e8m0/uint8 byte views agree: {same}")
            valid = sorted_ids < max_slot
            sid = sorted_ids.long()
            rows = torch.where(valid, sid.clamp(max=max_slot - 1), torch.zeros_like(sid))
            A = a_g[rows].float()
            slot_expert = torch.repeat_interleave(
                eids.long(), torch.full_like(eids, 16, dtype=torch.int64))
            ref = torch.zeros(num_slots, N, device=dev)
            for e in range(E):
                sel = slot_expert == e
                if sel.any(): ref[sel] = A[sel] @ R[e].t()
            ref[~valid] = 0.0
            err = (out.float() - ref).abs().max().item()
            rel = err / max(ref.abs().max().item(), 1e-9)
            tag = "e8m0" if as_e8m0 else "u8  "
            st = "PASS" if rel < 2e-2 else "FAIL"
            ok = ok and rel < 2e-2
            print(f"E={E} N={N} K={K} scale={tag}: rel {rel:.2e} -> {st}")

    # A float32 scale must be rejected, not silently converted per forward.
    E, N, K = 4, 64, 128
    w = torch.randint(-128, 127, (E, N, K // 2), dtype=torch.int8, device=dev)
    s32 = torch.pow(2.0, torch.randint(-9, 0, (E, N, K // 32), device=dev).float())
    ids = torch.arange(E * 16, dtype=torch.int32, device=dev)
    eids = torch.arange(E, dtype=torch.int32, device=dev)
    out = torch.zeros(E * 16, N, dtype=torch.float16, device=dev)
    try:
        tri.mxfp4_w4a16_gemm_ptx_direct(
            torch.zeros(E * 16, K, device=dev, dtype=torch.float16),
            w, s32, ids, eids, out, E * 16)
        print("float32 scale rejected: False -> FAIL")
        ok = False
    except ValueError:
        print("float32 scale rejected: True")

    # perf vs Triton at real shape
    E, N, K = 6, 4096, 4096
    w = torch.randint(-128, 127, (E, N, K // 2), dtype=torch.int8, device=dev)
    s = byte_scale(
        (127 + torch.randint(-9, 0, (E, N, K // 32), device=dev)).to(torch.uint8), True)
    slots = E * 16
    a_g = (torch.randn(slots, K, device=dev) * 0.05).half()
    sorted_ids = torch.arange(slots, dtype=torch.int32, device=dev)
    eids = torch.arange(E, dtype=torch.int32, device=dev)
    out = torch.zeros(slots, N, dtype=torch.float16, device=dev)
    for name, fn in (
        ("PTX-direct", lambda: tri.mxfp4_w4a16_gemm_ptx_direct(
            a_g, w, s, sorted_ids, eids, out, slots)),
        ("Triton", lambda: tri.mxfp4_w4a16_gemm(
            a_g, w, s, sorted_ids, eids, out, sentinel=slots)),
    ):
        for _ in range(3): fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20): fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 20 * 1000
        print(f"{name}: {ms:7.2f} ms  {E*N*(K//2)/ms/1e6:5.0f} GB/s")
    print("DIRECT VALIDATION:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
