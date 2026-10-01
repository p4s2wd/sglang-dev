"""Unit test for mxfp4_w4a16_gemm: kernel vs reference dequant + matmul.

Contract under test: slot-indexed grouped GEMM over MXFP4-packed experts.
Run: CUDA_VISIBLE_DEVICES=<sm75 gpu> python test_mxfp4_w4a16.py
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

spec = importlib.util.spec_from_file_location(
    "m", _PY + "/sglang/kernels/ops/moe/mxfp4_w4a16_kernels.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def make_grouped_layout(token_experts, block_m, num_experts, m_total, dev):
    """Per-expert padded layout (moe_align semantics); pad slots carry sentinel."""
    counts = torch.bincount(token_experts, minlength=num_experts)
    blocks = (counts + block_m - 1) // block_m
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=dev),
        blocks.to(torch.int32),
    )
    parts, slot_of = [], {}
    cursor = 0
    for e in range(num_experts):
        idx = torch.nonzero(token_experts == e).flatten().to(torch.int32)
        for j in range(idx.numel()):
            slot_of[int(idx[j])] = cursor + j
        cursor += idx.numel()
        pad = int(blocks[e]) * block_m - idx.numel()
        parts.append(torch.cat([idx, torch.full((pad,), m_total, dtype=torch.int32, device=dev)]))
        cursor += pad
    return torch.cat(parts), expert_ids, slot_of


def main():
    torch.manual_seed(0)
    dev = "cuda:0"
    E, N, K, M = 8, 128, 256, 37
    ok_all = True

    for scale_dtype in ("f32", "u8"):
        codes = torch.randint(0, 16, (E, N, K), dtype=torch.int64)
        packed = (codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)).to(torch.uint8).view(torch.int8)
        if scale_dtype == "f32":
            scale = torch.pow(2.0, torch.randint(-9, 1, (E, N, K // 32)).float())
        else:
            scale = torch.randint(118, 128, (E, N, K // 32), dtype=torch.uint8)

        a = (torch.randn(M, K, device=dev) * 0.1).half()
        w, s = packed.to(dev), scale.to(dev)
        token_experts = torch.randint(0, E, (M,), device=dev)
        block_m = 16
        sorted_ids, expert_ids, slot_of = make_grouped_layout(token_experts, block_m, E, M, dev)

        out = torch.zeros(sorted_ids.numel(), N, device=dev, dtype=torch.float16)
        m.mxfp4_w4a16_gemm(a, w, s, sorted_ids, expert_ids, out)
        torch.cuda.synchronize()

        wref = m.dequant_mxfp4_reference(w, s)
        err_max = 0.0
        for mm in range(M):
            e = int(token_experts[mm])
            ref = a[mm].float() @ wref[e].t()
            got = out[slot_of[mm]].float()
            err_max = max(err_max, (got - ref).abs().max().item())
        ok = err_max < 0.05
        ok_all = ok_all and ok
        print(f"scale_dtype={scale_dtype}: max abs err {err_max:.5f} -> {'PASS' if ok else 'FAIL'}")

    return ok_all


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
