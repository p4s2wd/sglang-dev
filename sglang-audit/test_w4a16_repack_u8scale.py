"""Correctness for the repacked-layout W4A16 GEMM after the uint8-scale change.

The repacked kernel used to take pre-expanded float32 scales. It now takes the
raw UE8M0 byte and rebuilds 2^(byte-127) with the fp16 bit identity
(byte-112)<<10, which is what lets the repacked path stay memory-neutral: an
fp32 scale copy costs E*N*K/32*4 = 1.6 GiB per card on top of the weights, and
that copy -- not the repack buffer, which is the same size as the packed one --
was the entire memory objection to repacking.

Two independent checks, because a shared bug would excuse itself otherwise:
  1. vs dequant_mxfp4_reference + torch matmul (absolute correctness)
  2. vs the direct-layout kernel, which is the production path and was already
     validated against the same reference (rel 3e-4)
Plus the num_valid guard: sorted/eids are capacity-sized torch.empty buffers, so
slots past *num_valid hold garbage that passes the sentinel test and indexes the
weights out of bounds.
"""
import os as _os
import pathlib as _pb


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

import importlib.util  # noqa: E402
import torch  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "m", _PY + "/sglang/kernels/ops/moe/mxfp4_w4a16_kernels.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

torch.cuda.set_device(0)
dev = "cuda:0"


def make_grouped_layout(token_experts, block_m, num_experts, m_total, dev):
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
        parts.append(torch.cat(
            [idx, torch.full((pad,), m_total, dtype=torch.int32, device=dev)]))
        cursor += pad
    return torch.cat(parts), expert_ids, slot_of


def run_case(E, N, K, M, scale_lo, scale_hi, capacity_extra):
    """capacity_extra > 0 exercises the num_valid guard with garbage slots."""
    codes = torch.randint(0, 16, (E, N, K), dtype=torch.int64)
    packed = (codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)).to(torch.uint8)
    # Real checkpoint scale-byte range is [118, 126]; keep the test inside it so
    # the fp16 bit identity is exercised where the model actually uses it.
    scale = torch.randint(scale_lo, scale_hi, (E, N, K // 32), dtype=torch.uint8)

    a = (torch.randn(M, K, device=dev) * 0.1).half()
    w, s = packed.to(dev), scale.to(dev)
    token_experts = torch.randint(0, E, (M,), device=dev)
    block_m = 16
    sorted_ids, expert_ids, slot_of = make_grouped_layout(
        token_experts, block_m, E, M, dev)

    n_real = sorted_ids.numel()
    n_cap = n_real + capacity_extra
    if capacity_extra:
        # Capacity-sized buffers as production allocates them: torch.empty, so
        # the tail is uninitialized. Fill it with values that PASS the sentinel
        # test (a real token id) -- exactly the dangerous case.
        sbuf = torch.empty(n_cap, dtype=torch.int32, device=dev)
        sbuf[:n_real] = sorted_ids
        sbuf[n_real:] = torch.randint(0, M, (capacity_extra,), dtype=torch.int32,
                                      device=dev)
        ebuf = torch.empty(expert_ids.numel() + 8, dtype=torch.int32, device=dev)
        ebuf[:expert_ids.numel()] = expert_ids
        ebuf[expert_ids.numel():] = torch.randint(
            0, E, (8,), dtype=torch.int32, device=dev)
        num_valid = torch.full((1,), n_real, dtype=torch.int32, device=dev)
    else:
        sbuf, ebuf = sorted_ids, expert_ids
        num_valid = torch.full((1,), n_cap, dtype=torch.int32, device=dev)

    sentinel = M  # pad slots carry id == m_total

    # 1. absolute reference
    wref = m.dequant_mxfp4_reference(w, s)

    # 2. production direct-layout kernel
    out_d = torch.zeros(n_cap, N, device=dev, dtype=torch.float16)
    m.mxfp4_w4a16_gemm_ptx_direct(a, w.view(torch.int8), s, sbuf, ebuf, out_d,
                                 sentinel, num_valid=num_valid)
    torch.cuda.synchronize()

    # 3. repacked kernel under test
    w_rep = m.repack_mxfp4_for_ptx(w)
    out_r = torch.zeros(n_cap, N, device=dev, dtype=torch.float16)
    m.mxfp4_w4a16_gemm_ptx(a, w_rep, s, sbuf, ebuf, out_r, sentinel,
                           num_valid=num_valid)
    torch.cuda.synchronize()

    err_ref = err_cross = 0.0
    for mm in range(M):
        e = int(token_experts[mm])
        ref = a[mm].float() @ wref[e].t()
        err_ref = max(err_ref, (out_r[slot_of[mm]].float() - ref).abs().max().item())
        err_cross = max(err_cross,
                        (out_r[slot_of[mm]] - out_d[slot_of[mm]]).abs().max().item())

    # A guard failure shows up as corruption in the tail or a crash; check the
    # pad slots stayed zero, which is what the sentinel contract promises.
    pad_ok = True
    for slot in range(n_real):
        if int(sbuf[slot]) >= sentinel and out_r[slot].abs().max().item() > 0:
            pad_ok = False
    scale_ref = max(wref.abs().max().item(), 1.0)
    return err_ref / scale_ref, err_cross, pad_ok


ok_all = True
print("repacked kernel, uint8 UE8M0 scales (fp16 bit identity):")
for (E, N, K, M, lo, hi, extra) in [
    (8, 128, 256, 37, 118, 127, 0),
    (8, 128, 256, 37, 118, 127, 64),   # garbage slots past num_valid
    (4, 64, 128, 12, 118, 127, 0),
    (16, 256, 512, 61, 118, 127, 32),
]:
    rel, cross, pad_ok = run_case(E, N, K, M, lo, hi, extra)
    good = rel < 5e-3 and cross == 0.0 and pad_ok
    ok_all = ok_all and good
    print(f"  E{E:<3d} N{N:<4d} K{K:<4d} M{M:<3d} cap+{extra:<3d} "
          f"rel_vs_ref {rel:.3e}  vs_direct {cross:.3e}  pad_zero={pad_ok} "
          f"{'PASS' if good else 'FAIL'}")

print("REPACKED W4A16 (uint8 scales):", "PASS" if ok_all else "FAIL")
raise SystemExit(0 if ok_all else 1)
