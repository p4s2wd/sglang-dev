import sys, torch
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
from sglang.kernels.ops.attention.dsa.triton_mqa_logits import mqa_paged_smallq
from sglang.srt.layers.attention.dsv4.indexer import fp8_paged_mqa_logits_torch

torch.manual_seed(1)
B, H, D, PAGE = 2, 64, 128, 64
for max_pages, seq in [(3, 150), (16, 1000), (32, 2048)]:
    nphys = max_pages * B + 4
    k_vals = (torch.randn(nphys, PAGE, 1, D, device="cuda:0") * 10).to(torch.float8_e4m3fn)
    k_scales = (torch.rand(nphys, PAGE, 1, 1, device="cuda:0") * 0.02 + 0.001)
    flat = torch.empty(nphys, PAGE * (D + 4), dtype=torch.uint8, device="cuda:0")
    flat[:, : PAGE * D] = k_vals.view(nphys, PAGE * D).view(torch.uint8)
    flat.view(nphys, -1).unflatten(-1, (PAGE * (D + 4),))[:, PAGE * D :].view(
        torch.float32
    ).view(nphys, PAGE)[:] = k_scales.view(nphys, PAGE)
    kv = flat.view(nphys, PAGE, 1, D + 4)
    q = torch.randn(B, 1, H, D, device="cuda:0").to(torch.float8_e4m3fn)
    w = torch.rand(B, H, device="cuda:0")
    sl = torch.full((B,), seq, dtype=torch.int32, device="cuda:0")
    pt = torch.randperm(nphys, generator=torch.Generator(device="cpu").manual_seed(2))[: B * max_pages].reshape(B, max_pages).cuda().to(torch.int64)
    pt[1, max(2, max_pages - 1)] = -1
    ms = max_pages * PAGE
    o_ref = fp8_paged_mqa_logits_torch(q, kv, w, sl, pt, None, ms, clean_logits=False)
    o_new = torch.zeros(B, ms, dtype=torch.float32, device="cuda:0")
    mqa_paged_smallq(q, kv, w, sl, pt, ms, o_new)
    ref = o_ref[:, :ms] if o_ref.shape[1] >= ms else o_ref
    holes = (pt < 0)[:, : ref.shape[1] // 64].repeat_interleave(64, dim=1)
    cmp_mask = ~holes[:, : ref.shape[1]]
    denom = ref.abs().max()
    err = ((o_new[:, : ref.shape[1]] - ref).abs() * cmp_mask).max() / denom
    print(f"pages={max_pages} seq={seq}: max rel err {err:.2e} shape_ref {tuple(ref.shape)}")
    assert err < 5e-3, err
print("PASS")
