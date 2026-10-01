import sys, torch
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
from sglang.kernels.ops.layernorm.mhc import hc_prenorm_smallm

torch.manual_seed(0)
for M, HK in [(1, 28672), (2, 28672), (16, 28672), (1, 1000)]:
    x = (torch.randn(M, HK, device="cuda:0", dtype=torch.bfloat16) * 0.5)
    xf, rs = hc_prenorm_smallm(x, 1e-6)
    xf_ref = x.float()
    rs_ref = torch.rsqrt(xf_ref.square().mean(-1) + 1e-6)
    assert torch.equal(xf, xf_ref), f"x_flat mismatch M={M} HK={HK}"
    err = (rs - rs_ref).abs().max().item() / rs_ref.abs().max().item()
    assert err < 1e-6, f"rsqrt rel err {err} M={M} HK={HK}"
    print(f"ok M={M} HK={HK} rsqrt rel err {err:.2e}")
print("PASS")
