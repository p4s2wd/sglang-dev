import sys, torch
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
from sglang.kernels.ops.attention.dsa.triton_mqa_logits import (
    fp16_mqa_logits_triton, mqa_logits_smallq)

torch.manual_seed(0)
H, D = 64, 128
for Q, K, N in [(1, 256, 256), (1, 2048, 2048), (7, 1000, 1024), (16, 4096, 4096)]:
    q = (torch.randn(Q, H, D, device="cuda:0") * 10).to(torch.float8_e4m3fn)
    k = (torch.randn(K, D, device="cuda:0") * 10).to(torch.float8_e4m3fn)
    scale = torch.rand(K, device="cuda:0") * 0.02 + 0.001
    w = torch.rand(Q, H, device="cuda:0")
    ks = torch.zeros(Q, dtype=torch.int32, device="cuda:0")
    ke = torch.full((Q,), min(K, 200 if Q > 1 else K), dtype=torch.int32, device="cuda:0")
    q64 = q.to(torch.float64); k64 = k.to(torch.float64)
    ref = torch.zeros(Q, N, dtype=torch.float64, device="cuda:0")
    for t in range(K):
        s = (q64 * k64[t]).sum(-1).clamp(min=0)  # [Q,H]
        ref[:, t] = (s * w.double()).sum(-1) * scale[t].double()
    ref[:, :] = 0.0
    for qq in range(Q):
        for t in range(int(ks[qq]), int(ke[qq])):
            s = (q64[qq] * k64[t]).sum(-1).clamp(min=0)
            ref[qq, t] = (s * w[qq].double()).sum() * scale[t].double()
    o_t = torch.zeros(Q, N, device="cuda:0"); o_f = torch.zeros(Q, N, device="cuda:0")
    fp16_mqa_logits_triton(q, k, scale, w, ks, ke, o_t)
    mqa_logits_smallq(q, k, scale, w, ks, ke, o_f)
    denom = ref.abs().max()
    et = (o_t.double() - ref).abs().max() / denom
    ef = (o_f.double() - ref).abs().max() / denom
    ok = torch.equal(o_f[:, N - 1:], torch.zeros_like(o_f[:, N - 1:])) if N > K else True
    print(f"Q={Q} K={K}: err_torch={et:.2e} err_fused={ef:.2e} tail_zero={ok}")
    assert ef < min(et, 2e-3), "fused must be at least as good"
print("PASS")
