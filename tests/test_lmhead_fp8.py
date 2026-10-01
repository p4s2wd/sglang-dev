"""TDD: fp8 lm_head must reproduce bf16 logits closely and match argmax."""
import sys, torch
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
from sglang.kernels.ops.quantization.fp8_w8a16 import w8a16_linear

def quantize_block_fp8(w_bf16):
    n, k = w_bf16.shape
    w = w_bf16.detach().cpu().float().view(n // 128, 128, k // 128, 128)
    amax = w.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
    scale = amax / 448.0
    wq = (w / scale).to(torch.float8_e4m3fn)
    return (
        wq.view(n, k).contiguous().cuda(),
        scale.view(n // 128, k // 128).float().cuda(),
    )

def main():
    torch.manual_seed(0)
    N, K = 16384, 4096
    W = (torch.randn(N, K) * 0.02).bfloat16().cuda()
    wq, s = quantize_block_fp8(W)
    for M in (1, 2, 4):
        x = (torch.randn(M, K) * 2.0).bfloat16().cuda()
        ref = (x @ W.t()).float()
        got = w8a16_linear(x, wq, s).float()
        rel = (got - ref).norm() / ref.norm()
        top1 = (got.argmax(-1) == ref.argmax(-1)).all().item()
        print(f"M={M}: rel_l2={rel:.2e} argmax_all_match={top1}")
        assert rel < 5e-2, rel
        assert top1, "argmax must match at greedy decoding"
    print("PASS")

if __name__ == "__main__":
    main()
