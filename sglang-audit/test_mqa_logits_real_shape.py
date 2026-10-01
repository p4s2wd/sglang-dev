"""MQA logits kernel at REAL indexer shapes: H=64, D=128 (SM75 SMEM limit)."""
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
    "m", _PY + "/sglang/kernels/ops/attention/dsa/triton_mqa_logits.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def main():
    torch.manual_seed(7)
    dev = "cuda:0"
    Q, H, D, K, N = 64, 64, 128, 4096, 4096
    qf = (torch.randn(Q, H, D, device=dev) * 0.5).clamp(-400, 400).to(torch.float8_e4m3fn)
    kf = (torch.randn(K, D, device=dev) * 0.5).clamp(-400, 400).to(torch.float8_e4m3fn)
    k_scale = torch.pow(2.0, torch.randint(-6, 1, (K,), device=dev).float())
    w = torch.rand(Q, H, device=dev)
    ke = torch.randint(1, K + 1, (Q,), device=dev, dtype=torch.int32)
    ks = torch.zeros(Q, device=dev, dtype=torch.int32)

    out = torch.zeros(Q, N, device=dev, dtype=torch.float32)
    m.fp16_mqa_logits_triton(qf, kf, k_scale, w, ks, ke, out)
    torch.cuda.synchronize()

    q16, k16 = qf.float(), kf.float()
    ref = torch.zeros(Q, N, device=dev)
    for qq in range(Q):
        e = int(ke[qq])
        s = (k16[:e] @ q16[qq].t()).clamp(min=0)
        s = (s * w[qq]).sum(dim=1) * k_scale[:e]
        ref[qq, :e] = s

    err = (out - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    print(f"real shape (Q=64,H=64,D=128,K=4096): max abs {err:.5f} rel {rel:.2e}")
    ok = rel < 1e-2
    print("MQA LOGITS REAL SHAPE:", "PASS" if ok else "FAIL")
    return ok

if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
