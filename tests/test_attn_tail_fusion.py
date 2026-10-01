"""TDD: fused attention-tail kernel must match the chained torch helpers."""
import sys, torch
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
from sglang.kernels.ops.attention.flash_mla_sm120_triton import (
    _merge_partial_attn, _apply_attn_sink, _fused_attn_tail)

def chain(outs, lses, sink):
    out, lse = outs[0], lses[0]
    for o, l in zip(outs[1:], lses[1:]):
        out, lse = _merge_partial_attn(out, lse, o, l)
    if sink is not None:
        out, lse = _apply_attn_sink(out, lse, sink)
    return out, lse


def chain_f64(outs, lses, sink):
    """Same closed form in float64: the value both paths approximate."""
    l = torch.stack([x.double() for x in lses])          # [P,B,1,H]
    o = torch.stack([x.double() for x in outs])          # [P,B,1,H,D]
    if sink is not None:
        l = torch.cat([l, sink.view(1, 1, 1, -1).expand(1, *l.shape[1:])], 0)
        o = torch.cat([o, torch.zeros_like(o[:1])], 0)
    m = l.amax(dim=0, keepdim=True)
    w = torch.where(l > -1e20, torch.exp(l - m), torch.zeros_like(l))
    t = w.sum(0)
    out = (w.unsqueeze(-1) * o).sum(0) / t.unsqueeze(-1)
    return out.float(), (m.squeeze(0) + torch.log(t)).squeeze(1)

def main():
    torch.manual_seed(0)
    dev = "cuda:0"
    B, H, D = 2, 64, 576
    for case in ("2p+sink", "3p+sink", "1p+sink", "2p-no-sink"):
        outs, lses = [], []
        n = {"2p+sink": 2, "3p+sink": 3, "1p+sink": 1, "2p-no-sink": 2}[case]
        for p in range(n):
            o = torch.randn(B, 1, H, D, dtype=torch.float16, device=dev)
            l = torch.randn(B, 1, H, device=dev) * 3
            if p == 0:
                l[:, :, :5] = float("-inf")  # empty-partial rows
            outs.append(o); lses.append(l)
        sink = torch.randn(H, device=dev) if "sink" in case else None
        co, cl = chain([o.clone() for o in outs], [l.clone() for l in lses], sink)
        fo, fl = _fused_attn_tail(outs, lses, sink)
        do = (fo.float() - co.float()).abs().max().item()
        good = torch.isfinite(cl)
        dl = (fl - cl).abs()[good].max().item() if good.any() else 0.0
        nm = (torch.isfinite(cl) != torch.isfinite(fl)).sum().item()
        ref_o, ref_l = chain_f64(outs, lses, sink)
        ef = (fo.float() - ref_o).abs().max().item()
        ec = (co.float() - ref_o).abs().max().item()
        print(f"{case}: max|dOut|={do:.2e} max|dLse|={dl:.2e} finite-mismatch={nm} "
              f"err_fused={ef:.2e} err_chain={ec:.2e}")
        assert do < 1.5e-2 and dl < 2e-3 and nm == 0, case
        assert ef <= ec + 1e-5, f"fused must not be less accurate than the chain"
    print("PASS")

if __name__ == "__main__":
    main()
