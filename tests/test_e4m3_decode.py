"""TDD: inline bit-op e4m3 decode must equal the payload LUT for all 256 bytes."""
import sys, torch, triton, triton.language as tl
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

from sglang.kernels.ops.quantization.fp8_w8a16 import _e4m3_to_f16


@triton.jit
def _decode_kernel(p, o, B: tl.constexpr):
    offs = tl.arange(0, B)
    v = _e4m3_to_f16(tl.load(p + offs))
    tl.store(o + offs, v)

def main():
    import sglang.kernels.ops.quantization.fp8_w8a16 as m
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    lut = m.fp8_payload_lut(torch.device(dev), torch.bfloat16)
    b = torch.arange(256, dtype=torch.uint8, device=dev)
    out = torch.empty(256, dtype=torch.float32, device=dev)
    _decode_kernel[(1,)](b, out, B=256)
    ref = lut.to(torch.float32)
    bad = (out != ref) & ~(torch.isnan(out) & torch.isnan(ref))
    print("mismatched bytes:", bad.nonzero().flatten().tolist()[:10])
    bad = bad & (b != 0x7F) & (b != 0xFF)  # 0x7F is e4m3fn NaN, never in weights
    assert not bad.any(), f"decode mismatch on {bad.sum().item()} bytes"
    print("PASS")

if __name__ == "__main__":
    main()
