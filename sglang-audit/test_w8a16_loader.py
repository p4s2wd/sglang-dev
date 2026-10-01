"""Loader integration test for the sub-80 W8A16 path.

The kernel itself is covered by test_w8a16.py. What is tested here is the glue:
that process_weights_after_loading keeps the FP8 payload (instead of widening
it), converts e8m0 scales to real multipliers exactly once, and that apply()
then dispatches to the kernel and returns the same numbers the widening path
would have produced. A silent failure here means every dense projection in the
model is wrong, or that the weights were widened anyway and the memory was never
freed -- which is the whole point of the change.
"""
import os
import sys

import torch


def _find_repo():
    import pathlib

    r = os.environ.get("SGLANG_REPO")
    if r:
        return r
    d = pathlib.Path(__file__).resolve().parent
    for _ in range(8):
        for cand in (d, d / "sglang"):
            if (cand / "python" / "sglang").is_dir():
                return str(cand)
        d = d.parent
    raise RuntimeError("set SGLANG_REPO to the sglang checkout")


sys.path.insert(0, _find_repo() + "/python")

# The sub-80 path is gated behind this env var, and the flag is read in
# Fp8Config's constructor, so it has to be set before the config is built.
os.environ["SGLANG_ALLOW_SUB80_QUANT"] = "1"

from torch import nn

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

torch.cuda.set_device(0)
dev = "cuda:0"
BS = 128


class FakeLayer(nn.Module):
    """Just enough of a quantized Linear for the method's hooks."""

    def __init__(self, n, k, scale_dtype=torch.float32, ue8m0=False):
        super().__init__()
        g = torch.Generator(device=dev).manual_seed(0)
        w = (torch.randn(n, k, generator=g, device=dev) * 0.05).to(
            torch.float8_e4m3fn
        )
        exp = torch.randint(
            -8, -1, ((n + BS - 1) // BS, (k + BS - 1) // BS), generator=g, device=dev
        )
        scale = torch.pow(2.0, exp.float())
        self.weight = nn.Parameter(w, requires_grad=False)
        stored = scale if not ue8m0 else (127.0 + exp.float())
        p = nn.Parameter(stored.to(scale_dtype), requires_grad=False)
        p.format_ue8m0 = ue8m0
        self.weight_scale_inv = p
        self.orig_dtype = torch.float16
        self.weight_block_size = [BS, BS]


def make_method():
    cfg = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[BS, BS],
    )
    m = Fp8LinearMethod(cfg)
    m.sub80_dequant = True
    m.sub80_w8a16 = True
    m.block_quant = True
    m.weight_block_size = [BS, BS]
    return m


def reference(x, w, scale):
    n, k = w.shape
    full = (scale.repeat_interleave(BS, dim=0)[:n]
            .repeat_interleave(BS, dim=1)[:, :k])
    return x.float() @ (w.float() * full).t()


ok = True
for label, ue8m0 in (("fp32 scales", False), ("e8m0 scales", True)):
    n, k = 4096, 4096
    layer = FakeLayer(n, k, scale_dtype=torch.float32, ue8m0=ue8m0)
    orig_scale = (layer.weight_scale_inv.float().clone())
    method = make_method()

    x = (torch.randn(1, k, device=dev) * 0.5).half()
    # Reference from the ORIGINAL stored scale, before the hook touched it.
    true_scale = (torch.exp2(orig_scale - 127.0) if ue8m0 else orig_scale)
    ref = reference(x, layer.weight.data, true_scale)

    method.process_weights_after_loading(layer)

    kept_fp8 = layer.weight.dtype == torch.float8_e4m3fn
    widened = getattr(method, "dequantized_to_16bit", False)
    out = method.apply(layer, x)
    torch.cuda.synchronize()
    rel = (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)

    good = kept_fp8 and not widened and rel < 5e-3
    ok = ok and good
    print(f"{label:>14}: kept_fp8={kept_fp8} widened={widened} "
          f"w8a16_active={getattr(method, 'w8a16_active', False)} "
          f"rel={rel:.1e} {'ok' if good else 'FAIL'}")

    # The memory claim: the fp16 copy must not exist.
    nbytes = layer.weight.numel() * layer.weight.element_size()
    print(f"{'':>14}  weight resident {nbytes/1024/1024:.1f} MiB "
          f"(fp16 copy would be {layer.weight.numel()*2/1024/1024:.1f} MiB)")

# Ragged shape must still take the kernel, not silently widen.
layer = FakeLayer(640, 1000)
method = make_method()
method.process_weights_after_loading(layer)
kept = layer.weight.dtype == torch.float8_e4m3fn
x = (torch.randn(1, 1000, device=dev) * 0.5).half()
ref = reference(x, layer.weight.data, layer.weight_scale_inv.float())
out = method.apply(layer, x)
rel = (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
good = kept and rel < 5e-3
ok = ok and good
print(f"{'ragged 640x1000':>14}: kept_fp8={kept} rel={rel:.1e} "
      f"{'ok' if good else 'FAIL'}")

# A layer the kernel cannot index (no block scale) must still widen correctly.
layer = FakeLayer(4096, 4096)
del layer.weight_scale_inv
method = make_method()
method.process_weights_after_loading(layer)
widened = getattr(method, "dequantized_to_16bit", False)
good = widened and layer.weight.dtype == torch.float16
ok = ok and good
print(f"{'no scale':>14}: widened={widened} dtype={layer.weight.dtype} "
      f"{'ok' if good else 'FAIL'}")

print("\nW8A16 LOADER INTEGRATION:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
