"""Which per-layer linears are NOT FP8, and therefore fall to cuBLAS at decode.

The true DECODE ranking shows cuBLAS internal::gemvx::kernel at 10.1% of device
time, 672 calls at 44.3 us each. Every dense projection is supposed to go through
the W8A16 GEMV kernel (483-495 GB/s measured), so something is bypassing it. The
likely cause is weights that are BF16 in the checkpoint rather than FP8: a bf16
linear has no block scale, so it cannot use the W8A16 path and lands in cuBLAS.

Enumerate per-layer weights by dtype straight from the checkpoint, then compute
what a GEMV over each one should cost at 616 GB/s so the 44.3 us can be judged.
672 calls over 8 stage-traces and ~7.6 steps is 11 calls per stage per step, and a
stage owns 11 of the 43 layers, so the signature is exactly one bf16 linear per
layer per step.
"""
import json, re, collections, sys
from safetensors import safe_open

D = "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731"
idx = json.load(open(D + "/model.safetensors.index.json"))
BYTES = {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4}

meta = {}
for f in sorted(set(idx["weight_map"].values())):
    with safe_open(D + "/" + f, framework="pt") as sf:
        for k in sf.keys():
            t = sf.get_slice(k)
            n = 1
            for x in t.get_shape():
                n *= x
            meta[k] = (tuple(t.get_shape()), t.get_dtype(), n)

# Per-layer, per-module dtype + shape, using layer 1 as the representative.
rows = collections.OrderedDict()
for k, (shape, dt, n) in meta.items():
    if not k.startswith("layers.1."):
        continue
    if ".scale" in k or "_scale_inv" in k:
        continue
    mod = k[len("layers.1."):]
    rows[mod] = (shape, dt, n)

print("layer-1 weights by dtype:")
print("%-38s %-16s %-8s %10s %9s" % ("module", "shape", "dtype", "bytes", "us@616"))
nonfp8 = []
for mod, (shape, dt, n) in sorted(rows.items()):
    if len(shape) != 2:
        continue
    b = n * BYTES.get(dt, 2)
    print("%-38s %-16s %-8s %10d %9.2f"
          % (mod[:38], "x".join(map(str, shape)), dt, b, b / 616e3 * 1e3))
    if dt != "F8_E4M3":
        nonfp8.append((mod, shape, dt, b))

print("\nNON-FP8 2-D per-layer weights (cannot use the W8A16 GEMV):")
tot = 0
for mod, shape, dt, b in nonfp8:
    print("  %-36s %-16s %-6s %8.2f MB  floor %.2f us"
          % (mod[:36], "x".join(map(str, shape)), dt, b / 1e6, b / 616e3 * 1e3))
    tot += b
print("  total per layer %.2f MB, x43 layers = %.3f GB" % (tot / 1e6, tot * 43 / 1e9))
