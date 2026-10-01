"""Where the HBM actually goes, and how much of it is load-time widening.

The DSpark draft costs 1.46 GB/card at TP8 and the target leaves only 0.78 GB, so
the gap is ~0.68 GB/card. Before guessing which weights to reclaim, measure it.
On SM75 the FP8 payload is kept as-is only when _keep_fp8_for_w8a16 accepts the
layer (1-byte float, 2-D, block scale covering the grid); everything else is
widened to fp16 at load, which costs exactly one extra byte per element. wo_a is
widened unconditionally by a separate streaming path. So the reclaimable memory
is the FP8 footprint of every tensor that ends up widened -- computable from the
checkpoint index alone, no model load needed.

Sharding is reported per module pattern because it differs: attention projections
are column/row-parallel (split by TP), MoE experts are not split by expert
parallelism at ep_size=1, and everything is divided across PP stages.
"""
import json, sys, collections, re

d = sys.argv[1]
idx = json.load(open(d + "/model.safetensors.index.json"))
from safetensors import safe_open
meta = {}
for f in sorted(set(idx["weight_map"].values())):
    with safe_open(d + "/" + f, framework="pt") as sf:
        for k in sf.keys():
            t = sf.get_slice(k)
            n = 1
            for x in t.get_shape():
                n *= x
            meta[k] = (tuple(t.get_shape()), t.get_dtype())

# Group by module kind, ignoring the layer index.
def kind(k):
    k = re.sub(r"^layers\.\d+\.", "layers.N.", k)
    k = re.sub(r"^mtp\.\d+\.", "mtp.N.", k)
    return k

# Which tensors are 2-D FP8 with a block scale that the W8A16 kernel can index?
def w8a16_ok(name, shape, dtype):
    if dtype != "F8_E4M3" or len(shape) != 2:
        return False
    scale = meta.get(name[: -len(".weight")] + ".scale") or meta.get(name[: -len(".weight")] + ".weight_scale_inv")
    if scale is None:
        return False
    ss, sd = scale
    if len(ss) != 2:
        return False
    n, kk = shape
    return ss[0] >= (n + 127) // 128 and ss[1] >= (kk + 127) // 128

rows = collections.defaultdict(lambda: [0, 0, 0])  # fp8 bytes, widened bytes, count
for k, (shape, dt) in meta.items():
    n = 1
    for x in shape:
        n *= x
    if not k.startswith("layers."):
        continue
    kk = kind(k)
    if dt == "F8_E4M3":
        if ".wo_a." in k:
            rows[kk][1] += n            # widened by the streaming dequant path
        elif w8a16_ok(k, shape, dt):
            rows[kk][0] += n            # stays FP8, kernel dequantizes
        else:
            rows[kk][1] += n            # widened by _dequantize_layer_to_16bit
    elif dt in ("BF16", "F16", "F32") and k.endswith(".weight"):
        rows[kk][2] += n * (2 if dt != "F32" else 4)

tot_fp8 = sum(v[0] for v in rows.values())
tot_wid = sum(v[1] for v in rows.values())
tot_16 = sum(v[2] for v in rows.values())
print("target layers only, checkpoint bytes:")
print("  kept FP8 (W8A16)      %8.2f GB" % (tot_fp8 / 1e9))
print("  WIDENED from fp8      %8.2f GB  -> costs +%.2f GB as fp16" % (tot_wid / 1e9, tot_wid / 1e9))
print("  already 16-bit        %8.2f GB" % (tot_16 / 1e9))
print("\nper module (fp8 kept / widened / 16-bit), GB:")
for kk, v in sorted(rows.items(), key=lambda x: -(x[1][0] + x[1][1] + x[1][2]))[:14]:
    if v[0] + v[1] + v[2] < 1e8:
        continue
    print("  %-42s %7.3f %7.3f %7.3f" % (kk[:42], v[0] / 1e9, v[1] / 1e9, v[2] / 1e9))
print("\nreclaimable if every widened tensor stayed FP8: %.2f GB total" % (tot_wid / 1e9))
print("  per card at TP8/PP1: %.3f GB   per card at TP2/PP4: %.3f GB"
      % (tot_wid / 8 / 1e9, tot_wid / 8 / 1e9))
print("(MoE experts are not TP-split at ep_size=1, so the real per-card number")
print(" depends on which of these are expert weights -- see the per-module rows.)")
