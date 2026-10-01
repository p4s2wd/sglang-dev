"""Is it safe to store the compressor wkv/wgate weights as fp16 instead of bf16?

compressor.py:365 hardcodes wkv_gate_dtype = torch.bfloat16, but on SM75 the model's
activations are float16, so linear_bf16_fp32 falls into its fp32-copy branch on every
call. Casting the weight once at load to the activation dtype would remove that copy,
but bf16 has an 8-bit exponent and fp16 only 5, so a bf16 value above 65504 becomes inf
in fp16 and a tiny one flushes to zero. Check the actual checkpoint values for every
compressor wkv/wgate tensor before making the change.
"""
import collections, json
import torch
from safetensors import safe_open

D = "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731"
idx = json.load(open(D + "/model.safetensors.index.json"))
byfile = collections.defaultdict(list)
for k, f in idx["weight_map"].items():
    if "compressor.wkv.weight" in k or "compressor.wgate.weight" in k:
        byfile[f].append(k)

mx, mn, nbad, tot, n = 0.0, float("inf"), 0, 0, 0
for f, keys in byfile.items():
    with safe_open(D + "/" + f, framework="pt") as sf:
        for k in keys:
            t = sf.get_tensor(k).float()
            a = t.abs()
            mx = max(mx, a.max().item())
            nz = a[a > 0]
            if nz.numel():
                mn = min(mn, nz.min().item())
            h = t.half()
            nbad += int((h.isinf() | ((a > 0) & (h == 0))).sum().item())
            tot += t.numel()
            n += 1
print("%d compressor wkv/wgate tensors, %d elements" % (n, tot))
print("max abs %.6f   min nonzero abs %.3e" % (mx, mn))
print("fp16 max is 65504 -> overflow: %s" % ("NONE" if mx < 65504 else "YES"))
print("inf-or-flushed-to-zero in fp16: %d (%.2e%%)" % (nbad, 100.0 * nbad / max(tot, 1)))
