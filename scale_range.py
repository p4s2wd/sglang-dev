"""Measure the real UE8M0 scale byte range in the checkpoint.

Reads tensor bytes directly from the safetensors data section using the header's
data_offsets, so it needs no safetensors/torch glue. The answer decides whether
the sub-80 PTX kernel can rebuild the scale as an fp16 bit pattern: 2**(b-127)
is exactly representable in fp16 only for b in [113, 142] (subnormals below
113 and overflow above 142 both break the bit trick).
"""
import glob
import json
import os
import struct

ckpt = os.environ.get("DSV4_CKPT", "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731")
files = sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))
print(f"files={len(files)}")

lo, hi = 255, 0
n = 0
ntensors = 0
for f in files:
    with open(f, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(hlen))
        base = 8 + hlen
        base += (-base) % 8
        for name, info in meta.items():
            if name == "__metadata__":
                continue
            if ".ffn.experts." not in name or not name.endswith(".scale"):
                continue
            if info["dtype"] != "F8_E8M0":
                continue
            s0, s1 = info["data_offsets"]
            fh.seek(base + s0)
            buf = fh.read(s1 - s0)
            mn, mx = min(buf), max(buf)
            lo = min(lo, mn)
            hi = max(hi, mx)
            n += len(buf)
            ntensors += 1
            if ntensors == 1:
                print(f"  first scale tensor {name} bytes={len(buf)} "
                      f"min={mn} max={mx} s0={s0}")

print(f"scale tensors={ntensors} elems={n:,}")
print(f"byte range [{lo}, {hi}]  ->  2^({lo}-127) .. 2^({hi}-127)")
print(f"  = {2.0**(lo-127):.3e} .. {2.0**(hi-127):.3e}")
print(f"fp16 exact window: b in [113, 142]")
print("IN-WINDOW" if lo >= 113 and hi <= 142 else "OUT OF WINDOW")
