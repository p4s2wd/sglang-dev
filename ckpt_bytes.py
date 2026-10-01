"""Aggregate the checkpoint's tensor bytes by role and dtype.

Reads only the safetensors headers (no tensor data), so it is fast and safe on
a 156 GB checkpoint. Prints, per (role, dtype), the element count, element
size and total bytes, plus what the per-card footprint would be for a given
TPxPP split. This is the number that decides whether the fp32-scale design fits
in 22 GiB at all.
"""
import glob
import json
import os
import struct
import sys
from collections import defaultdict

ckpt = sys.argv[1] if len(sys.argv) > 1 else "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731"
TP = int(sys.argv[2]) if len(sys.argv) > 2 else 2
PP = int(sys.argv[3]) if len(sys.argv) > 3 else 4

files = sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))
if not files:
    files = sorted(glob.glob(os.path.join(ckpt, "*.safetensors-*")))
print(f"files={len(files)}")

SIZE = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3FN": 1, "F8_E5M2FN": 1,
        "F8_E8M0FN": 1, "U8": 1, "I8": 1, "I32": 4, "I64": 8, "F64": 8, "U32": 4}


def role(name):
    if ".experts." in name:
        kind = "routed"
    elif ".shared_experts." in name:
        kind = "shared"
    else:
        kind = "other"
    part = "scale" if name.endswith(".scale") or "scale" in name.split(".")[-1] else "weight"
    # subdivide "other" by first module-ish component
    if kind == "other":
        comps = [c for c in name.split(".") if not c.isdigit()]
        mod = comps[1] if len(comps) > 2 else comps[0]
        kind = "other:" + mod
    return kind, part


agg = defaultdict(lambda: [0, 0])  # (role, dtype) -> [elems, bytes]
dtypes = defaultdict(int)
n = 0
for f in files:
    with open(f, "rb") as fh:
        hdr = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(hdr))
    for name, info in meta.items():
        if name == "__metadata__":
            continue
        dt = info["dtype"]
        shp = info["shape"]
        elems = 1
        for s in shp:
            elems *= s
        esz = SIZE.get(dt)
        if esz is None:
            dtypes[dt] += 1
            esz = 0
        k, p = role(name)
        a = agg[(k, p, dt)]
        a[0] += elems
        a[1] += elems * esz
        n += 1

if dtypes:
    print("UNKNOWN DTYPES:", dict(dtypes))

GiB = 1024 ** 3
tot = 0
print(f"{'role':22s} {'part':7s} {'dtype':12s} {'elems':>14s} {'GiB':>9s}")
for (k, p, dt), (elems, byts) in sorted(agg.items(), key=lambda x: -x[1][1]):
    tot += byts
    print(f"{k:22s} {p:7s} {dt:12s} {elems:14,d} {byts/GiB:9.3f}")
print(f"{'TOTAL':42s} {tot/GiB:9.3f} GiB")
print(f"per-card at TP{TP}xPP{PP} ({TP*PP}-way): {tot/GiB/(TP*PP):.3f} GiB")
print(f"22 GiB card budget for weights+KV+act: {tot/GiB/(TP*PP):.3f} GiB weights")
