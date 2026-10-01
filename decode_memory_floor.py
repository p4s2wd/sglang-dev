"""Decode memory floor: bytes each token must read, and the tok/s that implies.

A batch-1 decode is bandwidth bound: every weight is read once per token and no
arithmetic hides behind it. So the ceiling is (bytes that must move) / (achieved
bandwidth). This computes the first term from the checkpoint itself rather than
from guessed projection shapes, and reports what each optimization is worth.
"""
import json
import os
import re
import struct
import sys
from collections import defaultdict

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731"
BW = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0  # measured GB/s, not 616 peak

ESZ = {"F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "F16": 2, "BF16": 2,
       "F32": 4, "I8": 1, "U8": 1, "I32": 4, "I64": 8}

index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
wmap = index["weight_map"]

shards = defaultdict(list)
for name, shard in wmap.items():
    shards[shard].append(name)

sizes = {}
for shard in shards:
    with open(os.path.join(CKPT, shard), "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        numel = 1
        for d in meta["shape"]:
            numel *= d
        sizes[name] = numel * ESZ.get(meta["dtype"], 1)

cfg = json.load(open(os.path.join(CKPT, "config.json")))
L = cfg["num_hidden_layers"]
N_ROUTED = cfg["n_routed_experts"]
ACTIVE = cfg["num_experts_per_tok"]

MiB = 1024 ** 2
GiB = 1024 ** 3

# Scale bytes are 1/128 of the payload for block-scaled FP8; the sub-80 loader
# keeps them at native width, so they ride along with the weight.
def is_scale(name):
    return "scale" in name or "scale_inv" in name

dense = 0
expert_total = 0
for name, nbytes in sizes.items():
    if ".experts." in name:
        expert_total += nbytes
    else:
        dense += nbytes

# What a token touches:
#  - every dense weight (attention, indexer, dense MLP, shared expert, heads)
#  - ACTIVE of the N_ROUTED routed experts, in every layer
# expert_total is all experts in all layers, so one expert is
# expert_total / (L * N_ROUTED); a token reads ACTIVE of them per layer.
expert_per_layer = expert_total / L
expert_each = expert_per_layer / N_ROUTED
expert_per_token = expert_each * ACTIVE * L

print(f"checkpoint: dense (non-routed) {dense/GiB:7.2f} GiB, "
      f"routed experts {expert_total/GiB:7.2f} GiB")
print(f"layers={L} routed_experts={N_ROUTED} active_per_token={ACTIVE}")
print(f"routed per layer {expert_per_layer/MiB:8.1f} MiB; one expert "
      f"{expert_each/MiB:6.2f} MiB; a token reads {expert_per_token/MiB:.1f} MiB")

# The sub-80 path widens FP8 dense linears to fp16 at load: 2x the bytes.
fp8_bytes = dense
fp16_bytes = dense * 2

print(f"\nbytes a token must read:")
print(f"  dense weights, FP8-resident : {fp8_bytes/GiB:7.3f} GiB")
print(f"  dense weights, fp16 (today) : {fp16_bytes/GiB:7.3f} GiB")
print(f"  active experts (MXFP4)      : {expert_per_token/MiB:7.1f} MiB")

TP = int(os.environ.get("TP", "2"))
print(f"\nper card at TP{TP} (both dense linears and each expert are split):")
for label, dbytes in (("today: fp16 dense", fp16_bytes),
                      ("FP8 dense resident", fp8_bytes)):
    total = (dbytes + expert_per_token) / TP
    ms = total / (BW * 1e9) * 1e3
    print(f"  {label:>22}: {total/GiB:6.3f} GiB/token -> floor "
          f"{ms:5.1f} ms = {1000/ms:5.1f} tok/s at {BW:.0f} GB/s")

# Compare with what is measured, to show how much is NOT the memory floor.
MEASURED_MS = 71.0  # 13.14 tok/s on the real server
floor = (fp16_bytes + expert_per_token) / TP / (BW * 1e9) * 1e3
print(f"\nmeasured today : {MEASURED_MS:5.1f} ms/token (13.1 tok/s)")
print(f"memory floor   : {floor:5.1f} ms/token ({1000/floor:5.1f} tok/s)")
print(f"above the floor: {MEASURED_MS-floor:5.1f} ms/token "
      f"({100*(MEASURED_MS-floor)/MEASURED_MS:.0f}% of today is NOT weight traffic)")
print(f"\n=> the target is {'reachable' if 1000/floor >= 50 else 'NOT reachable'}"
      f" by removing non-traffic overhead alone: the floor itself is "
      f"{1000/floor:.0f} tok/s.")
