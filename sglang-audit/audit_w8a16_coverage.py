"""Which dense weights actually reach the W8A16 path, and which are still fp16?

The kernel test predicted W8A16 saves 24.5 ms/token; the server saved ~3. The
gap has to be coverage: weights that the loader widens before the quant method
sees them (wo_a's streaming dequant), weights with no block scale, weights that
are not 2D, weights that are not 1-byte float. This applies the predicate from
fp8.py::_keep_fp8_for_w8a16 to the checkpoint's own tensor list -- no GPU, no
server -- and prices each bucket at the measured per-byte rates.

Rates (measured on this machine): fp16 cuBLAS GEMV ~372 GB/s, W8A16 ~483 GB/s
of payload bytes. TP2 splits every column/row-parallel dim across 2 cards, so
per-card bytes are half the checkpoint bytes for the sharded weights.
"""
import json
import os
import re
import struct
import sys
import collections

CK = os.environ.get("DSV4_CKPT",
                    "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731")
ESZ = {"F8_E4M3": 1, "F8_E8M0": 1, "F16": 2, "BF16": 2, "F32": 4,
       "I8": 1, "U8": 1}
BLOCK = 128  # W8A16 scale block

idx = json.load(open(os.path.join(CK, "model.safetensors.index.json")))
wm = idx["weight_map"]
by_shard = collections.defaultdict(list)
for k, v in wm.items():
    by_shard[v].append(k)

tensors = {}
for shard in by_shard:
    f = open(os.path.join(CK, shard), "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    for name, m in json.loads(f.read(n)).items():
        if name == "__metadata__":
            continue
        ne = 1
        for d in m["shape"]:
            ne *= d
        tensors[name] = (m["shape"], m["dtype"], ne * ESZ.get(m["dtype"], 1))

GiB = 1024**3
dense = {k: v for k, v in tensors.items()
         if ".experts." not in k or "shared_experts" in k}

def is_1byte_float(dt):
    return dt in ("F8_E4M3", "F8_E8M0")

buckets = collections.defaultdict(lambda: [0, 0])  # name -> [bytes, count]
reasons = collections.Counter()
for name, (shape, dt, nbytes) in sorted(dense.items()):
    if "scale" in name or name.endswith(".bias"):
        continue
    b = buckets
    if not is_1byte_float(dt):
        b["not-1byte (already 16b)"][0] += nbytes
        b["not-1byte (already 16b)"][1] += 1
        reasons[f"dtype={dt}"] += 1
        continue
    if len(shape) != 2:
        b["not-2D"][0] += nbytes
        b["not-2D"][1] += 1
        continue
    # The checkpoint names the block scale "<prefix>.scale"; the loader renames
    # it to the layer attribute weight_scale_inv. (The first version of this
    # audit looked for "<prefix>.weight_scale_inv" in the checkpoint, found
    # none, and mis-bucketed every FP8 weight as "no block scale".)
    scale = name.rsplit(".", 1)[0] + ".scale"
    if scale not in tensors:
        b["no block scale"][0] += nbytes
        b["no block scale"][1] += 1
        reasons["missing " + name.rsplit(".", 1)[-1]] += 1
        continue
    sshape = tensors[scale][0]
    n, k = shape
    need = ((n + BLOCK - 1) // BLOCK, (k + BLOCK - 1) // BLOCK)
    if tuple(sshape) != need:
        b["scale grid mismatch"][0] += nbytes
        b["scale grid mismatch"][1] += 1
        reasons[f"scale {tuple(sshape)} != {need} for {name.rsplit('.', 1)[-1]}"] += 1
        continue
    if "wo_a" in name:
        b["fp8 but wo_a (streaming widen)"][0] += nbytes
        b["fp8 but wo_a (streaming widen)"][1] += 1
        continue
    b["fp8 -> W8A16"][0] += nbytes
    b["fp8 -> W8A16"][1] += 1

print("per-token dense weight bytes (checkpoint totals; TP2 card = half):")
tot = 0
for key in sorted(buckets, key=lambda x: -buckets[x][0]):
    nb, cnt = buckets[key]
    tot += nb
    print(f"  {nb/GiB:7.3f} GiB  x{cnt:<4d} {key}")
print(f"  {tot/GiB:7.3f} GiB  TOTAL")

card = {k: v[0] / 2 for k, v in buckets.items()}
w8 = card.get("fp8 -> W8A16", 0)
fp16 = (card.get("fp8 but wo_a (streaming widen)", 0) * 2
        + card.get("not-1byte (already 16b)", 0))
print(f"\nper card per token: W8A16 {w8/GiB*1024:6.0f} MiB at 483 GB/s = "
      f"{w8/483e9*1e3:5.2f} ms")
print(f"                    fp16  {fp16/GiB*1024:6.0f} MiB at 372 GB/s = "
      f"{fp16/372e9*1e3:5.2f} ms")
print(f"                    wo_a alone if W8A16: "
      f"{card.get('fp8 but wo_a (streaming widen)', 0)/483e9*1e3:5.2f} ms "
      f"(vs {card.get('fp8 but wo_a (streaming widen)', 0)*2/372e9*1e3:5.2f} "
      f"ms as fp16)")

# Break the "already 16-bit" bucket down by role: if the big ones are real
# linear layers (not norms/gates/embeddings), they are the next FP8-resident
# target, because they are the only remaining weights that can shed bytes.
print("\n16-bit weights by role (checkpoint bytes):")
role = collections.defaultdict(lambda: [0, 0])
for name, (shape, dt, nbytes) in sorted(dense.items()):
    if "scale" in name or name.endswith(".bias"):
        continue
    if is_1byte_float(dt) or len(shape) != 2:
        continue
    parts = name.split(".")
    r = parts[1] if len(parts) > 1 and parts[0] == "layers" else parts[0]
    leaf = parts[-1]
    key = f"{r}.{leaf}" if r in ("attn", "mlp", "moe") else f"{r}"
    role[key][0] += nbytes
    role[key][1] += 1
for k in sorted(role, key=lambda x: -role[x][0])[:12]:
    nb, cnt = role[k]
    print(f"  {nb/GiB:7.3f} GiB x{cnt:<4d} {k}")

print("\ntop scale/shape mismatches:")
for r, c in reasons.most_common(6):
    print(f"  x{c:<4d} {r[:100]}")
