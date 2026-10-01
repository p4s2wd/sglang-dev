"""Exact per-token byte counts from the checkpoint, no dtype guessing.

Two earlier analyses (decode_floor.py, bf16_linears.py) computed tensor bytes with a
dtype table {F8_E4M3:1, BF16:2, F16:2, F32:4} and .get(dt, 2) as fallback -- but the
routed experts are stored as I8 (int8-packed fp4), which is NOT in the table, so every
expert byte count came out 2x too large. That inflated the routed share of the
bandwidth floor (6.493 GB/token) and the total (13.206 GB/token), and every tok/s
ceiling derived from them.

Sanity check on the model itself: the checkpoint is ~156 GB. If experts were 277 GB
the model could not fit that description. Read the real itemsize per dtype from
safetensors metadata and recompute: bytes per expert, routed bytes per token
(43 layers x topk 6), dense bytes per token (all 43 layers every step), and the
resulting bandwidth floors at 616 GB/s.
"""
import json, collections

D = "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731"
idx = json.load(open(D + "/model.safetensors.index.json"))

# exact itemsize per dtype from safetensors metadata
import struct
sizes = {}
files = sorted(set(idx["weight_map"].values()))
for f in files[:2]:
    with open(D + "/" + f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n))
    for k, v in meta.items():
        if k == "__metadata__":
            continue
        import math
        itemsize = {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4, "I8": 1,
                    "I64": 8, "U8": 1, "F64": 8}.get(v["dtype"])
        if itemsize is None:
            print("UNKNOWN dtype", v["dtype"])
            continue
        sizes[v["dtype"]] = itemsize

print("dtypes seen:", sorted(sizes))

# per-expert bytes for one expert, and per-layer dense bytes
def tbytes(name, f):
    with open(D + "/" + f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n))
    tot = 0
    for k, v in meta.items():
        if k != name:
            continue
        nel = 1
        for x in v["shape"]:
            nel *= x
        return nel * sizes[v["dtype"]]
    return None

# gather layer-1 expert bytes
exp_keys = [k for k in idx["weight_map"] if k.startswith("layers.1.ffn.experts.0.")]
per_exp = 0
for k in exp_keys:
    b = tbytes(k, idx["weight_map"][k])
    if b:
        per_exp += b
        print("  %-44s %10.2f MB" % (k[len("layers.1.ffn.experts.0."):], b / 1e6))
print("bytes per expert: %.2f MB" % (per_exp / 1e6))

# dense (non-expert) per-layer bytes
dense = 0
dense_list = []
for k, f in idx["weight_map"].items():
    if k.startswith("layers.1.") and ".experts." not in k:
        b = tbytes(k, f)
        if b:
            dense += b
            dense_list.append((k[len("layers.1."):], b))
print("\ndense bytes per layer: %.2f MB" % (dense / 1e6))
for k, b in sorted(dense_list, key=lambda x: -x[1])[:8]:
    print("  %-44s %10.2f MB" % (k[:44], b / 1e6))

NL, TOPK, TP = 43, 6, 2
routed_tok = NL * TOPK * per_exp / TP
dense_tok = NL * dense / TP
print("\nper token, per (rank,stage) at TP2/PP4 (a stage owns ~11 layers):")
routed_stage = 11 * TOPK * per_exp / TP
dense_stage = 11 * dense / TP
print("  routed experts: %.3f GB   dense: %.3f GB" % (routed_stage / 1e9, dense_stage / 1e9))
print("per token across all 8 cards: routed %.3f GB + dense %.3f GB = %.3f GB"
      % (NL * TOPK * per_exp / 1e9, NL * dense / 1e9,
         (NL * TOPK * per_exp + NL * dense) / 1e9))
tot = routed_stage + dense_stage
print("\nbandwidth floor per stage: %.3f GB / 616 GB/s = %.2f ms" % (tot / 1e9, tot / 616e6 * 1e3))
print("single-stream floor (4 stages serial): %.2f ms -> %.1f tok/s"
      % (4 * tot / 616e6 * 1e3, 1000 / (4 * tot / 616e6 * 1e3)))
