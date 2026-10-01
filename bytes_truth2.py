"""Exact per-token byte counts from the checkpoint, reading each file header once.

Earlier analyses (decode_floor.py, bf16_linears.py) computed bytes with a dtype table
{F8_E4M3:1, BF16:2, F16:2, F32:4} and a .get(dt, 2) fallback, but the routed experts
are I8 (int8-packed fp4) and their scales are F8_E8M0 -- neither was in the table, so
both fell back to 2 bytes and every expert byte count came out 2x too large. That
inflated the routed share of the bandwidth floor and every tok/s ceiling derived from
it. The model is ~156 GB on disk, which experts-at-277-GB cannot be consistent with.
"""
import json, struct, collections

D = "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731"
idx = json.load(open(D + "/model.safetensors.index.json"))
SZ = {"F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F16": 2, "F32": 4, "I8": 1,
      "I64": 8, "U8": 1, "F64": 8, "BOOL": 1, "I32": 4}

# one pass over file headers
meta = {}
for f in sorted(set(idx["weight_map"].values())):
    with open(D + "/" + f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        m = json.loads(fh.read(n))
    for k, v in m.items():
        if k == "__metadata__":
            continue
        nel = 1
        for x in v["shape"]:
            nel *= x
        dt = v["dtype"]
        if dt not in SZ:
            print("UNKNOWN dtype", dt, k)
            continue
        meta[k] = (tuple(v["shape"]), dt, nel * SZ[dt])

print("tensors read: %d" % len(meta))
tot_all = sum(v[2] for v in meta.values())
print("checkpoint total: %.1f GB" % (tot_all / 1e9))

# per-expert bytes (one expert, layer 1)
per_exp = sum(v[2] for k, v in meta.items() if k.startswith("layers.1.ffn.experts.0."))
n_exp_l1 = len({k.split(".")[3] for k in meta if k.startswith("layers.1.ffn.experts.")})
print("experts per layer: %d, bytes per expert: %.2f MB" % (n_exp_l1, per_exp / 1e6))
# breakdown
for k in sorted(k for k in meta if k.startswith("layers.1.ffn.experts.0.")):
    print("   %-24s %-12s %-8s %.2f MB" % (k.split(".")[-2] + "." + k.split(".")[-1],
                                           "x".join(map(str, meta[k][0])), meta[k][1],
                                           meta[k][2] / 1e6))

dense = {k[len("layers.1."):]: v[2] for k, v in meta.items()
         if k.startswith("layers.1.") and ".experts." not in k}
d_tot = sum(dense.values())
print("\ndense (non-expert) bytes per layer: %.2f MB" % (d_tot / 1e6))
for k, b in sorted(dense.items(), key=lambda x: -x[1])[:10]:
    print("   %-42s %8.2f MB" % (k[:42], b / 1e6))

NL, TOPK = 43, 6
routed_all = NL * TOPK * per_exp
dense_all = NL * d_tot
print("\nper token, read across all 8 cards (TP2 shards, PP4 splits layers):")
print("  routed: %.3f GB   dense: %.3f GB   total: %.3f GB"
      % (routed_all / 1e9, dense_all / 1e9, (routed_all + dense_all) / 1e9))
# per (rank,stage): a stage owns 11 layers, TP2 halves each tensor
for name, per_tok in (("routed", routed_all), ("dense", dense_all)):
    stage = per_tok / 8 * (11 / (NL / 4))  # 11 of 11 layers in stage, /8 = /TP2 /PP4
    print("  %s per (rank,stage): %.3f GB" % (name, stage / 1e9))
stage_tot = (routed_all + dense_all) / 8
print("  TOTAL per (rank,stage): %.3f GB -> %.2f ms at 616 GB/s"
      % (stage_tot / 1e9, stage_tot / 616e6 * 1e3))
print("\nbandwidth floors:")
print("  single card streaming everything: %.3f GB / 616 = %.2f ms -> %.1f tok/s"
      % ((routed_all + dense_all) / 1e9, (routed_all + dense_all) / 616e6 * 1e3,
         1000 / ((routed_all + dense_all) / 616e6 * 1e3)))
t = stage_tot / 616e6 * 1e3
print("  TP2/PP4, 4 stages serial: 4 x %.2f = %.2f ms -> %.1f tok/s" % (t, 4 * t, 1000 / (4 * t)))
