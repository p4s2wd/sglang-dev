"""Byte accounting for DeepSeek-V4-Flash: what the 156 GB actually is, and what
a KV cache of N tokens costs. Decides whether 256K-512K context and 50-80 tok/s
are reachable on 8x 22 GiB cards, and which change frees the memory.

Run on the machine holding the checkpoint.
"""
import json
import os
import re
import sys
from collections import defaultdict

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731"

index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
wmap = index["weight_map"]

# Per-tensor dtype+shape live in the shard headers; read just the headers.
import struct

def shard_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

shards = defaultdict(list)
for name, shard in wmap.items():
    shards[shard].append(name)

sizes = {}
for shard, names in shards.items():
    hdr = shard_header(os.path.join(CKPT, shard))
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        esz = {"F8_E4M3": 1, "F8_E5M2": 1, "F16": 2, "BF16": 2, "F32": 4,
               "I8": 1, "U8": 1, "F8_E8M0": 1}.get(meta["dtype"], 1)
        numel = 1
        for d in meta["shape"]:
            numel *= d
        sizes[name] = (numel * esz, meta["dtype"], meta["shape"])

def cat(name):
    if ".experts." in name:
        return "moe_routed_experts"
    if "shared_experts" in name:
        return "moe_shared_expert"
    if "indexer" in name:
        return "indexer"
    if "mla_attn" in name or "attn" in name:
        return "attention"
    if "mlp" in name:
        return "dense_mlp"
    if "eh_proj" in name or "hproj" in name or "hc_" in name or "mhc" in name:
        return "hyper_connection"
    if "embed" in name or "lm_head" in name:
        return "embed_lmhead"
    return "other"

tot = defaultdict(int)
for name, (nbytes, dt, shape) in sizes.items():
    tot[cat(name)] += nbytes

GiB = 1024 ** 3
print(f"checkpoint tensors: {len(sizes)}")
for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
    print(f"  {v/GiB:8.2f} GiB  {k}")
print(f"  {sum(tot.values())/GiB:8.2f} GiB  TOTAL")

# What the sub-80 loader does: FP8/MXFP4 linears are widened to fp16 at load.
# Attention/indexer/dense-mlp fp8 -> fp16 doubles; routed experts stay packed.
widen = sum(v for k, v in tot.items()
            if k in ("attention", "indexer", "dense_mlp", "moe_shared_expert",
                     "hyper_connection"))
print(f"\nfp8->fp16 widening on non-routed linears adds {widen/GiB:.2f} GiB")
print(f"  -> resident becomes {(sum(tot.values()) + widen)/GiB:.2f} GiB")
print(f"  -> per card at TP2xPP4 (weights split 8 ways): "
      f"{(sum(tot.values()) + widen)/GiB/8:.2f} GiB")

# KV cost per token. Layout from csrc/deepseek_v4/store.cuh:
#   compressed latent: 448 B fp8 nope + 128 B bf16 rope = 576 B, + 8 B scales
#   indexer: index_head_dim fp8 per token
cfg = json.load(open(os.path.join(CKPT, "config.json")))
L = cfg["num_hidden_layers"]
idx_bytes = cfg["index_head_dim"]
per_tok_layer = 576 + 8 + idx_bytes
print(f"\nKV per token per layer = 576 + 8 + {idx_bytes} = {per_tok_layer} B")
print(f"KV per token all {L} layers = {per_tok_layer * L / 1024:.1f} KiB")
for ctx in (1024, 8192, 65536, 131072, 262144, 524288):
    total = per_tok_layer * L * ctx / GiB
    print(f"  ctx {ctx:>7}: {total:8.2f} GiB total, "
          f"{total/4:6.2f} GiB per PP4 stage, {total/2:6.2f} per PP2 stage")
