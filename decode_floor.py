"""Is single-stream decode already at the memory-bandwidth floor?

Decode is 15.80 tok/s and did not move at all when the power limit went 150 -> 190 W
(+40% on prefill, -0.7% here), which says it is bound by memory bandwidth, not
compute. If so, 50 tok/s single-stream is unreachable by kernel work: the only
levers are reading fewer bytes per token or verifying more tokens per byte read.

Compute the bytes that MUST be read per token from the checkpoint itself: dense
weights are read in full every token, and for MoE only the routed experts that
actually fire are read. Compare the implied floor with the measured rate. If
measured/floor is near 1, decode is done and the remaining path is speculative
decoding; if it is well under 1, there is a real kernel win left.
"""
import json, sys, collections
from safetensors import safe_open

d = "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731"
cfg = json.load(open(d + "/config.json"))
idx = json.load(open(d + "/model.safetensors.index.json"))

print("key config:")
for k in ("hidden_size", "num_hidden_layers", "n_routed_experts", "num_experts_per_tok",
          "moe_intermediate_size", "intermediate_size", "n_shared_experts",
          "first_k_dense_replace", "q_lora_rank", "kv_lora_rank",
          "num_attention_heads", "head_dim", "v_head_dim", "qk_rope_head_dim"):
    if k in cfg:
        print("  %-24s %s" % (k, cfg[k]))

meta = {}
for f in sorted(set(idx["weight_map"].values())):
    with safe_open(d + "/" + f, framework="pt") as sf:
        for k in sf.keys():
            t = sf.get_slice(k)
            n = 1
            for x in t.get_shape():
                n *= x
            meta[k] = (tuple(t.get_shape()), t.get_dtype(), n)

BYTES = {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4}
NL = cfg["num_hidden_layers"]

# Classify per-layer weights: dense (always read) vs routed-expert (read only if
# the expert fires). Scale tensors ride with their weight at ~1/128 the size.
dense = collections.defaultdict(int)
expert_total = collections.defaultdict(int)
for k, (shape, dt, n) in meta.items():
    if not k.startswith("layers."):
        continue
    if ".scale" in k or "_scale_inv" in k:
        continue
    b = n * BYTES.get(dt, 2)
    kk = k
    if ".experts." in kk:
        expert_total["experts"] += b
    else:
        kk2 = kk.split(".")
        if kk2[0] == "layers":
            kk2.pop(1)
        dense[".".join(kk2[:3])] += b

n_experts = cfg.get("n_routed_experts") or cfg.get("num_experts") or 1
k_act = cfg.get("num_experts_per_tok") or cfg.get("top_k") or 1
# Routed experts exist in every MoE layer; read only k_act of n_experts per token.
moe_layers = len({k.split(".")[1] for k in meta
                  if k.startswith("layers.") and ".experts." in k})
per_layer_experts = expert_total["experts"] / max(1, moe_layers)
expert_read_per_token = per_layer_experts * (k_act / n_experts) * moe_layers

dense_sum = sum(dense.values())
print("\nper-token weight bytes that must be read:")
print("  dense (all layers, read every token)   %8.3f GB" % (dense_sum / 1e9))
for k, v in sorted(dense.items(), key=lambda x: -x[1])[:6]:
    print("     %-40s %7.3f GB" % (k[:40], v / 1e9))
print("  routed experts: %d MoE layers, %d experts, %d active"
      % (moe_layers, n_experts, k_act))
print("  expert bytes on disk (all layers)      %8.3f GB" % (expert_total["experts"] / 1e9))
print("  expert bytes read per token            %8.3f GB" % (expert_read_per_token / 1e9))
tot = dense_sum + expert_read_per_token
print("  TOTAL per token                        %8.3f GB" % (tot / 1e9))

PEAK = 616e9
floor = tot / PEAK
print("\nbandwidth floor at 616 GB/s: %.1f ms/token -> %.2f tok/s" % (floor * 1e3, 1 / floor))
for meas in (15.80, 26.55 / 4, 72.8 / 8):
    print("  measured %5.2f tok/s = %.0f%% of the floor" % (meas, 100 * meas * floor))
print("\nfp16-widened dense would read 2x those bytes; FP8-resident reads them once.")
