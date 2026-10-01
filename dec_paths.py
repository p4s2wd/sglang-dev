"""Where decode's 64 ms/token goes: dense linears vs experts vs attention.

The floor says 13.21 GB must be read per token, so 616 GB/s would be 21.4 ms.
Measured is 64 ms/token (15.62 tok/s) = 33% of the floor. The repacked expert
kernel benchmarks at 137 GB/s, so its 6.49 GB is 11.9 ms/token -- only 19% of the
64 ms. The dense half (6.713 GB read in full every token) has never been measured at
the shape decode actually uses.

Dense runs W8A16: FP8 weight dequantized on the fly against an fp16 activation at
batch 1-2 rows. That is a GEMV, and GEMVs are latency- and launch-bound rather than
bandwidth-bound, so the open question is what bandwidth it really achieves. Shapes
are taken from the checkpoint itself rather than from config keys, because this
model's config has no kv_lora_rank and guessing produced a KeyError.
"""
import json, re, sys, time, collections
sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")
import torch
from safetensors import safe_open

D = "/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731"
idx = json.load(open(D + "/model.safetensors.index.json"))
dev = torch.device("cuda")
TP = 2

# One representative FP8 2-D dense weight per distinct module pattern, layer 1.
want = collections.OrderedDict()
pat = re.compile(r"^layers\.1\.(.+)\.weight$")
for k, f in idx["weight_map"].items():
    m = pat.match(k)
    if not m or ".experts." in k:
        continue
    want.setdefault(m.group(1), (k, f))
# Norm weights are 1-D and do not go through the linear kernel at all.

shapes = {}
for name, (k, f) in want.items():
    with safe_open(D + "/" + f, framework="pt") as sf:
        t = sf.get_slice(k)
        shapes[name] = tuple(t.get_shape())

print("layer-1 dense FP8 weights from the checkpoint:")
for n, s in shapes.items():
    print("  %-28s %s" % (n, s))

from sglang.kernels.ops.quantization.fp8_w8a16 import w8a16_linear


def bench(fn, iters=150, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


NL = 43
print("\nW8A16 at decode shape, per-rank TP=%d" % TP)
print("%-26s %14s %9s %9s %9s" % ("module", "shape(N,K)", "M=1 ms", "M=2 ms", "GB/s"))
tot = {1: 0.0, 2: 0.0}
for name, sh in shapes.items():
    if len(sh) != 2:
        continue
    N, K = sh
    Nr = max(1, N // TP)
    w = torch.randint(0, 255, (Nr, K), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    sc = torch.ones(max(1, Nr // 128), max(1, K // 128), dtype=torch.float32, device=dev)
    r = {}
    for M in (1, 2):
        x = torch.randn(M, K, dtype=torch.float16, device=dev)
        try:
            r[M] = bench(lambda: w8a16_linear(x, w, sc))
        except Exception as e:
            r[M] = float("nan")
            print("  %-24s FAILED %s" % (name, str(e)[:56]))
    gb = Nr * K / 1e9
    print("%-26s %6dx%-7d %9.4f %9.4f %9.1f"
          % (name[:26], Nr, K, r[1], r[2], gb / r[1] * 1e3 if r[1] == r[1] else float("nan")))
    for M in (1, 2):
        if r[M] == r[M]:
            tot[M] += r[M] * NL

print("\nper token, all %d layers of dense linears:" % NL)
for M in (1, 2):
    print("  M=%d  %6.1f ms" % (M, tot[M]))
print("expert path (repacked, measured): 11.9 ms/token")
print("attention: 2.136 ms/call x 11 layers/stage x 4 stages = %.1f ms/token" % (2.136 * 44))
print("measured total at bs=1: %.1f ms/token" % (1000 / 15.62))
