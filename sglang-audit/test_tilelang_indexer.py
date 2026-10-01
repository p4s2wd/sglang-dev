"""Can the TileLang indexer kernel run on SM75?

The sub-90 branch of the model hook sets SGLANG_OPT_USE_TILELANG_INDEXER=False
with the comment "no TileLang" and routes the indexer to the torch fallback.
That fallback gathers the KV once per query row, so a 512-row prefill chunk
against a 262K context materializes ~18 GB and dies around 20K tokens.

The comment may be an assumption rather than a measurement. TileLang JIT-compiles
for the target architecture (unlike the AOT sgl_kernel wheel, which we already
confirmed has no sm_75 cubin), so check it directly: if it compiles and runs
here, the indexer problem is a flag, not a rewrite.

Cache layout matters and was read off the consumer rather than guessed. Both the
torch fallback and the kernel flatten a block to block_size*(head_dim+4) bytes
and then split at SCALE_OFFSET = block_size*head_dim, so a block is
[all 64 payloads][all 64 fp32 scales] -- NOT payload/scale interleaved per
position. The first version of this test interleaved them and compared against
nothing meaningful.
"""
import os
import sys
import traceback

import torch


def _find_repo():
    import pathlib

    r = os.environ.get("SGLANG_REPO")
    if r:
        return r
    d = pathlib.Path(__file__).resolve().parent
    for _ in range(8):
        for cand in (d, d / "sglang"):
            if (cand / "python" / "sglang").is_dir():
                return str(cand)
        d = d.parent
    raise RuntimeError("set SGLANG_REPO")


sys.path.insert(0, _find_repo() + "/python")

torch.cuda.set_device(0)
dev = "cuda:0"
print(f"device: {torch.cuda.get_device_properties(0).name}")

FP8 = torch.float8_e4m3fn
BLOCK, HEAD, HEADS = 64, 128, 64

batch, num_blocks = 8, 4
seq_len = num_blocks * BLOCK
num_phys = batch * num_blocks + 8

g = torch.Generator(device=dev).manual_seed(0)

# Per block: BLOCK*HEAD payload bytes, then BLOCK*4 scale bytes.
payload = (torch.randn(num_phys, BLOCK, HEAD, generator=g, device=dev) * 0.3
           ).to(FP8).view(torch.uint8)
scale_f32 = (torch.rand(num_phys, BLOCK, 1, generator=g, device=dev) * 0.02 + 0.01)
scales = scale_f32.view(torch.uint8).reshape(num_phys, BLOCK * 4)
flat = torch.cat([payload.reshape(num_phys, BLOCK * HEAD), scales], dim=1)
# The consumer's view: [num_phys, block_size, 1, head_dim + 4].
cache = flat.view(num_phys, BLOCK, 1, HEAD + 4)

q = (torch.randn(batch, 1, HEADS, HEAD, generator=g, device=dev) * 0.3).to(FP8)
weight = torch.rand(batch, HEADS, generator=g, device=dev)
seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=dev)
page_table = torch.arange(batch * num_blocks, dtype=torch.int32,
                          device=dev).view(batch, num_blocks)
max_seq_len = seq_len


def reference():
    """The torch fallback's math, spelled out, as ground truth."""
    # Each gathered row is one BLOCK of 64*132 bytes: 64*128 payloads then 64
    # fp32 scales. Slice on uint8, make contiguous, then reinterpret.
    blocks = flat[page_table.reshape(-1)]                    # [b*nb, 8448]
    pb = blocks[:, : BLOCK * HEAD].contiguous().view(FP8)
    p = pb.view(batch, seq_len, HEAD).to(torch.bfloat16)
    sb = blocks[:, BLOCK * HEAD:].contiguous().view(torch.float32)
    sc = sb.view(batch, seq_len)
    qf = q[:, 0].to(torch.bfloat16)                          # [b, H, D]
    s = torch.bmm(p, qf.transpose(1, 2))                    # [b, S, H]
    s = torch.relu(s) * weight.unsqueeze(1)
    return s.sum(dim=2) * sc


ref = reference()
print(f"reference {tuple(ref.shape)} max|ref|={ref.abs().max():.3f} "
      f"finite={torch.isfinite(ref).all().item()}")

try:
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
        tilelang_fp8_paged_mqa_logits,
    )
    print("import: OK")
except Exception as e:
    print(f"import: FAILED {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit(1)

try:
    out = tilelang_fp8_paged_mqa_logits(
        q, cache, weight, seq_lens, page_table, None, max_seq_len, False,
    )
    torch.cuda.synchronize()
except Exception:
    print("launch: FAILED")
    traceback.print_exc()
    print("\nConclusion: TileLang does not run on this architecture; the hook's "
          "assumption is confirmed and the indexer needs a different "
          "formulation.")
    raise SystemExit(1)

print(f"launch: OK, out {tuple(out.shape)} {out.dtype}")
o = out[:, :seq_len].float()
rel = (o - ref.float()).abs().max().item() / max(ref.abs().max().item(), 1e-9)
print(f"correctness: rel={rel:.2e} {'PASS' if rel < 5e-2 else 'FAIL'}")
raise SystemExit(0 if rel < 5e-2 else 1)
