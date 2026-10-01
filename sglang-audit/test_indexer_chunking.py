"""KV-chunked torch indexer logits must equal the unchunked math.

The fallback used to gather every page of every row at once. That peak, not the
arithmetic, is what capped this server at ~20K prompt tokens: at a 262K context
the gather and the [batch, seq, num_heads] score tile run to tens of GB. The fix
walks the row in page chunks with the head reduction inside each chunk.

Chunking along KV is exact -- output position j depends only on key j and scale
j -- so the test is an equality check, not a tolerance check, apart from the
float32 accumulation the original also used. Two configurations are run: the
default budget (one chunk, the path decode takes) and a forced 1 MiB budget that
puts each page in its own chunk, which is what a large prefill batch takes.
"""
import os
import sys

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
FP8 = torch.float8_e4m3fn
BLOCK, HEAD, HEADS = 64, 128, 64


def build(batch, num_blocks, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    num_phys = batch * num_blocks + 8
    payload = (torch.randn(num_phys, BLOCK, HEAD, generator=g, device=dev) * 0.3
               ).to(FP8).view(torch.uint8)
    scale_f32 = (torch.rand(num_phys, BLOCK, 1, generator=g, device=dev) * 0.02
                 + 0.01)
    scales = scale_f32.view(torch.uint8).reshape(num_phys, BLOCK * 4)
    flat = torch.cat([payload.reshape(num_phys, BLOCK * HEAD), scales], dim=1)
    cache = flat.view(num_phys, BLOCK, 1, HEAD + 4)

    q = (torch.randn(batch, 1, HEADS, HEAD, generator=g, device=dev) * 0.3).to(FP8)
    weight = torch.rand(batch, HEADS, generator=g, device=dev)
    # Ragged rows, so the mask path is exercised too.
    seq_lens = torch.randint(BLOCK, num_blocks * BLOCK + 1, (batch,),
                             dtype=torch.int32, device=dev)
    page_table = torch.arange(batch * num_blocks, dtype=torch.int32,
                              device=dev).view(batch, num_blocks)
    return flat, cache, q, weight, seq_lens, page_table


def reference(flat, q, weight, seq_lens, page_table, num_blocks):
    """The original unchunked math, spelled out independently."""
    batch = q.shape[0]
    blocks = flat[page_table.reshape(-1)]
    pb = blocks[:, : BLOCK * HEAD].contiguous().view(FP8)
    p = pb.view(batch, num_blocks * BLOCK, HEAD).to(torch.bfloat16)
    sb = blocks[:, BLOCK * HEAD:].contiguous().view(torch.float32)
    sc = sb.view(batch, num_blocks * BLOCK)
    qf = q[:, 0].to(torch.bfloat16)
    s = torch.bmm(p, qf.transpose(1, 2))
    s = torch.relu(s) * weight.unsqueeze(1)
    s = s.sum(dim=2) * sc
    pos = torch.arange(num_blocks * BLOCK, device=dev).unsqueeze(0)
    return s.masked_fill(pos >= seq_lens.unsqueeze(1), 0.0)


ok = True
for batch, num_blocks in ((8, 4), (64, 6)):
    flat, cache, q, weight, seq_lens, page_table = build(batch, num_blocks)
    max_seq_len = num_blocks * BLOCK
    ref = reference(flat, q, weight, seq_lens, page_table, num_blocks)

    from sglang.srt.layers.attention.dsv4.indexer import fp8_paged_mqa_logits_torch

    for tag in ("default budget", "forced 1 MiB budget"):
        if tag.startswith("forced"):
            os.environ["SGLANG_INDEXER_LOGITS_KV_CHUNK_MB"] = "1"
        else:
            os.environ.pop("SGLANG_INDEXER_LOGITS_KV_CHUNK_MB", None)
        # Re-import so the module-level budget constant is recomputed.
        import importlib
        import sglang.srt.layers.attention.dsv4.indexer as idx
        importlib.reload(idx)

        out = idx.fp8_paged_mqa_logits_torch(
            q, cache, weight, seq_lens, page_table, None, max_seq_len, False,
        )
        torch.cuda.synchronize()
        o = out[:, :max_seq_len].float()
        rel = (o - ref.float()).abs().max().item() / max(ref.abs().max().item(), 1e-9)
        good = rel < 1e-5
        ok = ok and good
        print(f"batch={batch:<3} blocks={num_blocks} {tag:<20} rel={rel:.2e} "
              f"{'ok' if good else 'FAIL'}")

os.environ.pop("SGLANG_INDEXER_LOGITS_KV_CHUNK_MB", None)
print("\nINDEXER KV CHUNKING:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
