"""Correctness across seq_len bucket boundaries.

Bucketing captures one decode graph per KV width and picks between them on the
batch's longest sequence. The risk is a request that crosses a bucket boundary
mid-decode: it starts in the 2048 graph and, once its KV passes 2048, must move
to a wider one. If the dispatch or the metadata cache were wrong, attention would
read a page table narrower than the sequence and silently drop context.

Test: run the same greedy prompt long enough to cross a boundary, and compare the
first N tokens against a run of the identical prompt on a server with bucketing
off. Any divergence past the boundary means the wide bucket is not seeing the
full context.

Usage: python capture_long.py <out.json> <words> <newtok>
"""
import json
import sys
import urllib.request

words, newtok, out_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
# A filler that is not repeated verbatim, so the prefix cache cannot serve the
# prompt from a previous request and the comparison actually runs the model.
import hashlib

filler = " ".join(
    f"item{i}:{hashlib.md5(str(i).encode()).hexdigest()[:6]}" for i in range(words)
)
prompt = (
    f"Below is a list of coded values: {filler}. "
    "Repeat the last three coded values exactly, then stop.\n"
)

req = urllib.request.Request(
    "http://127.0.0.1:30000/generate",
    data=json.dumps({
        "text": prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": newtok},
    }).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=1800) as r:
    out = json.loads(r.read())
info = out["meta_info"]
json.dump({"prompt_tokens": info["prompt_tokens"],
           "completion_tokens": info["completion_tokens"],
           "text": out["text"]},
          open(out_path, "w"))
print(f"prompt={info['prompt_tokens']} new={info['completion_tokens']} -> {out_path}")
