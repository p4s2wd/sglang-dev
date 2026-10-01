#!/bin/bash
# Find the prompt length at which the real-weight server starts failing, to
# separate a genuine capacity limit from a profiler-only artifact.
#
# The torch sparse-MLA fallback materializes (s_q x topk x head_dim) fp32
# intermediates inside _gather_and_dequant, so the peak scales with the number of
# query tokens in the batch -- a short prompt fits, a long one does not.
#
# Usage: ./probe_len_limit.sh [max_tokens_to_try]
set -u
PORT="${PORT:-30000}"
PY=/data/nvme/sglang-codex/.venv/bin/python

"$PY" - "$PORT" <<'PY'
import json, sys, time
import urllib.request

port = sys.argv[1]
URL = f"http://127.0.0.1:{port}/generate"


def call(text, max_new=8):
    body = json.dumps({
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    return out, time.perf_counter() - t0


print(f"{'prompt_tok':>10} {'wall_s':>8}  result")
words = "history of the roman empire and its provinces "
last_ok = 0
for n_words in (4, 16, 32, 64, 128, 192, 256, 320, 448, 640, 900, 1200):
    prompt = words * n_words
    try:
        out, dt = call(prompt)
        pt = out["meta_info"]["prompt_tokens"]
        last_ok = pt
        print(f"{pt:>10} {dt:>8.2f}  ok")
    except Exception as e:
        print(f"{n_words*9:>10} {'-':>8}  FAILED: {type(e).__name__}: "
              f"{str(e)[:70]}")
        print(f"limit: last working prompt was {last_ok} tokens")
        break
else:
    print(f"no failure up to {last_ok} tokens")
PY
