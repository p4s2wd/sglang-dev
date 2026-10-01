"""Prefill throughput probe that cannot be fooled by the radix prefix cache.

The original probe sends a fixed prompt, so a second run on the same server is a
prefix-cache hit and reports thousands of tok/s for work that never happened.
Every prompt here carries a fresh salt, and a warmup prompt (also salted) runs
first so Triton JIT cost lands outside the measurement.
"""
import json
import sys
import time
import urllib.request
import uuid

URL = "http://127.0.0.1:30000/generate"
WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
LABEL = sys.argv[2] if len(sys.argv) > 2 else ""
WARMUP = "--no-warmup" not in sys.argv

FILLER = ("The quick brown fox jumps over the lazy dog while the committee "
          "reviews the annual report and the engineers calibrate the sensor. ")


def post(text):
    body = json.dumps({
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        j = json.load(r)
    dt = time.perf_counter() - t0
    return j.get("meta_info", {}).get("prompt_tokens"), dt


def make_text(words):
    # A unique leading token defeats the prefix cache: no two runs share a prefix.
    return " ".join([f"{uuid.uuid4().hex[:8]} {FILLER}" for _ in range(words // 17 + 1)])


if WARMUP:
    post(make_text(600))

n, dt = post(make_text(WORDS))
n = n or int(WORDS * 1.3)
print(f"{LABEL:22s} prompt_tokens={n:7d}  wall={dt:7.2f}s  prefill={n / dt:8.1f} tok/s")
