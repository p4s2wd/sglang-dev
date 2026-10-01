"""Measure prefill-only throughput: one long prompt, max_new_tokens=1.

The scheduler logs an "input throughput (token/s)" per prefill batch, but that
number is computed per chunk and includes the gap between chunks, so it is noisy
at small chunk sizes. Wall-clock over the whole prompt is what a user waits for.

Run with the SAME prompt on two servers that differ only in chunked_prefill_size.
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:30000/generate"
# Word count target; ~1.3 tokens per word for this filler.
WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
LABEL = sys.argv[2] if len(sys.argv) > 2 else ""

filler = ("The quick brown fox jumps over the lazy dog while the committee "
          "reviews the annual report and the engineers calibrate the sensor. ")
text = " ".join([f"{i} {filler}" for i in range(WORDS // 17 + 1)])

body = json.dumps({
    "text": text,
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
}).encode()
req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})

t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=3600) as r:
    j = json.load(r)
dt = time.perf_counter() - t0
mi = j.get("meta_info", {})
# prompt_tokens is authoritative; fall back to the word estimate.
n = mi.get("prompt_tokens") or int(WORDS * 1.3)
print(f"{LABEL:22s} prompt_tokens={n:7d}  wall={dt:7.2f}s  "
      f"prefill={n / dt:8.1f} tok/s")
