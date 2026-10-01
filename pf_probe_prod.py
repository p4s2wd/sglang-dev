#!/usr/bin/env python
"""Prefill throughput probe against a running server, cache-defeated, median-of-N.

Why this is not `probe_prefill_rate2.py`: that one is hardcoded to :30000 (the
dev server, now dead) and reads the server's own "input throughput (token/s)"
line, which the scheduler computes per chunk and therefore includes inter-chunk
pipeline gaps. Wall clock around the request is what the caller actually waits
for, and that is the number the prefill target has to be stated against.

Two measurement traps this deliberately avoids:

  1. Radix prefix cache. A repeated prompt is a cache hit and reports thousands
     of tok/s for work that never happened. Every prompt here carries a fresh
     uuid4 token at position 0, so no two runs share a prefix.
  2. Thermal / clock drift. This box loses ~6% across a session. Rounds are
     therefore interleaved across lengths and we report the MEDIAN per length,
     not a single sequential pass.

Usage:
  pf_probe_prod.py                      # default ladder, port 8200
  pf_probe_prod.py --port 30000
  pf_probe_prod.py --lens 4000,13000 --rounds 3
"""
import argparse
import json
import random
import statistics
import time
import urllib.request
import uuid

# ~1.3 tokens per "word" for this tokenizer; lengths below are TOKEN targets
# and get converted to a word count that lands near them.
FILLER = (
    "The quick brown fox jumps over the lazy dog while the committee "
    "reviews the annual report and the engineers calibrate the sensor. "
)


def post(url, payload, timeout=3600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, time.perf_counter() - t0


def make_text(approx_tokens, salt):
    # ~17 tokens per FILLER repetition; pad with single tokens to land close.
    reps = max(1, approx_tokens // 17)
    return f"{salt} " + " ".join([FILLER] * reps)


def one_round(url, approx_tokens, timeout):
    salt = uuid.uuid4().hex
    out, dt = post(url, {
        "text": make_text(approx_tokens, salt),
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
    }, timeout=timeout)
    ntok = out["meta_info"]["prompt_tokens"]
    return ntok, dt, ntok / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--lens", default="2000,4000,8000,13000,24000",
                    help="comma separated TOKEN targets")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    url = f"http://{a.host}:{a.port}/generate"
    lens = [int(x) for x in a.lens.split(",")]

    if not a.no_warmup:
        # move Triton JIT + allocator warmup out of every measurement
        n, dt, r = one_round(url, 600, a.timeout)
        print(f"warmup: {n} tok in {dt:.1f}s ({r:.0f} tok/s)", flush=True)

    results = {L: [] for L in lens}
    # interleave so thermal drift hits all lengths equally
    for rnd in range(a.rounds):
        for L in lens:
            n, dt, r = one_round(url, L, a.timeout)
            results[L].append(r)
            print(f"  round {rnd} len~{L:>6} -> {n:>6} tok  {dt:7.2f}s  "
                  f"{r:7.1f} tok/s", flush=True)

    print()
    print("%9s %9s %9s %9s %9s" % ("target", "median", "min", "max", "spread"))
    summary = {}
    for L in lens:
        v = sorted(results[L])
        med = statistics.median(v)
        spread = (v[-1] - v[0]) / med * 100 if med else 0.0
        summary[L] = {"median": med, "min": v[0], "max": v[-1], "all": v}
        print("%9d %9.1f %9.1f %9.1f %8.1f%%"
              % (L, med, v[0], v[-1], spread))

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"port": a.port, "rounds": a.rounds,
                       "summary": summary}, f, indent=2)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
