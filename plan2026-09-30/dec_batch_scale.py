#!/usr/bin/env python
"""Decode throughput vs batch size, to separate bandwidth from overhead.

At bs=1 this box does about 27 tok/s on a short prompt, which is far enough
below what the hardware ought to manage that the cause matters. Two very
different answers are consistent with that number:

  weight-bandwidth bound. Every token re-reads every weight, so the step costs
  the same regardless of batch and throughput scales with batch size. Nothing
  to fix in the kernels; the only lever is reading fewer bytes per token.

  latency bound. A 4-stage pipeline passing one token's activation between
  stages, plus a 2-way all-reduce inside each stage, is mostly pure latency at
  bs=1 -- there is almost no data to move. Batch size amortises it and
  throughput climbs faster than the arithmetic.

The measurement needs no restart and no profiler, so it is not distorted by
either. All requests share one prompt so prefill is paid once and the radix
cache keeps it resident; what varies is only how many decode steps run at once.

Usage: dec_batch_scale.py [--ctx-tokens 8000] [--newtok 120] [--levels 1,2,4,8]
"""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

FILLER = "history of the roman empire "


def post(url, body, timeout=3600):
    req = urllib.request.Request(
        url + "/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--ctx-tokens", type=int, default=8000)
    ap.add_argument("--newtok", type=int, default=120)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    prompt = FILLER * max(1, args.ctx_tokens // 5)

    w = post(url, {"text": prompt, "sampling_params": {
        "temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}})
    ptok = w["meta_info"]["prompt_tokens"]
    print(f"prompt_tokens={ptok}  (shared by every request, radix-cached)\n")
    print(f"{'bs':>3} {'rounds':>7} {'aggregate tok/s':>17} {'per-req tok/s':>14} "
          f"{'vs bs=1':>8}")

    base = None
    for bs in [int(x) for x in args.levels.split(",")]:
        agg = []
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=bs) as ex:
                out = list(ex.map(
                    lambda _: post(url, {"text": prompt, "sampling_params": {
                        "temperature": 0.0, "max_new_tokens": args.newtok,
                        "ignore_eos": True}}), range(bs)))
            dt = time.perf_counter() - t0
            ntok = sum(o["meta_info"]["completion_tokens"] for o in out)
            agg.append(ntok / dt)
        med = statistics.median(agg)
        if base is None:
            base = med
        print(f"{bs:>3} {len(agg):>7} {med:>17.2f} {med / bs:>14.2f} "
              f"{med / base:>7.2f}x", flush=True)


if __name__ == "__main__":
    main()