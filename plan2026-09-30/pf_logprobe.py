#!/usr/bin/env python
"""Prefill throughput, measured as prompt tokens over wall time.

Needed because `chunked-prefill-size` is a memory knob that was set for a
reason unrelated to speed, and the only way to settle "is 256 leaving anything
on the table" is to price the thing the knob actually affects.

Each point sends a prompt that has never been seen, so the radix cache cannot
turn a prefill into a cache hit, and asks for a single token so decode does not
enter the measurement. prompt_tokens / wall is the honest prefill rate; the
log's own instantaneous `input throughput` swings by three orders of magnitude
between chunks and is useless as a summary.

Wall time includes request overhead, which is why the short-prompt column is
small and low-variance while the long ones carry real prefill.

Usage: pf_logprobe.py --lens 2000,16000,64000,128000 --out res/pf.json
"""
import argparse
import json
import os
import time
import urllib.request

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
         "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
         "oscar", "papa"]


def post(url, payload, timeout=3600):
    req = urllib.request.Request(
        url + "/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--lens", default="2000,16000,64000,128000")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"

    results = []
    # Measured on this tokenizer: one repetition of WORDS is 11.5 tokens, not
    # 5. Guessing it wrong overshoots the context limit and the server answers
    # 400, which looks like a server fault and is not one.
    TOKENS_PER_REP = 11.5
    print(f"{'want':>8} {'prompt tok':>11} {'wall s':>9} {'tok/s':>10}")
    for want in [int(x) for x in args.lens.split(",")]:
        # A nonce in the middle keeps the prompt out of the radix cache while
        # leaving its token count essentially unchanged.
        reps = max(1, int(want / TOKENS_PER_REP))
        prompt = (" ".join(WORDS) + " ") * reps
        prompt = prompt[: len(prompt) // 2] + f" zz{os.getpid()}{want} " \
            + prompt[len(prompt) // 2:]
        t0 = time.perf_counter()
        d = post(url, {"text": prompt, "sampling_params": {
            "temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}})
        dt = time.perf_counter() - t0
        n = d.get("meta_info", {}).get("prompt_tokens")
        tps = n / dt if n else None
        results.append({"want": want, "prompt_tokens": n, "seconds": round(dt, 2),
                        "tok_s": round(tps, 1) if tps else None})
        print(f"{want:>8} {str(n):>11} {dt:>9.1f} "
              f"{('%.1f' % tps) if tps else '-':>10}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()