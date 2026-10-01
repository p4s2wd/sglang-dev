#!/usr/bin/env python
"""Decode ms/token as a function of context length.

Why this needs its own probe: every decode number in the earlier investigation
came from a one-line prompt, where the compressed KV cache is nearly empty and
each attention call gathers only a handful of topk tiles. The operating point
that actually hurts is a long session, and there the selection stage saturates
-- so short-context decode is the wrong place to look for the cause.

Method (borrowed from the earlier dec_ctx3.py): prefill is far too expensive to
include (a 128K prompt at ~1300 tok/s is ~100 s), so price only the decode
window. Two cache-warm calls, one generating N tokens and one generating 1,
differenced. The cached prefill and the queue overhead cancel; what is left is
N-1 decode steps. That also means a warm radix cache is required, hence the
throwaway warm call before timing.

Reports ms/token and tok/s per context point, plus the spread across repeated
decode windows, because the failure mode being chased is a slowdown and a
single noisy sample would be indistinguishable from clock drift.

    dec_ctx_probe.py --lens 4000,16000,64000,128000 [--port 8200] [--n 41]
"""
import argparse
import json
import random
import statistics
import time
import urllib.request

WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
]


def post(url, payload, timeout):
    req = urllib.request.Request(
        url + "/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def filler(seed, n_words):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(n_words)) + f" end{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="2000,8000,32000,64000,128000",
                    help="comma-separated target prompt token counts")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--n", type=int, default=41, help="tokens in the timed call")
    ap.add_argument("--reps", type=int, default=3, help="decode windows per point")
    ap.add_argument("--timeout", type=int, default=2500)
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    # The filler is ~1 token per word for this vocabulary, so seed with the
    # target and correct using the prompt_tokens the server actually reports.
    print("%9s %9s %11s %10s %9s %s" % (
        "want_tok", "got_tok", "ms/token", "tok/s", "spread", "note"))
    out = []
    for want in (int(x) for x in args.lens.split(",")):
        txt = filler(want, want)
        try:
            t0 = time.time()
            j = post(url, {"text": txt,
                           "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}},
                     args.timeout)
            got = j["meta_info"]["prompt_tokens"]
            uncached_s = time.time() - t0

            # Warm the radix cache so the timed calls skip prefill entirely.
            post(url, {"text": txt,
                       "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}},
                 args.timeout)

            samples = []
            for _ in range(args.reps):
                t1 = time.time()
                post(url, {"text": txt,
                           "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}},
                     args.timeout)
                base = time.time() - t1
                t2 = time.time()
                post(url, {"text": txt,
                           "sampling_params": {"temperature": 0.0,
                                               "max_new_tokens": args.n}},
                     args.timeout)
                samples.append((time.time() - t2 - base) * 1e3 / (args.n - 1))

            med = statistics.median(samples)
            spread = (max(samples) - min(samples)) / med * 100 if med else 0.0
            print("%9d %9d %11.2f %10.2f %8.1f%%  uncached prefill %.1fs" % (
                want, got, med, 1000.0 / med, spread, uncached_s), flush=True)
            out.append({"want": want, "got": got, "ms_per_token": med,
                        "tok_s": 1000.0 / med, "spread_pct": spread,
                        "samples": samples})
        except Exception as e:
            print("%9d  FAIL %s" % (want, str(e)[:80]), flush=True)
            out.append({"want": want, "error": str(e)[:200]})

    print(json.dumps(out))


if __name__ == "__main__":
    main()
