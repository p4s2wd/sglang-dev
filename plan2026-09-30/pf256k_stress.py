#!/usr/bin/env python
"""256K prefill reliability, measured the way it actually failed.

The original OOM did not happen on a fresh process. It happened after the
server had already served GSM8K, batch-scaling sweeps and parity runs, and
fragmentation accumulates with a process's lifetime -- the caching allocator
retains freed blocks in whatever size they were first used at. So a 256K prefill
on a freshly started server is not the test; the test is a 256K prefill after
the server has been used.

This therefore fragments first, on purpose, with a mix that produces many
different allocation sizes (varied prompt lengths, batch sizes 1 through 8,
decode-heavy and prefill-heavy), and only then sends cold 256K prompts.

Cold means a nonce every 20 words. Without it the radix cache absorbs the
prefix and reports 4000-5000 tok/s, which is a cache hit wearing a prefill's
numbers -- the first version of this test did exactly that and nearly passed.

Usage: pf256k_stress.py [--rounds 3] [--port 8200]
"""
import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# One group = 20 reps of the unit plus a nonce, measured at 104.99 tokens.
TOK_PER_GROUP = 104.99
MAX_CONTEXT = 262144
# The nonce is a variable-width random integer, so tokens-per-group is not
# constant -- measured 104.99 and 106.00 on the same filler. Predicting it was
# wrong twice, so this does not predict: on a 400 it shrinks and retries, and
# reports whatever the server says the prompt actually was.
TOK_PER_GROUP_EST = 106.0


def post(url, body, timeout=1500):
    req = urllib.request.Request(
        url + "/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def build(rng, target_tokens):
    groups = max(1, int(target_tokens / TOK_PER_GROUP_EST))
    unit = "history of the roman empire "
    return "".join(unit * 20 + f" n{rng.randrange(10 ** 12)} "
                   for _ in range(groups))


def call(url, rng, target_tokens, newtok):
    return post(url, {"text": build(rng, target_tokens), "sampling_params": {
        "temperature": 0.0, "max_new_tokens": newtok, "ignore_eos": True}})


def cold_prefill(url, rng, target, newtok, tries=4):
    """Returns (ok, prompt_tokens, wall, note). Shrinks on a length rejection."""
    for attempt in range(tries):
        t0 = time.time()
        try:
            d = call(url, rng, target, newtok)
            return True, d["meta_info"].get("prompt_tokens"), time.time() - t0, ""
        except urllib.error.HTTPError as e:
            if e.code != 400 or attempt == tries - 1:
                return False, None, time.time() - t0, f"HTTP {e.code}"
            target = int(target * 0.97)
        except Exception as e:
            return False, None, time.time() - t0, f"{type(e).__name__}: {e}"[:120]
    return False, None, 0.0, "unreachable"


def fragment(url, rng, minutes=6):
    """Mixed load sized to make the allocator retain many block sizes."""
    t_end = time.time() + minutes * 60
    n = 0
    print("  fragmenting:", flush=True)
    while time.time() < t_end:
        bs = rng.choice([1, 1, 2, 4, 8])
        # Vary the length widely: 200 to 40k tokens, so allocation sizes differ
        # by orders of magnitude and freed blocks cannot be coalesced.
        tgt = rng.choice([200, 900, 4000, 16000, 40000])
        ntok = rng.choice([1, 16, 128, 400])
        with ThreadPoolExecutor(max_workers=bs) as ex:
            list(ex.map(lambda _: call(url, rng, tgt, ntok), range(bs)))
        n += bs
    print(f"  fragmented with {n} requests over {minutes} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--fragment-min", type=float, default=6.0)
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"

    print(f"256K prefill reliability: {args.rounds} cold prefill(s), each "
          f"after {args.fragment_min} min of mixed load\n")
    results = []
    for r in range(args.rounds):
        rng = random.Random(90210 + r * 7919)
        fragment(url, rng, args.fragment_min)
        # As close to the ceiling as a request may legally sit.
        target = 258000 + r * 1000
        newtok = 800
        ok, pt, dt, note = cold_prefill(url, rng, target, newtok)
        if ok:
            results.append((r, "PASS", pt, dt, pt / dt))
            print(f"[round {r}] PASS prompt={pt} +{newtok} = {pt + newtok} tok "
                  f"wall={dt:.0f}s prefill={pt / dt:.0f} tok/s", flush=True)
        else:
            results.append((r, "FAIL", None, dt, None))
            print(f"[round {r}] FAIL {note}", flush=True)

    npass = sum(1 for x in results if x[1] == "PASS")
    print(f"\n{npass}/{len(results)} 通过 "
          f"(全部在 {args.fragment_min} 分钟混合负载之后)")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())