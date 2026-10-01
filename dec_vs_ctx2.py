"""Decode throughput as a function of context length, measured through the radix cache.

The first attempt subtracted a prefill-only call from a prefill+decode call, but the
second call's prefill is served from the radix cache, so it is faster than the first and
the difference came out negative.

Use that instead: send the prompt once to populate the cache, then send it again with
max_new_tokens=1 to price the cached-prefill overhead, then again with N new tokens. The
difference of those two cached calls is the decode window for N-1 tokens.

Also cap the prompt by measured prompt_tokens rather than a word count, since the word
to token ratio here is not 1:1 and an over-long prompt is rejected with HTTP 400.
"""
import json, random, time, urllib.request

URL = "http://127.0.0.1:30000"
MAX_CTX = 236800


def post(payload):
    r = urllib.request.Request(URL + "/generate", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=3000).read().decode())


WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
         "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"]


def filler(seed, n):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(n)) + " end%d" % seed


def timed(payload, reps=3):
    best = None
    for _ in range(reps):
        t0 = time.time()
        post(payload)
        dt = time.time() - t0
        best = dt if best is None else min(best, dt)
    return best


print("%10s %9s %9s %9s" % ("prompt_tok", "cached_pf", "decode_ms", "decode_t/s"))
for words in (3000, 12000, 45000, 90000, 150000, 200000):
    txt = filler(words, words)
    try:
        j = post({"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        pt = j["meta_info"]["prompt_tokens"]
        if pt + 40 > MAX_CTX:
            print("%10d  skip (prompt %d exceeds budget)" % (words, pt))
            continue
        # warm again so both timed calls are fully cached
        post({"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        t1 = timed({"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        N = 33
        t2 = timed({"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": N}})
        dec = (t2 - t1) * 1e3 / (N - 1)
        print("%10d %9.1f %9.2f %9.2f" % (pt, t1 * 1e3, dec, 1000.0 / dec))
    except Exception as e:
        print("%10d  FAIL %s" % (words, str(e)[:60]))
