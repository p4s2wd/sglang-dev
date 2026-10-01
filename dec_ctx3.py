"""Decode throughput vs context length, trimmed to fit the time budget.

Long prefills dominate the cost (a 128K prompt at ~1300 tok/s is 100 s per uncached
call), so: three context points, one warm call each, and price the decode window as the
difference between two cache-warm calls (N tokens minus 1 token). The cached prefill and
queue overhead cancel; what is left is N-1 decode steps.

This matters because every decode number so far came from a one-line prompt, where the
compressed cache is nearly empty and each attention call gathers a small topk. At 256K
the c128 cache holds 2048 tokens and index_topk=512 saturates, so the real target
operating point may be slower than the 17.27 tok/s measured at short context.
"""
import json, random, sys, time, urllib.request

URL = "http://127.0.0.1:30000"
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
         "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"]


def post(payload, timeout=2500):
    r = urllib.request.Request(URL + "/generate", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read().decode())


def filler(seed, n):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(n)) + " end%d" % seed


print("%10s %10s %9s %9s" % ("prompt_tok", "cached_ms", "dec_ms", "decode_t/s"), flush=True)
for words in (int(x) for x in sys.argv[1].split(",")):
    txt = filler(words, words)
    try:
        t0 = time.time()
        j = post({"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        pt = j["meta_info"]["prompt_tokens"]
        warm = time.time() - t0
        post({"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        t1 = time.time()
        post({"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        c1 = time.time() - t1
        N = 41
        t2 = time.time()
        post({"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": N}})
        c2 = time.time() - t2
        dec = (c2 - c1) * 1e3 / (N - 1)
        print("%10d %10.1f %9.2f %9.2f  (uncached prefill %.1fs)"
              % (pt, c1 * 1e3, dec, 1000.0 / dec, warm), flush=True)
    except Exception as e:
        print("%10d  FAIL %s" % (words, str(e)[:70]), flush=True)
