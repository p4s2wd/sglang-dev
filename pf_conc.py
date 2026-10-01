"""Aggregate prefill throughput with N prompts in flight.

The prefill profile says one PP stage is only 30% busy, and 98% of the idle sits
in three gaps of 1506/1178/500 ms, all immediately before or after an nccl
SendRecv -- i.e. the stage waiting for proxy tensors from the stage before it.
Excluding those, the stage runs at 363 ms per 512-token chunk, which would be
1410 tok/s, against 880 measured. So the gap is pipeline bubble, not kernel
speed, and the question is whether more than one request in flight fills it:
with PP=4 a single request leaves three stages idle behind it, while two
interleaved requests can keep all four fed.

The user capped concurrency at 1-2 requests, so this measures 1 and 2, not a
sweep. What matters is aggregate tokens/s across the concurrent requests: if two
requests together beat one, the pipe was underfilled and the fix is scheduling,
not kernels.
"""
import json, sys, time, urllib.request

WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 13000
NS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "2"])]


def post(path, payload, timeout=1200):
    req = urllib.request.Request("http://127.0.0.1:30000" + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def one(seed):
    txt = "s%d " % seed + " ".join(["word"] * WORDS)
    t0 = time.time()
    o = post("/generate", {"text": txt,
                           "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
    return time.time() - t0, o["meta_info"]["prompt_tokens"]


# warm the shape
one(0)

import concurrent.futures as cf
for n in NS:
    best = None
    for trial in range(3):
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            t0 = time.time()
            res = list(ex.map(one, [trial * 10 + i for i in range(n)]))
            wall = time.time() - t0
        toks = sum(r[1] for r in res)
        rate = toks / wall
        if best is None or rate > best:
            best = rate
        print("  n=%d trial%d: %6.1f tok/s aggregate (%d tokens, %.1f s), per-req %.1f"
              % (n, trial, rate, toks, wall, toks / n / wall), flush=True)
    print("n=%d BEST %.1f tok/s aggregate" % (n, best), flush=True)
