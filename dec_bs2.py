"""Decode throughput at bs=1 and bs=2 against the bandwidth floor.

The floor computation says a single token must read 13.21 GB of weights (6.71 GB
dense, read in full every token, plus 6.49 GB of routed experts -- 6 of 256 firing
in each of 43 layers). At 616 GB/s that is 21.4 ms/token, i.e. 46.65 tok/s, so the
50 tok/s single-stream target sits ABOVE the physical floor and cannot be reached by
kernel work. Measured 15.80 is 34% of that floor.

Batching is the only way to amortize the dense half: two concurrent tokens share the
6.71 GB of dense weights but fire different experts, so two tokens cost
6.71 + 2*6.49 = 19.7 GB = 9.85 GB/token, a 62.6 tok/s aggregate ceiling. The user
capped concurrency at 1-2 requests, so bs=2 is the largest legitimate batch and the
bs=2 aggregate is the number that decides whether 50 tok/s is reachable at all.

Note MAXREQ=2 in the running config, so a probe asking for 4 concurrent requests
actually runs two waves of two and reports a meaningless average -- that is what
made an earlier bs sweep look non-monotonic. This probe requests exactly 1 and 2.
"""
import json, sys, time, urllib.request

def post(text, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:30000/generate",
                                 data=json.dumps({"text": text, "sampling_params":
                                     {"temperature": 0.0, "max_new_tokens": 96}}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    j = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    return j["meta_info"]["completion_tokens"], time.perf_counter() - t0

import concurrent.futures as cf
DENSE, EXP = 6.713, 6.493
PEAK = 616e9
print("%4s %10s %10s %12s %10s" % ("bs", "agg tok/s", "per-req", "GB/token", "of floor"))
for bs in (1, 2):
    best = 0.0
    for trial in range(3):
        with cf.ThreadPoolExecutor(max_workers=bs) as ex:
            t0 = time.perf_counter()
            res = list(ex.map(lambda i: post("count tokens %s please %d" % ("abc" * 8, i)),
                              range(bs)))
            wall = time.perf_counter() - t0
        toks = sum(r[0] for r in res)
        rate = toks / wall
        best = max(best, rate)
    gb = (DENSE + bs * EXP) / bs
    floor = PEAK / (gb * 1e9)
    print("%4d %10.2f %10.2f %12.2f %9.0f%%" % (bs, best, best / bs, gb, 100 * best / floor))
