#!/usr/bin/env python
"""Decode throughput at batch size N: the test of whether 50 tok/s is reachable.

Decode is PP4-serialised per token (4 stages x 17.13 ms = 68.5 ms = 14.6 tok/s
at bs=1, matching the measured 14.9), but the per-step cost is dominated by
weight reads that do not grow with batch size. So aggregate throughput should
scale with batch. This fires N concurrent greedy requests and reports aggregate
tok/s, which is the number that matters for serving.
"""
import json, sys, time, urllib.request, threading

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
NEWTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 64
URL = "http://127.0.0.1:30000/generate"

# Warm the graph for this batch size first.
def one(i):
    body = {"text": f"Count from {i*100} upward: ",
            "sampling_params": {"temperature": 0.0, "max_new_tokens": NEWTOK}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    o = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return time.perf_counter() - t0, o["meta_info"]["completion_tokens"]

for warm in range(2):
    ts = [one(i) for i in range(N)]
    time.sleep(1)

results = []
for trial in range(3):
    out = [None] * N
    def worker(i):
        out[i] = one(i)
    th = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    t0 = time.perf_counter()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.perf_counter() - t0
    toks = sum(c for _, c in out)
    results.append((wall, toks))
    print("trial %d: %d reqs x %d tok  %.2fs  %.2f tok/s aggregate  (%.2f/req)"
          % (trial + 1, N, NEWTOK, wall, toks / wall, N * NEWTOK / wall / N * NEWTOK / NEWTOK))
best = min(r[0] for r in results)
print("BEST bs=%d: %.2f tok/s aggregate" % (N, sum(r[1] for r in results) / sum(r[0] for r in results)))
