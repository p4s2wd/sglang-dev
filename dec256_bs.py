"""Aggregate decode throughput at the real target: 256K context, 1 and 2 concurrent.

The user's constraint is 1-2 concurrent requests, and the target is 256K context, so the
decision-relevant number is aggregate throughput with two 216K-token requests co-terminated.
Short-context scaling (bs=2 gave 1.83x bs=1) may not hold when attention is gathering a
saturated topk=512 for every request, since the extra work is real bytes rather than
exposed latency.

Method: fill the cache with two long prompts, then drive both to completion and time the
decode window as before. Long prefills dominate (811 s for 216K), so this runs detached.
"""
import json, random, threading, time, urllib.request

URL = "http://127.0.0.1:30000"
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
         "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"]


def post(payload, timeout=4000):
    r = urllib.request.Request(URL + "/generate", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read().decode())


def filler(seed, n):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(n)) + " end%d" % seed


N = 41
for nreq in (1, 2):
    texts = [filler(900000 + i, 200000) for i in range(nreq)]
    try:
        # warm every prompt into the radix cache (slow: one long prefill each)
        for t in texts:
            post({"text": t, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        # price the cached prefill with the same concurrency
        t0 = time.time()
        th = [threading.Thread(target=post, args=({"text": t,
              "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}},)) for t in texts]
        [x.start() for x in th]; [x.join() for x in th]
        c1 = time.time() - t0
        t0 = time.time()
        th = [threading.Thread(target=post, args=({"text": t,
              "sampling_params": {"temperature": 0.0, "max_new_tokens": N}},)) for t in texts]
        [x.start() for x in th]; [x.join() for x in th]
        c2 = time.time() - t0
        wall = (c2 - c1) * 1e3
        per_req = wall / (N - 1)
        print("nreq=%d  wall/step %.2f ms  per-request %.2f tok/s  AGGREGATE %.2f tok/s"
              % (nreq, per_req, 1000.0 / per_req, 1000.0 / per_req * nreq), flush=True)
    except Exception as e:
        print("nreq=%d FAIL %s" % (nreq, str(e)[:80]), flush=True)
