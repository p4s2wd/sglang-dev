"""Aggregate decode at long context, sized to fit the KV pool.

The previous version asked for two 200K-word prompts, which are ~288K tokens each and
were rejected with HTTP 400. That failure is itself the finding: the pool holds 236800
tokens, so two 216K-token requests cannot coexist at all -- "256K context" and "2
concurrent requests" are mutually exclusive by construction, not by tuning.

So measure the two operating points that do fit:
  nreq=1 at ~216K tokens (the 256K target, single stream)
  nreq=2 at ~110K tokens each (2 concurrent, half the context)
and report per-request and aggregate throughput for each. Long prefills dominate the
runtime, so this runs detached.
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


def run(texts, label):
    N = 41
    try:
        for t in texts:
            post({"text": t, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        for t in texts:
            post({"text": t, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})

        def burst(n_new):
            t0 = time.time()
            th = [threading.Thread(target=post, args=({"text": t,
                  "sampling_params": {"temperature": 0.0, "max_new_tokens": n_new}},))
                  for t in texts]
            [x.start() for x in th]
            [x.join() for x in th]
            return time.time() - t0

        c1 = burst(1)
        c2 = burst(N)
        wall = (c2 - c1) * 1e3 / (N - 1)
        print("%-22s wall/step %7.2f ms  per-request %6.2f tok/s  AGGREGATE %6.2f tok/s"
              % (label, wall, 1000.0 / wall, 1000.0 / wall * len(texts)), flush=True)
    except Exception as e:
        print("%-22s FAIL %s" % (label, str(e)[:80]), flush=True)


run([filler(900001, 150000)], "1req @ ~216K")
run([filler(900002, 76000), filler(900003, 76000)], "2req @ ~110K each")
