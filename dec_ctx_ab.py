"""Decode throughput at a topk-saturating context, for the split A/B.

topk saturates past ~65536 prompt tokens (the c128 compressed cache exceeds
index_topk=512), so 86K is the shortest context where every attention call gathers a
full 512 tiles -- the shape the split is meant to shorten. Prefill there is ~115 s
rather than the 811 s a 216K prompt costs.

Decode window = (cached prefill + 41 tokens) - (cached prefill + 1 token), which cancels
the cached prefill and queue overhead. The server's own Decode batch log line confirms
the context really was loaded.
"""
import json, random, time, urllib.request

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


txt = filler(424242, 76000)
try:
    post({"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
    post({"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
    res = []
    for rep in range(3):
        t0 = time.time()
        post({"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        c1 = time.time() - t0
        t0 = time.time()
        post({"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 41}})
        c2 = time.time() - t0
        res.append((c2 - c1) * 1e3 / 40)
    res.sort()
    print("ctx: decode %.2f ms/token -> %.2f tok/s  (median of 3: %s)"
          % (res[1], 1000.0 / res[1], ["%.2f" % (1000.0 / x) for x in res]), flush=True)
except Exception as e:
    print("FAIL %s" % str(e)[:90], flush=True)
