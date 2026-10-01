"""Does decode throughput depend on context length? The 17 tok/s number may be short-context.

Every decode number in this investigation came from captures driven with a one-line
prompt, so the compressed KV cache was nearly empty and each attention call gathered a
small topk. The objective's target is a 256K context, where the c128 cache holds 2048
tokens and index_topk=512 saturates -- 4x the tiles per call. If attention time scales
with topk (it does: the per-head kernel is linear in tiles), decode at 256K could be far
slower than the 17.27 tok/s measured at short context, and the topk-split idea would be
aimed at the real operating point rather than a benchmark artifact.

Measure decode tok/s for a single request as a function of prompt length, using the
server's own timing over a fixed number of generated tokens. Prefill time is excluded by
measuring only the decode window via /generate's meta_info (completion_tokens and the
e2e latency minus a prefill-only probe of the same prompt).
"""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:30000"


def post(path, payload):
    r = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=3000).read().decode())


def filler(n_words):
    # distinct text so the radix cache cannot reuse a prefix
    import random
    rnd = random.Random(n_words)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
             "hotel", "india", "juliet", "kilo", "lima", "mike", "november"]
    return " ".join(rnd.choice(words) for _ in range(n_words)) + " %d" % n_words


print("%9s %9s %9s %10s %10s" % ("prompt_tok", "prefill", "decode_ms", "decode_t/s", "note"))
for target in (2000, 8000, 32000, 64000, 128000, 200000):
    txt = filler(int(target * 1.35))
    try:
        # prefill-only: 1 new token
        t0 = time.time()
        j1 = post("/generate", {"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})
        t1 = time.time() - t0
        # prefill + N decode tokens
        t0 = time.time()
        j2 = post("/generate", {"text": txt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 33}})
        t2 = time.time() - t0
        pt = j1["meta_info"].get("prompt_tokens", 0)
        ct = j2["meta_info"]["completion_tokens"]
        dec_ms = (t2 - t1) * 1e3 / max(ct - 1, 1)
        print("%9d %9d %9.1f %10.2f" % (pt, pt, dec_ms, 1000.0 / dec_ms))
    except Exception as e:
        print("%9s  FAIL %s" % (target, str(e)[:60]))
