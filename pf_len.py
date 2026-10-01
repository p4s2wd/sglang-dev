"""Prefill throughput vs prompt length, cache defeated on every request.

Two probes disagreed: one reported ~850 tok/s, the other ~1000. Both were
correct -- the first sends 14000 filler WORDS, which tokenizes to ~24600 tokens,
while the second sends 13000 tokens. So the rate depends on prompt length, and
the single number I have been quoting is meaningless without saying which length
it belongs to.

The mechanism is already understood: attention gathers topk=512 entries from the
growing KV pool, so early chunks of a prompt gather from a nearly-empty,
L2-resident pool while late chunks gather 512 scattered entries from a pool that
is tens of megabytes. Longer prompts spend proportionally more time in the slow
regime. That is inherent to sparse attention over a growing pool, not a bug.

The user's target is >=1000 tok/s and the stated context goal is 256K, so what
matters is the rate at the lengths they will actually run. This measures the
curve with a unique uuid prefix per request so nothing is served from the prefix
cache.
"""
import json, sys, time, urllib.request, uuid

def post(text, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:30000/generate",
                                 data=json.dumps({"text": text, "sampling_params":
                                     {"temperature": 0.0, "max_new_tokens": 1}}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    j = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    return j["meta_info"]["prompt_tokens"], time.perf_counter() - t0

FILLER = ("The quick brown fox jumps over the lazy dog while the committee "
          "reviews the annual report and the engineers calibrate the sensor. ")

def make(ntok):
    # Unique uuid prefix per repetition defeats the prefix cache; ~30 tokens each.
    reps = max(1, ntok // 30)
    return " ".join(["%s %s" % (uuid.uuid4().hex[:8], FILLER) for _ in range(reps)])

post(make(400))
print("%10s %8s %9s %9s" % ("target", "tokens", "wall s", "tok/s"))
for ntok in (4000, 8000, 16000, 32000):
    rates = []
    for _ in range(1):
        try:
            n, dt = post(make(ntok))
            rates.append(n / dt)
        except Exception as e:
            print("  %d failed: %s" % (ntok, type(e).__name__)); break
    if rates:
        print("%10d %8d %9.2f %9.1f" % (ntok, n, dt, max(rates)))
