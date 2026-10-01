"""Does the prefill benchmark's filler text inflate the measured rate?

Two probes of mine disagree at nearly the same prompt length: filler sentences at
~13K tokens measured ~815 tok/s, while "word" repeated 13000 times measured ~1000.
Same length, same server, same code path -- so the difference is the TEXT.

That matters because attention gathers topk=512 entries per query token from the
KV pool, and the indices come from the indexer. A prompt of one repeated token is
maximally degenerate: every position's topk set is nearly identical and clustered,
so the gathers hit the same pages and stay in L2. Natural text spreads them over
the pool. If that is what is happening, the repetitive probe has been flattering
every prefill number I have reported, and the honest rate is the lower one.

Both texts are measured in the same session, interleaved, with a uuid prefix so
neither is served from the radix cache.
"""
import json, time, urllib.request, uuid

FILLER = ("The quick brown fox jumps over the lazy dog while the committee "
          "reviews the annual report and the engineers calibrate the sensor. ")

def post(text, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:30000/generate",
                                 data=json.dumps({"text": text, "sampling_params":
                                     {"temperature": 0.0, "max_new_tokens": 1}}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    j = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    m = j["meta_info"]
    return m["prompt_tokens"], m.get("cached_tokens", 0), time.perf_counter() - t0

def natural(ntok):
    reps = max(1, ntok // 30)
    return " ".join(["%s %s" % (uuid.uuid4().hex[:6], FILLER) for _ in range(reps)])

def repetitive(ntok):
    # Same token count, one repeated word: degenerate indexer output.
    return uuid.uuid4().hex[:6] + " " + " ".join(["word"] * ntok)

post(natural(500))
print("interleaved, same session, uuid-prefixed so nothing is cached")
print("%-12s %8s %7s %8s %9s" % ("text", "tokens", "cached", "wall s", "tok/s"))
nat, rep = [], []
for trial in range(3):
    for kind, mk, acc in (("natural", natural, nat), ("repetitive", repetitive, rep)):
        n, cached, dt = post(mk(12000))
        acc.append(n / dt)
        print("%-12s %8d %7d %8.2f %9.1f" % (kind, n, cached, dt, n / dt), flush=True)
nat.sort(); rep.sort()
print("\nmedian natural %.1f tok/s   median repetitive %.1f tok/s   inflation %.2fx"
      % (nat[1], rep[1], rep[1] / nat[1]))
