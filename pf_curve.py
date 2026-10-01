"""Honest prefill throughput curve vs prompt length, interleaved for thermal drift.

Two things were conflated in every prefill number quoted so far. First, the probe
asks for 14000 filler WORDS, which tokenizes to ~24600 tokens, so "854 tok/s" was
a 24.6K-token measurement, not a 14K one. Second, the rate genuinely falls with
length: attention gathers topk=512 entries per query token from the KV pool, so a
long prompt spends proportionally more of its chunks gathering from a large,
scattered pool instead of a small L2-resident one.

A single number is meaningless without its length, so the >=1000 tok/s target is
only answerable against the lengths actually run. This sweeps lengths in an
interleaved order -- every length gets an equal share of early and late positions
in the session -- so the machine's ~6% per-session thermal drift cannot masquerade
as a length effect, and reports medians.
"""
import json, sys, time, urllib.request, uuid
from collections import defaultdict

FILLER = ("The quick brown fox jumps over the lazy dog while the committee "
          "reviews the annual report and the engineers calibrate the sensor. ")
TARGETS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else
                            ["4000", "8000", "12000", "16000", "24000"])]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def post(text, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:30000/generate",
                                 data=json.dumps({"text": text, "sampling_params":
                                     {"temperature": 0.0, "max_new_tokens": 1}}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    j = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    m = j["meta_info"]
    return m["prompt_tokens"], m.get("cached_tokens", 0), time.perf_counter() - t0


def make(ntok):
    reps = max(1, ntok // 30)
    return " ".join(["%s %s" % (uuid.uuid4().hex[:6], FILLER) for _ in range(reps)])


post(make(500))
res = defaultdict(list)
real = defaultdict(list)
for r in range(ROUNDS):
    for n in TARGETS:
        tok, cached, dt = post(make(n))
        res[n].append(tok / dt)
        real[n].append(tok)
        print("  round%d target%6d -> %6d tok  %6.1f tok/s (cached %d)"
              % (r, n, tok, tok / dt, cached), flush=True)

print("\nmedian prefill throughput by prompt length (natural text, cache defeated)")
print("%10s %9s %9s %9s %9s" % ("target", "actual", "median", "best", "worst"))
for n in TARGETS:
    v = sorted(res[n])
    print("%10d %9d %9.1f %9.1f %9.1f" % (n, int(sum(real[n]) / len(real[n])),
                                          v[len(v) // 2], v[-1], v[0]))
