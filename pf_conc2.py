"""Prefill throughput vs concurrency, with the prefix cache defeated per request.

Reinterpreting the prefill trace changed the diagnosis. Stage PP3 logged 88
headshared calls; the stage owns 11 of the 43 layers, so that is 8 chunks, not 4.
Device time is therefore 1379/8 = 172 ms per 512-token chunk, while wall time is
598 ms per chunk -- the stage is 29% busy, matching the measured 30%. If the four
pipeline stages overlapped, wall per chunk would be about one stage's 172 ms;
598 ms is close to 4 x 172 = 688 ms, i.e. the chunks are moving through PP0, PP1,
PP2, PP3 one at a time with no pipelining.

That is what a single request does through a 4-stage pipeline: there is only one
microbatch, so nothing to interleave. The earlier concurrency test that showed no
gain was invalid -- it reused seeds, so trials 1 and 2 were largely prefix-cache
hits (trial 0 reported 11408 tok/s, which is a cache hit, not a measurement).
This version puts a unique uuid in every request so nothing is served from cache,
and reports AGGREGATE tokens/s, which is what a filled pipeline should raise. The
user capped concurrency at 1-2 requests, so 1, 2 and 3 is the whole relevant
range.
"""
import json, sys, time, urllib.request, uuid
import concurrent.futures as cf

WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 13000
NS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "2", "3"])]
FILLER = ("The quick brown fox jumps over the lazy dog while the committee "
          "reviews the annual report and the engineers calibrate the sensor. ")


def make(seed):
    # Unique uuid per repetition: no two requests share a prefix, so the radix
    # cache cannot serve any of this from an earlier trial.
    reps = max(1, WORDS // 30)
    return " ".join(["%s-%s %s" % (seed, uuid.uuid4().hex[:6], FILLER) for _ in range(reps)])


def post(text, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:30000/generate",
                                 data=json.dumps({"text": text, "sampling_params":
                                     {"temperature": 0.0, "max_new_tokens": 1}}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    j = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    return j["meta_info"]["prompt_tokens"], time.perf_counter() - t0


post(make("warm"))
print("aggregate prefill tokens/s, prefix cache defeated on every request")
for n in NS:
    best = 0.0
    for trial in range(2):
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            t0 = time.perf_counter()
            res = list(ex.map(lambda i: post(make("t%d-%d" % (trial, i))), range(n)))
            wall = time.perf_counter() - t0
        toks = sum(r[0] for r in res)
        rate = toks / wall
        best = max(best, rate)
        print("  n=%d trial%d: %6.1f agg tok/s  (%d tok each, wall %.1f s, per-req %.1f)"
              % (n, trial, rate, res[0][0], wall, toks / n / wall), flush=True)
    print("n=%d BEST aggregate %.1f tok/s" % (n, best), flush=True)
