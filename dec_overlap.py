"""Does decode ever reach bs=2 when two requests are truly co-terminated?

MAXREQ=8 does admit two requests (12 of 76 decode steps showed #running-req: 2), yet
aggregate throughput at "bs=2" was 30.19 tok/s -- the same as at MAXREQ=2. Either the
two requests barely overlap (so most steps are still bs=1 and the number is a mix), or
they overlap and bs=2 genuinely costs nearly 2x bs=1, which would mean batching buys
nothing and the 125 tok/s bs=2 floor is unreachable for a different reason.

Distinguish them: send N requests with IDENTICAL prompt and output length so they
start together and finish together, then read the server's own batch histogram for
that window. If steps are bs=2 throughout and throughput is still ~30, batching is
not the lever. If steps are bs=2 and throughput rises, the earlier probe was just
measuring poor overlap.
"""
import json, subprocess, sys, time, urllib.request
import concurrent.futures as cf

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 64
LOG = sys.argv[3] if len(sys.argv) > 3 else "/data/nvme/sglang-codex/logs/mr-m8.log"
# Same text for every request so the steps are the same length; a uuid keeps the
# radix cache from serving any of them as a prefix hit.
def txt(i):
    return "Explain in detail how a steam turbine converts heat into work. ref=%s" % uuid_hex(i)


def uuid_hex(i):
    import hashlib
    return hashlib.md5(("%d-%f" % (i, time.time())).encode()).hexdigest()[:10]


def post(i, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:30000/generate",
                                 data=json.dumps({"text": txt(i), "sampling_params":
                                     {"temperature": 0.0, "max_new_tokens": LEN}}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    j = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    return j["meta_info"]["completion_tokens"], time.perf_counter() - t0


# Mark the log so the histogram can be restricted to this window.
mark = time.strftime("%H:%M:%S")
post(0)  # warm
time.sleep(1)
with cf.ThreadPoolExecutor(max_workers=N) as ex:
    t0 = time.perf_counter()
    res = list(ex.map(lambda i: post(i), range(N)))
    wall = time.perf_counter() - t0
toks = sum(r[0] for r in res)
print("N=%d  tokens=%d  wall=%.2fs  aggregate=%.2f tok/s  per-req=%.2f"
      % (N, toks, wall, toks / wall, toks / N / wall))
time.sleep(2)
txt_log = subprocess.run(["grep", "-aE", "Decode batch", LOG], capture_output=True, text=True).stdout
lines = txt_log.splitlines()
win = lines[-int(toks / N * 1.6) - 4:] if lines else []
import re
from collections import Counter
c = Counter()
for L in win:
    m = re.search(r"#running-req: (\d+)", L)
    if m:
        c[int(m.group(1))] += 1
print("decode-step batch histogram over the last %d lines: %s"
      % (len(win), dict(sorted(c.items()))))
