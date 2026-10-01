"""Decode step time vs batch size, with co-terminated equal-length requests.

bs=1 runs 62 ms/token while bs=2 runs 68.8 ms for two tokens (34.4 ms/token), so the
marginal token is almost free and the per-step cost is dominated by a fixed
component. If that is right, larger batches scale nearly linearly in tokens per step
and aggregate throughput should climb steeply -- which matters because the objective
is 50 tok/s and single-stream sits at 17.

Earlier probes could not see this because MAXREQ=2 with PP=4 caps admission at
max(2//4,1)=1 request per batch (scheduler.py:1175, :3627), so every step was bs=1
regardless of how many clients connected. MAXREQ=16 raises that cap to 4.

Requests here share one prompt and one output length so they start and finish
together; each carries a unique uuid so none is served from the radix cache. The
server's own batch histogram is printed per point so a "bs=4" number can only be
believed if the steps really were bs=4.
"""
import json, re, subprocess, sys, time, urllib.request
from collections import Counter
import concurrent.futures as cf

BS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "2", "4"])]
LOG = sys.argv[3] if len(sys.argv) > 3 else "/data/nvme/sglang-codex/logs/mr-m32.log"
LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 48


def post(i, n, timeout=1800):
    # Same body for every request so step counts match; uuid makes each unique.
    body = ("Explain in detail how a steam turbine converts heat into mechanical "
            "work, covering the role of pressure stages. ref=%d-%f" % (i, time.time()))
    req = urllib.request.Request("http://127.0.0.1:30000/generate",
                                 data=json.dumps({"text": body, "sampling_params":
                                     {"temperature": 0.0, "max_new_tokens": n}}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    j = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    return j["meta_info"]["completion_tokens"], time.perf_counter() - t0


def hist(nlines):
    txt = subprocess.run(["grep", "-aE", "Decode batch", LOG],
                         capture_output=True, text=True).stdout.splitlines()
    c = Counter()
    for L in txt[-nlines:]:
        m = re.search(r"#running-req: (\d+)", L)
        if m:
            c[int(m.group(1))] += 1
    return dict(sorted(c.items()))


post(0, 4)
print("%4s %10s %10s %12s %10s  %s" % ("bs", "agg tok/s", "ms/step", "tokens/step", "per-req", "observed batch histogram"))
for bs in BS:
    best = (0.0, None, None)
    for trial in range(3):
        with cf.ThreadPoolExecutor(max_workers=bs) as ex:
            t0 = time.perf_counter()
            res = list(ex.map(lambda i: post(i, LEN), range(bs)))
            wall = time.perf_counter() - t0
        toks = sum(r[0] for r in res)
        rate = toks / wall
        if rate > best[0]:
            best = (rate, wall, toks)
    rate, wall, toks = best
    # steps ~= tokens per request, since they are co-terminated
    steps = toks / bs
    ms_step = wall * 1e3 / max(steps, 1)
    print("%4d %10.2f %10.1f %12.2f %10.2f  %s"
          % (bs, rate, ms_step, toks / bs / (wall / steps) if False else toks / steps,
             rate / bs, hist(int(steps * 1.5) + 6)))
