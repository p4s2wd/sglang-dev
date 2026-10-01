"""Decode scaling with the batch histogram restricted to the measurement window.

The previous version estimated decode steps as tokens/bs and read the last that-many
log lines. When fewer requests run concurrently than were requested, that estimate is
wrong, so the window sampled the tail where requests had already drained -- it
reported "{2: 78}" for an 8-request run and made the result unreadable.

Fix: record the log's line count before the run and histogram only lines appended
during it. That reports the batch sizes actually used, which is the only way to tell a
real bs=8 step from four waves of two.
"""
import json, re, subprocess, sys, time, urllib.request
from collections import Counter
import concurrent.futures as cf

BS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["4", "8"])]
LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 48
LOG = sys.argv[3] if len(sys.argv) > 3 else "/data/nvme/sglang-codex/logs/mr-m32.log"


def post(i, n, timeout=1800):
    body = ("Explain in detail how a steam turbine converts heat into mechanical "
            "work, covering the role of pressure stages. ref=%d-%f" % (i, time.time()))
    req = urllib.request.Request("http://127.0.0.1:30000/generate",
                                 data=json.dumps({"text": body, "sampling_params":
                                     {"temperature": 0.0, "max_new_tokens": n}}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    j = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    return j["meta_info"]["completion_tokens"], time.perf_counter() - t0


def nlines():
    return int(subprocess.run(f"grep -acE 'Decode batch' {LOG} 2>/dev/null || echo 0",
                              shell=True, capture_output=True, text=True).stdout.strip())


def hist(a, b):
    txt = subprocess.run(f"grep -aE 'Decode batch' {LOG} | sed -n '{a+1},{b}p'",
                         shell=True, capture_output=True, text=True).stdout.splitlines()
    c = Counter()
    for L in txt:
        m = re.search(r"#running-req: (\d+)", L)
        if m:
            c[int(m.group(1))] += 1
    return dict(sorted(c.items())), len(txt)


post(0, 4)
print("%4s %11s %10s %11s %10s  %s" % ("req", "agg tok/s", "ms/step", "tok/step", "per-req", "batch histogram in window"))
for bs in BS:
    best = (0.0, 0, 0, {}, 0)
    for trial in range(3):
        a = nlines()
        with cf.ThreadPoolExecutor(max_workers=bs) as ex:
            t0 = time.perf_counter()
            res = list(ex.map(lambda i: post(i, LEN), range(bs)))
            wall = time.perf_counter() - t0
        b = nlines()
        toks = sum(r[0] for r in res)
        rate = toks / wall
        h, nst = hist(a, b)
        if rate > best[0]:
            best = (rate, wall, toks, h, nst)
    rate, wall, toks, h, nst = best
    nst = max(nst, 1)
    print("%4d %11.2f %10.1f %11.2f %10.2f  %s"
          % (bs, rate, wall * 1e3 / nst, toks / nst, rate / bs, h))
