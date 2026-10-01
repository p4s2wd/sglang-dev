#!/bin/bash
# Aggregate decode throughput with two requests in flight.
#
# The user capped concurrency at 1-2, so batch scaling is the only remaining
# lever that does not need a kernel rewrite: two requests share the weight read
# that dominates a decode step. This fires N pairs simultaneously and reports
# aggregate tokens/s, which is what "50 tok/s" means under a 2-request cap.
set -u
PORT="${PORT:-30000}"
PAIRS="${PAIRS:-2}"
OUT="${OUT:-64}"
PY=/data/nvme/sglang-codex/.venv/bin/python

"$PY" - "$PORT" "$PAIRS" "$OUT" <<'PY'
import json, sys, threading, time
import urllib.request

port, pairs, out_tok = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
URL = f"http://127.0.0.1:{port}/generate"
results = {}
lock = threading.Lock()


def run(i):
    body = json.dumps({
        "text": f"Count from {100 + i * 7} upward, one number per line. ",
        "sampling_params": {"temperature": 0.0, "max_new_tokens": out_tok},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        j = json.loads(r.read())
    with lock:
        results[i] = j["meta_info"].get("completion_tokens", out_tok)


for trial in range(3):
    # Warm the graph for this shape on trial 0.
    ts = [threading.Thread(target=run, args=(i,)) for i in range(pairs)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    total = sum(results.values())
    print(f"trial {trial}: {pairs} reqs x {out_tok} tok  {dt:5.2f}s  "
          f"{total/dt:6.2f} tok/s aggregate  ({total/dt/pairs:5.2f}/req)")
PY
