#!/bin/bash
# Decode throughput vs concurrent requests.
#
# Decode at batch 1 is weight-bandwidth bound: the same weight read serves every
# token in the batch, so aggregate tok/s should scale roughly linearly until
# activation traffic or compute takes over. This measures where that curve bends,
# which decides whether 50-80 tok/s is reachable by batching alone.
#
# Usage: ./probe_batch_scaling.sh [batch sizes...]
set -u
PORT="${PORT:-30000}"
PY=/data/nvme/sglang-codex/.venv/bin/python

"$PY" - "$PORT" "${@:-1 2 4 8 16}" <<'PY'
import concurrent.futures as futures
import json, sys, time
import urllib.request

port = sys.argv[1]
sizes = [int(a) for a in sys.argv[2:]] or [1, 2, 4, 8, 16]
URL = f"http://127.0.0.1:{port}/generate"
NEWTOK = 48


def one(seed):
    body = json.dumps({
        "text": f"Explain item {seed} of a long list of technical topics in detail.",
        "sampling_params": {"temperature": 0.0, "max_new_tokens": NEWTOK},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["meta_info"]["completion_tokens"]


# Warm the graph for each shape once so capture cost is not timed.
for b in sizes:
    with futures.ThreadPoolExecutor(max_workers=b) as ex:
        list(ex.map(one, range(b)))

print(f"{'bs':>4} {'tok_total':>10} {'wall_s':>8} {'agg_t/s':>8} {'per-req':>8}")
base = None
for b in sizes:
    t0 = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=b) as ex:
        toks = sum(ex.map(one, range(b)))
    dt = time.perf_counter() - t0
    agg = toks / dt
    if base is None:
        base = agg
    print(f"{b:>4} {toks:>10} {dt:>8.2f} {agg:>8.2f} {agg/b:>8.2f}"
          f"   scaling {agg/base:>5.2f}x")
PY
