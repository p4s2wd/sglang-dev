#!/usr/bin/env python
"""Decode throughput at a list of batch sizes, interleaved, against :8200.

Why this exists: decode_bs.py is hardcoded to the dead :30000 dev server and
runs each batch size as a separate sequential block, so the box's ~6-10%
thermal drift lands entirely on whichever arm ran last. Same discipline as
pf_probe_prod.py: interleave the arms, take the median per arm.

For each batch size it reports:
  aggregate tok/s   (N concurrent requests all generating NEWTOK tokens)
  per-request ms/token
and it also verifies the thing the experiment is actually about: whether the
decode CUDA graph was used. The scheduler logs `cuda graph: True/False` per
Decode batch line, so --log tells us what fraction of steps hit the graph.

Usage:
  dec_probe.py                          # bs 1,2,4,8 --rounds 3
  dec_probe.py --bs 1,2,4,8 --newtok 256 --rounds 3
  dec_probe.py --log /data/nvme/sglang/logs/serve-prod.log
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog while the committee "
    "reviews the annual report and the engineers calibrate the sensor. "
)


def one(url, salt_idx, newtok, timeout):
    # Distinct prompts so the radix cache cannot answer any part of this.
    text = f"Count from {salt_idx * 977} upward: " + FILLER * 3
    body = {
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": newtok},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    dt = time.perf_counter() - t0
    return dt, out["meta_info"]["completion_tokens"]


def run_bs(url, bs, newtok, timeout, salt_base):
    out = [None] * bs

    def worker(i):
        out[i] = one(url, salt_base + i, newtok, timeout)

    th = [threading.Thread(target=worker, args=(i,)) for i in range(bs)]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    toks = sum(c for _, c in out)
    return wall, toks


def graph_fraction(log_path, since_pos):
    """Fraction of Decode batch lines with cuda graph: True after since_pos."""
    if not log_path:
        return None, 0
    try:
        with open(log_path, "r", errors="replace") as f:
            f.seek(since_pos)
            tail = f.read()
    except OSError:
        return None, 0
    lines = [l for l in tail.splitlines() if "Decode batch" in l]
    if not lines:
        return None, 0
    tr = sum(1 for l in lines if "cuda graph: True" in l)
    return tr / len(lines), len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--bs", default="1,2,4,8")
    ap.add_argument("--newtok", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--log", default="/data/nvme/sglang/logs/serve-prod.log")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    url = f"http://{a.host}:{a.port}/generate"
    bss = [int(x) for x in a.bs.split(",")]

    try:
        urllib.request.urlopen(f"http://{a.host}:{a.port}/health", timeout=10)
    except Exception as e:
        print(f"server not healthy: {e}")
        raise SystemExit(1)

    # Warm every graph bucket once so JIT/allocator cost is not in the numbers.
    salt = 100000
    for _ in range(a.warmup):
        for bs in bss:
            run_bs(url, bs, a.newtok, a.timeout, salt)
            salt += bs
    print(f"warmup done ({a.warmup} x {bss})", flush=True)

    log_pos = 0
    if a.log:
        try:
            log_pos = __import__("os").path.getsize(a.log)
        except OSError:
            log_pos = 0
    # mark position before the measured rounds
    try:
        with open(a.log, "rb") as f:
            f.seek(0, 2)
            log_pos = f.tell()
    except OSError:
        log_pos = 0

    results = {bs: [] for bs in bss}
    salt = 900000
    for rnd in range(a.rounds):
        for bs in bss:
            wall, toks = run_bs(url, bs, a.newtok, a.timeout, salt)
            salt += bs + 13
            agg = toks / wall
            results[bs].append(agg)
            per_req = wall / (toks / bs) * 1000 if toks else 0
            print(f"  round {rnd} bs={bs:<2} {wall:6.2f}s {toks:>5} tok  "
                  f"{agg:7.1f} tok/s agg  {per_req:7.1f} ms/tok/req",
                  flush=True)

    gfrac, glines = graph_fraction(a.log, log_pos)
    print()
    print("%4s %9s %9s %9s %8s" % ("bs", "median", "min", "max", "spread"))
    summary = {}
    for bs in bss:
        v = sorted(results[bs])
        med = statistics.median(v)
        spread = (v[-1] - v[0]) / med * 100 if med else 0.0
        summary[bs] = {"median": med, "min": v[0], "max": v[-1], "all": v}
        print("%4d %9.1f %9.1f %9.1f %7.1f%%"
              % (bs, med, v[0], v[-1], spread))
    if gfrac is not None:
        print(f"\ndecode-graph usage during measurement: {gfrac*100:.1f}% "
              f"of {glines} Decode batch lines")

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"port": a.port, "rounds": a.rounds,
                       "newtok": a.newtok, "summary": summary,
                       "graph_fraction": gfrac}, f, indent=2)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
