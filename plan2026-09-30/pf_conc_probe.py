#!/usr/bin/env python
"""Does `pp_max_micro_batch_size` throttle PREFILL?

Why this exists: the previous round's D1 recommendation was "raise
--cuda-graph-max-bs-decode to 4/8". The log kills that: `cuda graph: True`
on 1336/1336 decode lines and `#running-req` never exceeds 2, because
`scheduler.py:1194-1199` auto-sets

    pp_max_micro_batch_size = max_running_requests // pp_size = 8 // 4 = 2

and `get_num_allocatable_reqs()` (scheduler.py:3790) returns

    pp_budget = pp_max_micro_batch_size - running_bs

whose four call sites are ALL on the prefill path (`adder.can_run_list`).
So the same knob that pins the decode micro-batch at 2 also caps each prefill
batch at `2 - running_bs` *sequences*, regardless of `max_prefill_tokens=16384`.

That only matters when prompts are short enough that token-count batching would
admit more than 2 sequences -- exactly the agentic shape (many medium prompts
competing). If the aggregate prefill rate stops scaling at K=2 concurrent
prompts, the cap is binding and the fix is a scheduling knob, not a kernel.

Usage:
  pf_conc_probe.py                       # lens 1000,4000  K 1,2,4,8
  pf_conc_probe.py --lens 500,4000 --ks 1,2,4,8 --rounds 3
"""
import argparse
import collections
import json
import os
import re
import statistics
import threading
import time
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog while the committee "
    "reviews the annual report and the engineers calibrate the sensor. "
)
LOG = "/data/nvme/sglang/logs/serve-prod.log"


def make_text(approx_tokens, salt):
    reps = max(1, approx_tokens // 17)
    return f"{salt} " + " ".join([FILLER] * reps)


def post(url, payload, timeout=3600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, time.perf_counter() - t0


def fire(url, k, approx_tokens, salt_base, timeout):
    """K concurrent pure-prefill requests; returns (wall, total prompt toks)."""
    res = [None] * k

    def worker(i):
        out, _ = post(url, {
            "text": make_text(approx_tokens, f"s{salt_base + i}"),
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        }, timeout=timeout)
        res[i] = out["meta_info"]["prompt_tokens"]

    th = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    return wall, sum(x or 0 for x in res)


def window_stats(log, mark):
    """Prefill scheduling facts from exactly the measured window."""
    try:
        # Binary: `mark` is a byte offset from os.path.getsize, and text-mode
        # seek only accepts offsets from tell() on the same file.
        with open(log, "rb") as f:
            f.seek(mark)
            window = f.read().decode("utf-8", "replace")
    except OSError:
        return {}
    newseq = collections.Counter()
    newtok = collections.Counter()
    pending = collections.Counter()
    qreq = collections.Counter()
    for l in window.splitlines():
        if "Prefill batch" not in l:
            continue
        m = re.search(r"#new-seq: (\d+), #new-token: (\d+)", l)
        if m:
            newseq[int(m.group(1))] += 1
            newtok[int(m.group(2))] += 1
        m = re.search(r"#queue-req: (\d+), #pending-token: (\d+)", l)
        if m:
            qreq[int(m.group(1))] += 1
            pending[int(m.group(2))] += 1
    return {"new_seq": dict(sorted(newseq.items())),
            "new_token": dict(sorted(newtok.items())),
            "queue_req": dict(sorted(qreq.items())),
            "pending_token": dict(sorted(pending.items())),
            "lines": sum(newseq.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--lens", default="1000,4000")
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--detail", action="store_true",
                    help="print the scheduling histogram for every cell")
    a = ap.parse_args()

    url = f"http://{a.host}:{a.port}/generate"
    lens = [int(x) for x in a.lens.split(",")]
    ks = [int(x) for x in a.ks.split(",")]

    try:
        urllib.request.urlopen(f"http://{a.host}:{a.port}/health", timeout=10)
    except Exception as e:
        print(f"server not healthy: {e}")
        raise SystemExit(1)

    salt = 1_000_000
    for _ in range(a.warmup):
        for L in lens:
            for k in ks:
                fire(url, k, L, salt, a.timeout)
                salt += k + 7

    cells = {(L, k): [] for L in lens for k in ks}
    sched = {}
    salt = 5_000_000
    for rnd in range(a.rounds):
        for L in lens:
            for k in ks:
                try:
                    mark = os.path.getsize(a.log)
                except OSError:
                    mark = 0
                wall, toks = fire(url, k, L, salt, a.timeout)
                salt += k + 13
                rate = toks / wall if wall else 0.0
                cells[(L, k)].append(rate)
                st = window_stats(a.log, mark)
                if st:
                    sched[(L, k)] = st
                print(f"  round {rnd} L={L:<5} K={k:<2} {wall:6.2f}s "
                      f"{toks:>6} tok  {rate:8.1f} tok/s"
                      + (f"  new-seq={st.get('new_seq')} pend={st.get('pending_token')}"
                         if st else ""), flush=True)

    print("\n=== aggregate prefill throughput (tok/s), median of rounds ===")
    hdr = "".join(f"{'K=' + str(k):>12}" for k in ks)
    print(f"{'prompt':>8}{hdr}")
    summary = {}
    for L in lens:
        row = f"{L:>8}"
        for k in ks:
            v = sorted(cells[(L, k)])
            med = statistics.median(v)
            summary[f"{L}/{k}"] = {"median": med, "all": v}
            row += f"{med:>12.1f}"
        print(row)

    print("\n=== speedup over K=1 (does concurrency still pay?) ===")
    for L in lens:
        base = statistics.median(cells[(L, 1)])
        row = f"{L:>8}"
        for k in ks:
            med = statistics.median(cells[(L, k)])
            row += f"{med / base:>11.2f}x" if base else f"{'-':>12}"
        print(row)

    print("\n=== ideal scaling (K=1 x K) -- how far from linear? ===")
    for L in lens:
        base = statistics.median(cells[(L, 1)])
        row = f"{L:>8}"
        for k in ks:
            med = statistics.median(cells[(L, k)])
            row += f"{med / (base * k):>11.2f}x" if base else f"{'-':>12}"
        print(row)

    if a.detail:
        print("\n=== scheduler admission behaviour per cell ===")
        for key in sorted(sched):
            st = sched[key]
            print(f"  L={key[0]} K={key[1]}: prefill lines={st['lines']}")
            print(f"      #new-seq   {st['new_seq']}")
            print(f"      #new-token {st['new_token']}")
            print(f"      #queue-req {st['queue_req']}")
            print(f"      #pending   {st['pending_token']}")

    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out), exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump({"lens": lens, "ks": ks, "rounds": a.rounds,
                       "summary": summary,
                       "sched": {f"{k[0]}/{k[1]}": v for k, v in sched.items()}},
                      f, indent=2)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
