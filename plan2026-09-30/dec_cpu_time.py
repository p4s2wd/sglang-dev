#!/usr/bin/env python
"""How much of the decode step is CPU time, measured without a profiler.

Needed because the py-spy profile said 93% of the bs=1 decode step is the host
waiting, and that could not be reconciled with the trace: the four pipeline
stages' kernels sum to 26.0 ms against a 37.2 ms step, so the GPU is ~70% busy
and only ~11 ms should be unaccounted. Either the profiler sampled mostly idle
time, or the gap is elsewhere. Both readings cannot be right, and a profiler
that distorts throughput tenfold is a poor instrument for the question.

/proc/<pid>/stat answers it directly and costs nothing: utime+stime is CPU time
actually consumed, wall time is what elapsed. Their ratio needs no attribution
to any Python frame, so it cannot be fooled by where the samples happened to
land. Resolution is one clock tick, which over a few thousand tokens is worth
well under a percent.

Each pipeline stage is its own process. They are serialised at bs=1, so the
stage with the most CPU time sets the step, and that is the number to look at.

Usage: dec_cpu_time.py [--newtok 3000] [--port 8200]
"""
import argparse
import json
import os
import time
import urllib.request

CLK = os.sysconf("SC_CLK_TCK")


def procs():
    """One scheduler process per (pp, tp), with their cpu times."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip()
        except OSError:
            continue
        # /proc/<pid>/comm is capped at 15 characters by the kernel, so
        # "sglang::scheduler_PP3_TP0" is stored as "sglang::schedul" and the
        # full stage name has to come from the cmdline.
        if not comm.startswith("sglang::sched"):
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read().decode(errors="replace")
            name = "sglang::" + raw.split("\x00")[0].strip()[-10:]
        except OSError:
            name = comm
        try:
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().rsplit(") ", 1)[1].split()
        except (OSError, IndexError):
            continue
        # after the comm field, fields[0] is state; utime/stime are 11 and 12
        utime, stime = int(fields[11]), int(fields[12])
        out[(name, int(pid))] = (utime + stime) / CLK
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--newtok", type=int, default=3000)
    ap.add_argument("--ctx-tokens", type=int, default=8000)
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"

    prompt = "history of the roman empire " * max(1, args.ctx_tokens // 5)
    # Warm first so the timed call decodes rather than prefills. One call, so
    # the server is not concurrently doing anything else while we measure.
    warm = {"text": prompt, "sampling_params": {
        "temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}}
    req = urllib.request.Request(url + "/generate", data=json.dumps(warm).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=600).read()

    before = procs()
    if not before:
        print("no scheduler processes found")
        return
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps({"text": prompt, "sampling_params": {
            "temperature": 0.0, "max_new_tokens": args.newtok,
            "ignore_eos": True}}).encode(),
        headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    wall = time.perf_counter() - t0
    after = procs()
    n = d["meta_info"]["completion_tokens"]

    print(f"completion_tokens = {n}   wall = {wall:.1f} s   "
          f"({n / wall:.2f} tok/s, {1000 * wall / n:.1f} ms/token)")
    print(f"{'stage':<28} {'cpu s':>8} {'cpu ms/token':>14} {'share of wall':>15}")
    rows = []
    for key in sorted(before, key=lambda k: k[1]):
        name, pid = key
        used = after.get(key, 0.0) - before[key]
        rows.append((name, used, 1000 * used / n, 100 * used / wall))
    for name, used, per_tok, share in sorted(rows, key=lambda r: -r[2]):
        print(f"{name:<28} {used:>8.2f} {per_tok:>14.1f} {share:>14.1f}%")
    if rows:
        busiest = max(rows, key=lambda r: r[2])
        print(f"\n最忙的一级: {busiest[0]}  {busiest[2]:.1f} ms CPU / token "
              f"({busiest[3]:.0f}% of the {1000 * wall / n:.1f} ms step)")
        print("若这一级 CPU 时间接近整步耗时 -> 该级是 CPU 瓶颈;"
              "若明显小于 -> 它在等 GPU 或等对端")


if __name__ == "__main__":
    main()