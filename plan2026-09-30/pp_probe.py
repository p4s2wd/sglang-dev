#!/usr/bin/env python
"""Decisive probe: what actually constrains the decode batch size?

Evidence so far is self-contradictory, and the contradiction decides whether
the graph-raise experiment (D1) is worth doing at all:

  * `#running-req` never exceeds 2, `#new-seq` never exceeds 2  -> a cap of
    2, which is exactly `max_running_requests // pp_size` = 8 // 4, the value
    `scheduler.py:1194-1199` assigns to `pp_max_micro_batch_size`.
  * yet `#queue-req` is 0 in all 1424 log lines, which says nothing is being
    held back by `pp_budget = pp_max_micro_batch_size - running_bs`.

Those cannot both be true unless the log line reports the micro-batch rather
than the whole running batch. So: fire N concurrent requests, then dump the raw
scheduler lines from exactly that window and report the distributions.

Usage: pp_probe.py [--bs 8] [--newtok 256]
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


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, time.perf_counter() - t0


def server_info(host, port):
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/get_server_info", timeout=30
        ) as r:
            txt = r.read().decode("utf-8", "replace")
    except Exception as e:
        return {"error": str(e)}
    keys = ("pp_max_micro_batch_size", "max_running_requests", "pp_size",
            "tp_size", "cuda_graph_max_bs_decode", "cuda_graph_bs_decode",
            "cuda_graph_backend_decode", "pp_async_batch_depth",
            "max_total_tokens", "decode_log_interval")
    found = {}
    for k in keys:
        m = re.search(rf"^\s*{k}\s*[:=]\s*(.+)$", txt, re.M)
        if m:
            found[k] = m.group(1).strip()
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--newtok", type=int, default=256)
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--out", default="/data/nvme/sglang-codex/plan2026-09-30/res/pp_probe.txt")
    a = ap.parse_args()

    url = f"http://{a.host}:{a.port}/generate"
    lines_out = []

    info = server_info(a.host, a.port)
    lines_out.append("== /get_server_info (key fields) ==")
    for k, v in info.items():
        lines_out.append(f"  {k} = {v}")

    try:
        with open(a.log, "rb") as f:
            f.seek(0, 2)
            mark = f.tell()
    except OSError:
        mark = 0
    lines_out.append(f"\nlog offset before run: {mark}")

    # Fire N concurrent requests; nothing is warmed up, so this is the real
    # admission behaviour rather than a steady-state artefact.
    out = [None] * a.bs
    lat = [0.0] * a.bs

    def worker(i):
        body = {
            "text": f"Count from {i * 977} upward: " + FILLER * 3,
            "sampling_params": {"temperature": 0.0,
                                "max_new_tokens": a.newtok},
        }
        t0 = time.perf_counter()
        o, _ = post(url, body)
        lat[i] = time.perf_counter() - t0
        out[i] = o["meta_info"]["completion_tokens"]

    th = [threading.Thread(target=worker, args=(i,)) for i in range(a.bs)]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    toks = sum(x or 0 for x in out)

    lines_out.append(
        f"\n== fired {a.bs} concurrent requests, {a.newtok} newtok each ==\n"
        f"  wall={wall:.2f}s  tokens={toks}  aggregate={toks / wall:.1f} tok/s\n"
        f"  per-request latency: min={min(lat):.2f}s "
        f"median={statistics.median(lat):.2f}s max={max(lat):.2f}s"
    )

    time.sleep(3)
    try:
        with open(a.log, "r", errors="replace") as f:
            f.seek(0, 2) if mark == 0 else None
            f.seek(mark)
            window = f.read()
    except OSError as e:
        lines_out.append(f"log read failed: {e}")
        window = ""

    run = collections.Counter()
    que = collections.Counter()
    newseq = collections.Counter()
    newtok_c = collections.Counter()
    graph = collections.Counter()
    prefill_lines = []
    pat = re.compile(
        r"#running-req: (\d+).*?#queue-req: (\d+).*?"
        r"gen throughput \(token/s\): ([\d.]+)"
    )
    pat_pre = re.compile(r"#new-seq: (\d+), #new-token: (\d+)")
    for l in window.splitlines():
        if "Decode batch" in l:
            m = pat.search(l)
            if m:
                run[int(m.group(1))] += 1
                que[int(m.group(2))] += 1
                graph["True" if "cuda graph: True" in l else "False"] += 1
        elif "Prefill batch" in l:
            m = pat_pre.search(l)
            if m:
                newseq[int(m.group(1))] += 1
                newtok_c[int(m.group(2))] += 1
            if len(prefill_lines) < 6:
                prefill_lines.append(re.sub(r"^\[[^\]]*\] ", "", l))

    lines_out.append("\n== Decode batch lines in window ==")
    lines_out.append(f"  running-req histogram : {dict(sorted(run.items()))}")
    lines_out.append(f"  queue-req   histogram : {dict(sorted(que.items()))}")
    lines_out.append(f"  cuda graph            : {dict(graph)}")
    lines_out.append(f"  total decode lines    : {sum(run.values())}")

    lines_out.append("\n== Prefill batch lines in window ==")
    lines_out.append(f"  #new-seq histogram    : {dict(sorted(newseq.items()))}")
    lines_out.append(f"  #new-token histogram  : {dict(sorted(newtok_c.items()))}")
    for l in prefill_lines:
        lines_out.append(f"  {l}")

    lines_out.append("\n== raw window (first 40 scheduler lines) ==")
    n = 0
    for l in window.splitlines():
        if "Decode batch" in l or "Prefill batch" in l or "queue" in l.lower():
            lines_out.append("  " + re.sub(r"^\[[^\]]*\] ", "", l))
            n += 1
            if n >= 40:
                break

    text = "\n".join(lines_out)
    print(text)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(text + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
