#!/usr/bin/env python
"""Decode throughput vs context, read from the server's own accounting.

The earlier dec_ctx_probe.py prices decode by differencing two calls and
subtracting a cached prefill. That fails once the pool is loaded: the second
call can evict and recompute, so the difference stops meaning "N-1 decode
steps". At a 92K-token context it reported 536 tok/s with a 119% spread, and
the samples (0.65 ms/token = 1540 tok/s) are not physically available on this
box. A measurement that cannot be wrong in the direction it did is not worth
repairing.

So this reads what the scheduler itself reports. Every decode step emits a
"Decode batch" line carrying the running token count and gen throughput, already
averaged over the step and already restricted to the decode phase. Taking the
median of those lines over one generation gives the number directly, with no
subtraction and no assumption that a prefill was cached.

Only PP0 lines are read; the four pipeline stages each log the same step.

Usage: dec_ctx_logprobe.py --lens 8000,32000,128000,256000 [--n 160] [--port 8200]
"""
import argparse
import json
import re
import statistics
import time
import urllib.request

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
         "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
         "oscar", "papa", "papa", "quebec", "romeo"]

LINE = re.compile(
    r"\[(?P<date>\d{4}-\d{2}-\d{2}) (?P<hms>\d{2}:\d{2}:\d{2}) "
    r"(?P<rank>PP\d+ TP\d+)\] "
    r"Decode batch, #running-req: (?P<run>\d+), #full token: (?P<full>\d+),"
)
TP = re.compile(r"gen throughput \(token/s\): (?P<tp>[0-9.]+)")


def post(url, payload, timeout=3600):
    req = urllib.request.Request(
        url + "/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def log_lines(path):
    with open(path, "r", errors="replace") as f:
        return f.readlines()


def parse_since(lines, since_idx, hms_lo, hms_hi):
    """Decode-batch throughputs for PP0 TP0 inside a clock window."""
    out = []
    for ln in lines[since_idx:]:
        m = LINE.search(ln)
        if not m or m.group("rank") != "PP0 TP0":
            continue
        if not (hms_lo <= m.group("hms") <= hms_hi):
            continue
        t = TP.search(ln)
        if t:
            out.append((int(m.group("full")), float(t.group("tp"))))
    return out


def build_prompt(target_words):
    return "history of the roman empire " * target_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--lens", default="8000,32000,128000,256000")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--log", default="/data/nvme/sglang/logs/serve-prod.log")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    # 5 tokens per word for this filler, measured earlier on this model.
    results = []
    print(f"{'want tok':>9} {'prompt tok':>11} {'full tok(med)':>14} "
          f"{'tok/s(med)':>11} {'p10':>8} {'p90':>8} {'steps':>6}")
    for want in [int(x) for x in args.lens.split(",")]:
        words = max(1, want // 5)
        prompt = build_prompt(words)
        n0 = len(log_lines(args.log))
        t0 = time.strftime("%H:%M:%S")
        # one warm call so the prefill is cached and only decode is measured
        post(url, {"text": prompt, "sampling_params": {
            "temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}})
        n1 = len(log_lines(args.log))
        t1 = time.strftime("%H:%M:%S")
        d = post(url, {"text": prompt, "sampling_params": {
            "temperature": 0.0, "max_new_tokens": args.n,
            "ignore_eos": True}})
        t2 = time.strftime("%H:%M:%S")
        lines = log_lines(args.log)
        pts = parse_since(lines, n1, t1, t2)
        if not pts:
            print(f"{want:>9} {'-':>11} {'(no decode lines)':>14}")
            continue
        # The first decode line after a long prefill averages that prefill into
        # its throughput, which shows up as a 0.4 tok/s sample beside 20 tok/s
        # neighbours. It is a transition artefact, not a measurement, so drop
        # anything below half the median rather than letting it widen the range
        # and hide a real effect.
        raw = sorted(p[1] for p in pts)
        med_raw = statistics.median(raw)
        tps = [t for t in raw if t >= 0.5 * med_raw]
        fulls = sorted(p[0] for p in pts)
        med = statistics.median(tps)
        rec = {"want": want, "prompt_tokens": d.get("meta_info", {}).get(
            "prompt_tokens"), "full_token_median": statistics.median(fulls),
            "tok_s_median": med, "p10": tps[len(tps) // 10],
            "p90": tps[len(tps) * 9 // 10], "steps": len(tps),
            "dropped": len(raw) - len(tps), "raw_median": med_raw}
        results.append(rec)
        print(f"{want:>9} {rec['prompt_tokens']:>11} "
              f"{rec['full_token_median']:>14.0f} {med:>11.2f} "
              f"{rec['p10']:>8.2f} {rec['p90']:>8.2f} {len(tps):>6}"
              f"{('  dropped ' + str(rec['dropped'])) if rec['dropped'] else ''}",
              flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()