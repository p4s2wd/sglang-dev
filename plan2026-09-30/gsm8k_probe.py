#!/usr/bin/env python
"""Repeatable GSM8K-shaped probe: same prompts, same budget, saved to JSON.

Written to settle one question -- is a decode regression or a behaviour change
attributable to the parallel top-K -- so it has to be byte-comparable between
two server configurations. Fixed prompt set, greedy, fixed max_new_tokens, and
it records completion tokens and wall time alongside the text, because a change
that only shows up as speed is as real as one that shows up as text.

Usage: gsm8k_probe.py <out.json> [--n 3] [--max-new 2048] [--port 8200]
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA = "/data/nvme/sglang-codex/gsm8k_test.jsonl"


def one(url, question, max_new, timeout=1800):
    body = json.dumps({
        "text": question,
        "sampling_params": {
            "temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
            "max_new_tokens": max_new, "ignore_eos": False,
        },
        "stream": False,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return {
        "text": d["text"],
        "completion_tokens": d.get("meta_info", {}).get("completion_tokens"),
        "prompt_tokens": d.get("meta_info", {}).get("prompt_tokens"),
        "seconds": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--conc", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DATA)][: args.n]
    url = f"http://127.0.0.1:{args.port}/generate"
    with ThreadPoolExecutor(max_workers=args.conc) as ex:
        out = list(ex.map(
            lambda r: one(url, r["question"], args.max_new), rows))
    recs = [{"question": r["question"], "gold": r["answer"].split("####")[-1],
             **o} for r, o in zip(rows, out)]
    with open(args.out, "w") as f:
        json.dump(recs, f, indent=1)
    for i, r in enumerate(recs):
        print(f"[{i}] {r['completion_tokens']} tok in {r['seconds']}s  "
              f"gold={r['gold'].strip()[:12]!r}")
        print(f"     {r['text'][:160]!r}")


if __name__ == "__main__":
    main()