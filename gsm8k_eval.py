#!/usr/bin/env python
"""GSM8K accuracy for the SM75 build, against the running server.

Replicates benchmark/gsm8k/bench_sglang.py exactly -- same 5-shot prompt
construction, same gold/pred number extraction, same stop tokens -- but talks
to /generate over plain HTTP. The repo harness needs `sglang.test`, which the
target's editable install does not expose, and its sgl-program layer adds
nothing here (a single prompt, no branching).

Greedy (temperature 0), which is what the published sglang GSM8K numbers use.

Usage: gsm8k_eval.py --num-questions 200 --parallel 8 [--out results.jsonl]
"""
import argparse
import ast
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

INVALID = -9999999
URL = os.environ.get("GSM8K_URL", "http://127.0.0.1:30000/generate")
DATA_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
            "master/grade_school_math/data/test.jsonl")


def get_answer_value(answer_str):
    """Verbatim from bench_sglang.py: the last number in the string."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    return "".join(get_one_example(lines, i, True) + "\n\n" for i in range(k))


def load_lines(path):
    if not os.path.isfile(path):
        import urllib.request as u
        tmp = path + ".part"
        with u.urlopen(DATA_URL, timeout=120) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, path)
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


STOP = ["Question", "Assistant:", "<|separator|>"]


def post(payload, timeout=1800):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--data-path", type=str, default="gsm8k_test.jsonl")
    ap.add_argument("--out", type=str, default="gsm8k_results.jsonl")
    ap.add_argument("--label", type=str, default="")
    args = ap.parse_args()

    lines = load_lines(args.data_path)
    few_shot = get_few_shot_examples(lines, args.num_shots)

    n = min(args.num_questions, len(lines))
    prompts, labels = [], []
    for i in range(n):
        prompts.append(few_shot + get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels), "gold answer unparseable"

    # Warmup outside the measurement: first request pays Triton JIT for any
    # shape the server has not seen yet.
    # `stop` is a SamplingParams field -- at the top level of the request body
    # it is silently ignored, the answer runs on into the next question, and
    # since the extractor takes the LAST number in the text, accuracy collapses.
    post({"text": prompts[0],
          "sampling_params": {"temperature": 0.0, "max_new_tokens": 8,
                              "stop": STOP}})

    preds = [None] * n
    tok_counts = [0] * n
    lock = threading.Lock()
    done = [0]

    def work(i):
        body = post({
            "text": prompts[i],
            "sampling_params": {"temperature": 0.0,
                                "max_new_tokens": args.max_new_tokens,
                                "stop": STOP},
        })
        preds[i] = get_answer_value(body["text"])
        tok_counts[i] = body.get("meta_info", {}).get("completion_tokens", 0)
        with lock:
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == n:
                acc = sum(p == l for p, l in zip(preds[:done[0]], labels[:done[0]])
                          ) / done[0]
                print(f"  {done[0]}/{n}  running acc {acc:.3f}", flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        list(ex.map(work, range(n)))
    dt = time.perf_counter() - t0

    correct = sum(p == l for p, l in zip(preds, labels))
    invalid = sum(p == INVALID for p in preds)
    out_tok = sum(tok_counts)
    with open(args.out, "w") as f:
        for i in range(n):
            f.write(json.dumps({"i": i, "gold": labels[i], "pred": preds[i],
                                "ok": preds[i] == labels[i],
                                "out_tok": tok_counts[i]}) + "\n")

    print(f"=== GSM8K {args.label} n={n} shots={args.num_shots} greedy ===")
    print(f"Accuracy: {correct / n:.4f}  ({correct}/{n})")
    print(f"Invalid:  {invalid / n:.4f}  ({invalid})")
    # If a large share hit the token cap, the stop strings are not firing and
    # the accuracy number is meaningless (the extractor reads the last number
    # out of a runaway continuation).
    capped = sum(t >= args.max_new_tokens for t in tok_counts)
    print(f"Capped:   {capped / n:.4f}  ({capped} of {n} hit max_new_tokens)")
    print(f"wall {dt:.1f}s   output {out_tok} tok   {out_tok / dt:.1f} tok/s")


if __name__ == "__main__":
    sys.exit(main())
