#!/usr/bin/env python
r"""MMLU accuracy for the SM75 build, against the running server.

Replicates sglang's simple_eval_mmlu prompt/extraction contract:
  - prompt: QUERY_TEMPLATE_MULTICHOICE (zero-shot, "Think step by step",
            'Answer: $LETTER' as the last line)
  - extract: ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
  - score:   exact letter match

Two deviations, both forced by this checkpoint and both stated in the report:
  1. Completion mode, not chat completion: the checkpoint ships no chat
     template (tokenizer_config.json has no chat_template), so there is no
     role framing to apply.
  2. Data from cais/mmlu through the hf-mirror endpoint (huggingface.co is
     not routable from the target). Subjects are sampled with a fixed seed
     (random.Random(0)) the way the repo harness subsamples.

Usage: HF_ENDPOINT=https://hf-mirror.com python mmlu_eval.py \
           --num-questions 300 --parallel 2 --out mmlu_results.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("MMLU_URL", "http://127.0.0.1:30000/generate")
ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([A-D])")
# run_eval.py's generate/completion mode defaults max_tokens to 2048; that is
# what this uses. It also defaults stop to ["Question", "Assistant:",
# "<|separator|>"], which is right for GSM8K's concatenated few-shot block but
# wrong for MMLU: an MMLU prompt is a single question, and a completion that
# writes the word "Question" while reasoning would be cut off before it ever
# emits 'Answer: X' and would score 0. So no stop strings here -- the budget
# and EOS end the generation, and the unparsed/capped counts below make any
# truncation visible instead of silently scoring it wrong.
STOP = None
QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
LETTERS = "ABCD"


def post(payload, timeout=1800):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def load_subjects(subjects, per_subject):
    rows = []
    for s in subjects:
        d = _load(s)
        if d is None:
            continue
        take = min(per_subject, d.num_rows)
        picked = random.Random(0).sample(range(d.num_rows), take)
        for i in picked:
            r = d[i]
            rows.append({
                "subject": s,
                "question": r["question"],
                "choices": list(r["choices"]),
                "answer": LETTERS[int(r["answer"])],
            })
    return rows


def _load(subject):
    """datasets >=5.0 rejects trust_remote_code (cais/mmlu is plain parquet,
    no loading script), older versions want it. Try the modern call first.

    Only the `test` split is ever used: a subject that ships no test split is
    recorded and skipped rather than substituted with `train`, which would
    score memorised material and inflate the headline.
    """
    from datasets import load_dataset

    def go(**kw):
        return load_dataset("cais/mmlu", subject, **kw)

    try:
        return go(split="test")
    except ValueError:
        SKIPPED.append((subject, "no test split"))
        return None
    except TypeError:
        try:
            return go(split="test", trust_remote_code=True)
        except ValueError:
            SKIPPED.append((subject, "no test split"))
            return None


SKIPPED: list = []


def _configs():
    from datasets import get_dataset_config_names
    try:
        return get_dataset_config_names("cais/mmlu")
    except TypeError:
        return get_dataset_config_names("cais/mmlu", trust_remote_code=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-questions", type=int, default=300)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--subjects", type=str, default="",
                    help="comma list; default = all 57")
    ap.add_argument("--out", type=str, default="mmlu_results.jsonl")
    ap.add_argument("--label", type=str, default="")
    args = ap.parse_args()

    if args.subjects:
        subjects = args.subjects.split(",")
        per = max(1, args.num_questions // len(subjects))
    else:
        from datasets import get_dataset_config_names
        subjects = sorted(_configs())
        subjects = [s for s in subjects if s not in ("all",)]
        per = max(1, round(args.num_questions / len(subjects)))
    rows = load_subjects(subjects, per)
    rows = rows[: args.num_questions]
    n = len(rows)
    print(f"subjects={len(subjects)} per_subject={per} n={n}", flush=True)

    prompts = [QUERY_TEMPLATE.format(question=r["question"], A=r["choices"][0],
                                     B=r["choices"][1], C=r["choices"][2],
                                     D=r["choices"][3]) for r in rows]

    post({"text": prompts[0],
          "sampling_params": {"temperature": 0.0, "max_new_tokens": 8}})

    preds = [None] * n
    capped = [0] * n
    lock = threading.Lock()
    done = [0]

    def work(i):
        sp = {"temperature": 0.0, "max_new_tokens": args.max_new_tokens}
        if STOP:
            sp["stop"] = STOP
        body = post({"text": prompts[i], "sampling_params": sp})
        txt = body["text"]
        m = ANSWER_PATTERN.search(txt)
        preds[i] = m.group(1).upper() if m else None
        capped[i] = body.get("meta_info", {}).get("completion_tokens", 0) \
            >= args.max_new_tokens
        with lock:
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == n:
                ok = sum(p == rows[j]["answer"]
                         for j, p in enumerate(preds[:done[0]]))
                print(f"  {done[0]}/{n}  running acc {ok / done[0]:.3f}", flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        list(ex.map(work, range(n)))
    dt = time.perf_counter() - t0

    ok = sum(p == r["answer"] for p, r in zip(preds, rows))
    unparsed = sum(p is None for p in preds)
    ncap = sum(capped)
    with open(args.out, "w") as f:
        for i in range(n):
            f.write(json.dumps({"subject": rows[i]["subject"],
                                "gold": rows[i]["answer"], "pred": preds[i],
                                "ok": preds[i] == rows[i]["answer"]}) + "\n")

    by_subject = {}
    for r, p in zip(rows, preds):
        a, b = by_subject.setdefault(r["subject"], [0, 0])
        by_subject[r["subject"]] = (a + (p == r["answer"]), b + 1)

    print(f"=== MMLU {args.label} n={n} zero-shot greedy completion ===")
    print(f"Accuracy: {ok / n:.4f}  ({ok}/{n})")
    print(f"Unparsed (no 'Answer: X'): {unparsed}")
    print(f"Capped at max_new_tokens:  {ncap}")
    if SKIPPED:
        print(f"Skipped subjects: {SKIPPED}")
    print(f"wall {dt:.1f}s")
    worst = sorted(by_subject.items(), key=lambda kv: kv[1][0] / kv[1][1])[:5]
    print("lowest subjects:", ", ".join(f"{k} {a}/{b}" for k, (a, b) in worst))


if __name__ == "__main__":
    sys.exit(main())
