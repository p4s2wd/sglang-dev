#!/usr/bin/env python
"""Why do some MMLU completions not contain 'Answer: X'?

Pulls the unparsed cases at a bigger token budget and prints the tail so the
formatting the model actually uses is visible.
"""
import json
import os
import re
import urllib.request

from datasets import load_dataset

URL = "http://127.0.0.1:30000/generate"
PAT = re.compile(r"(?i)Answer\s*:\s*([A-D])")
T = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before answering."
    "\n\n{q}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}"
)


def gen(text, mx):
    body = json.dumps({"text": text,
                       "sampling_params": {"temperature": 0.0,
                                           "max_new_tokens": mx}}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


LET = "ABCD"
bad = []
d = load_dataset("cais/mmlu", "college_chemistry", split="test",
                 trust_remote_code=True)
for i in range(8):
    r = d[i]
    p = T.format(q=r["question"], a=r["choices"][0], b=r["choices"][1],
                 c=r["choices"][2], d=r["choices"][3])
    o = gen(p, 900)
    t = o["text"]
    ct = o["meta_info"]["completion_tokens"]
    m = PAT.search(t)
    if not m:
        bad.append((i, ct, LET[int(r["answer"])], t))

print(f"unparsed at 900-token budget: {len(bad)} of 8")
for i, ct, gold, t in bad[:3]:
    print(f"--- idx {i}  completion_tokens {ct}  gold {gold}")
    print("TAIL:", repr(t[-350:]))
