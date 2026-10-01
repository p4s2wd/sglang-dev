#!/usr/bin/env python
"""Correctness at the batch size that hits the 50 tok/s target.

The headline is 59 tok/s aggregate at bs=8, which is what makes the objective's
throughput target real. A throughput number is worthless if the batched path
produces different text, so run a set of prompts twice: once alone (bs=1) and
once with all eight in flight together, and compare the outputs token for token.
Greedy decoding should make them identical apart from where numerical
non-determinism in the MoE reduction can flip a token.
"""
import json, sys, time, urllib.request, threading

URL = "http://127.0.0.1:30000/generate"
PROMPTS = [
    "Calculate step by step: 17 * 23 = ",
    "Write a Python function that reverses a linked list, with type hints:",
    "The three laws of thermodynamics are:",
    "Translate to French: The weather is pleasant today.",
    "Explain what a hash map is in one paragraph:",
    "List the first 10 prime numbers:",
    "Balance this equation: CH4 + O2 -> ",
    "What is the capital of Australia? Answer in one word.",
]


def gen(text, maxtok=48):
    body = {"text": text,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": maxtok,
                                "ignore_eos": True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["text"]


# Reference: one at a time.
ref = []
for p in PROMPTS:
    ref.append(gen(p))
    time.sleep(0.3)

# Batched: all eight concurrently.
out = [None] * len(PROMPTS)
def worker(i):
    out[i] = gen(PROMPTS[i])
th = [threading.Thread(target=worker, args=(i,)) for i in range(len(PROMPTS))]
for t in th: t.start()
for t in th: t.join()

same = diff = 0
for i, (a, b) in enumerate(zip(ref, out)):
    if a == b:
        same += 1
    else:
        diff += 1
        # how far do they agree?
        n = 0
        for ca, cb in zip(a, b):
            if ca != cb: break
            n += 1
        print("prompt %d differs after %d/%d chars" % (i, n, len(a)))
        print("   solo : %r" % a[:90])
        print("   batch: %r" % b[:90])
print("\nbs=8 batched vs solo: %d identical, %d differ (of %d)"
      % (same, diff, len(PROMPTS)))
print("NOTE: this stack is known to be non-reproducible run-to-run even at bs=1,")
print("so a difference is only a regression if it diverges immediately.")
