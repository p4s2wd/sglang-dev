#!/usr/bin/env python
"""Isolate the BOS repetition: does a leading BOS make the model burn its budget?

The earlier probe showed that a prompt already containing <|begin_of_sentence|>
returns an EMPTY completion on both /v1/completions and /generate, and one of them
timed out. That is the shape of the degeneration the user sees through pi: the
model emits the BOS token again and again instead of answering. The prior run only
looked at the first 90 characters, so a repetition later in the output would be
missed, and max_tokens=40 is too short to show it.

This measures, per endpoint and per prompt form:
  * how many BOS the model emitted, counted over the WHOLE completion
  * the completion length, and whether the budget was exhausted
  * finish_reason, which distinguishes "ran out of tokens mid-repetition" from
    "stopped normally"

Prompts are salted so the radix prefix cache cannot serve a previous run.
"""
import json
import urllib.request
import uuid

URL = "http://127.0.0.1:8200"
BOS = "<｜begin▁of▁sentence｜>"


def post(path, body, timeout=300):
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def probe(name, path, body, extract, maxtok=256):
    salt = uuid.uuid4().hex[:8]
    body = dict(body)
    if "sampling_params" in body:
        body["sampling_params"] = dict(body["sampling_params"], max_new_tokens=maxtok)
    else:
        body.setdefault("max_tokens", maxtok)
    try:
        r = post(path, body)
    except Exception as e:
        print(f"{name:38s} FAILED {type(e).__name__}: {str(e)[:70]}")
        return
    try:
        text, fin = extract(r)
    except Exception:
        print(f"{name:38s} PARSE FAILED")
        return
    n = text.count(BOS)
    stripped = text.replace(BOS, "")
    print(f"{name:38s} len={len(text):4d} bos={n:<4d} rest={len(stripped):4d} "
          f"finish={fin}")
    print(f"{'':38s} {repr(text[:100])}")
    if n >= 3:
        print(f"{'':38s} *** BOS REPRODUCED ({n} times) ***")
    print()


def main():
    plain = "Explain what a pipeline stage is in one sentence."
    print("=" * 78)
    print("A. prompt WITHOUT a leading BOS")
    print("=" * 78)
    probe("chat/completions", "/v1/chat/completions",
          {"model": "deepseek-v4-flash", "temperature": 0.0,
           "messages": [{"role": "user", "content": plain}]},
          lambda r: (r["choices"][0]["message"].get("content") or "",
                     r["choices"][0].get("finish_reason")))
    probe("completions", "/v1/completions",
          {"model": "deepseek-v4-flash", "temperature": 0.0, "prompt": plain},
          lambda r: (r["choices"][0]["text"], r["choices"][0].get("finish_reason")))
    probe("generate", "/generate",
          {"text": plain,
           "sampling_params": {"temperature": 0.0, "max_new_tokens": 256}},
          lambda r: (r["text"], r["meta_info"].get("finish_reason")))

    print("=" * 78)
    print("B. prompt WITH a leading BOS (what the model then has to continue)")
    print("=" * 78)
    probe("chat/completions (msg has BOS)", "/v1/chat/completions",
          {"model": "deepseek-v4-flash", "temperature": 0.0,
           "messages": [{"role": "user", "content": BOS + plain}]},
          lambda r: (r["choices"][0]["message"].get("content") or "",
                     r["choices"][0].get("finish_reason")))
    probe("completions (prompt has BOS)", "/v1/completions",
          {"model": "deepseek-v4-flash", "temperature": 0.0,
           "prompt": BOS + plain},
          lambda r: (r["choices"][0]["text"], r["choices"][0].get("finish_reason")))
    probe("generate (text has BOS)", "/generate",
          {"text": BOS + plain,
           "sampling_params": {"temperature": 0.0, "max_new_tokens": 256}},
          lambda r: (r["text"], r["meta_info"].get("finish_reason")))

    print("=" * 78)
    print("C. multi-turn: does the reply's own content feed BOS back in?")
    print("=" * 78)
    # pi keeps a conversation; if a previous reply ever started with BOS, the next
    # turn would carry it. Simulate two turns with the first reply being BOS-heavy.
    try:
        r1 = post("/v1/chat/completions",
                  {"model": "deepseek-v4-flash", "temperature": 0.0,
                   "messages": [{"role": "user", "content": plain}]})
        first = r1["choices"][0]["message"].get("content") or ""
        r2 = post("/v1/chat/completions",
                  {"model": "deepseek-v4-flash", "temperature": 0.0,
                   "messages": [{"role": "user", "content": plain},
                                {"role": "assistant", "content": first},
                                {"role": "user", "content": "Now say OK."}]})
        second = r2["choices"][0]["message"].get("content") or ""
        print(f"turn1 bos={first.count(BOS)}  {repr(first[:70])}")
        print(f"turn2 bos={second.count(BOS)}  {repr(second[:70])}")
        if second.count(BOS) >= 3:
            print("*** BOS REPRODUCED in multi-turn ***")
    except Exception as e:
        print("multi-turn failed:", type(e).__name__, str(e)[:80])


if __name__ == "__main__":
    main()
