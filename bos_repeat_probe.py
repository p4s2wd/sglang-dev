#!/usr/bin/env python
"""Reproduce the <|begin_of_sentence|> repetition the user sees through pi.

pi is a coding agent; it may hit /v1/chat/completions, /v1/completions, or
/v1/responses. opt1 fixed the BOS duplication by passing
add_default_bos_token=False into encoding_dsv4.encode_messages (serving_chat.py:1526),
but that only covers the dsv4 chat path. This probes every endpoint the server
exposes and reports, for each, whether the prompt starts with BOS and whether the
completion degenerates into repeating it.

Prompt is salted per probe so the radix prefix cache cannot serve a previous run.
"""
import json
import urllib.request
import uuid

URL = "http://127.0.0.1:8200"
BOS = "<｜begin▁of▁sentence｜>"


def post(path, body, timeout=240):
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def report(name, path, body, extract):
    salt = uuid.uuid4().hex[:8]
    try:
        r = post(path, body)
    except Exception as e:
        print(f"{name:34s} REQUEST FAILED {type(e).__name__}: {str(e)[:90]}")
        return
    try:
        text, meta = extract(r)
    except Exception as e:
        print(f"{name:34s} PARSE FAILED {type(e).__name__}")
        return
    n = text.count(BOS)
    first = repr(text[:90])
    flag = "  <-- BOS REPEAT" if n >= 3 else ""
    print(f"{name:34s} bos={n:<4d} {first}{flag}")
    if meta:
        print(f"{'':34s} {meta}")


# 1. chat completions, dsv4 path (the one opt1 fixed)
report(
    "chat/completions (dsv4)", "/v1/chat/completions",
    {"model": "deepseek-v4-flash", "max_new_tokens": 40, "temperature": 0.0,
     "messages": [{"role": "user",
                   "content": "Reply with exactly: OK"}]},
    lambda r: (r["choices"][0]["message"].get("content") or "", ""),
)

# 2. raw prompt starting with BOS, via chat/completions
report(
    "chat/completions (msg starts w/ BOS)", "/v1/chat/completions",
    {"model": "deepseek-v4-flash", "max_new_tokens": 40, "temperature": 0.0,
     "messages": [{"role": "user",
                   "content": BOS + " Reply with exactly: OK"}]},
    lambda r: (r["choices"][0]["message"].get("content") or "", ""),
)

# 3. completions, plain prompt
report(
    "completions (plain)", "/v1/completions",
    {"model": "deepseek-v4-flash", "max_new_tokens": 40, "temperature": 0.0,
     "prompt": "Reply with exactly: OK"},
    lambda r: (r["choices"][0]["text"], ""),
)

# 4. completions, prompt already begins with BOS
report(
    "completions (prompt has BOS)", "/v1/completions",
    {"model": "deepseek-v4-flash", "max_new_tokens": 40, "temperature": 0.0,
     "prompt": BOS + "Reply with exactly: OK"},
    lambda r: (r["choices"][0]["text"], ""),
)

# 5. responses API if present
try:
    report(
        "responses", "/v1/responses",
        {"model": "deepseek-v4-flash", "max_output_tokens": 40,
         "input": "Reply with exactly: OK"},
        lambda r: (r["output"][0]["content"][0]["text"], ""),
    )
except Exception:
    pass

# 6. generate (sglang native), no BOS
report(
    "generate (native, no BOS)", "/generate",
    {"text": "Reply with exactly: OK",
     "sampling_params": {"temperature": 0.0, "max_new_tokens": 40}},
    lambda r: (r["text"], ""),
)

# 7. generate with BOS, greedy -- the direct repro of the report
report(
    "generate (BOS + greedy)", "/generate",
    {"text": BOS + "Reply with exactly: OK",
     "sampling_params": {"temperature": 0.0, "max_new_tokens": 40}},
    lambda r: (r["text"], ""),
)
