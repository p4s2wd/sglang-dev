#!/usr/bin/env python
"""End-to-end check for the BOS degeneration reported via pi, against the running
server with the fix deployed.

The failure is self-propagating, so the test has to model the whole loop rather
than a single call:

  turn 1     a clean question                     -> must produce real content
  turn 2     that content fed back in             -> must not contain BOS
  poisoned   a history where an earlier turn already degenerated into BOS
             (exactly the state pi got stuck in)  -> must recover
  current    BOS inside the live user message      -> must be handled

A single successful request proves nothing here, because the bug only appears once
BOS has entered the history. The poisoned case is the one that matters.

Two details that a naive probe gets wrong, both of which cost time already:

1. `max_tokens` must leave room for thinking. This checkpoint reasons before it
   answers and the reasoning parser moves that text into `reasoning_content`.
   With a small budget the whole allowance is consumed by thinking, so `content`
   comes back empty with finish_reason=length -- a healthy reply that looks like a
   total failure. Assertions therefore look at content + reasoning together.
2. Prompts are salted so the radix prefix cache cannot serve an earlier turn.
"""
import json
import urllib.request
import uuid

URL = "http://127.0.0.1:8200"
BOS = "<｜begin▁of▁sentence｜>"
BUDGET = 2048


def chat(messages, max_tokens=BUDGET, timeout=300):
    body = {"model": "deepseek-v4-flash", "temperature": 0.0,
            "max_tokens": max_tokens, "messages": messages}
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    ch = r["choices"][0]
    msg = ch["message"]
    return (msg.get("content") or ""), (msg.get("reasoning_content") or ""), ch.get(
        "finish_reason"
    )


fails = 0


def check(name, cond, detail=""):
    global fails
    if not cond:
        fails += 1
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{('  ' + detail) if detail else ''}")


print("1. clean single turn")
try:
    q = f"[{uuid.uuid4().hex[:8]}] Say OK."
    t, rc, fin = chat([{"role": "user", "content": q}])
    check("produced content", len(t.strip()) > 0, f"len={len(t)} finish={fin}")
    check("no BOS anywhere", (t + rc).count(BOS) == 0)
except Exception as e:
    check("request succeeded", False, f"{type(e).__name__}: {str(e)[:70]}")

print("2. two-turn conversation, reply fed back")
try:
    q = f"[{uuid.uuid4().hex[:8]}] Name one river in France."
    a1, a1r, _ = chat([{"role": "user", "content": q}])
    check("turn1 has content", len(a1.strip()) > 0, f"len={len(a1)}")
    check("turn1 has no BOS", (a1 + a1r).count(BOS) == 0)
    a2, a2r, fin2 = chat([{"role": "user", "content": q},
                          {"role": "assistant", "content": a1},
                          {"role": "user", "content": "Now say OK."}])
    check("turn2 has content", len(a2.strip()) > 0, f"len={len(a2)} finish={fin2}")
    check("turn2 has no BOS", (a2 + a2r).count(BOS) == 0)
except Exception as e:
    check("two-turn succeeded", False, f"{type(e).__name__}: {str(e)[:70]}")

print("3. POISONED history -- the state pi was stuck in")
print("   (an earlier turn already degenerated into BOS)")
try:
    q = f"[{uuid.uuid4().hex[:8]}] Read PROGRESS.md and summarize."
    t, rc, fin = chat([{"role": "user", "content": q},
                       {"role": "assistant", "content": BOS * 400},
                       {"role": "user", "content": "Now say OK."}])
    check("recovered: has content", len(t.strip()) > 0, f"len={len(t)} finish={fin}")
    check("recovered: no BOS", (t + rc).count(BOS) == 0)
    print(f"       reply: {t[:70]!r}")
except Exception as e:
    check("poisoned history recovered", False, f"{type(e).__name__}: {str(e)[:70]}")

print("4. BOS in the current user message")
try:
    q = BOS + f"[{uuid.uuid4().hex[:8]}] Say OK."
    t, rc, fin = chat([{"role": "user", "content": q}])
    check("has content", len(t.strip()) > 0, f"len={len(t)} finish={fin}")
    check("no BOS", (t + rc).count(BOS) == 0)
except Exception as e:
    check("handled", False, f"{type(e).__name__}: {str(e)[:70]}")

print()
print(f"FAILS = {fails}")
