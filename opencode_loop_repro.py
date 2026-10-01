#!/usr/bin/env python
"""Replay the opencode tool-call loop, and test what actually breaks it.

The pi investigation produced a noisy, partly stochastic failure. This session is a
much cleaner instrument: 34 consecutive assistant turns where the tool call, the
tool output (3342 chars) and the output token count (364) are all byte-identical.
The model re-runs the same grep, receives the same bytes it already had, and issues
the same call again, until the user aborts.

Two details that matter for reading the results:

  * This loop has no reasoning at all (`reasoning` tokens are 0 everywhere, the
    provider runs without thinking), so it is a different failure from the pi
    thinking-loop -- and a worse one for an agent, because the only output is a
    tool call that has already been made.
  * The replay sends the history as it stood when the loop began, then the tool
    result, and asks for the next action. Greedy is expected to re-issue the same
    call. That is the baseline; anything that changes it is attributable.

Sampling parameters are supplied by the caller so the same conversation can be
tested under each. opencode's provider sets no options, so the live path is
greedy (model generation_config temperature 0).
"""
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

DB = Path.home() / ".local/share/opencode/storage/../opencode.db"
DB = Path.home() / ".local/share/opencode/opencode.db"
SESSION = "ses_f12a25898ffe1ypJUExVYX9aQm"
URL = "http://127.0.0.1:8200/v1/chat/completions"

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description":
        "Run a shell command.", "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read", "description":
        "Read a file.", "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
]


def load():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cur = con.cursor()
    rows = cur.execute(
        "select id,data from message where session_id=? order by time_created",
        (SESSION,)).fetchall()
    msgs = []
    for mid, raw in rows:
        d = json.loads(raw)
        parts = [json.loads(x[0]) for x in cur.execute(
            "select data from part where message_id=? order by time_created", (mid,))]
        msgs.append((mid, d, parts))
    return msgs


def to_openai(mid, d, parts):
    """opencode stores tool RESULTS inside the assistant message, as `tool` parts
    alongside the `toolCall` info. The previous version emitted the tool_calls and
    silently dropped the outputs, which collapsed a 100K-token conversation to 3.5K
    and made every replay meaningless. Reasoning parts are likewise part of what
    the model saw, so they are inlined into the assistant text.
    """
    role = d.get("role")
    texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
    think = [p.get("reasoning", "") or p.get("text", "")
             for p in parts if p.get("type") == "reasoning"]
    tools = [p for p in parts if p.get("type") == "tool"]

    if role == "user":
        return {"role": "user", "content": "\n".join(t for t in texts if t.strip())}

    if role == "assistant":
        body = "\n".join(t for t in texts if t.strip())
        if any(t.strip() for t in think):
            body = ("\n".join(t for t in think if t.strip()) + "\n\n" + body).strip()
        out = [{"role": "assistant", "content": body or None}]
        for i, p in enumerate(tools):
            st = p.get("state") or {}
            cid = p.get("callID") or f"call_{i}"
            out[0].setdefault("tool_calls", []).append({
                "id": cid, "type": "function",
                "function": {"name": p.get("tool") or "bash",
                             "arguments": json.dumps(st.get("input") or {})}})
            out.append({"role": "tool", "tool_call_id": cid,
                        "content": st.get("output") or ""})
        return out
    return None


def build(msgs, upto):
    """Conversation as it stood just before message index `upto`."""
    out = []
    for mid, d, parts in msgs[:upto]:
        conv = to_openai(mid, d, parts)
        if isinstance(conv, list):
            out.extend(conv)
        elif conv:
            out.append(conv)
    return [m for m in out if m]


def ask(messages, extra, max_tokens=2000, timeout=1200):
    body = {"model": "deepseek-v4-flash", "messages": messages,
            "max_tokens": max_tokens, "stream": False, "tools": TOOLS}
    body.update(extra or {})
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    ch = r["choices"][0]
    return ch, r.get("usage", {}), round(time.time() - t0)


def call_of(ch):
    tc = ch["message"].get("tool_calls") or []
    if not tc:
        return None
    try:
        args = json.loads(tc[0]["function"]["arguments"])
    except Exception:
        args = {"raw": tc[0]["function"]["arguments"][:80]}
    return (tc[0]["function"]["name"], json.dumps(args, sort_keys=True)[:110])


def main():
    msgs = load()
    # Find where the lock-in starts: first message with output == 364.
    first = None
    for i, (mid, d, parts) in enumerate(msgs):
        if (d.get("tokens") or {}).get("output") == 364:
            first = i
            break
    # Replay from INSIDE the loop: the lock is not present at the first repeat, it
    # is the accumulation of identical (call, result) pairs that builds it.
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else first + 23
    conv = build(msgs, idx)
    print(f"messages={len(msgs)}  first repeat at {first}  replaying from {idx}")
    print(f"replay context: {len(conv)} messages\n")

    # The exact command the live session looped on, taken from the DB rather than
    # retyped, so "did it loop again" is a comparison against what actually happened.
    looped_cmd = None
    for mid, d, parts in msgs[first:]:
        for pp in parts:
            if pp.get("type") == "tool":
                looped_cmd = json.dumps((pp.get("state") or {}).get("input") or {},
                                        sort_keys=True)
                break
        if looped_cmd:
            break
    print(f"基准(线上循环的命令): {looped_cmd[:150]}...\n")

    reps = 3
    cases = [
        ("greedy (live config)", {"temperature": 0.0}),
        ("temperature 0.6", {"temperature": 0.6}),
        ("temp 0.6 + rep_pen 1.1", {"temperature": 0.6, "repetition_penalty": 1.1}),
    ]
    for label, extra in cases:
        hits = 0
        outs = []
        for _ in range(reps):
            ch, usage, secs = ask(conv, extra)
            c = call_of(ch)
            outs.append(usage.get("completion_tokens"))
            if c and c[1] == looped_cmd:
                hits += 1
        print(f"{label:26s} 复现同一 grep: {hits}/{reps}   out={outs}")
        sys.stdout.flush()
    return
if __name__ == "__main__":
    main()
