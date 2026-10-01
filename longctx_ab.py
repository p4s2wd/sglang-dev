#!/usr/bin/env python
"""A/B instrument for long-context behaviour on the local server.

Replays the real session's final state -- the conversation that pi actually sent,
with assistant thinking inlined, ending on a tool result so the model is asked for
the next action rather than answering a fresh question. That state is deterministic
(two identical greedy runs), so a change in the numbers is attributable to the
server config, not to sampling noise.

Three things are measured, because long-context degradation shows up in all three
and only the last two are actually actionable:

  repetition   counts of the two sentences that ran away in the real session.
               Greedy decoding in a long thinking stream is the known mechanism.
  dsml_leak    the model emitting `<|DSML|tool_calls>` as plain text instead of a
               structured tool call. When this fires the agent is broken even though
               the HTTP response looks fine, so it is easy to miss.
  tool_call    whether a structured tool call was produced at all.

Run it once per server config and diff the output.
"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from abort_poison_repro import SESSION, to_openai, metrics  # noqa: E402

URL = "http://127.0.0.1:8200/v1/chat/completions"
MODEL = "deepseek-v4-flash"
DSML = "<｜DSML｜"


def ask(messages, extra=None, max_tokens=16000, timeout=1800):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "stream": False}
    body.update(extra or {})
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    ch = r["choices"][0]
    return ch, r.get("usage", {})


def main():
    rows = [json.loads(l) for l in SESSION.read_text().splitlines()]
    msgs = [r for r in rows if r.get("type") == "message"]
    conv = [m for m in (to_openai(r) for r in msgs[:-1]) if m]

    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    ch, usage = ask(conv, extra={"temperature": 0.0})
    msg = ch["message"]
    think = msg.get("reasoning_content") or ""
    text = msg.get("content") or ""
    st = metrics(think, text)
    tool_calls = msg.get("tool_calls") or []

    print(f"[{tag}] prompt={usage.get('prompt_tokens')} "
          f"reasoning={usage.get('reasoning_tokens')} "
          f"completion={usage.get('completion_tokens')} "
          f"finish={ch.get('finish_reason')}")
    print(f"[{tag}] reconsider x{st['reconsider']}  look_moe x{st['look_moe']}  "
          f"dup_span={st['dup_span']}  think_chars={len(think)}")
    print(f"[{tag}] structured_tool_calls={len(tool_calls)}  "
          f"dsml_leak={DSML in text}")
    print(f"[{tag}] text_head: {text[:130]!r}")


if __name__ == "__main__":
    main()
