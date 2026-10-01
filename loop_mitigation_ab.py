#!/usr/bin/env python
"""A/B the mitigations against the reproduced loop.

Reproduction (deterministic, greedy, tools present, the real 104K session state):
    finish_reason = length, 16000/16000 tokens spent, 67198 chars of output,
    "Actually, let me reconsider" x219 -- the same runaway as the real session's
    aborted turn, which had it x119.

So this is a real, reproducible failure of the serving path, not a one-off.

The budget is cut to 4000 tokens for the comparison. The loop starts long before
that, and the metric is a count within a fixed budget, so it stays comparable
across configs while costing a quarter of the time. Greedy at 4000 is included as
the baseline so the shorter budget is measured on the same footing.

Excluding the KV-cache hypothesis is also re-checked here rather than assumed from
the earlier no-tools run, because that run never reached the loop.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from abort_poison_repro import SESSION, to_openai  # noqa: E402

URL = "http://127.0.0.1:8200/v1/chat/completions"
TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                    "required": ["command"]}}},
]
BUDGET = 4000


def ask(messages, extra, max_tokens=BUDGET, timeout=2400):
    body = {"model": "deepseek-v4-flash", "messages": messages,
            "max_tokens": max_tokens, "stream": False, "tools": TOOLS}
    body.update(extra)
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    ch = r["choices"][0]
    msg = ch["message"]
    return ch, msg, r.get("usage", {})


def score(msg):
    s = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
    return {
        "reconsider": s.count("Actually, let me reconsider"),
        "look_moe": s.count("Let me look at the MoE kernel"),
        "chars": len(s),
        "tool_calls": len(msg.get("tool_calls") or []),
    }


CASES = [
    ("greedy (baseline)", {"temperature": 0.0}),
    ("temperature 0.6", {"temperature": 0.6}),
    ("temperature 1.0", {"temperature": 1.0}),
    ("greedy + repetition_penalty 1.05", {"temperature": 0.0, "repetition_penalty": 1.05}),
    ("greedy + frequency_penalty 0.3", {"temperature": 0.0, "frequency_penalty": 0.3}),
]


def main():
    rows = [json.loads(l) for l in SESSION.read_text().splitlines()]
    msgs = [r for r in rows if r.get("type") == "message"]
    conv = [m for m in (to_openai(r) for r in msgs[:-1]) if m]
    print(f"context: {len(conv)} messages, budget {BUDGET}\n")
    for label, extra in CASES:
        t0 = time.time()
        try:
            ch, msg, usage = ask(conv, extra)
            s = score(msg)
            print(f"{label:36s} finish={str(ch.get('finish_reason')):7s} "
                  f"comp={usage.get('completion_tokens'):5d} "
                  f"reconsider x{s['reconsider']:4d} "
                  f"look_moe x{s['look_moe']:3d} "
                  f"chars={s['chars']:6d} tool_calls={s['tool_calls']} "
                  f"({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"{label:36s} FAILED {type(e).__name__}: {str(e)[:80]}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
