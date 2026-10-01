#!/usr/bin/env python
"""Reproduce the loop the user actually saw: an agentic tool-calling session.

The single-request sweep showed no degeneration even at 58K prompt tokens, so the
loop is not a long-context decoding failure. What the report describes --

    "Let me look at the MoE kernel and the overall pipeline.
     Actually, let me reconsider. Let me look at the MoE kernel ...
     Let me look at the MoE kernel and the overall pipeline."

-- is a whole-conversation failure: the same sentence recurs, and with it the same
tool call recurs. In a coding agent that means the session never advances.

This probe therefore drives a real tool-calling loop instead of one completion.
A stub `read_file` tool is defined; the model is asked to optimise the MoE kernel;
whatever it requests is answered with plausible file content and fed back, for up
to N turns. No code runs -- the point is to observe the choice sequence.

Loop detection is on the (tool_name, argument) pairs, because that is the part
that is actually wasted work: re-reading the same file is a no-op, so a repeated
call is a stalled session even if the prose varies.

Greedy decoding is the default here, matching the deployed generation_config
(temperature 0). A deterministic policy that lands in a fixed point cannot escape
it, so the temperature comparison at the end is the informative part.
"""
import json
import sys
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8200"
ROOT = Path("/data/nvme/sglang-codex")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the project and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "Project-relative path, e.g. PROGRESS.md"},
            },
            "required": ["path"],
        },
    },
}]

TASK = ("We are optimising this SGLang fork for sm75 GPUs. Read the relevant "
        "source and tell me what to optimise next. Start by investigating.")


def stub_result(path):
    # Not truncated: a truncated stub makes the model believe there is "the rest"
    # to read, which induces repeats for a reason that has nothing to do with the
    # bug under investigation. Re-reading a file must return what it returned
    # before, exactly as a real agent would see.
    p = ROOT / path
    if p.exists() and p.is_file():
        text = p.read_text(errors="replace")
        return text or "(empty file)"
    # Plausible stand-in so the model has something to react to either way.
    return (f"// {path}\n// (stub content for offline loop probe)\n"
            "def moe_combine_kernel(...):\n    # expert combine\n    pass\n")


def turn(messages, temperature, max_tokens=800, timeout=900):
    body = {"model": "deepseek-v4-flash", "temperature": temperature,
            "max_tokens": max_tokens, "messages": messages, "tools": TOOLS}
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    ch = r["choices"][0]
    return ch["message"], ch.get("finish_reason"), r.get("usage", {})


def run(temperature, max_turns=8, verbose=False):
    messages = [{"role": "user", "content": TASK}]
    calls = []
    for i in range(max_turns):
        msg, fin, usage = turn(messages, temperature)
        tc = msg.get("tool_calls") or []
        text = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()

        if not tc:
            print(f"  turn{i}: no tool call, finish={fin}, content={text[:70]!r}")
            return calls, i, "answered"

        c = tc[0]["function"]
        key = (c["name"], c.get("arguments", "")[:60])
        calls.append(key)
        print(f"  turn{i}: {c['name']}({c.get('arguments','')[:50]}) "
              f"finish={fin} content={text[:60]!r}")
        if verbose and text:
            print(f"          prose: {text[:150]!r}")
        sys.stdout.flush()

        messages.append({"role": "assistant", "content": text,
                         "tool_calls": tc})
        messages.append({"role": "tool", "tool_call_id": tc[0]["id"],
                         "name": c["name"],
                         "content": stub_result(json.loads(c["arguments"]).get("path", ""))})
    return calls, max_turns, "exhausted"


def repeated(calls):
    """Longest suffix run that repeats a call already made, plus total dupes."""
    seen, dup = set(), 0
    for k in calls:
        if k in seen:
            dup += 1
        seen.add(k)
    return dup, len(set(calls))


if __name__ == "__main__":
    temp = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    turns = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(f"=== agentic loop probe, temperature={temp}, max_turns={turns} ===")
    calls, n, why = run(temp, turns, verbose=True)
    dup, uniq = repeated(calls)
    print(f"\noutcome={why} turns={n} tool_calls={len(calls)} unique={uniq} "
          f"repeated={dup}")
    if calls and dup >= 2:
        print("VERDICT: LOOP -- the same action is being retried repeatedly")
    elif calls:
        print("VERDICT: no loop -- distinct actions each turn")
