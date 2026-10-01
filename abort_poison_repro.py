#!/usr/bin/env python
"""Fidelity fix for the replay, then A/B the mitigation.

The first replay reconstructed only 74K prompt tokens against a real 105K, because
it kept assistant `text` and dropped assistant `thinking`. That is the part of the
history that matters most here: the repetitive thinking is precisely what primes the
next turn to repeat. So the replay was easier than the real thing and could not
reproduce the loop. Thinking is now inlined, which is what pi does when the model
declares `requiresThinkingAsText` / `requiresReasoningContentOnAssistantMessages`.

Then the mitigation is measured rather than assumed. The failure is greedy lock-in
inside the thinking stream, so the candidate knobs are a small temperature and a
repetition penalty; both are supported by pi's `samplingParams` passthrough, so no
server change is needed to test them.

Loop metrics are counts of the two sentences that actually ran away in the real
session (116x and 54x there), plus the longest immediately-repeated span, which
catches a cycle that happens to use different words.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

SESSION = Path("/home/shuang/.pi/agent/sessions/--data-nvme-sglang-codex--/"
               "2026-09-29T06-42-48-517Z_01a0ebe6-d444-702a-9c6a-12ff37fdd84a.jsonl")
LOCAL = "http://127.0.0.1:8200/v1/chat/completions"
MODEL = "deepseek-v4-flash"


def to_openai(rec, inline_thinking=True):
    m = rec["message"]
    role, content = m.get("role"), m.get("content")
    blocks = content if isinstance(content, list) else []

    if role == "user":
        return {"role": "user", "content": "\n".join(
            b.get("text", "") for b in blocks if b.get("type") == "text")}

    if role == "assistant":
        text = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
        think = "\n".join(b.get("thinking", "") for b in blocks
                          if b.get("type") == "thinking")
        if inline_thinking and think:
            text = (think + "\n\n" + text).strip()
        msg = {"role": "assistant", "content": text or None}
        calls = [b for b in blocks if b.get("type") == "toolCall"]
        if calls:
            msg["tool_calls"] = [{
                "id": c.get("id", f"call_{i}"), "type": "function",
                "function": {"name": c["name"],
                             "arguments": json.dumps(c.get("arguments", c.get("input", {})))},
            } for i, c in enumerate(calls)]
        return msg

    if role == "toolResult":
        return {"role": "tool", "tool_call_id": m.get("toolCallId", ""),
                "content": "\n".join(b.get("text", "") for b in blocks
                                     if b.get("type") == "text")}
    return None


def ask(messages, extra=None, max_tokens=16000, timeout=1800):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "stream": False}
    body.update(extra or {})
    req = urllib.request.Request(
        LOCAL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    ch = r["choices"][0]
    msg = ch["message"]
    think = msg.get("reasoning_content") or ""
    return {"text": msg.get("content") or "", "think": think,
            "finish": ch.get("finish_reason"), "usage": r.get("usage", {})}


def longest_repeated_span(text, min_len=40):
    """Length of the longest span that immediately repeats at least twice."""
    best = 0
    n = len(text)
    step = 10
    for size in range(min_len, 400, step):
        i = 0
        while i + 2 * size <= n:
            if text[i:i + size] == text[i + size:i + 2 * size]:
                best = max(best, size)
                break
            i += step
        if best >= size:
            break
    return best


def metrics(think, text):
    s = think + "\n" + text
    return {"reconsider": s.count("Actually, let me reconsider"),
            "look_moe": s.count("Let me look at the MoE kernel"),
            "dup_span": longest_repeated_span(s),
            "chars": len(s)}


def main():
    rows = [json.loads(l) for l in SESSION.read_text().splitlines()]
    msgs = [r for r in rows if r.get("type") == "message"]
    aborted = msgs[-1]
    prefix = [m for m in (to_openai(r) for r in msgs[:-1]) if m]
    follow = "Continue. What is the single most impactful optimization left, and why?"

    cases = [
        ("thinking OMITTED (old replay, 74K)", dict(inline=False)),
        ("thinking INLINED, greedy (faithful)", dict(inline=True, extra={"temperature": 0.0})),
        ("thinking INLINED, temp 0.6", dict(inline=True, extra={"temperature": 0.6})),
        ("thinking INLINED, greedy + rep_penalty 1.05",
         dict(inline=True, extra={"temperature": 0.0, "repetition_penalty": 1.05})),
    ]
    for label, cfg in cases:
        inline = cfg.get("inline", True)
        extra = cfg.get("extra")
        conv = [to_openai(r, inline_thinking=inline) for r in msgs[:-1]]
        conv = [m for m in conv if m] + [{"role": "user", "content": follow}]
        print(f"=== {label} ===")
        try:
            out = ask(conv, extra=extra)
            st = metrics(out["think"], out["text"])
            print(f"  prompt={out['usage'].get('prompt_tokens')} "
                  f"comp={out['usage'].get('completion_tokens')} "
                  f"reasoning={out['usage'].get('reasoning_tokens')} "
                  f"finish={out['finish']}")
            print(f"  reconsider x{st['reconsider']}  look_moe x{st['look_moe']}  "
                  f"dup_span={st['dup_span']}  chars={st['chars']}")
            print(f"  text: {out['text'][:160]!r}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:160]}")
        print()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
