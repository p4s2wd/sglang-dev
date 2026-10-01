#!/usr/bin/env python
"""Reproduce the long-context repetition loop and locate its threshold.

Symptom (from pi analysing this project): once the conversation gets long, the
model stops making progress and cycles a handful of sentences forever, e.g.
    "Let me look at the MoE kernel and the overall pipeline."
    "Actually, let me reconsider. Let me look at the MoE kernel to understand if
     there's a way to optimize it."

Two things have to be separated before anything can be blamed on a component:

1. Is it length-dependent? A sweep over context size says where it starts. A
   threshold is what makes the next A/B meaningful -- without it, "it did not loop
   this time" is just luck.
2. Is it greedy lock-in or corrupted context? Greedy decoding is famous for
   falling into repetition loops once the distribution flattens. If a little
   sampling fixes it, the context is merely weak; if sampling loops too, something
   upstream is destroying the representation.

Context is real project files, not filler, because the report came from reading
this project and lorem ipsum does not exercise the same attention patterns.

Loop detection counts how often a fixed-size window recurs in the output, so it
reports degree rather than a yes/no.
"""
import json
import sys
import urllib.request
import uuid
from pathlib import Path

URL = "http://127.0.0.1:8200"
ROOT = Path("/data/nvme/sglang-codex")

# Ordered so the sweep changes how much context is present, not what it is about.
FILES = ["PROGRESS.md", "PREFILL_PROFILE.md", "sglang-analysis-report.md",
         "llama.cpp-analysis-report.md", "sglang-sm75-progress.md"]


def build_context(target_tokens, tokenizer=None):
    """Concatenate project files until the request carries ~target_tokens."""
    parts, total = [], 0
    for name in FILES:
        p = ROOT / name
        if not p.exists():
            continue
        text = p.read_text(errors="replace")
        parts.append(f"=== FILE: {name} ===\n{text}")
        total += len(text) // 3  # ~3 chars/token for code+prose mix
        if total >= target_tokens:
            break
    return "\n\n".join(parts)


def ask(context, question, max_tokens=1024, temperature=0.0, timeout=900):
    body = {"model": "deepseek-v4-flash", "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user",
                          "content": context + "\n\n=== QUESTION ===\n" + question}]}
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    ch = r["choices"][0]
    msg = ch["message"]
    return {"content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "finish": ch.get("finish_reason"),
            "usage": r.get("usage", {})}


def loop_score(text, window=80, min_repeat=3):
    """How many times the most-repeated window-length slice occurs."""
    if len(text) < window * min_repeat:
        return 0
    best, seen = 0, {}
    step = max(1, window // 4)
    for i in range(0, len(text) - window, step):
        seen[text[i:i + window]] = seen.get(text[i:i + window], 0) + 1
    return max(seen.values())


Q = ("Based on the documents above, what is the single most impactful optimization "
     "left to do, and why? Answer in three sentences.")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        sizes = [2000, 8000, 16000, 32000, 64000]
        for target in sizes:
            ctx = build_context(target)
            out = ask(ctx, Q)
            both = out["content"] + out["reasoning"]
            sc = loop_score(both)
            flag = "LOOP" if sc >= 3 else "ok  "
            print(f"[{flag}] ctx~{target:6d}  finish={out['finish']:8s} "
                  f"prompt={out['usage'].get('prompt_tokens'):6} "
                  f"comp={out['usage'].get('completion_tokens'):5} "
                  f"rep={sc:3d}  {out['content'][:90]!r}")
            sys.stdout.flush()
    elif mode == "temp":
        ctx = build_context(64000)
        for temp in (0.0, 0.3, 0.7):
            out = ask(ctx, Q, temperature=temp)
            both = out["content"] + out["reasoning"]
            sc = loop_score(both)
            flag = "LOOP" if sc >= 3 else "ok  "
            print(f"[{flag}] temp={temp}  finish={out['finish']:8s} rep={sc:3d}  "
                  f"{out['content'][:90]!r}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
