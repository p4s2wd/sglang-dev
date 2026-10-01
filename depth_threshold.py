#!/usr/bin/env python
"""Find the context depth at which the loop starts.

What the repeats established, so this does not repeat them:

  greedy                 loop on 1/3 runs
  temperature 0.6        loop on 2/3
  temp 0.6 + rep_pen 1.1 loop on 2/5
  fp8 vs bf16 KV cache   byte-identical output

So sampling parameters shift the odds but do not remove the failure, and the KV
cache is not involved. That leaves a capability limit, which means the useful
question is not "how do I stop it" but "below what depth is it reliable". A depth
threshold is actionable: pi can be told the real window and compact before the
model degrades, instead of the 1M that the config advertises.

Depth is varied by dropping messages from the FRONT of the real session, keeping
the tail intact. Keeping the tail matters: the loop is fed by the accumulated
repetitive thinking, so a prefix that ends earlier in the conversation is a
genuinely easier problem rather than the same problem with fewer tokens.

Every depth is run more than once. An earlier single-run comparison in this
investigation reversed its own conclusion, so single samples are not trusted here.
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
    {"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
BUDGET = 4000


def ask(messages, extra=None, timeout=2400):
    body = {"model": "deepseek-v4-flash", "messages": messages,
            "max_tokens": BUDGET, "stream": False, "tools": TOOLS}
    body.update(extra or {"temperature": 0.0})
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    m = r["choices"][0]["message"]
    s = (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")
    return {"reconsider": s.count("Actually, let me reconsider"),
            "prompt": r["usage"].get("prompt_tokens"),
            "comp": r["usage"].get("completion_tokens"),
            "secs": round(time.time() - t0),
            "hit_budget": r["choices"][0].get("finish_reason") == "length"}


def main():
    rows = [json.loads(l) for l in SESSION.read_text().splitlines()]
    msgs = [r for r in rows if r.get("type") == "message"][:-1]
    full = [m for m in (to_openai(r) for r in msgs) if m]

    depths = [12, 20, 28, 36, len(full)]
    reps = 3
    print(f"{'msgs':>5} {'prompt_tok':>10} {'runs':>5}  {'reconsider per run':<22} "
          f"{'comp':<20} verdict")
    for depth in depths:
        conv = full[-depth:] if depth < len(full) else full
        runs = [ask(conv) for _ in range(reps)]
        # A run counts as looped when it burned the whole budget, which is the
        # signature that matters: the turn is wasted, not merely verbose.
        looped = sum(1 for x in runs if x["hit_budget"] or x["reconsider"] >= 5)
        print(f"{len(conv):>5} {runs[0]['prompt']:>10} {reps:>5}  "
              f"{str([x['reconsider'] for x in runs]):<22} "
              f"{str([x['comp'] for x in runs]):<20} "
              f"{'LOOPS' if looped >= 2 else ('borderline' if looped else 'clean')}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
