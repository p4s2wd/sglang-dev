#!/usr/bin/env python
"""Greedy parity capture + compare, port-parameterised.

`capture_greedy.py` hardcodes :30000 (the dead dev server) and has no compare
mode, so it cannot be used to police an A/B against production on :8200.

Byte-identical greedy output across two arms is the cheapest strong evidence
that a scheduling/knurl change did not alter model behaviour: any wrong nibble,
wrong scale or wrong address surfaces as a divergent token. Temperature 0, no
sampling randomness, fixed seed.

  greedy.py capture <out.json> [--port 8200] [--n 6]
  greedy.py compare <a.json> <b.json>
"""
import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "Calculate step by step: 17 * 23 = ",
    "Write a Python function that reverses a linked list, with type hints:",
    "The rare word 'defenestration' means ",
    "List the first 12 elements of the periodic table in order, one per line:",
    "Translate to French, keep the sentence structure: The server was down all night, but nobody called.",
    "Explain in three sentences why the sky is blue at noon and red at sunset:",
]


def gen(url, prompt, max_new=48, timeout=600):
    body = json.dumps({
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0, "top_p": 1.0, "top_k": -1,
            "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "max_new_tokens": max_new, "ignore_eos": False,
        },
        "stream": False,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def capture(args):
    url = f"http://{args.host}:{args.port}/generate"
    rows = []
    for i, p in enumerate(PROMPTS[:args.n]):
        try:
            d = gen(url, p, args.max_new, args.timeout)
            text = d.get("text") or (d.get("meta_info") or {}).get("text", "")
            rows.append({"prompt": p, "text": text})
            print(f"[{i}] {len(text):>4} chars: {text[:70]!r}")
        except Exception as e:
            rows.append({"prompt": p,
                         "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"[{i}] ERROR {type(e).__name__}: {str(e)[:120]}")
    with open(args.out, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(rows)} rows -> {args.out}")


def compare(args):
    a = json.load(open(args.a))
    b = json.load(open(args.b))
    if len(a) != len(b):
        print(f"ROW COUNT DIFFERS: {len(a)} vs {len(b)}")
        return 2
    bad = 0
    for i, (ra, rb) in enumerate(zip(a, b)):
        ta, tb = ra.get("text", ""), rb.get("text", "")
        if ta == tb:
            print(f"[{i}] IDENTICAL ({len(ta)} chars)")
            continue
        bad += 1
        if ra.get("error") or rb.get("error"):
            print(f"[{i}] ERROR MISMATCH\n     a={ra.get('error')}\n     b={rb.get('error')}")
            continue
        j = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]),
                 min(len(ta), len(tb)))
        print(f"[{i}] DIFFER at char {j} (len {len(ta)} vs {len(tb)})")
        print(f"     a: ...{ta[max(0, j - 30):j + 40]!r}")
        print(f"     b: ...{tb[max(0, j - 30):j + 40]!r}")
    print()
    if bad == 0:
        print(f"PARITY OK: all {len(a)} prompts byte-identical")
        return 0
    print(f"PARITY FAIL: {bad}/{len(a)} prompts differ")
    return 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture")
    c.add_argument("out")
    c.add_argument("--port", type=int, default=8200)
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--n", type=int, default=len(PROMPTS))
    c.add_argument("--max-new", type=int, default=48)
    c.add_argument("--timeout", type=int, default=600)
    c.set_defaults(func=capture)

    d = sub.add_parser("compare")
    d.add_argument("a")
    d.add_argument("b")
    d.set_defaults(func=compare)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
