"""Capture greedy continuations from a running server for parity comparison.

Byte-identical greedy output across two builds is the strongest cheap evidence
that a weight-layout or kernel change did not alter model behaviour: any wrong
nibble, wrong scale or wrong address shows up as a divergent token, usually
immediately. Temperature 0, no sampling randomness, fixed seed.

Usage: python capture_greedy.py <outfile> [n_prompts]
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:30000/generate"

# Prompts chosen to exercise different machinery: arithmetic (multi-token
# numeric continuation), code (indentation/brackets), recall of an unusual
# string (tests the KV gather), and a long factual list (tests topk selection).
PROMPTS = [
    "Calculate step by step: 17 * 23 = ",
    "Write a Python function that reverses a linked list, with type hints:",
    "The rare word 'defenestration' means ",
    "List the first 12 elements of the periodic table in order, one per line:",
    "Translate to French, keep the sentence structure: The server was down all night, but nobody called.",
    "Explain in three sentences why the sky is blue at noon and red at sunset:",
]


def gen(prompt, max_new=48):
    body = json.dumps({
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0, "top_p": 1.0, "top_k": -1,
            "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "max_new_tokens": max_new, "ignore_eos": False,
        },
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


out_path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else len(PROMPTS)
rows = []
for i, p in enumerate(PROMPTS[:n]):
    try:
        d = gen(p)
        text = d.get("text") or (d.get("meta_info") or {}).get("text", "")
        rows.append({"prompt": p, "text": text})
        print(f"[{i}] {len(text)} chars: {text[:70]!r}")
    except Exception as e:
        rows.append({"prompt": p, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        print(f"[{i}] ERROR {type(e).__name__}: {str(e)[:120]}")

with open(out_path, "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
print(f"wrote {len(rows)} rows -> {out_path}")
