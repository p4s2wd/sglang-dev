#!/bin/bash
# Throughput + a few quality probes against a running server.
# Usage: ./probe_perf.sh [rounds] [tokens_per_round]
PORT="${PORT:-30000}"
PY=/data/nvme/sglang-codex/.venv/bin/python
ROUNDS="${1:-6}"
NTOK="${2:-64}"

"$PY" - "$PORT" "$ROUNDS" "$NTOK" <<'PY'
import json, sys, time
import urllib.request

port, rounds, ntok = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
URL = f"http://127.0.0.1:{port}/generate"

def call(prompt, max_new, temperature=0.0):
    body = json.dumps({
        "text": prompt,
        "sampling_params": {"temperature": temperature, "max_new_tokens": max_new},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    return out, dt

# warmup (JIT/caches) then timed rounds
call("Hello.", 8)
print(f"{'round':>5} {'in':>4} {'out':>4} {'sec':>7} {'tok/s':>7}")
tot_tok, tot_sec = 0, 0.0
for i in range(rounds):
    out, dt = call("Explain briefly why the sky appears blue during the day.", ntok)
    n = out["meta_info"]["completion_tokens"]
    pt = out["meta_info"]["prompt_tokens"]
    tot_tok += n; tot_sec += dt
    print(f"{i:>5} {pt:>4} {n:>4} {dt:>7.2f} {n/dt:>7.2f}")
print(f"decode throughput: {tot_tok/tot_sec:.2f} tok/s over {tot_tok} tokens")

print("\n--- quality probes (greedy) ---")
for prompt in (
    "Q: What is 17 * 4? A:",
    "The three primary colors in painting are red, blue, and",
    "Write a Python function that returns the nth Fibonacci number:",
    "Translate to French: Good morning, how are you?",
):
    out, dt = call(prompt, 48)
    print(f"\n[{prompt[:52]}]  ({dt:.1f}s)")
    print("  " + out["text"].replace("\n", "\n  ")[:400])
PY
