#!/bin/bash
# Long-context probe: does a single request with a ~200K-token prompt work?
#
# Sends a prompt built to a target token count (not a word count, which cannot
# reach 200K in a shell loop), asks for a short answer, and reports the wall
# time. A long prompt exercises the parts a short one does not: the KV pool at
# full depth, the DSA indexer over 200K candidates, and the sparse-MLA gather
# chunking that bounds the per-call peak.
set -u
PORT="${PORT:-30000}"
PY=/data/nvme/sglang-codex/.venv/bin/python

"$PY" - "$PORT" "${@:-1000 20000 100000 200000}" <<'PY'
import json, random, string, sys, time
import urllib.request, urllib.error

port = sys.argv[1]
targets = [int(a) for a in sys.argv[2:]] or [1000, 20000, 100000, 200000]
URL = f"http://127.0.0.1:{port}/generate"

# A ~1 token per word filler, plus a needle so the answer is checkable.
FILLER = "alpha bravo charlie delta echo foxtrot golf hotel india juliet "


def call(prompt, max_new=24):
    body = json.dumps({
        "text": prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    return out, time.perf_counter() - t0


print(f"{'target':>9} {'prompt_tok':>11} {'wall_s':>8} {'prefill_t/s':>12}  result")
for tgt in targets:
    reps = tgt // 9 + 1
    # A random nonce at the START, so the radix/prefix cache cannot absorb the
    # prompt: without it a repeated probe reports the cache-hit speed (~23K t/s)
    # instead of the prefill speed, and any utilization sampled alongside is 0%.
    nonce = "".join(random.choice(string.ascii_letters) for _ in range(24))
    # Put a needle near the start so a correct answer is verifiable.
    prompt = (f"{nonce} The secret number is 4171. Remember it. "
              + FILLER * reps
              + " What was the secret number?")
    try:
        out, dt = call(prompt)
        pt = out["meta_info"]["prompt_tokens"]
        txt = out["text"].strip().replace("\n", " ")[:40]
        print(f"{tgt:>9} {pt:>11} {dt:>8.1f} {pt/dt:>12.0f}  ok: {txt!r}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:90] if hasattr(e, "read") else ""
        print(f"{tgt:>9} {'-':>11} {'-':>8}  HTTP {e.code}: {detail}")
    except Exception as e:
        print(f"{tgt:>9} {'-':>11} {'-':>8}  FAILED {type(e).__name__}: "
              f"{str(e)[:70]}")
        break
PY
