#!/bin/bash
# Capture prefill- and decode-stage Chrome traces from a running sglang server.
#
# The server's /start_profile with profile_by_stage=true keys the torch profiler
# off each batch's forward_mode: a prefill batch starts the EXTEND capture, and
# the first decode batch force-flushes it and starts the DECODE capture. So one
# request yields two separate traces -- no need to time two requests.
#
# The CUPTI buffer the GPU activity recorder needs is ~314 MiB per rank, and on
# a 22 GiB card holding 156G of weights there is no room for it: the server dies
# with CUDA OOM the moment /start_profile lands. CPU-only recording skips that
# buffer, so the default ACTS is '"CPU"'. For a GPU/kernel trace, restart the
# server with a config that leaves headroom (smaller weights config, or a
# dummy-weight server) and pass ACTS='"CPU","GPU"'.
#
# Usage: ACTS='"CPU","GPU"' ./profile_probe.sh [num_steps] [words] [max_new]
set -u
ACTS="${ACTS:-[\"CPU\"]}"
cd /data/nvme/sglang-codex
. ./env.sh

PORT="${PORT:-30000}"
STEPS="${1:-20}"
WORDS="${2:-900}"
NEWTOK="${3:-40}"
PY=/data/nvme/sglang-codex/.venv/bin/python

OUT=/data/nvme/sglang-codex/profiles/$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
echo "output_dir=$OUT"

# A prompt long enough to span more than one chunked-prefill batch (512), so the
# EXTEND trace contains several prefill batches to average over.
PROMPT=$("$PY" -c "print('history of the roman empire ' * $WORDS)")

"$PY" - "$PORT" "$OUT" "$STEPS" "$PROMPT" "$NEWTOK" "$ACTS" <<'PY'
import json, sys, time
import urllib.request

port, out_dir, steps, prompt, newtok = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
acts = json.loads(sys.argv[6])
print(f"activities={acts}", flush=True)
BASE = f"http://127.0.0.1:{port}"


def post(path, body=None):
    data = json.dumps(body).encode() if body is not None else b"{}"
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.read().decode()


def generate(text, max_new):
    t0 = time.perf_counter()
    raw = post("/generate", {
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
    })
    dt = time.perf_counter() - t0
    out = json.loads(raw)
    n = out["meta_info"]["completion_tokens"]
    return out, dt, n


# Warm up so JIT/graph capture cost lands outside the trace.
generate("hello", 8)
generate(prompt[:2000], 8)
print("warmup done", flush=True)

print(post("/start_profile", {
    "output_dir": out_dir,
    "activities": acts,
    "profile_by_stage": True,
    "record_shapes": True,
    "with_stack": False,
    "num_steps": steps,
    "profile_prefix": "dsv4",
}), flush=True)

t0 = time.perf_counter()
out, dt, n = generate(prompt, newtok)
print(f"profiled request: prompt={out['meta_info']['prompt_tokens']}tok "
      f"completion={n}tok wall={dt:.2f}s", flush=True)

# The decode capture stops itself after num_steps; give it time to write.
time.sleep(20)
print(post("/stop_profile"), flush=True)
PY

echo "=== traces ==="
ls -la "$OUT" 2>/dev/null | tail -12
