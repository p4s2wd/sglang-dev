#!/bin/bash
# Re-capture a decode trace with Python stacks so elementwise kernels can be
# attributed to the model source lines that launch them.
#
# The production trace cannot do this: CUDA graph replay joins every kernel to
# cudaGraphLaunch, so the op->kernel correlation is lost, and with_stack=False
# drops the Python frames that would otherwise identify the caller. Stacks are
# only affordable on the dummy-weight server (they multiply the event count).
set -u
cd /data/nvme/sglang-codex
. ./env.sh
PORT="${PORT:-30000}"
STEPS="${1:-10}"
WORDS="${2:-600}"
NEWTOK="${3:-30}"
PY=/data/nvme/sglang-codex/.venv/bin/python
OUT=/data/nvme/sglang-codex/profiles/$(date +%Y%m%d-%H%M%S)-stack
mkdir -p "$OUT"
echo "output_dir=$OUT"
PROMPT=$("$PY" -c "print('history of the roman empire ' * $WORDS)")
"$PY" - "$PORT" "$OUT" "$STEPS" "$PROMPT" "$NEWTOK" <<'PY'
import json, sys, time
import urllib.request

port, out_dir, steps, prompt, newtok = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
BASE = f"http://127.0.0.1:{port}"


def post(path, body=None):
    data = json.dumps(body).encode() if body is not None else b"{}"
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.read().decode()


def generate(text, max_new):
    raw = post("/generate", {
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
    })
    return json.loads(raw)


# Clear any profiler state a previous failed run left behind, then warm up.
# /stop_profile raises when nothing is running, so treat it as best-effort.
def best_effort(path, body=None):
    try:
        return post(path, body)
    except Exception as e:
        return f"ignored: {type(e).__name__}"


print(best_effort("/stop_profile"), flush=True)
time.sleep(8)
generate("hello", 8)
print("warmup done", flush=True)

print(post("/start_profile", {
    "output_dir": out_dir,
    "activities": ["CPU", "GPU"],
    "profile_by_stage": True,
    "record_shapes": True,
    "with_stack": True,
    "num_steps": steps,
    "profile_prefix": "stk",
}), flush=True)

out, = (generate(prompt, newtok),)
print(f"profiled: prompt={out['meta_info']['prompt_tokens']}tok", flush=True)
time.sleep(25)
print(post("/stop_profile"), flush=True)
PY
ls -la "$OUT" 2>&1 | tail -4
