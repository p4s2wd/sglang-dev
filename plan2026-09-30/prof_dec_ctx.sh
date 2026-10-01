#!/bin/bash
# Per-kernel decode accounting at a chosen context length.
#
# Why: the decode census in SM75_DSV4_DECODE_PROFILE.md was taken on a one-line
# prompt, where the compressed KV cache is nearly empty. Long-context decode is
# 2-3x slower and the question is which kernel family grows. Guessing from the
# source is unreliable here -- the selection path (c128 / c4 / swa, index_topk)
# has too many interacting pieces -- so measure the same census at two context
# lengths and read off the difference.
#
# The prompt is warmed first so the profiled window contains decode steps and not
# a prefill: the radix cache makes the second call skip prefill, and the trace
# would otherwise be dominated by chunked prefill kernels.
#
#   ./prof_dec_ctx.sh <tag> <words> [steps] [newtok] [port]
#
# words=40 is the historical short-context point (~440 tokens); words=9000 is
# roughly 40K tokens.
set -u
cd /data/nvme/sglang-codex

TAG="${1:-decctx}"
WORDS="${2:-9000}"
STEPS="${3:-12}"
NEWTOK="${4:-24}"
PORT="${5:-8200}"
PY=/data/nvme/sglang/.venv/bin/python
OUT=/data/nvme/sglang-codex/profiles/$(date +%Y%m%d-%H%M%S)-$TAG
mkdir -p "$OUT"
echo "OUT=$OUT  words=$WORDS steps=$STEPS port=$PORT"

"$PY" - "$OUT" "$STEPS" "$WORDS" "$NEWTOK" "$PORT" <<'PY'
import json, sys, time, urllib.request

out_dir, steps, words, newtok, port = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
URL = "http://127.0.0.1:" + port


def post(path, body=None, timeout=2400):
    data = json.dumps(body).encode() if body is not None else b"{}"
    req = urllib.request.Request(URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


PROMPT = "history of the roman empire " * words
try:
    post("/stop_profile")
except Exception:
    pass

# Warm pass: fills the radix cache so the profiled call decodes instead of
# prefilling. This is the expensive one at long context.
t0 = time.perf_counter()
w = json.loads(post("/generate", {"text": PROMPT,
                                  "sampling_params": {"temperature": 0.0,
                                                      "max_new_tokens": 1}}))
print(f"warm prompt_tokens={w['meta_info']['prompt_tokens']} "
      f"wall={time.perf_counter()-t0:.1f}s", flush=True)

print(post("/start_profile", {
    "output_dir": out_dir, "activities": ["CPU", "GPU"],
    "profile_by_stage": True, "record_shapes": True,
    "with_stack": False, "num_steps": steps, "profile_prefix": "p"}), flush=True)

t0 = time.perf_counter()
o = json.loads(post("/generate", {"text": PROMPT,
                                  "sampling_params": {"temperature": 0.0,
                                                      "max_new_tokens": newtok}}))
wall = time.perf_counter() - t0
ct = o["meta_info"].get("completion_tokens")
print(f"profiled prompt_tokens={o['meta_info']['prompt_tokens']} "
      f"completion={ct} wall={wall:.2f}s"
      + (f"  -> {ct/wall:.2f} tok/s" if ct and wall else ""), flush=True)

time.sleep(20)
try:
    print(post("/stop_profile"))
except Exception as e:
    print("stop:", type(e).__name__)
PY

sleep 8
ls -1 "$OUT" 2>/dev/null | head -5
echo "OUT_DIR=$OUT"
