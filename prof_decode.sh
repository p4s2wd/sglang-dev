#!/bin/bash
# Profile decode on the currently-running server and print the top kernels.
#
# The invocation matters: activities must be ["CPU","GPU"] and num_steps must be
# set, otherwise /start_profile returns 200 but records nothing and /stop_profile
# later reports "Profiling is not in progress".
set -u
cd /data/nvme/sglang-codex
. ./env.sh
STEPS="${1:-12}"
WORDS="${2:-40}"
NEWTOK="${3:-24}"
TAG="${4:-dec}"
PY=/data/nvme/sglang-codex/.venv/bin/python
OUT=/data/nvme/sglang-codex/profiles/$(date +%Y%m%d-%H%M%S)-$TAG
mkdir -p "$OUT"
echo "OUT=$OUT"
PROMPT=$("$PY" -c "print('history of the roman empire ' * $WORDS)")
"$PY" - "$OUT" "$STEPS" "$PROMPT" "$NEWTOK" <<'PY'
import json, sys, time, urllib.request
out_dir, steps, prompt, newtok = (
    sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
def post(path, body=None):
    data = json.dumps(body).encode() if body is not None else b"{}"
    req = urllib.request.Request("http://127.0.0.1:30000" + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=900).read().decode()
try:
    post("/stop_profile")
except Exception:
    pass
post("/generate", {"text": prompt,
                   "sampling_params": {"temperature": 0.0, "max_new_tokens": 4}})
print(post("/start_profile", {
    "output_dir": out_dir, "activities": ["CPU", "GPU"],
    "profile_by_stage": True, "record_shapes": True,
    "with_stack": False, "num_steps": steps, "profile_prefix": "p"}), flush=True)
t0 = time.perf_counter()
o = json.loads(post("/generate", {"text": prompt,
                                  "sampling_params": {"temperature": 0.0,
                                                      "max_new_tokens": newtok}}))
print(f"prompt={o['meta_info']['prompt_tokens']} wall={time.perf_counter()-t0:.2f}s",
      flush=True)
time.sleep(20)
try:
    print(post("/stop_profile"))
except Exception as e:
    print("stop:", type(e).__name__)
PY
sleep 8
ls "$OUT" 2>/dev/null | head -4
