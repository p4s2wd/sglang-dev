#!/bin/bash
# usage: prof_cpu2.sh
# Name the elementwise kernels by Python call site. The earlier attempt passed
# activities=["GPU"] with with_stack=True and got no python_function events at
# all -- torch only records Python frames when CPU activity is enabled. Enable
# both. Elementwise work is 13.6% of prefill device time over ~4800 launches, and
# the fused-vs-eager norm probe shows a 3-9x penalty for eager, so identifying
# which sites are eager is worth more than guessing.
set -u
cd /data/nvme/sglang-codex
. ./env.sh
OUT=profiles/cs2-$(date +%s)
mkdir -p $OUT
timeout 900 .venv/bin/python - <<PY
import json, time, urllib.request
def post(path, payload=None):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request("http://127.0.0.1:30000"+path, data=data,
                                 headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(req, timeout=900).read().decode()
try: post("/stop_profile")
except Exception: pass
post("/generate", {"text":"warm ", "sampling_params":{"temperature":0.0,"max_new_tokens":2}})
print(post("/start_profile", {"output_dir":"$OUT","activities":["CPU","GPU"],
      "profile_by_stage":True,"record_shapes":False,"with_stack":True,
      "num_steps":2,"profile_prefix":"cs"}), flush=True)
for i in range(2):
    o = json.loads(post("/generate", {"text":"salt%d %s" % (i, " ".join(["word"]*11000)),
          "sampling_params":{"temperature":0.0,"max_new_tokens":1}}))
    print("prompt=%s" % o["meta_info"]["prompt_tokens"], flush=True)
time.sleep(10)
try: print(post("/stop_profile"))
except Exception as e: print("stop:", type(e).__name__)
PY
sleep 8
echo "OUTDIR=$OUT"
