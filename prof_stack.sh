#!/bin/bash
# usage: prof_stack.sh
# Elementwise/copy kernels are 13.6% of prefill device time across ~4800 launches,
# but the earlier ranking truncated kernel names to 58 chars, so one
# at::native::vectorized_elementwise_kernel line covers dozens of distinct lambdas
# and cannot name a fusion target. Prefill runs eager (no CUDA graph), so a
# with_stack capture attributes each kernel to its Python call site.
set -u
cd /data/nvme/sglang-codex
OUT=profiles/stk-$(date +%s)
mkdir -p $OUT
. ./env.sh
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
print(post("/start_profile", {"output_dir":"$OUT","activities":["GPU"],
      "profile_by_stage":True,"record_shapes":False,"with_stack":True,
      "num_steps":2,"profile_prefix":"st"}), flush=True)
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
ls $OUT | head -4
