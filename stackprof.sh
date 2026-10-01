#!/bin/bash
# Profile with Python stacks so the per-layer element-wise kernels can be traced
# to the model code that launches them. Decode's floor is 4 PP stages x 17.13 ms
# and 5.28 ms of that is small kernels, but CUDA-graph capture erases the CPU-side
# op name, so the only way to attribute them is the Python stack.
set -u
OUT="/data/nvme/sglang-codex/profiles/stack-$(date +%s)"
mkdir -p "$OUT"
cd /data/nvme/sglang-codex
.venv/bin/python - <<PY
import json, time, urllib.request
def post(path, body=None):
    data = json.dumps(body).encode() if body is not None else b"{}"
    req = urllib.request.Request("http://127.0.0.1:30000"+path, data=data,
                                 headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(req, timeout=900).read().decode()
try: post("/stop_profile")
except Exception: pass
post("/generate", {"text":"Warm up. ", "sampling_params":{"temperature":0.0,"max_new_tokens":4}})
print(post("/start_profile", {"output_dir":"$OUT","activities":["CPU","GPU"],
      "profile_by_stage":True,"record_shapes":False,"with_stack":True,
      "num_steps":3,"profile_prefix":"s"}), flush=True)
o = json.loads(post("/generate", {"text":"Explain pipeline parallelism briefly. ",
      "sampling_params":{"temperature":0.0,"max_new_tokens":24}}))
print("prompt=%s wall ok" % o["meta_info"]["prompt_tokens"], flush=True)
time.sleep(15)
try: print(post("/stop_profile"))
except Exception as e: print("stop:", type(e).__name__)
PY
sleep 8
ls "$OUT" | head -4
echo "OUTDIR=$OUT"
