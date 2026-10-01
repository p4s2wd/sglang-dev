#!/bin/bash
# Server-side prefill profile. The first attempt ran torch.profiler inside the
# client process and captured zero kernels -- the kernels live in the scheduler
# workers, so the profile has to be requested from the server.
set -u
OUT="/data/nvme/sglang-codex/profiles/pf2-$(date +%s)"
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
post("/generate", {"text":"warm ", "sampling_params":{"temperature":0.0,"max_new_tokens":2}})
print(post("/start_profile", {"output_dir":"$OUT","activities":["GPU"],
      "profile_by_stage":True,"record_shapes":False,"with_stack":False,
      "num_steps":4,"profile_prefix":"pf"}), flush=True)
for i in range(3):
    o = json.loads(post("/generate", {"text":"salt%d %s" % (i, " ".join(["word"]*13000)),
          "sampling_params":{"temperature":0.0,"max_new_tokens":1}}))
    print("prompt=%s" % o["meta_info"]["prompt_tokens"], flush=True)
time.sleep(12)
try: print(post("/stop_profile"))
except Exception as e: print("stop:", type(e).__name__)
PY
sleep 8
ls "$OUT" | head -6
echo "OUTDIR=$OUT"
