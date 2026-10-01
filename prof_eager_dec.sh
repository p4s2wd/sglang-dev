#!/bin/bash
# usage: prof_eager_dec.sh
# Capture decode with CUDA graphs DISABLED so its kernels carry CPU ops and Python
# frames, to attribute the elementwise work that cannot be attributed otherwise.
#
# With graphs on, 4269 of 4330 decode kernels have no External id, so no CPU op and no
# stack -- the join against an eager PREFILL trace only recovers ops whose (name,grid,
# block) triple also occurs in prefill, and decode-shaped ops (batch 1-2 rows) do not.
# That left 5.73 ms/token of the 9.50 ms/token elementwise family UNMATCHED, the
# largest unattributed item in decode.
#
# Running decode eager costs throughput, but attribution is what is being measured
# here, not speed. Same request shape as the graphed capture so the call counts line up.
set -u
cd /data/nvme/sglang-codex
. ./env.sh
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 50
for i in $(seq 1 15); do L=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$L" = "0" ] && break; sleep 10; done
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 2 4 eagerdec --disable-cuda-graph >/dev/null 2>&1 </dev/null &
for i in $(seq 1 70); do sleep 10; grep -qaE "fired up|out of memory|Traceback" logs/eagerdec.log 2>/dev/null && break; done
grep -qaE "fired up" logs/eagerdec.log || { echo "SERVER FAILED"; grep -aE "out of memory|Traceback" logs/eagerdec.log|tail -2|cut -c1-140; exit 1; }
OUT=profiles/eagerdec-$(date +%s)
mkdir -p $OUT
timeout 900 .venv/bin/python - <<PY
import json, time, urllib.request
def post(p, pl=None):
    r = urllib.request.Request("http://127.0.0.1:30000"+p, data=json.dumps(pl or {}).encode(),
                               headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(r, timeout=900).read().decode()
try: post("/stop_profile")
except Exception: pass
post("/generate", {"text":"warm ", "sampling_params":{"temperature":0.0,"max_new_tokens":4}})
print(post("/start_profile", {"output_dir":"$OUT","activities":["CPU","GPU"],
      "profile_by_stage":True,"record_shapes":True,"with_stack":True,
      "num_steps":200,"profile_prefix":"eg"}), flush=True)
j = json.loads(post("/generate", {"text":"Explain how a steam turbine works in detail. ref=%f" % time.time(),
      "sampling_params":{"temperature":0.0,"max_new_tokens":24}}))
print("completion:", j["meta_info"]["completion_tokens"], flush=True)
time.sleep(10)
try: print(post("/stop_profile"))
except Exception as e: print("stop:", type(e).__name__)
PY
sleep 8
echo "OUTDIR=$OUT"
