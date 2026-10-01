#!/bin/bash
# usage: prof_dec_stack.sh
# Name the decode kernels that are real work, by Python call site and with shapes.
# The true DECODE ranking (dec_top.py, 8 stages, 78 ms device time per token) is:
#   nccl SendRecv 25.7%  (PP comm, structural)
#   per-head attn 16.4%  (correct kernel at bs=1)
#   cuBLAS gemvx  10.1%  672 calls at 44.3 us -- a tiny GEMV costing 4x its bytes
#   w4a16_v3       9.8%
#   _w8a16_gemv    9.3%
#   elementwise   ~16.3% across ~24000 calls
# The gemvx is the surprise: every dense projection in this model goes through
# W8A16 (_w8a16_gemv_kernel), so something is falling back to cuBLAS. Candidates
# are the ReplicatedLinear/compressor/router layers that get quant_config=None and
# stay bf16. Capture CPU+GPU with shapes and stacks so gemvx and the biggest
# elementwise families can be attributed to a line of sglang code.
set -u
cd /data/nvme/sglang-codex
. ./env.sh
OUT=profiles/dst-$(date +%s)
mkdir -p $OUT
timeout 900 .venv/bin/python - <<PY
import json, time, urllib.request
import concurrent.futures as cf
def post(path, payload=None):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request("http://127.0.0.1:30000"+path, data=data,
                                 headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(req, timeout=900).read().decode()
try: post("/stop_profile")
except Exception: pass
post("/generate", {"text":"warm up please ", "sampling_params":
                   {"temperature":0.0,"max_new_tokens":8}})
print(post("/start_profile", {"output_dir":"$OUT","activities":["CPU","GPU"],
      "profile_by_stage":True,"record_shapes":True,"with_stack":True,
      "num_steps":3,"profile_prefix":"ds"}), flush=True)
def one(i):
    return json.loads(post("/generate", {"text":"Describe the process of photosynthesis in detail %d" % i,
          "sampling_params":{"temperature":0.0,"max_new_tokens":40}}))["meta_info"]
with cf.ThreadPoolExecutor(max_workers=2) as ex:
    ms = list(ex.map(one, range(2)))
print("completion:", [m["completion_tokens"] for m in ms], flush=True)
time.sleep(12)
try: print(post("/stop_profile"))
except Exception as e: print("stop:", type(e).__name__)
PY
sleep 8
echo "OUTDIR=$OUT"
