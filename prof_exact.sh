#!/bin/bash
# usage: prof_exact.sh <TOKENS>
# Capture a decode profile whose step count is known by construction, to settle the
# per-stage device work per token. Every prior estimate hinged on how many decode
# steps a trace holds, and the per-layer kernel counts disagree: hc_split_sinkhorn
# shows 88 calls over 11 layers (=8 steps if once per layer, 4 if twice), and
# attention shows 88 calls but layers with compress_ratios 0 make only one call per
# step while the rest make two. The report's headline budget -- "each stage does
# 9.75 ms of device work per token, so 4 stages serialize to 39.0 ms, so single-stream
# 50 tok/s is unreachable" -- is wrong by whatever factor this is off, and 2x is the
# difference between "impossible" and "reachable".
#
# So: drive exactly ONE request for exactly N tokens. bs=1 throughout, so the number
# of decode steps is exactly N (minus the prefill). Then per-stage device work per
# token = busy / N, measured rather than inferred.
set -u
cd /data/nvme/sglang-codex
. ./env.sh
N=${1:-32}
OUT=profiles/exact-$(date +%s)
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
post("/generate", {"text":"warm up ", "sampling_params":{"temperature":0.0,"max_new_tokens":4}})
print(post("/start_profile", {"output_dir":"$OUT","activities":["GPU"],
      "profile_by_stage":True,"num_steps":400,"profile_prefix":"ex"}), flush=True)
j = json.loads(post("/generate", {"text":"Explain how a steam turbine works in detail. ref=%f" % time.time(),
      "sampling_params":{"temperature":0.0,"max_new_tokens":$N}}))
print("completion tokens:", j["meta_info"]["completion_tokens"], flush=True)
time.sleep(10)
try: print(post("/stop_profile"))
except Exception as e: print("stop:", type(e).__name__)
PY
sleep 8
echo "OUTDIR=$OUT"
